"""Opt-in, forward-scoped sharing of eager DiT rotary position tensors.

The unchanged upstream rotary module computes the same float32 frequencies,
cosines and sines once for compatible layers in a single DiT forward. Nothing
is retained across Euler calls: inference-mode position tensors have no version
counter, so caching them across calls would need an unsafe immutability promise
or a synchronizing comparison. No model weights or solver arithmetic change.
"""

import hashlib
import weakref

import torch


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        return None


def _signature(rotary):
    frequency = rotary.inv_freq
    return (id(frequency), _version(frequency), tuple(frequency.shape),
            frequency.dtype, frequency.device, rotary.dim, rotary._theta)


class _RotaryBlocks:
    """Own CPU descriptors; all actual GPU rotary tensors are local to a call."""

    def __init__(self, dit):
        from dots_tts.modules.backbone.layers import RotaryEmbedding

        self._dit = weakref.ref(dit)
        self._original = dit.run_blocks
        self._descriptors = {}
        self.stats = {"forwards": 0, "rotary_computations": 0,
                      "rotary_layer_reuses": 0, "uncacheable_layers": 0}
        for block in dit.blocks:
            attention = block.attn
            if not attention.rotary_bias:
                continue
            rotary = attention.rotary
            # Restrict grouping to the pinned upstream implementation. A
            # subclass may have behavior not represented by inv_freq alone.
            if type(rotary) is not RotaryEmbedding:
                continue
            signature = _signature(rotary)
            if signature[1] is None or rotary.inv_freq.dtype != torch.float32:
                continue
            values = rotary.inv_freq.detach().cpu().contiguous().numpy().tobytes()
            group = (rotary.dim, rotary._theta, tuple(rotary.inv_freq.shape),
                     hashlib.sha256(values).digest())
            self._descriptors[id(rotary)] = (signature, group)

    def __call__(self, *, x, all_mods, attn_mask=None, pos_ids=None):
        from dots_tts.modules.backbone.inference_utils import project_attention

        dit = self._dit()
        if dit is None:
            raise RuntimeError("The rotary reuse adapter outlived its DiT.")
        self.stats["forwards"] += 1
        # Only this call shares positions. A different/mutated tensor passed to
        # the next Euler call is always recomputed, including inference tensors.
        positions = pos_ids
        rotary_tensors = {}
        sdpa_mask = attn_mask
        if sdpa_mask is not None:
            if sdpa_mask.ndim == 2:
                sdpa_mask = sdpa_mask[:, None, None, :]
            elif sdpa_mask.ndim == 3:
                sdpa_mask = sdpa_mask[:, None, :, :]

        def attention(_layer_index, block, attn_in):
            nonlocal positions
            module = block.attn
            # Preserve the upstream failure/dispatch behavior for flex
            # attention, which requires a BlockMask the eager path lacks.
            if module.attn_backend == "flex":
                return module(attn_in, mask=attn_mask, pos_ids=pos_ids)
            cosine = sine = None
            if module.rotary_bias:
                if positions is None:
                    positions = torch.arange(attn_in.shape[1], device=attn_in.device,
                                             dtype=torch.float32)
                rotary = module.rotary
                descriptor = self._descriptors.get(id(rotary))
                shareable = (descriptor is not None
                             and descriptor[0] == _signature(rotary)
                             and rotary.inv_freq.device == positions.device)
                group = descriptor[1] if shareable else None
                if group is not None and group in rotary_tensors:
                    cosine, sine = rotary_tensors[group]
                    self.stats["rotary_layer_reuses"] += 1
                else:
                    frequencies = rotary(positions)
                    cosine, sine = frequencies.cos(), frequencies.sin()
                    self.stats["rotary_computations"] += 1
                    if group is not None:
                        rotary_tensors[group] = (cosine, sine)
                    else:
                        self.stats["uncacheable_layers"] += 1
            output, _key, _value = project_attention(
                module, attn_in, num_heads=module.num_heads,
                head_dim=module.head_dim, rotary_cos=cosine, rotary_sin=sine,
                attn_mask=sdpa_mask,
            )
            return output

        output, final_modulation = dit.run_modulated_blocks(
            x=x, all_mods=all_mods, attention=attention,
        )
        return dit.apply_final_layer(output, final_modulation)


def configure_rotary_cache(runtime, enabled=True):
    """Install once on a CPU-fused eager runtime; no default application change.

    ``runtime.model._t1000_rotary_reuse.stats`` exposes counters for diagnostics.
    Descriptor construction may copy tiny frequency buffers to CPU once; timed
    forwards perform only Python metadata checks and no CUDA scalar reads.
    Setting ``enabled=False`` restores this instance's original run_blocks.
    """
    if not isinstance(enabled, bool):
        raise ValueError("The rotary cache flag must be a boolean.")
    model = runtime.model
    existing = getattr(model, "_t1000_rotary_reuse", None)
    if not enabled:
        if existing is not None:
            dit = existing._dit()
            if dit is not None:
                dit.run_blocks = existing._original
            del model._t1000_rotary_reuse
        return runtime
    if runtime.optimize or model._optimize_enabled or existing is not None:
        return runtime
    from dots_tts.modules.backbone.dit_inference import FusedAdaLNDiT

    dit = model.core.velocity_field_predictor
    if not isinstance(dit, FusedAdaLNDiT):
        raise RuntimeError("Rotary reuse requires an already fused eager DiT.")
    adapter = _RotaryBlocks(dit)
    dit.run_blocks = adapter
    model._t1000_rotary_reuse = adapter
    return runtime
