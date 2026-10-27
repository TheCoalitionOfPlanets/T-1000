"""Experimental SOAR layer offload with original, unquantized weights.

Offloaded transformer blocks live in CPU RAM and execute on CUDA. Accelerate
copies each block's weights to the GPU for its forward pass, then releases them.
This saves resident VRAM at the cost of repeated CPU-to-GPU transfers.
"""

from pathlib import Path

import torch
from torch import nn


class _OffloadedModel(nn.Module):
    """Keep the runtime's normal API while staging its one device transfer."""

    def __init__(self, model, *, gpu_layers, cpu_offload):
        super().__init__()
        self._wrapped = model
        self._cpu_offload = cpu_offload
        self._gpu_layers = gpu_layers
        self._placed_device = None

        llm = model.core.llm
        base = llm.model
        layers = base.layers
        if not isinstance(layers, nn.ModuleList) or not layers:
            raise RuntimeError("SOAR offload requires Qwen transformer decoder blocks.")
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, int):
            raise ValueError("gpu_layers must be an integer.")
        if not 0 <= gpu_layers <= len(layers):
            raise ValueError(f"gpu_layers must be between 0 and {len(layers)}.")

        # SOAR's output head aliases its input embedding. Keep that alias when
        # the embedding gets a new CUDA Parameter; never copy both separately.
        llm.tie_weights()
        if llm.get_output_embeddings().weight is not llm.get_input_embeddings().weight:
            raise RuntimeError("SOAR offload requires tied input/output embeddings.")
        if llm.get_output_embeddings().bias is not None:
            raise RuntimeError("SOAR offload requires a bias-free tied output head.")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("_wrapped"), name)

    def to(self, device):
        """Place modules without ever moving the complete model to CUDA."""
        target = torch.device(device)
        if target.type != "cuda":
            raise RuntimeError("The layer-offload profile requires a CUDA GPU.")
        if self._placed_device is not None:
            if target != self._placed_device:
                raise RuntimeError("Reload the offloaded runtime to change its CUDA device.")
            return self

        model = self._wrapped
        llm = model.core.llm
        base = llm.model
        layers = base.layers

        # Runtime has already cast core weights on CPU. Acoustic modules remain
        # resident on the GPU; the vocoder/speaker retain their normal FP32.
        for child in model.core.children():
            if child is not llm:
                child.to(target)
        model.vocoder.to(target)
        model.xvector_extractor.to(target)

        for child in base.children():
            if child is not layers:
                child.to(target)
        llm.tie_weights()
        if llm.get_output_embeddings().weight is not llm.get_input_embeddings().weight:
            raise RuntimeError("SOAR offload could not preserve tied embeddings.")
        head = llm.get_output_embeddings()
        for child in llm.children():
            if child is not base and child is not head:
                child.to(target)

        for index, block in enumerate(layers):
            if index < self._gpu_layers:
                block.to(target)
            else:
                self._cpu_offload(
                    block, execution_device=target, offload_buffers=True
                )

        self._placed_device = target
        return self


def load_offloaded_runtime(model_path, *, precision, gpu_layers=0):
    """Load SOAR with selected resident layers and CPU-backed remaining blocks.

    ``gpu_layers=0`` minimizes resident language-model weights; on the bundled
    28-layer model, ``gpu_layers=14`` retains half the decoder blocks on CUDA.
    Neither option guarantees a memory ceiling or faster inference.
    """
    if precision not in {"bfloat16", "float16"}:
        raise ValueError("The offload profile requires bfloat16 or float16 precision.")
    if not torch.cuda.is_available():
        raise RuntimeError("The layer-offload profile requires a CUDA GPU.")
    try:
        from accelerate import cpu_offload
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The layer-offload profile needs Accelerate. Install its dependencies "
            "with: .venv/bin/python -m pip install -r requirements/gpu-low-vram.txt"
        ) from exc

    from dots_tts.models.dots_tts.model import DotsTtsModel
    from dots_tts.runtime import DotsTtsRuntime

    model_path = Path(model_path)
    # Runtime's mutable decoding caches require normal tensors even if a caller
    # happens to enter inference_mode before loading the model.
    with torch.inference_mode(False):
        with torch.device("cpu"):
            model = DotsTtsModel.from_pretrained(model_path)
            model = _OffloadedModel(
                model, gpu_layers=gpu_layers, cpu_offload=cpu_offload
            )
        return DotsTtsRuntime(
            model,
            model_path,
            precision=precision,
            optimize=False,
            max_generate_length=128,
            max_sequence_length=512,
            vocoder_merge_steps=1,
            warmup_on_optimize=False,
        )
