"""Experimental SOAR loading with NF4 weights in the language model only.

This reduces weight memory; a particular GPU's fit, speed, and speech quality
still require measurement. Acoustic and speaker modules retain their normal
runtime precision, and the larger compiled inference caches stay disabled.
"""

from pathlib import Path

import torch
from torch import nn

from .vocoder import configure_vocoder_batching, read_vocoder_batch_size


def _replace_linears(module, linear4bit, compute_dtype):
    """Replace CPU language-model projections before any CUDA transfer."""
    replaced = 0
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            quantized = linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                compress_statistics=True,
                quant_type="nf4",
                device="cpu",
            )
            # Keep Params4bit supplied by the constructor: replacing it with a
            # plain Parameter would lose the lazy quantization metadata.
            quantized.load_state_dict(child.state_dict())
            quantized.requires_grad_(False)
            quantized.eval()
            setattr(module, name, quantized)
            replaced += 1
        else:
            replaced += _replace_linears(child, linear4bit, compute_dtype)
    return replaced


def load_low_vram_runtime(model_path, *, precision):
    """Load the experimental LLM-only NF4 profile without dense CUDA weights.

    The full checkpoint is initially loaded into CPU RAM. Runtime construction
    casts the still-unquantized core to its compute dtype, then moves modules
    to CUDA; each Params4bit weight quantizes during that device transfer.
    """
    compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
    if compute_dtype is None:
        raise ValueError("The low-VRAM profile requires bfloat16 or float16 precision.")
    vocoder_batch_size = read_vocoder_batch_size()

    try:
        from bitsandbytes.nn import Linear4bit
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The low-VRAM profile needs bitsandbytes. Install its dependencies "
            "with: .venv/bin/python -m pip install -r requirements/gpu-low-vram.txt"
        ) from exc

    from dots_tts.models.dots_tts.model import DotsTtsModel
    from dots_tts.modules.backbone.dit_inference import fuse_dit_for_inference
    from dots_tts.runtime import DotsTtsRuntime

    model_path = Path(model_path)
    # Lazy inference adapters contain mutable tensors: create ordinary tensors
    # even when the caller enters inference_mode before loading the runtime.
    with torch.inference_mode(False):
        with torch.device("cpu"):
            model = DotsTtsModel.from_pretrained(model_path)
            # Restrict quantization to transformer blocks. Embeddings and the
            # tied output head, plus acoustic/speaker precision, stay unchanged.
            replaced = _replace_linears(model.core.llm.model.layers, Linear4bit, compute_dtype)
            if not replaced:
                raise RuntimeError("The SOAR language model has no supported linear layers to quantize.")
            # Upstream otherwise fuses on the first decode, temporarily holding
            # original adaLN weights, a fused Linear and its concatenated copy
            # on CUDA. Fuse on CPU so a 4-GB GPU only receives the final weights.
            fuse_dit_for_inference(model.core)

        # Do not pre-quantize weights or call .to(device='cpu') on Params4bit.
        # A later dtype cast of already packed weights corrupts their storage.
        runtime = DotsTtsRuntime(
            model,
            model_path,
            precision=precision,
            optimize=False,
            max_generate_length=128,
            max_sequence_length=512,
            vocoder_merge_steps=1,
            warmup_on_optimize=False,
        )
        return configure_vocoder_batching(runtime, vocoder_batch_size)
