"""Isolated, opt-in SOAR vocoder experiments for measured CUDA comparisons.

Nothing imports or enables these adapters during ordinary application startup.
Use a fresh runtime for each experiment: precision conversion is intentionally
not reversible here, because casting weights back would retain rounded values.
Every candidate keeps the acoustic sampler and reference processing unchanged.
"""

import torch

from .vocoder import BufferedVocoderInference


VARIANTS = (
    "baseline", "native-lstm", "compiled-vocoder", "decoder-bf16",
    "decoder-fp16", "batch8",
)


class EightPatchVocoderInference(BufferedVocoderInference):
    """Experimental group size using the already validated stream/flush logic."""

    def __init__(self, inner, *, initial_unmerged=2, model=None):
        # The production constructor deliberately only accepts one or four.
        # This opt-in subclass changes that one invariant without widening the
        # production environment variable or copying the buffering algorithm.
        super().__init__(
            inner, batch_size=4, initial_unmerged=initial_unmerged, model=model,
        )
        self.batch_size = 8


def _unwrap(adapter):
    while isinstance(adapter, BufferedVocoderInference):
        adapter = adapter.inner
    return adapter


def _configure_native_lstm(inner):
    recurrent = inner.vocoder.dec_mi_layer[1]
    if recurrent.bidirectional or recurrent.lstm.bidirectional:
        raise ValueError("The native streaming LSTM must be unidirectional.")
    if not recurrent.lstm.batch_first:
        raise ValueError("The native streaming LSTM requires batch_first=True.")
    if recurrent.lstm.training and recurrent.lstm.dropout:
        raise ValueError("The native streaming LSTM requires inference mode.")

    def native_step(x, hidden):
        if hidden is None:
            hidden = inner._init_lstm_stream_state(x.size(0))
        # Preserve upstream hidden/cell layout [layers, batch, hidden] and
        # SLSTM's skip connection. Native CUDA dispatch may use cuDNN; this is
        # not a promise of speed or bitwise equality versus manual recurrence.
        output, next_hidden = recurrent.lstm(x, hidden)
        if recurrent.skip:
            output = output + x
        return output, next_hidden

    inner._lstm_stream_step = native_step


def configure_vocoder_experiment(runtime, variant):
    """Mutate one eager runtime and return JSON-serializable configuration.

    The caller must run memory, identical-latent, and actual speech checks.
    Compiler startup/warmup belongs outside warm generation measurements.
    Exceptions are propagated; there is no silent fallback to baseline.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown vocoder experiment {variant!r}; choose {VARIANTS}.")
    model = runtime.model
    if runtime.optimize or model._optimize_enabled:
        raise ValueError("Vocoder experiments require the eager runtime.")
    adapter = model._get_vocoder_inference()
    inner = _unwrap(adapter)
    metadata = {"variant": variant, "acoustic_sampler_changed": False}

    if variant == "native-lstm":
        _configure_native_lstm(inner)
        metadata["recurrent_backend"] = "native torch.nn.LSTM; CUDA backend chosen by PyTorch"
    elif variant == "compiled-vocoder":
        # Compile only the fixed-window convolutional decoder. In particular,
        # do not enable upstream optimize or allocate its full DiT KV cache.
        options = {"triton.cudagraphs": False}
        inner._decode_stream_window = torch.compile(
            inner._decode_stream_window, backend="inductor", dynamic=False,
            fullgraph=False, options=options,
        )
        metadata["compile"] = {
            "target": "VocoderInference._decode_stream_window",
            "backend": "inductor", "dynamic": False, "fullgraph": False,
            "options": options,
        }
    elif variant in {"decoder-bf16", "decoder-fp16"}:
        precision = torch.bfloat16 if variant == "decoder-bf16" else torch.float16
        # Upstream _decode_stream_latents and _decode_stream_window already
        # cast only decoder inputs to conv_pre's dtype. Reference encoding,
        # post projection and recurrent state stay in their original precision.
        inner.vocoder.decoder.to(dtype=precision)
        metadata["precision_scope"] = "vocoder.decoder parameters and floating buffers only"
    elif variant == "batch8":
        adapter = EightPatchVocoderInference(
            inner, initial_unmerged=model.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES,
            model=model,
        )
        model._vocoder_inference = adapter
        model._vocoder_inference_vocoder_id = id(model.vocoder)

    decoder = next(inner.vocoder.decoder.parameters())
    metadata.update(
        decoder_dtype=str(decoder.dtype),
        decoder_device=str(decoder.device),
        vocoder_batch_size=getattr(adapter, "batch_size", 1),
        first_unmerged_patches=getattr(adapter, "initial_unmerged", None),
        uses_upstream_optimized_mode=False,
    )
    return metadata
