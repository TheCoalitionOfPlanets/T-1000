"""Conservative, local cleanup for reference voices, without model downloads.

The spectral gate learns stationary background from quiet parts of the selected
audio. It is not speaker isolation: overlapping voices or music need a cleaner
recording. Cropping is sample-exact and happens before estimating any noise.
"""

import math

import numpy as np
from scipy import ndimage, signal


PROCESSING_VERSION = "reference-cleanup-v1"


def _rms(samples):
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _quiet_edges(samples, sample_rate):
    """Remove only quiet outer edges, retaining 120 ms around detected audio."""
    width = max(1, round(sample_rate * 0.01))
    starts = np.arange(0, len(samples), width)
    energy = np.add.reduceat(np.square(samples, dtype=np.float64), starts)
    counts = np.minimum(width, len(samples) - starts)
    levels = np.sqrt(energy / counts)
    high = float(np.percentile(levels, 90))
    low = float(np.percentile(levels, 10))
    # A recording without a distinct quiet floor should keep its full extent.
    threshold = max(1e-5, high * 0.015)
    if low < high * 0.35:
        threshold = max(threshold, low * 2)
    active = np.flatnonzero(levels > threshold)
    if not active.size:
        raise ValueError("The selected audio is silent or too quiet. Keep a clear spoken sentence.")
    padding = round(sample_rate * 0.12)
    start = max(0, int(starts[active[0]]) - padding)
    end = min(len(samples), int(starts[active[-1]]) + width + padding)
    return samples[start:end]


def _reduce_noise(samples, sample_rate):
    """Suppress steady hiss/hum conservatively; never boost spectral energy."""
    # Remove DC and subsonic rumble while retaining the fundamental of low voices.
    sos = signal.butter(2, 45, btype="highpass", fs=sample_rate, output="sos")
    values = signal.sosfiltfilt(sos, samples).astype(np.float32)
    width = 2 ** math.ceil(math.log2(sample_rate * 0.032))
    width = min(width, len(values))
    hop = max(1, width // 4)
    frequencies, times, spectrum = signal.stft(
        values, fs=sample_rate, window="hann", nperseg=width,
        noverlap=width - hop, boundary="zeros", padded=True,
    )
    power = np.abs(spectrum) ** 2
    # Exclude padded boundary frames from background estimation.
    interior = (times >= width / (2 * sample_rate)) & (
        times <= (len(values) - width / 2) / sample_rate
    )
    levels = np.sqrt(np.sum(power, axis=0))
    real_levels = levels[interior]
    if real_levels.size >= 4:
        quiet = float(np.percentile(real_levels, 15))
        loud = float(np.percentile(real_levels, 85))
        # This guard keeps a sustained vowel or continuous speech from becoming
        # its own noise profile. In that case only subsonic cleanup is applied.
        if quiet < loud * 0.35:
            background = interior & (levels <= quiet)
            if np.count_nonzero(background) >= 3:
                noise = np.mean(power[:, background], axis=1, keepdims=True)
                voice_band = np.maximum(noise[frequencies >= 80], 1e-30)
                flatness = float(np.exp(np.mean(np.log(voice_band))) / np.mean(voice_band))
                # Quiet but strongly tonal frames can be sustained or whispered
                # phonemes. Do not mistake those for a noise-only segment.
                if flatness >= 0.02:
                    gain = np.sqrt(np.maximum(0.16 ** 2, 1 - 1.8 * noise / np.maximum(power, 1e-30)))
                    # Smooth gain changes to avoid isolated musical-noise bins.
                    gain = ndimage.uniform_filter(gain, size=(3, 5), mode="nearest")
                    spectrum *= gain
    _, cleaned = signal.istft(
        spectrum, fs=sample_rate, window="hann", nperseg=width,
        noverlap=width - hop, boundary=True,
    )
    return cleaned[:len(samples)]


def process_reference(
    samples: np.ndarray,
    sample_rate: int,
    *,
    start_seconds: float = 0,
    end_seconds: float | None = None,
    noise_reduction: bool = True,
    auto_trim: bool = False,
    min_seconds: float = 1.0,
) -> np.ndarray:
    """Crop and optionally clean mono reference audio at its original rate.

    Times refer to the supplied source; the end is exclusive, rounded to the
    nearest sample. No gain normalization or amplification is performed. Quiet
    edge trimming is explicit because removing spoken words changes transcripts.
    The caller should transcribe and validate the returned audio before use.
    """
    try:
        values = np.asarray(samples, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("The reference contains invalid audio samples.") from exc
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Use nonempty, finite mono reference audio.")
    if (isinstance(sample_rate, (bool, np.bool_))
            or not isinstance(sample_rate, (int, np.integer))
            or not 8000 <= sample_rate <= 192000):
        raise ValueError("Use a sample rate between 8 kHz and 192 kHz.")
    if not isinstance(noise_reduction, (bool, np.bool_)) or not isinstance(auto_trim, (bool, np.bool_)):
        raise ValueError("Noise reduction and automatic trimming must be enabled or disabled.")
    if (isinstance(min_seconds, (bool, np.bool_))
            or not isinstance(min_seconds, (int, float, np.integer, np.floating))
            or not math.isfinite(min_seconds) or min_seconds <= 0):
        raise ValueError("The minimum recording duration must be finite and positive.")
    duration = len(values) / sample_rate
    end_seconds = duration if end_seconds is None else end_seconds
    for bound in (start_seconds, end_seconds):
        if (isinstance(bound, (bool, np.bool_))
                or not isinstance(bound, (int, float, np.integer, np.floating))
                or not math.isfinite(bound)):
            raise ValueError("Crop start and end must be finite times in seconds.")
    if start_seconds < 0 or end_seconds > duration or start_seconds >= end_seconds:
        raise ValueError("Keep a crop with 0 ≤ start < end within the reference recording.")
    start = round(start_seconds * sample_rate)
    end = round(end_seconds * sample_rate)
    values = values[start:end].copy()
    minimum = math.ceil(min_seconds * sample_rate)
    if len(values) < minimum:
        raise ValueError(f"Keep at least {min_seconds:g} seconds of speech after cropping or trimming.")
    if _rms(values) < 1e-5 or float(np.max(np.abs(values))) < 1e-4:
        raise ValueError("The selected audio is silent or too quiet. Keep a clear spoken sentence.")
    if noise_reduction:
        values = _reduce_noise(values, sample_rate)
    if auto_trim:
        values = _quiet_edges(values, sample_rate)
    if len(values) < minimum:
        raise ValueError(f"Keep at least {min_seconds:g} seconds of speech after cropping or trimming.")
    if not np.isfinite(values).all() or _rms(values) < 1e-5:
        raise ValueError("The cleaned audio is silent or too quiet. Try another crop or disable noise reduction.")
    return np.ascontiguousarray(values, dtype=np.float32)
