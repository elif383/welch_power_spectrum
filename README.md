# welch_power_spectrum

Computes the power spectral density using Welch's method of windowed, overlapping segments.

## Usage

```python
from welch_power_spectrum import welch_psd, get_frequencies

# A 5 Hz sine sampled at 100 Hz for 4 seconds
import math
fs = 100.0
data = [math.sin(2 * math.pi * 5.0 * i / fs) for i in range(400)]

freqs, psd = welch_psd(data, fs=fs, nperseg=64)
peak_bin = psd.index(max(psd))
print(f"Peak at {freqs[peak_bin]:.1f} Hz")  # Peak at 5.0 Hz

# Just the frequency axis for a given FFT length and sample rate
print(get_frequencies(256, fs))
```

## Why this exists

This library provides a zero-dependency, standard-library-only implementation of Welch's PSD estimator. The specific problem it solves: you need a power spectral density estimate in an environment where installing NumPy or SciPy is not possible or not desirable — locked-down CI, minimal containers, audit-sensitive deployments.

The trade-off is speed. The DFT is computed directly in O(N^2) rather than using a fast FFT. For moderate `nperseg` (a few hundred samples) and moderate-length inputs this is fine. For large `nfft` or long signals, use a library with a real FFT implementation.

## Edge cases you will hit

- **Even vs. odd `nfft`**: the returned one-sided spectrum has `nfft // 2 + 1` bins. For even `nfft`, the last bin is the Nyquist frequency and its power is *not* doubled (there is no folded negative-frequency counterpart). For odd `nfft`, the last bin is below Nyquist and *is* doubled.
- **`nfft < nperseg`**: the segment is truncated to `nfft` samples *after* windowing over the full `nperseg` samples. This means the window normalization still uses the full window, but only `nfft` samples enter the DFT.
- **Data shorter than `nperseg`**: a single segment is used and zero-padded to `nperseg`. No error is raised.
- **Per-segment detrending**: `detrend='constant'` and `detrend='linear'` operate on each segment independently, not on the global signal. This is standard Welch behaviour but differs from detrending the whole signal upfront.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

