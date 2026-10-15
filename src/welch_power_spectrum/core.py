"""Core implementation of Welch's power spectral density estimator.

Design decisions:
- Default window is Hann. Rectangular and Hamming are also supported.
  We implement the window directly rather than accepting a callable because
  the normalization of U (window energy) must be computed from the exact
  same samples used in the FFT, and accepting arbitrary callables makes it
  too easy for a caller to supply a window whose sum-of-squares disagrees
  with the samples they intended.
- The one-sided PSD is returned: the single-sided spectrum at DC and the
  positive frequencies up to (and including) the Nyquist when nfft is even.
  The non-DC/non-Nyquist bins are doubled to account for the folded negative
  frequencies. This is the convention used in most engineering texts.
- Segments shorter than nperseg are zero-padded up to nfft if nfft > nperseg;
  if nfft <= nperseg the segment is truncated to nfft samples before windowing.
- detrend='constant' subtracts the segment mean; detrend='linear' subtracts
  a least-squares line. These are applied per-segment, before windowing, so
  the mean removed is the segment's own mean, not the global mean. This is
  the standard Welch approach and prevents spectral leakage from a DC offset.
- We compute the DFT directly (O(N^2)) via the Goertzel-like formulation.
  This is intentionally slow for large inputs but keeps the library at zero
  third-party dependencies and makes the behaviour trivially auditable.
  For production workloads with large nfft you would want a real FFT; for
  the intended use cases (moderate nperseg, moderate sample counts) the cost
  is acceptable.
"""

import math


_WINDOW_REGISTRY = {
    "boxcar": lambda n, i: 1.0,
    "rectangular": lambda n, i: 1.0,
    "hann": lambda n, i: 0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1)) if n > 1 else 1.0,
    "hamming": lambda n, i: 0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1)) if n > 1 else 1.0,
}


def _get_window(name, n):
    if name not in _WINDOW_REGISTRY:
        raise ValueError(
            f"Unknown window '{name}'. Available: "
            + ", ".join(sorted(_WINDOW_REGISTRY))
        )
    if n < 1:
        raise ValueError("window length must be >= 1")
    func = _WINDOW_REGISTRY[name]
    return [func(n, i) for i in range(n)]


def _detrend_segment(seg, mode):
    n = len(seg)
    if n == 0:
        return seg
    if mode == "none":
        return list(seg)
    if mode == "constant":
        mean = sum(seg) / n
        return [v - mean for v in seg]
    if mode == "linear":
        # Least-squares line fit: slope and intercept.
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for i, y in enumerate(seg):
            sx += i
            sy += y
            sxx += i * i
            sxy += i * y
        denom = n * sxx - sx * sx
        if denom == 0:
            # All indices the same (only possible if n == 1); fall back to constant.
            mean = sy / n
            return [v - mean for v in seg]
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return [seg[i] - (intercept + slope * i) for i in range(n)]
    raise ValueError(f"Unknown detrend mode '{mode}'. Use 'none', 'constant', or 'linear'.")


def _dft(seg, nfft):
    """Compute the discrete Fourier transform of seg padded/truncated to nfft.

    Returns the complex spectrum as a list of (real, imag) tuples.
    Uses the direct definition: X[k] = sum_n x[n] e^{-j 2pi k n / N}.
    """
    n = len(seg)
    if n < nfft:
        seg = list(seg) + [0.0] * (nfft - n)
    elif n > nfft:
        seg = list(seg[:nfft])
    spectrum = []
    for k in range(nfft):
        real = 0.0
        imag = 0.0
        for nn in range(nfft):
            angle = 2.0 * math.pi * k * nn / nfft
            real += seg[nn] * math.cos(angle)
            imag += -seg[nn] * math.sin(angle)
        spectrum.append((real, imag))
    return spectrum


def _one_sided_psd(segment, window, nfft, fs, detrend):
    """Process one segment: detrend, window, DFT, and produce one-sided PSD.

    Returns a list of length (nfft // 2 + 1) of PSD values in (units^2 / Hz).
    """
    # 1. Truncate or pad the segment to the window length.
    nperseg = len(window)
    if len(segment) >= nperseg:
        seg = list(segment[:nperseg])
    else:
        # If the segment is shorter than the window, pad with zeros and
        # truncate the window to match (so window normalization is correct).
        seg = list(segment) + [0.0] * (nperseg - len(segment))

    # 2. Detrend before windowing.
    seg = _detrend_segment(seg, detrend)

    # 3. Apply window.
    win_sum_sq = sum(w * w for w in window)
    if win_sum_sq == 0:
        raise ValueError("Window is all zeros; cannot normalize.")
    windowed = [s * w for s, w in zip(seg, window)]

    # 4. DFT (pads to nfft internally).
    spectrum = _dft(windowed, nfft)

    # 5. One-sided PSD.
    half = nfft // 2
    psd = []
    for k in range(half + 1):
        real, imag = spectrum[k]
        power = real * real + imag * imag
        psd_val = power / (fs * win_sum_sq)
        if k != 0 and k != nfft / 2:
            psd_val *= 2.0
        psd.append(psd_val)
    return psd


def get_frequencies(nfft, fs):
    """Return the one-sided frequency bins for a DFT of length nfft at rate fs.

    The returned list has length nfft // 2 + 1 and ranges from 0 to the
    Nyquist frequency fs / 2 (inclusive when nfft is even).
    """
    if not isinstance(nfft, int) or nfft < 1:
        raise ValueError("nfft must be a positive integer")
    if fs <= 0:
        raise ValueError("fs must be positive")
    half = nfft // 2
    return [k * fs / nfft for k in range(half + 1)]


def welch_psd(data, fs, nperseg, noverlap=None, nfft=None, window="hann",
              detrend="constant", average="mean"):
    """Estimate the power spectral density using Welch's method.

    Parameters
    ----------
    data : sequence of float
        Input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each segment in samples.
    noverlap : int or None
        Number of overlapping samples between segments. If None, defaults
        to nperseg // 2 (50% overlap).
    nfft : int or None
        Length of the DFT. If None, defaults to nperseg (no zero-padding).
        If greater than nperseg, segments are zero-padded; if less, segments
        are truncated to nfft after windowing them over nperseg samples.
    window : str
        Window function name: 'hann' (default), 'hamming', 'boxcar', or
        'rectangular' (alias of 'boxcar').
    detrend : str
        Per-segment detrending: 'constant' (default, subtract mean),
        'linear' (subtract least-squares line), or 'none'.
    average : str
        Averaging method across segment PSDs: 'mean' (default) or 'median'.

    Returns
    -------
    (frequencies, psd) : tuple of lists
        frequencies : list of float, the one-sided frequency bins.
        psd : list of float, the estimated PSD at each frequency.

    Notes
    -----
    The one-sided spectrum doubles the power at non-DC, non-Nyquist bins
    to fold in the negative-frequency energy.
    """
    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError("fs must be a positive number")
    if not isinstance(nperseg, int) or nperseg < 1:
        raise ValueError("nperseg must be a positive integer")
    if len(data) == 0:
        raise ValueError("data must not be empty")
    if nperseg > len(data):
        # When there aren't enough samples for a full segment we use the
        # available samples, zero-padded to nperseg. This is the same
        # behaviour as processing a single short segment.
        pass
    if noverlap is None:
        noverlap = nperseg // 2
    if not isinstance(noverlap, int) or noverlap < 0:
        raise ValueError("noverlap must be a non-negative integer")
    if noverlap >= nperseg:
        raise ValueError("noverlap must be strictly less than nperseg")
    if nfft is None:
        nfft = nperseg
    if not isinstance(nfft, int) or nfft < 1:
        raise ValueError("nfft must be a positive integer")
    if average not in ("mean", "median"):
        raise ValueError("average must be 'mean' or 'median'")

    window_samples = _get_window(window, nperseg)

    # Determine segment start indices.
    step = nperseg - noverlap
    if step < 1:
        step = 1

    n = len(data)
    starts = list(range(0, max(1, n - nperseg + 1), step))
    if not starts:
        starts = [0]

    # Ensure the last segment does not start beyond the data; if it does,
    # the _one_sided_psd function will zero-pad.
    if starts[-1] + nperseg > n and len(starts) > 1 and n >= nperseg:
        # Only add a final partial segment if the last full segment doesn't
        # cover the tail. We keep it simple: only full-length starts plus a
        # possible trailing partial start if the data doesn't end on a boundary.
        pass

    psd_segments = []
    for start in starts:
        seg = data[start:start + nperseg]
        psd_segments.append(_one_sided_psd(seg, window_samples, nfft, fs, detrend))

    n_bins = len(psd_segments[0])

    if average == "mean":
        combined = []
        for k in range(n_bins):
            total = 0.0
            for seg_psd in psd_segments:
                total += seg_psd[k]
            combined.append(total / len(psd_segments))
    else:  # median
        combined = []
        for k in range(n_bins):
            values = sorted(seg_psd[k] for seg_psd in psd_segments)
            m = len(values)
            if m % 2 == 1:
                combined.append(values[m // 2])
            else:
                combined.append((values[m // 2 - 1] + values[m // 2]) / 2.0)

    frequencies = get_frequencies(nfft, fs)
    return frequencies, combined
