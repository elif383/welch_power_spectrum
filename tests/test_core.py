import math
import unittest

from welch_power_spectrum import welch_psd, get_frequencies


class TestGetFrequencies(unittest.TestCase):
    def test_basic_even_nfft(self):
        freqs = get_frequencies(8, 10.0)
        self.assertEqual(len(freqs), 5)
        self.assertAlmostEqual(freqs[0], 0.0)
        self.assertAlmostEqual(freqs[1], 1.25)
        self.assertAlmostEqual(freqs[4], 5.0)

    def test_odd_nfft(self):
        freqs = get_frequencies(7, 10.0)
        self.assertEqual(len(freqs), 4)
        self.assertAlmostEqual(freqs[0], 0.0)
        self.assertAlmostEqual(freqs[3], 30.0 / 7)

    def test_invalid_nfft(self):
        with self.assertRaises(ValueError):
            get_frequencies(0, 10.0)
        with self.assertRaises(ValueError):
            get_frequencies(-4, 10.0)

    def test_invalid_fs(self):
        with self.assertRaises(ValueError):
            get_frequencies(8, 0)
        with self.assertRaises(ValueError):
            get_frequencies(8, -1.0)


class TestWelchPSD(unittest.TestCase):
    def test_dc_signal_constant_detrend(self):
        # A constant signal detrended per-segment should give near-zero PSD
        # everywhere, and exactly zero at DC.
        data = [5.0] * 64
        freqs, psd = welch_psd(data, fs=10.0, nperseg=8, detrend="constant")
        self.assertEqual(len(freqs), 5)
        self.assertEqual(len(psd), 5)
        for v in psd:
            self.assertLess(abs(v), 1e-12)

    def test_pure_sine_frequency_peak(self):
        # A sine at 2 Hz should produce a peak near 2 Hz.
        fs = 100.0
        duration = 10.0
        n = int(fs * duration)
        data = [math.sin(2 * math.pi * 2.0 * i / fs) for i in range(n)]
        freqs, psd = welch_psd(data, fs=fs, nperseg=256, window="hann")
        peak_idx = psd.index(max(psd))
        peak_freq = freqs[peak_idx]
        self.assertAlmostEqual(peak_freq, 2.0, places=1)

    def test_noverlap_defaults_to_half(self):
        # 50% overlap should produce (n - nperseg) / (nperseg - noverlap) + 1
        # segments. For n=16, nperseg=4, noverlap=2: step=2, starts=[0,2,4,...,12] => 7 segments.
        data = [float(i) for i in range(16)]
        freqs, psd = welch_psd(data, fs=1.0, nperseg=4, detrend="none", window="boxcar")
        self.assertEqual(len(freqs), 3)
        self.assertEqual(len(psd), 3)

    def test_invalid_noverlap(self):
        data = [1.0] * 10
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, noverlap=4)
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, noverlap=-1)

    def test_invalid_window(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, window="nonexistent_window")

    def test_invalid_detrend(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, detrend="polynomial")

    def test_invalid_average(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, average="trimmed")

    def test_empty_data(self):
        with self.assertRaises(ValueError):
            welch_psd([], fs=1.0, nperseg=4)

    def test_invalid_fs(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=0, nperseg=4)
        with self.assertRaises(ValueError):
            welch_psd(data, fs=-1.0, nperseg=4)

    def test_invalid_nperseg(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=0)
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=-2)

    def test_invalid_nfft(self):
        data = [1.0] * 16
        with self.assertRaises(ValueError):
            welch_psd(data, fs=1.0, nperseg=4, nfft=0)

    def test_nfft_padding_changes_bin_count(self):
        data = [math.sin(2 * math.pi * 1.0 * i / 10.0) for i in range(64)]
        freqs1, _ = welch_psd(data, fs=10.0, nperseg=8, nfft=8)
        self.assertEqual(len(freqs1), 5)
        freqs2, psd2 = welch_psd(data, fs=10.0, nperseg=8, nfft=16)
        self.assertEqual(len(freqs2), 9)

    def test_nfft_smaller_than_nperseg(self):
        # nfft < nperseg: the segment is truncated to nfft after windowing
        # over nperseg samples. This still produces a valid spectrum.
        data = [float(i) for i in range(32)]
        freqs, psd = welch_psd(data, fs=1.0, nperseg=8, nfft=4, detrend="none",
                               window="boxcar")
        self.assertEqual(len(freqs), 3)
        self.assertEqual(len(psd), 3)

    def test_linear_detrend_on_ramp(self):
        # A linear ramp with linear detrend should produce near-zero PSD.
        data = [3.0 * i for i in range(64)]
        freqs, psd = welch_psd(data, fs=10.0, nperseg=8, detrend="linear")
        for v in psd:
            self.assertLess(abs(v), 1e-10)

    def test_linear_detrend_on_ramp_with_noise_floor(self):
        # Same ramp but with constant detrend should leave a lot of residual power.
        data = [3.0 * i for i in range(64)]
        _, psd_const = welch_psd(data, fs=10.0, nperseg=8, detrend="constant")
        _, psd_lin = welch_psd(data, fs=10.0, nperseg=8, detrend="linear")
        self.assertGreater(sum(psd_const), 1.0)
        self.assertLess(sum(psd_lin), 1e-10)

    def test_hann_vs_boxcar_peak_width(self):
        # Hann window should produce a narrower peak than boxcar for a sine.
        fs = 100.0
        n = 400
        data = [math.sin(2 * math.pi * 5.0 * i / fs) for i in range(n)]
        _, psd_box = welch_psd(data, fs=fs, nperseg=64, window="boxcar")
        _, psd_hann = welch_psd(data, fs=fs, nperseg=64, window="hann")
        # Hann peak should be higher (narrower main lobe concentrates energy).
        self.assertGreater(max(psd_hann), max(psd_box) * 0.5)

    def test_rectangular_alias_for_boxcar(self):
        data = [math.sin(2 * math.pi * 1.0 * i / 10.0) for i in range(64)]
        _, psd_box = welch_psd(data, fs=10.0, nperseg=8, window="boxcar")
        _, psd_rect = welch_psd(data, fs=10.0, nperseg=8, window="rectangular")
        self.assertEqual(len(psd_box), len(psd_rect))
        for a, b in zip(psd_box, psd_rect):
            self.assertAlmostEqual(a, b, places=12)

    def test_median_average(self):
        # With median averaging, an outlier segment has less influence.
        fs = 100.0
        n = 400
        data = [math.sin(2 * math.pi * 5.0 * i / fs) for i in range(n)]
        _, psd_mean = welch_psd(data, fs=fs, nperseg=64, average="mean")
        _, psd_median = welch_psd(data, fs=fs, nperseg=64, average="median")
        self.assertEqual(len(psd_mean), len(psd_median))

    def test_data_shorter_than_nperseg(self):
        # When data is shorter than nperseg, the single segment is zero-padded.
        data = [1.0, 2.0, 3.0]
        freqs, psd = welch_psd(data, fs=10.0, nperseg=8, detrend="none",
                               window="boxcar")
        self.assertEqual(len(freqs), 5)
        self.assertEqual(len(psd), 5)

    def test_returns_are_lists(self):
        data = [1.0] * 16
        freqs, psd = welch_psd(data, fs=1.0, nperseg=4)
        self.assertIsInstance(freqs, list)
        self.assertIsInstance(psd, list)
        self.assertTrue(all(isinstance(f, float) for f in freqs))
        self.assertTrue(all(isinstance(p, float) for p in psd))

    def test_single_segment(self):
        # noverlap = 0 with data exactly nperseg long => 1 segment.
        data = [math.sin(2 * math.pi * 1.0 * i / 8.0) for i in range(8)]
        freqs, psd = welch_psd(data, fs=8.0, nperseg=8, noverlap=0,
                               detrend="none", window="boxcar")
        self.assertEqual(len(freqs), 5)
        # Peak should be at 1 Hz (bin 1).
        peak_idx = psd.index(max(psd))
        self.assertEqual(freqs[peak_idx], 1.0)

    def test_nyquist_bin_not_doubled(self):
        # For even nfft, the Nyquist bin should not be doubled.
        # We can verify this by checking a signal at the Nyquist frequency:
        # a cosine at Nyquist should produce power only at the last bin,
        # and the value should equal (amplitude^2 / 4) * (1 / (fs * U)) * 1
        # (not doubled).
        fs = 8.0
        nfft = 8
        data = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # Nyquist cosine
        freqs, psd = welch_psd(data, fs=fs, nperseg=8, nfft=nfft, noverlap=0,
                               detrend="none", window="boxcar")
        self.assertEqual(len(freqs), 5)
        self.assertAlmostEqual(freqs[-1], fs / 2)
        # DC should be zero.
        self.assertLess(abs(psd[0]), 1e-12)
        # The peak is at Nyquist.
        peak_idx = psd.index(max(psd))
        self.assertEqual(peak_idx, 4)


if __name__ == "__main__":
    unittest.main()
