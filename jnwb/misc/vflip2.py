import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats
from scipy import ndimage

class vFLIP2:
    """
    vFLIP2 Analysis Class
    Analyzes electrophysiological data to identify spectrolaminar motifs based on
    power changes across frequency bands and laminar depth.
    """

    def __init__(self, data,
                 intdist=np.nan,
                 freqbinsize=1.0,
                 DataType='psd',
                 fsample=np.nan,
                 orientation='both',
                 layer4Thickness=np.nan,
                 plot_result=False,
                 omega_cut=6.0):

        # Input Validation
        if DataType not in ["psd", "raw", "raw_cut"]:
            raise ValueError("DataType must be 'psd', 'raw', or 'raw_cut'.")
        if orientation not in ["upright", "inverted", "both"]:
            raise ValueError("orientation must be 'upright', 'inverted', or 'both'.")

        self.plot_combined = False
        self.omega_cut = omega_cut

        # Handle inter-channel distance
        if np.isnan(intdist):
            try:
                val = float(input('Please enter the interchannel distance in mm (intdist): '))
                if val <= 0: raise ValueError
                self.intdist = val
            except:
                raise ValueError('Invalid interchannel distance entered.')
        else:
            self.intdist = intdist

        self.step = int(round(0.1 / self.intdist))  
        self.minrange_s = int(np.ceil(0.3 / self.intdist))

        if DataType == "psd":
            self.nonnormpowmat = data
            self.freqbinsize = freqbinsize
        elif DataType in ["raw", "raw_cut"]:
            if np.isnan(fsample):
                try:
                    val = float(input('Please enter the sampling rate (fsample): '))
                    if val <= 0: raise ValueError
                    self.fsample = val
                except:
                    raise ValueError('Invalid sampling rate entered.')
            else:
                self.fsample = fsample

            if DataType == "raw":
                trials = self._split_into_trials(data)
                self.nonnormpowmat = self._compute_psd_hanning(trials)
            elif DataType == "raw_cut":
                self.nonnormpowmat = self._compute_psd_hanning(data)

            self.freqbinsize = 1.0

        nanboolean = ~np.isnan(self.nonnormpowmat[:, 0])
        if np.sum(nanboolean) == 0:
            raise ValueError("Error using FLIPAnalysis: Empty matrix")

        valid_indices = np.where(nanboolean)[0]
        self.startrow = valid_indices[0]
        self.endrow = valid_indices[-1] 

        self.freqaxis = np.arange(1, self.nonnormpowmat.shape[1] + 1) * self.freqbinsize

        if orientation == 'both':
            self.orientation1 = 0
        elif orientation == 'upright':
            self.orientation1 = 1
        elif orientation == "inverted":
            self.orientation1 = -1

        laminae_thickness_mm = np.array([122.9, 396.9, 127.1, 211.4, 247.5, 226.3, 260.2]) / 1000.0
        if np.isnan(layer4Thickness):
            layer4 = np.sum(laminae_thickness_mm[2:5])
        else:
            layer4 = layer4Thickness

        self.minrange = int(np.ceil(layer4 / self.intdist))
        self.Results = self.flip_it()
        self.relpow = self._get_Window(self.startrow, self.endrow)

        if self.Results is not None and plot_result:
            self.plot_result()

    def _get_Window(self, proximalchannel, distalchannel):
        powspec_window = self.nonnormpowmat[proximalchannel : distalchannel + 1, :]
        maxpow = np.max(powspec_window, axis=0)
        maxpow[maxpow == 0] = np.nan
        relpow = powspec_window / maxpow
        return relpow

    def _get_freqbands(self, S1_meanpow, S2_meanpow):
        def find_longest_true_run(logical_array):
            d = np.diff(np.concatenate(([0], logical_array.astype(int), [0])))
            run_starts = np.where(d == 1)[0]
            run_ends = np.where(d == -1)[0] - 1
            run_lengths = run_ends - run_starts + 1
            if len(run_lengths) > 0:
                idx = np.argmax(run_lengths)
                return np.arange(run_starts[idx], run_ends[idx] + 1)
            else:
                return np.array([], dtype=int)

        lowfreqs = (self.freqaxis > 4) & (self.freqaxis < 70)
        highfreqs = (self.freqaxis > 40) & (self.freqaxis < 150)
        greater = (S1_meanpow > S2_meanpow)
        lesser = (S1_meanpow < S2_meanpow)

        longest_run_P1 = find_longest_true_run(greater & lowfreqs)
        longest_run_P2 = find_longest_true_run(lesser & lowfreqs)
        len_P1, len_P2 = len(longest_run_P1), len(longest_run_P2)

        deep_f, sup_f, orientation = [], [], 0
        ind_high = None

        if len_P1 >= 5 and len_P2 >= 5:
            freq_range_P1, freq_range_P2 = self.freqaxis[longest_run_P1], self.freqaxis[longest_run_P2]
            if np.min(freq_range_P1) < np.min(freq_range_P2):
                deep_f, orientation, ind_high = freq_range_P1, -1, lesser & highfreqs
            else:
                deep_f, orientation, ind_high = freq_range_P2, 1, greater & highfreqs
        elif len_P1 >= 5:
            deep_f, orientation, ind_high = self.freqaxis[longest_run_P1], -1, lesser & highfreqs
        elif len_P2 >= 5:
            deep_f, orientation, ind_high = self.freqaxis[longest_run_P2], 1, greater & highfreqs

        if orientation != 0 and ind_high is not None:
            longest_run_high = find_longest_true_run(ind_high)
            if len(longest_run_high) > 0:
                sup_f = self.freqaxis[longest_run_high]
                if len(sup_f) < 20 or sup_f[-1] < 70: sup_f = []
            else: sup_f = []

        return deep_f, sup_f, orientation

    def _peak_check(self, band, proximalchannel, distalchannel):
        peak_locations = np.where(band == np.max(band))[0]
        if len(peak_locations) == 0: return False
        peak_index = np.max(peak_locations) if np.mean(peak_locations) > len(band) / 2 else np.min(peak_locations)

        if peak_index == 0:
            peak_max_check = (proximalchannel == self.startrow) and (band[peak_index] > band[peak_index + 1])
        elif peak_index == len(band) - 1:
            peak_max_check = (distalchannel == self.endrow) and (band[peak_index] > band[peak_index - 1])
        else:
            peak_max_check = (band[peak_index] > band[peak_index + 1]) and (band[peak_index] > band[peak_index - 1])

        check1 = (peak_index < self.minrange_s) or (peak_index >= len(band) - self.minrange_s)
        return peak_max_check and check1

    def _crossover_channels(self, lowband, highband, proximalchannel, orientation):
        band_diff = np.abs(highband - lowband)
        def determine_cross(idx):
            b1 = highband if orientation > 0 else lowband
            b2 = lowband if orientation > 0 else highband
            if b1[idx] > b2[idx] and b2[idx+1] > b1[idx+1]:
                return idx if abs(band_diff[idx]) <= abs(band_diff[idx+1]) else idx + 1
            if idx + 2 < len(b1):
                if b1[idx] > b2[idx] and b1[idx+1] == b2[idx+1] and b2[idx+2] > b1[idx+2]:
                    return idx + 1
            return np.nan

        crossoverchannels = [determine_cross(i) for i in range(len(lowband) - 1)]
        crossoverchannels = [c for c in crossoverchannels if not np.isnan(c)]

        if not crossoverchannels: return np.nan
        if len(crossoverchannels) == 1: return crossoverchannels[0] + proximalchannel
        
        ratings = [np.sum(band_diff[:int(c)+1]) - np.sum(band_diff[int(c):]) for c in crossoverchannels]
        return crossoverchannels[np.argmax(ratings)] + proximalchannel

    def _evaluate_individual_goodness(self, lowband, highband):
        set_pval = 0.05
        def BandRegress(band):
            n = len(band)
            x = np.arange(1, n + 1)
            p, residuals, _, _, _ = np.polyfit(x, band, 2, full=True)
            y_pred = np.polyval(p, x)
            sst = np.sum((band - np.mean(band))**2)
            sse = np.sum((band - y_pred)**2)
            rsquared = 1 - (sse / sst) if sst != 0 else 0
            dof_model, dof_resid = 2, n - 3
            if dof_resid > 0 and sse > 0:
                f_stat = ((sst - sse) / dof_model) / (sse / dof_resid)
                pval = 1 - scipy.stats.f.cdf(f_stat, dof_model, dof_resid)
            else: pval = 1.0
            slope = 2 * p[0] * np.round(n / 2) + p[1]
            return slope, rsquared, pval

        low_slope, low_r2, low_pval = BandRegress(lowband)
        high_slope, high_r2, high_pval = BandRegress(highband)
        goodness = high_r2 * low_r2
        significant = (low_pval < set_pval) and (high_pval < set_pval)
        Gsign = 1 if (low_slope > 0 and high_slope < 0) else (-1 if (low_slope < 0 and high_slope > 0) else 0)
        return goodness * significant * Gsign

    def omega_fun(self):
        euc_distance = lambda g1, g2: np.sqrt(np.sum((g1 - g2)**2))
        best_split, best_omega = np.full(12, np.nan), -np.inf

        for proximalchannel in range(self.startrow, self.endrow - self.minrange + 2, self.step):
            for distalchannel in range(proximalchannel + self.minrange, self.endrow + 1, self.step):
                psd_normalized = self._get_Window(proximalchannel, distalchannel)
                self.minrange_s = int(np.floor(abs(proximalchannel - distalchannel) / 2))
                if self.minrange_s < 1: continue

                S1_meanpow = ndimage.uniform_filter1d(np.mean(psd_normalized[:self.minrange_s, :], axis=0), size=5)
                S2_meanpow = ndimage.uniform_filter1d(np.mean(psd_normalized[-self.minrange_s:, :], axis=0), size=5)
                Ps_dist = euc_distance(S1_meanpow, S2_meanpow)
                deep_f, sup_f, orientation = self._get_freqbands(S1_meanpow, S2_meanpow)

                if not len(deep_f) or not len(sup_f): continue

                lowband = np.mean(psd_normalized[:, np.isin(self.freqaxis, deep_f)], axis=1)
                highband = np.mean(psd_normalized[:, np.isin(self.freqaxis, sup_f)], axis=1)
                goodness = self._evaluate_individual_goodness(lowband, highband)

                if (self.orientation1 == -1 and goodness > 0) or (self.orientation1 == 1 and goodness < 0): goodness = 0

                omega = np.log(Ps_dist * euc_distance(lowband, highband) * abs(goodness) * abs(proximalchannel - distalchannel) * len(deep_f) * len(sup_f))
                highfreqmaxchannel, lowfreqmaxchannel = np.argmax(highband) + proximalchannel, np.argmax(lowband) + proximalchannel
                crossover_point = self._crossover_channels(lowband, highband, proximalchannel, orientation)

                good_fit = (omega != 0 and not np.isinf(omega)) and self._peak_check(lowband, proximalchannel, distalchannel) and \
                           self._peak_check(highband, proximalchannel, distalchannel) and abs(highfreqmaxchannel - lowfreqmaxchannel) >= self.minrange and \
                           not np.isnan(crossover_point) and ((lowfreqmaxchannel < crossover_point < highfreqmaxchannel) or (lowfreqmaxchannel > crossover_point > highfreqmaxchannel)) and \
                           (lowfreqmaxchannel != crossover_point != highfreqmaxchannel != lowfreqmaxchannel)

                if good_fit and (omega > best_omega):
                    best_split = [goodness, deep_f[0], deep_f[-1], sup_f[0], sup_f[-1], proximalchannel, distalchannel, lowfreqmaxchannel, highfreqmaxchannel, crossover_point, omega, orientation]
                    best_omega = omega
        return best_omega, best_split

    def flip_it(self):
        best_omega, best_split = self.omega_fun()
        if best_omega <= self.omega_cut: return None
        fields = ['goodnessvalue', 'startinglowfreq', 'endinglowfreq', 'startinghighfreq', 'endinghighfreq', 'proximalchannel', 'distalchannel', 'lowfreqmaxchannel', 'highfreqmaxchannel', 'crossoverchannel', 'omega', 'orientation']
        results = {f: best_split[i] for i, f in enumerate(fields)}
        class ResultsStruct:
            def __init__(self, **entries): self.__dict__.update(entries); self.relpow = None
        return ResultsStruct(**results)

    def _split_into_trials(self, data):
        samples_per_trial = int(self.fsample)
        n_channels, total_timepoints = data.shape
        num_trials = total_timepoints // samples_per_trial
        reshaped = data[:, :num_trials * samples_per_trial].reshape(n_channels, num_trials, samples_per_trial)
        return [reshaped[:, i, :] for i in range(num_trials)]

    def _compute_psd_hanning(self, data):
        is_list = isinstance(data, list)
        ntrials = len(data) if is_list else data.shape[2]
        nchan = data[0].shape[0] if is_list else data.shape[0]
        max_ndatsample = max(d.shape[1] for d in data) if is_list else data.shape[1]
        padding_len = int(2**np.ceil(np.log2(max_ndatsample)))
        pad_factor = padding_len / self.fsample
        freq_indices = np.round(np.arange(1, 151) * pad_factor).astype(int)
        powspctrm = np.zeros((ntrials, nchan, len(freq_indices)))

        for itrial in range(ntrials):
            dat = data[itrial] if is_list else data[:, :, itrial]
            dat = dat - np.mean(dat, axis=1, keepdims=True)
            tap = np.hanning(dat.shape[1]); tap /= np.linalg.norm(tap)
            data_tap = dat * tap[np.newaxis, :]
            data_tap_pad = np.pad(data_tap, ((0,0), (0, int(round(pad_factor * self.fsample)) - dat.shape[1])), 'constant')
            dum = np.fft.fft(data_tap_pad, axis=1)[:, freq_indices]
            anglein = (2 * np.pi / self.fsample) * np.arange(1, 151)
            dum *= np.exp(-1j * anglein)[np.newaxis, :]
            powspctrm[itrial, :, :] = np.abs(dum * np.sqrt(2 / (round(pad_factor * self.fsample))))**2
        return np.mean(powspctrm, axis=0)

    def plot_relpowMap(self, ax, plot_SLonly=False):
        if self.Results is None: return
        s_chan, e_chan = (self.Results.proximalchannel, self.Results.distalchannel) if plot_SLonly else (self.startrow, self.endrow)
        img_data = self._get_Window(s_chan, e_chan) if plot_SLonly else (self.nonnormpowmat / np.max(self.nonnormpowmat[self.Results.proximalchannel : self.Results.distalchannel + 1, :], axis=0))[s_chan : e_chan + 1, :]
        im = ax.imshow(img_data, aspect='auto', cmap='jet', extent=[self.freqaxis[0], self.freqaxis[-1], e_chan, s_chan])
        ax.set(title='LFP Relative Power', xlabel='Frequency (Hz)', ylabel='Channel Number')
        plt.colorbar(im, ax=ax, label='Relative Power'); im.set_clim(0.3, 1.0)
        self._plot_freqran(ax); self._plot_physMarkers(ax, self.freqaxis[-1], plot_SLonly)

    def plot_bandedrelpow(self, ax, plot_SLonly=False):
        if self.Results is None: return
        s_chan, e_chan = (self.Results.proximalchannel, self.Results.distalchannel) if plot_SLonly else (self.startrow, self.endrow)
        plot_data = self._get_Window(s_chan, e_chan) if plot_SLonly else (self.nonnormpowmat / np.max(self.nonnormpowmat[self.Results.proximalchannel : self.Results.distalchannel + 1, :], axis=0))[s_chan : e_chan + 1, :]
        l_idx = [np.searchsorted(self.freqaxis, f) for f in [self.Results.startinglowfreq, self.Results.endinglowfreq]]
        h_idx = [np.searchsorted(self.freqaxis, f) for f in [self.Results.startinghighfreq, self.Results.endinghighfreq]]
        ax.plot(np.mean(plot_data[:, l_idx[0] : l_idx[1] + 1], axis=1), np.arange(s_chan, e_chan + 1), 'b', lw=2, label='Low Band')
        ax.plot(np.mean(plot_data[:, h_idx[0] : h_idx[1] + 1], axis=1), np.arange(s_chan, e_chan + 1), 'r', lw=2, label='High Band')
        ax.invert_yaxis(); ax.set(xlim=(0, 1), xlabel='Relative Power', ylabel='Channel Number')
        self._plot_physMarkers(ax, 1, plot_SLonly)
        if not plot_SLonly: ax.add_patch(patches.Rectangle((0.95, self.Results.proximalchannel), 0.05, self.Results.distalchannel - self.Results.proximalchannel, lw=0, fc='yellow', alpha=0.5))
        ax.legend()

    def _plot_freqran(self, ax):
        yl = ax.get_ylim()
        for f, c in [(self.Results.startinglowfreq, 'b'), (self.Results.endinglowfreq, 'b'), (self.Results.startinghighfreq, 'r'), (self.Results.endinghighfreq, 'r')]:
            ax.vlines(f, yl[0], yl[1], colors=c, linestyles='--')

    def _plot_physMarkers(self, ax, textpos, plot_SLonly):
        off = self.Results.proximalchannel if plot_SLonly else 0
        for val, lbl in [(self.Results.crossoverchannel - off, 'Crossover'), (self.Results.lowfreqmaxchannel - off, 'Alpha/Beta Peak'), (self.Results.highfreqmaxchannel - off, 'Gamma Peak')]:
            ax.hlines(val, *ax.get_xlim(), colors='k', linestyles='-.', lw=0.5)
            ax.text(textpos, val, lbl, va='bottom', ha='right', fontsize=8)

    def plot_result(self, plot_SLonly=False):
        if self.Results is None: return
        fig = plt.figure(figsize=(12, 6)); gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, :2]); self.plot_relpowMap(ax1, plot_SLonly)
        ax2 = fig.add_subplot(gs[0, 2], sharey=ax1); self.plot_bandedrelpow(ax2, plot_SLonly)
        fig.suptitle(f'vFLIP2 Results: G = {self.Results.goodnessvalue:.4f}, Omega = {self.Results.omega:.4f}')
        plt.tight_layout(); plt.show()
