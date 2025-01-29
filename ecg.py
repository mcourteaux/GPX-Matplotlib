import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
import scipy.signal

#martijn:
#df = pd.read_csv("/home/martijn/Downloads/Telegram Desktop/Polar_H10_C8F81721_20250128_221511_ECG.txt", delimiter=";").iloc[5*60*130:, :]
df = pd.read_csv("/home/martijn/Downloads/Telegram Desktop/Polar_H10_C8F81721_20250129_083803_ECG.txt", delimiter=";")

#marija:
#df = pd.read_csv("/home/martijn/Downloads/Telegram Desktop/Polar_H10_C8F81721_20250129_002728_ECG.txt", delimiter=";")

df.reset_index(drop=True, inplace=True)
print(df)

ts = df.iloc[:,1] * 1e-9
val = df.iloc[:,3] * 1e-3

ts = ts - ts.iloc[0]

kernel = np.ones(5)
kernel = kernel / np.sum(kernel)
val_filt = np.convolve(val, kernel, mode="valid")
ts_filt = np.convolve(ts, kernel, mode="valid")

hr = 70 / 60 * 1.3
interval = (ts.iloc[1000] - ts.iloc[500]) / 500
print("Interval:", interval, "Freq:", 1/interval)

fig, ax = plt.subplots(1, 2)

peaks, props = scipy.signal.find_peaks(val, distance=1 / (interval * hr))
# s / (s * b/s)
median_peak = np.median(val[peaks])
peak_treshold_max = 2.5 * median_peak
peak_treshold_min = 0.5 * median_peak

ax[0].plot(ts, val, label="raw", alpha=0.9)
#ax[0].plot(ts, baseline, label="baseline", alpha=0.5)
ax[0].plot(ts_filt, val_filt, label="smoothed", alpha=0.2)
ax[0].plot(ts[peaks], val[peaks], "x")
ax[0].axhline(y=peak_treshold_min, color='k', linestyle=':')
ax[0].axhline(y=peak_treshold_max, color='k', linestyle=':')
ax[0].legend()

median_hr = np.median(peaks[1:] - peaks[:-1]) * interval
mean_hr   = np.mean  (peaks[1:] - peaks[:-1]) * interval
print("Median HR:", median_hr, "bps =", median_hr * 60, "bpm")
print("Mean HR:", mean_hr, "bps =", mean_hr * 60, "bpm")

mean_len = int(0.9 / interval)
offset = int(0.3 / interval)
mean_ecg = np.zeros(mean_len)
mean_count = 0
for peak in peaks[230:]:
    peak_height = val[peak]
    if peak_height < peak_treshold_max and peak_height > peak_treshold_min and peak_height > 0:
        if peak-offset >= 0 and peak-offset+mean_len <= len(val):
            sample = val.iloc[peak-offset:peak+mean_len-offset]
            mean_ecg += sample.values
            mean_count += 1
            if mean_count == 40: break
mean_ecg /= mean_count
print("Averaged", mean_count, "heartbeats")
ax[1].set_title("Mean heartbeat (%d samples)" % mean_count)
ax[1].plot(np.arange(mean_len) * interval, mean_ecg)

mean_peaks, props = scipy.signal.find_peaks(mean_ecg, distance=int(0.2 / interval), height=0.04);
ax[1].plot(mean_peaks * interval, mean_ecg[mean_peaks], "x")
if len(mean_peaks) != 2:
    print("Cannot auto-detect QT")
else:
    Tw_p = mean_peaks[1]

    # Find T-wave slope
    mean_ecg_diff = mean_ecg[1:] - mean_ecg[:-1]
    ax[1].plot(np.arange(mean_len - 1) * interval, mean_ecg_diff, linestyle='dashed', alpha=0.05)
    Tw_slope_sr = int(0.1 / median_hr / interval)
    ax[1].axvspan(Tw_p * interval, (Tw_p + Tw_slope_sr) * interval, alpha=0.05, color='orange')
    Tw_diff_peak = Tw_p + np.argmin(mean_ecg_diff[Tw_p : Tw_p + Tw_slope_sr])
    Tw_diff_max_diff = mean_ecg_diff[Tw_diff_peak]
    Tw_diff_threshold = Tw_diff_max_diff * 0.5
    Tw_down_slope_idxs = np.arange(Tw_p, Tw_p + Tw_slope_sr)[mean_ecg_diff[Tw_p : Tw_p + Tw_slope_sr] < Tw_diff_threshold]
    Tw_down_slope_idxr = (Tw_down_slope_idxs[0], Tw_down_slope_idxs[-1])
    #ax[1].axvline(Tw_diff_peak * interval, alpha=0.4, color='green')
    Tw_down_slope_tr = tuple(i * interval for i in Tw_down_slope_idxr)
    #ax[1].axvspan(Tw_down_slope_tr[0], Tw_down_slope_tr[1], alpha=0.2, color='blue')
    Tw_slope_xs = np.arange(*Tw_down_slope_idxr) * interval
    Tw_coeff = np.polyfit(Tw_slope_xs, mean_ecg[slice(*Tw_down_slope_idxr)], 1)
    xs = np.array([Tw_down_slope_tr[0] - 0.05, Tw_down_slope_tr[1] + 0.05])
    ax[1].plot(xs, xs * Tw_coeff[0] + Tw_coeff[1], linestyle=':')

    # Estimate Q
    QTcolor = 'green'
    Q = mean_peaks[0] - int(0.065 / interval)
    isoelectric = np.mean(mean_ecg[Q - 10:Q])
    ax[1].axvline(Q * interval, color=QTcolor)
    ax[1].axhline(isoelectric, color=QTcolor, linestyle='dashed')

    # Solve T-slope for isoelectric
    # t * Tw_coeff[0] + Tw_coeff[1] = isoelectric
    # => t = (isoelectric - Tw_coeff[1]) / Tw_coeff[0]
    T = (isoelectric - Tw_coeff[1]) / Tw_coeff[0]
    ax[1].axvline(T, color=QTcolor)
    QT_interval = T - Q * interval
    QTc_interval = QT_interval + 2 * (median_hr * 60 - 60) / 1000
    print("QT-interval:", QT_interval * 1000, "ms")
    print("QTc-interval:", QTc_interval * 1000, "ms")

    ax[1].text(x=(T + Q * interval) * 0.5,
               y=(int(mean_ecg[mean_peaks[0]] * 5) - 0.5) / 5,
               s="QT=%.0fms\nQTc=%.0fms" % (QT_interval * 1000, QTc_interval * 1000),
               ha="center", va="center", color=QTcolor,
               fontdict={"size": 14})
    ax[1].text(x=T + 0.15,
               y=(int(mean_ecg[mean_peaks[0]] * 5) - 0.5) / 5,
               s="HR=%.0fbpm" % (median_hr * 60),
               ha="center", va="center", color='red',
               fontdict={"size": 14})


ax[1].xaxis.set_major_locator(MultipleLocator(0.2))
ax[1].xaxis.set_minor_locator(MultipleLocator(0.04))
ax[1].yaxis.set_major_locator(MultipleLocator(0.2))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.04))
ax[1].minorticks_on()

ax[1].grid(which='major', color='r', alpha=0.5)
ax[1].grid(which='minor', color='r', linestyle=':', alpha=0.25)
ax[1].set_xlabel("time (s)")
ax[1].set_ylabel("potential (mV)")
plt.show()
