import gpxpy
import gpxpy.gpx

# Parsing an existing file:
# -------------------------

filename = 'Downloads/W11D2_Steady_5_30_MVN_inference_.gpx'
#filename = 'Downloads/W11D1_Poporun_.gpx'
#filename = 'Downloads/W09D1_Gentse_Feesten_eisen_hun_tol_.gpx'
#filename = 'Downloads/W10D2_Snappy_wind_tegen_stroll_terug_.gpx'
#filename = 'Downloads/Telegram Desktop/Marija_Stojchevska_2023-08-05_10-22-18.GPX'
#filename = 'Downloads/Telegram Desktop/Marija_Stojchevska_2023-08-01_18-36-37.GPX'

gpx_file = open(filename, 'r')

gpx = gpxpy.parse(gpx_file)


times = []
latitudes = []
longitudes = []
elevations = []

start = gpx.tracks[0].segments[0].points[0].time
for track in gpx.tracks:
    for segment in track.segments:
        print("Segment length:", len(segment.points))
        for point in segment.points:
            #print('Point at ({0},{1}) -> {2}: {3}'.format(point.latitude, point.longitude, point.elevation, point.time))
            #times.append((point.time - start).total_seconds())
            latitudes.append(point.latitude)
            longitudes.append(point.longitude)
            elevations.append(point.elevation)

            #print(point.time)
            time = point.time.replace(microsecond=0)
            times.append(time)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.DataFrame({'time': times, 'lat': latitudes, 'lon': longitudes, 'ele': elevations})
df.set_index(df.time, inplace=True, drop=True)


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.deg2rad(lat1), np.deg2rad(lon1), np.deg2rad(lat2), np.deg2rad(lon2)

    a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a)) * 1000 # distance in meter

# Approximate formula, based on Euclidean distance. Math rocks!
#def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
#    if to_radians:
#        lat1, lon1, lat2, lon2 = np.deg2rad(lat1), np.deg2rad(lon1), np.deg2rad(lat2), np.deg2rad(lon2)
#    dlat = lat1 - lat2
#    dlon = lon1 - lon2
#    s = np.cos(lat1)
#    return earth_radius * np.sqrt(dlat ** 2 + (s*dlon)**2) * 1000


length = haversine(df.lat, df.lon, df.lat.shift(), df.lon.shift())

# TODO: smoothen less, and first resample to 1s intervals to densify it, then smoothen properly.
win = 5
smooth_lat = df.lat
smooth_lon = df.lon
smooth_ele = df.ele

print(smooth_lat.isna().any())
print(smooth_lon.isna().any())
print(smooth_ele.isna().any())


resample = '250ms'
smooth_lat = smooth_lat.resample(resample).interpolate(method='linear')
smooth_lon = smooth_lon.resample(resample).interpolate(method='linear')
smooth_ele = smooth_ele.resample(resample).interpolate(method='linear')
time_step = (smooth_lat.index.values[2] - smooth_lat.index.values[1]).astype(np.float32) * 1e-9
print(smooth_lat.isna().any())
print(smooth_lon.isna().any())
print(smooth_ele.isna().any())
print(time_step)
smooth_dt = np.ones_like(smooth_lat) * time_step

print(smooth_lon)

smooth_lat = smooth_lat.rolling(win, min_periods=1, center=True).mean()
smooth_lon = smooth_lon.rolling(win, min_periods=1, center=True).mean()
smooth_ele = smooth_ele.rolling(120, min_periods=1, center=True).mean()
print(smooth_lat.isna().any())
print(smooth_lon.isna().any())
print(smooth_ele.isna().any())
smooth_length = haversine(smooth_lat, smooth_lon, smooth_lat.shift(), smooth_lon.shift())
smooth_run = smooth_length.cumsum()
print(smooth_run)

dt = df.time.diff().dt.seconds
dt = dt.rolling("40s", min_periods=1, center=True).mean()
run = length.cumsum()
strava_speed = length / dt


#fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios':[2, 1, 1, 1]}, figsize=(8, 8))
fig = plt.figure(figsize=(8, 10))
ax0 = fig.add_subplot(3,2,1)
ax1 = fig.add_subplot(3,1,2)
ax2 = fig.add_subplot(3,1,3, sharex=ax1)
ax3 = fig.add_subplot(3,2,2)

ax0.set_title("GPS Track")
ax0.plot(smooth_lon, smooth_lat, linewidth=1, color="orange")
ax0.scatter(longitudes, latitudes, s=1)
ax0.set_aspect(1/np.cos(np.deg2rad(smooth_lat.mean())))
ax0.scatter(longitudes[0], latitudes[0], s=23, color='green', zorder=2)
ax0.scatter(longitudes[-1], latitudes[-1], s=23, color='black', zorder=2)
ax0.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax0.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

print("mean dt:", dt.mean())

def shrink(x):
    x = np.concatenate([[x[0]], x])
    hp = 0.5 * (x[1:] - x[:-1])
    lp = 0.5 * (x[1:] + x[:-1])
    print("mean abs", np.abs(hp).mean())
    print("max  abs", np.abs(hp).max())
    shrink_amount = np.median(np.abs(hp)) * 0.8
    print("shrink_amount", shrink_amount)
    hp_orig = hp
    hp = np.sign(hp) * (np.maximum(np.abs(hp) - shrink_amount, 0.0))
    hp *= np.abs(hp_orig).max() / np.abs(hp).max()
    #hp *= np.abs(hp_orig).mean() / np.abs(hp).mean()

    rec0 = lp + hp
    rec1 = lp - hp
    rec1 = np.concatenate([rec1[1:], lp[-1:]])

    return 0.5 * (rec0 + rec1)


win1, win2, win3 = ("10s", "10s", "10s")
nom = smooth_length
nom = nom.rolling(win1, min_periods=1, center=True).mean()
nom = nom.rolling(win2, min_periods=1, center=True).median()
nom = nom.rolling(win3, min_periods=1, center=True).mean()
smooth_speed = nom / smooth_dt

def recursive_shrink(x, level=6):
    if level <= 0:
        return x
    chop = x[x.shape[0] // 4 * 4:]
    x = x[:x.shape[0] // 4 * 4]
    ds = 0.5 * (x[::2] + x[1::2])
    delta = (x[1::2] - x[0::2])
    ds = recursive_shrink(ds, level=level-1)
    delta = recursive_shrink(delta, level=level-1)
    x0 = ds - delta * 0.5
    x1 = ds + delta * 0.5
    rec = np.stack([x0, x1], axis=-1).flatten()
    for i in range(10*(7-level)):
        rec = shrink(rec)
    rec = np.concatenate([rec, chop])
    return rec

print("Shrinking...")
orig_smooth_speed = smooth_speed.copy()
smooth_speed = smooth_speed.fillna(0.0).values
#for i in range(500000):
#    smooth_speed = shrink(smooth_speed)
#    print()
smooth_speed = recursive_shrink(smooth_speed)

smooth_speed = pd.Series(index=orig_smooth_speed.index, data=smooth_speed)

smooth_speed *= orig_smooth_speed.mean() / smooth_speed.mean()

def compute_grade_adjusted_pace(distance_series_m, time_series_s, elevation_series_m):
    # Calculate the gradient for each segment
    gradient_series = elevation_series_m.diff() / distance_series_m

    # Calculate the raw pace for each segment in seconds per meter
    raw_pace_series_s_per_m = time_series_s / distance_series_m

    # Adjust the pace based on the gradient
    # Here we're using a simple rule that adds 6 seconds per meter for uphill grades
    # and subtracts 8 seconds per meter for downhill grades
    mile = 1609
    uphill_adjustment = -600/mile * np.maximum(gradient_series, 0)
    downhill_adjustment = -800/mile * np.minimum(gradient_series, 0)
    adjusted_pace_series_s_per_m = raw_pace_series_s_per_m + uphill_adjustment + downhill_adjustment

    # Convert to minutes per kilometer
    adjusted_pace_series_min_per_km = adjusted_pace_series_s_per_m * 1000/60

    return adjusted_pace_series_min_per_km

#smooth_elevation = df.ele.rolling(15, min_periods=1, center=True).mean()
smooth_slope_adjusted_pace = compute_grade_adjusted_pace(np.maximum(smooth_speed,1e-3) * smooth_dt, smooth_dt, smooth_ele)

def speed_to_pace(speed):
    return 60/(3.6*np.maximum(speed,1e-2))


ax1.set_title("Speed")
ax1.plot(smooth_run, orig_smooth_speed, color="gray", alpha=0.2, label='Raw Speed')
ax1.plot(smooth_run, smooth_speed, color="red", label='Speed')
ax1.set_ylim((0, ax1.get_ylim()[1] + 1))
ax1.set_ylabel("Speed (m/s)")
ax1.set_xlabel("Distance covered (m)")
ax1t = ax1.twinx()
ax1t.grid()
ax1t.plot(smooth_run, speed_to_pace(smooth_speed), color="blue", label='Pace')
ax1t.plot(smooth_run, smooth_slope_adjusted_pace, color="green", label='GAP', alpha=0.5)
ax1t.axhline(speed_to_pace(smooth_speed.median()), color='blue', linestyle=':')
ax1t.set_ylim((1.5, 7.5))
ax1t.invert_yaxis()
ax1t.set_ylabel("Pace (min/km)")
ax1.legend(loc=2)
ax1t.legend(loc=1)

ax2.fill_between(smooth_run, df.ele.min()-1, smooth_ele, label="Elevation")
ax2.set_title("Elevation")
ax2.set_ylabel("Elevation (m)")
ax2.set_xlabel("Distance covered (m)")
ax2.set_ylim(df.ele.min()-1, df.ele.max()+10)
ax2.legend(loc=2)

ax2t = ax2.twinx()
ax2t.plot(smooth_run, smooth_dt.cumsum(), color="orange", linewidth=1, label="Time taken")
ax2t.set_ylabel("Time taken")
ax2t.legend(loc=4)

# Create a function to convert seconds to hh:mm:ss format
def format_time(seconds, _):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
ax2t.yaxis.set_major_formatter(ticker.FuncFormatter(format_time))


smooth_pace = speed_to_pace(smooth_speed)
ax3.set_title("Pace Histogram")
ax3.hist(smooth_pace[smooth_pace < 7.0], bins=40, density=True)
ax3.invert_xaxis()
ax3.set_xlabel("Pace (min/km)")
plt.setp(ax3.get_yticklabels(), visible=False)

for a in [ax1, ax1t, ax2, ax2t]:
    print((smooth_run.min(), smooth_run.max()))
    a.set_xlim((smooth_run.min(), smooth_run.max()))

fig.suptitle(Path(filename).name, fontweight="bold")

plt.tight_layout()
plt.show()


