import numpy as np
from scipy.signal import savgol_filter

def smooth_average_curve(time, temperature, smooth_window=11, poly_order=3):
    """
    Combine multiple stitched trials into a single smooth, evenly-spaced curve,
    using the full union of time ranges (not just the overlap).

    Parameters
    ----------
    temperature : list of array-like
        List of temperature arrays, one per trial (already time-offset/stitched).
    time : list of array-like
        List of time arrays, one per trial, matching `temperature`.
    smooth_window : int
        Window length for Savitzky-Golay smoothing (must be odd).
    poly_order : int
        Polynomial order for Savitzky-Golay smoothing.

    Returns
    -------
    common_t : np.ndarray
        Evenly-spaced time array.
    avg_temp_smooth : np.ndarray
        Averaged and smoothed temperature values.
    """
    if not isinstance(temperature[0], (list, np.ndarray)):
        temperature = [temperature]
        time = [time]

    trials_t = []
    trials_temp = []
    for t, temp in zip(time, temperature):
        t = np.asarray(t, dtype=float)
        temp = np.asarray(temp, dtype=float)

        # Drop NaNs (in either array) before sorting
        mask = ~np.isnan(t) & ~np.isnan(temp)
        t = t[mask]
        temp = temp[mask]

        if len(t) < 2:
            continue  # skip trials with insufficient data

        order = np.argsort(t)
        trials_t.append(t[order])
        trials_temp.append(temp[order])

    if len(trials_t) == 0:
        raise ValueError("No valid trials with at least 2 non-NaN points were found.")

    # 1. Estimate dt automatically
    total_points = sum(len(t) for t in trials_t)
    total_span = max(t.max() for t in trials_t) - min(t.min() for t in trials_t)

    if total_span <= 0:
        raise ValueError(
            f"Total time span is {total_span} (<=0). "
            "Check that your time array has more than one distinct value."
        )

    dt = total_span / total_points

    if dt <= 0:
        raise ValueError(f"Computed dt={dt} is invalid (<=0).")

    # 2. Define common grid using the UNION of all trial ranges
    t_min = min(t.min() for t in trials_t)
    t_max = max(t.max() for t in trials_t)
    common_t = np.arange(t_min, t_max, dt)

    # 3. Interpolate each trial onto the common grid
    interp_temps = np.array([
        np.interp(common_t, t, temp, left=np.nan, right=np.nan)
        for t, temp in zip(trials_t, trials_temp)
    ])

    # 4. Average across trials, ignoring NaNs
    avg_temp = np.nanmean(interp_temps, axis=0)

    # 5. Smooth the averaged curve
    window = min(smooth_window, len(avg_temp) - 1)
    if window % 2 == 0:
        window -= 1
    if window <= poly_order:
        window = poly_order + 1 + (poly_order % 2 == 0)

    avg_temp_smooth = savgol_filter(avg_temp, window_length=window, polyorder=poly_order)

    return common_t, avg_temp_smooth