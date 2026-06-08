import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

#Hola mundo


def importer(name):
    csv_path = Path.cwd() / name
    df = pd.read_csv(csv_path, sep=';', decimal=',', header=0, engine='python')
    df.columns = [c.strip() for c in df.columns]
    arrays = {col: pd.to_numeric(df[col], errors='coerce').to_numpy() for col in df.columns}
    return arrays


def finder(target_temp, array):
    """Returns the index of the first value in array that is <= target_temp.
    Used to find where one run's starting temperature matches another run's cooling curve.
    Returns -1 if no such value is found."""
    for i in range(len(array)):
        if array[i] <= target_temp:
            return i
    return -1


def exponential(x, a, b):
    """Newton's law of cooling: T(t) = a·exp(b·t) + T_ambient"""
    return a * np.exp(b * x) + T_AMBIENT


# ── Load raw data ────────────────────────────────────────────────────────────

data_20 = importer("Temp Measurements 20-05-26.csv")
data_27 = importer("Temp Measurements 27-05-26.csv")
data_03 = importer("Temp Measurements 03-06-26.csv") 
data_05 = importer("Temp Measurements 05-06-26.csv") 

# Time arrays (one per run; all materials in a run share the same time axis)
time1 = data_20["Time (s) Run #1"]
time2 = data_20["Time (s) Run #3"]
time3 = data_27["Time (s) Run #1"]
time4 = data_03["Time (s) Run #1"]
time5 = data_03["Time (s) Run #2"]
time6 = data_05["Time (s) Run #2"]
time7 = data_05["Time (s) Run #3"]

time = [time1, time2, time3, time4, time5, time6, time7]  # for easy iteration

# Temperature arrays
# Run 1: aluminium probe wasn't placed on time → drop first 2000 samples
aluminium1 = data_20["Temperature 1 (°C) Run #1"][2000:]
time1_Al   = time1[:-2000]          # trim time to match

aluminium2 = data_20["Temperature 3 (°C) Run #3"]
aluminium3 = data_27["Temperature 3 (°C) Run #1"]
aluminium4 = data_03["Temperature 2 (°C) Run #1"]
aluminium5 = data_03["Temperature 2 (°C) Run #2"]

aluminium = [aluminium1, aluminium2, aluminium3, aluminium4, aluminium5]  # for easy iteration

tape1 = data_20["Temperature 2 (°C) Run #1"]
tape2 = data_20["Temperature 2 (°C) Run #3"]
tape3 = data_27["Temperature 2 (°C) Run #1"]
tape4 = data_03["Temperature 1 (°C) Run #1"]
tape5 = data_03["Temperature 3 (°C) Run #2"]
tape6 = data_05["Temperature 3 (°C) Run #2"][2000:]
time6_tape = time6[2000:]          # trim time to match
tape7 = data_05["Temperature 1 (°C) Run #3"][2000:]
time7_tape = time7[2000:]          # trim time to match

tape = [tape1, tape2, tape3, tape4, tape5, tape6, tape7]  # for easy iteration

glass1 = data_20["Temperature 3 (°C) Run #1"]
glass2 = data_20["Temperature 1 (°C) Run #3"]
glass3 = data_27["Temperature 1 (°C) Run #1"]
glass4 = data_03["Temperature 3 (°C) Run #2"]
glass5 = data_03["Temperature 1 (°C) Run #2"]

glass = [glass1, glass2, glass3, glass4, glass5]  # for easy iteration

copper1 = data_05["Temperature 1 (°C) Run #2"][2000:]
copper2 = data_05["Temperature 2 (°C) Run #2"][2000:]
time6_copper = time6[2000:]          # trim time to match
copper3 = data_05["Temperature 2 (°C) Run #3"][2000:]
time7_copper = time7[2000:]          # trim time to match
copper4 = data_05["Temperature 3 (°C) Run #3"]

copper = [copper1, copper2, copper3, copper4]  # for easy iteration

# ── State constants ───────────────────────────────────

sigma = 5.67e-8  # W/m²K⁴, Stefan-Boltzmann constant
T_AMBIENT = 22.0  # °C — measured room temperature


# ── Stitch runs onto a continuous timeline ───────────────────────────────────
# Strategy: find the index in run 2 where the temperature matches the *start*
# of the adjacent run, then offset that run's time axis so the curves join
# seamlessly.

def max_temperature(array):
    """Returns the array of the run with the highest initial temperature."""
    first_values = []
    for i in range(len(array)):
        first_values.append(array[i][0])
    winner = first_values.index(max(first_values))
    return winner


def stitch_times(time_full, array_material):
    """Reads the index of the maximum temperature in the material array, then finds the corresponding
    time for this index in every time array (offset), then deletes the array of maximum temperature (the reference)
    from the full time array, creating time_ref, then adds the offset to every value in every time in 
    the full time array, creating the stitched time_ref."""
    #print(f"Max temperature array index is {max_temperature(array_material)}.")
    ref_run_time = time_full.pop(max_temperature(array_material))
    ref_run_material = array_material.pop(max_temperature(array_material))
    #print(f"Reference run time array is {ref_run_time}.")
    
    for i, item in enumerate(array_material):
        idx = finder(item[0], ref_run_material) # This is the index position where the initial tempeture is in the reference
        #print(f"Index found for material {item} is {idx}.")
        if idx == -1:
            raise ValueError(f"Could not find temperature {item[0]:.2f} in reference run data.")
        else:
            offset = ref_run_time[idx] # Corresponding time for the temperature found in ref_run_material
            #print(f"Offset for material {item} is {offset:.2f}.")
            time_array = np.array(time_full[i]) # Convert the time array of the current run to a numpy array for easier manipulation
            time_full[i] = time_array + offset # Add the offset to every value in the time array of the current run
    return time_full, ref_run_time, array_material, ref_run_material

# Aluminium
time_Al = [time1_Al, time2, time3, time4, time5] # Only the first 5 runs have aluminium data
time_Al_stitched, ref_time_Al, aluminium_stitched, ref_aluminium = stitch_times(time_Al, aluminium)
time_Al_stitched.append(ref_time_Al) # Add the reference time array back to the stitched time arrays
time_Al = np.concatenate([np.concatenate(time_Al_stitched), ref_time_Al])
aluminium_stitched.append(ref_aluminium) # Add the reference aluminium array back to the stitched aluminium arrays
aluminium = np.concatenate([np.concatenate(aluminium_stitched), ref_aluminium])

# Tape
time_tape = [time1, time2, time3, time4, time5, time6_tape, time7_tape] # All runs have tape data
time_tape_stitched, ref_time_tape, tape_stitched, ref_tape = stitch_times(time_tape, tape)
time_tape_stitched.append(ref_time_tape) # Add the reference time array back to the stitched time arrays
time_tape = np.concatenate([np.concatenate(time_tape_stitched), ref_time_tape])
tape_stitched.append(ref_tape) # Add the reference tape array back to the stitched tape arrays
tape = np.concatenate([np.concatenate(tape_stitched), ref_tape])

# Glass
time_glass = [time1, time2, time3, time4, time5] # Only the first 5 runs have glass data
time_glass_stitched, ref_time_glass, glass_stitched, ref_glass = stitch_times(time_glass, glass)
time_glass_stitched.append(ref_time_glass) # Add the reference time array back to the stitched time arrays
time_glass = np.concatenate([np.concatenate(time_glass_stitched), ref_time_glass])
glass_stitched.append(ref_glass) # Add the reference glass array back to the stitched glass arrays
glass = np.concatenate([np.concatenate(glass_stitched), ref_glass])

# Copper
time_copper = [time6_copper, time6_copper, time7_copper, time7] # Only the last 2 runs have copper data
time_copper_stitched, ref_time_copper, copper_stitched, ref_copper = stitch_times(time_copper, copper)
time_copper_stitched.append(ref_time_copper) # Add the reference time array back to the stitched time arrays
time_copper = np.concatenate([np.concatenate(time_copper_stitched), ref_time_copper])
copper_stitched.append(ref_copper) # Add the reference copper array back to the stitched copper arrays
copper = np.concatenate([np.concatenate(copper_stitched), ref_copper])

# ── Clean (remove NaN / Inf) ─────────────────────────────────────────────────

def clean(time_arr, temp_arr):
    mask = ~(np.isnan(temp_arr) | np.isinf(temp_arr) |
             np.isnan(time_arr) | np.isinf(time_arr))
    return time_arr[mask], temp_arr[mask]

time_Al_c, al_c = clean(time_Al, aluminium)
time_tape_c, tape_c = clean(time_tape, tape)
time_glass_c, glass_c = clean(time_glass, glass)
time_copper_c, copper_c = clean(time_copper, copper)

# ── Exponential fits ─────────────────────────────────────────────────────────
# Fit each run independently, then report mean ± std across runs.
# The run-to-run spread is the physically meaningful uncertainty — it captures
# probe placement, ambient drift, and stitching imperfections that the
# covariance matrix cannot see.

def fit_single_run(time, temp):
    """Fit one run; return (a, b) or None on failure."""
    try:
        popt, _ = curve_fit(exponential, time, temp,
                            p0=(temp[0] - T_AMBIENT, -0.001), maxfev=10000)
        return popt
    except RuntimeError:
        return None

def goodness_of_fit(time, temp, popt):
    """
    Returns a dict with R² (least-squares) and χ² (chi-squared) metrics.
    
    R²  — fraction of variance explained by the fit. 1.0 = perfect.
    χ²  — sum of squared residuals weighted by variance. ~1.0/dof = good fit.
    χ²_reduced — χ² divided by degrees of freedom. Should be ≈ 1 for a good fit.
    """
    y_pred = exponential(time, *popt)
    residuals = temp - y_pred
    
    # ── R² (coefficient of determination) ───────────────────────────────────
    ss_res = np.sum(residuals ** 2)               # sum of squared residuals
    ss_tot = np.sum((temp - np.mean(temp)) ** 2)  # total variance
    r_squared = 1 - (ss_res / ss_tot)
    
    # ── χ² ──────────────────────────────────────────────────────────────────
    # σ is estimated from the residuals themselves (no external error bars).
    # This is standard when individual measurement uncertainties are unknown.
    sigma = np.std(residuals, ddof=len(popt))     # ddof = number of fit params
    if sigma == 0:
        chi2 = np.nan
        chi2_red = np.nan
    else:
        chi2 = np.sum((residuals / sigma) ** 2)
        dof  = len(temp) - len(popt)              # degrees of freedom
        chi2_red = chi2 / dof
    
    return {
        'R2':       r_squared,
        'chi2':     chi2,
        'chi2_red': chi2_red,
        'dof':      dof,
    }


def fit_per_run_pairs(times, temps, label):
    results = []
    for i, (t, temp) in enumerate(zip(times, temps), 1):
        t_arr    = np.asarray(t)
        temp_arr = np.asarray(temp)
        t_c, temp_c = clean(t_arr, temp_arr)
        popt = fit_single_run(t_c, temp_c)
        
        if popt is not None:
            gof = goodness_of_fit(t_c, temp_c, popt)
            results.append(popt)
            print(f"  Run {i}: a = {popt[0]:.4f},  b = {popt[1]:.2e} | "
                  f"R² = {gof['R2']:.4f},  "
                  f"χ²_red = {gof['chi2_red']:.4f}  (dof = {gof['dof']})")
        else:
            print(f"  Run {i}: fit failed")

    if not results:
        print(f"{label}: all runs failed.\n")
        return None

    results = np.array(results)
    mean = results.mean(axis=0)
    std  = results.std(axis=0, ddof=1) if results.shape[0] > 1 else np.array([0.0, 0.0])
    sem = std / np.sqrt(results.shape[0]) if results.shape[0] > 1 else np.array([0.0, 0.0])

    print(f"{label} summary — "
          f"a = {mean[0]:.4f} ± {sem[0]:.4f},  "
          f"b = {mean[1]:.2e} ± {sem[1]:.2e}\n")
    return mean, std

print("Exponential fit  T(t) = a·exp(b·t) + T_ambient,  per-run results:\n")

# Fit per stitched run pair for each material
popt_Al, sem_Al   = fit_per_run_pairs(time_Al_stitched, aluminium_stitched, "Aluminium")
popt_tape, sem_tape = fit_per_run_pairs(time_tape_stitched, tape_stitched, "Tape")
popt_glass, sem_glass = fit_per_run_pairs(time_glass_stitched, glass_stitched, "Glass")
popt_copper, sem_copper = fit_per_run_pairs(time_copper_stitched, copper_stitched, "Copper")

datasets = [
    (time_Al_c,       al_c,        popt_Al,      sem_Al,      0.04,  'Aluminium'),
    (time_tape_c,     tape_c,      popt_tape,    sem_tape,    0.95,  'Tape'),
    (time_glass_c,    glass_c,     popt_glass,   sem_glass,   0.9,    'Glass'),
    (time_copper_c,   copper_c,    popt_copper,  sem_copper,  0.02,   'Copper'),
]

print("Global goodness-of-fit  —  mean fit vs. full concatenated dataset:\n")

for t, temp, popt, sem, emissivity, label in datasets:
    if popt is None:
        print(f"  {label}: no fit available, skipping.")
        continue

    gof = goodness_of_fit(t, temp, popt)
    print(f"  {label}: k = {popt[1]:.5f} ± {sem[1]:.5f} s⁻¹ || R² = {gof['R2']:.4f},  "
          f"χ²_red = {gof['chi2_red']:.4f}  (dof = {gof['dof']})")

print()

# Computation of total radiative energy lost by each material, using the Stefan-Boltzmann law:
# P = εσA(T⁴ - T_env⁴)

print("--- TOTAL RADIATIVE ENERGY LOST ---")

total_energies = []
energy_errors = []
total_energies_model = []
energy_errors_model = []

for t, temp, popt, sem, emissivity, label in datasets:
    time_model = np.linspace(t.min(), t.max(), 5000)
    T_model_K = exponential(time_model, *popt) + 273.15
    T_K = temp + 273.15
    T_env_K = T_AMBIENT + 273.15
    dt = np.mean(np.diff(t, prepend=t[0]))  # Time intervals (with zero prepended for the first point)

    P_rad_flux = sigma * emissivity * (T_K**4 - T_env_K**4) 
    E_total = np.sum(P_rad_flux * dt)
    
    A = popt[0]
    K = popt[1]
    # P_uncertainty = (4 * sigma * emissivity * (T_K**3))   # Uncertainty in power due to uncertainty in cooling constant k
    P_uncertainty = np.sqrt((4 * sigma * emissivity * np.exp(K * t) * (A * np.exp(K * t) - T_env_K)**3 * sem[0])**2 + (4 * sigma * emissivity * A * t * np.exp(K * t) * (A * np.exp(K * t) - T_env_K)**3 * sem[1])**2)  # Uncertainty in power due to uncertainty in cooling constant k and A
    P_rel_uncertainty = P_uncertainty / P_rad_flux * 100  # Relative uncertainty in power due to uncertainty in k

    E_error = np.sqrt(np.sum((P_uncertainty * dt)**2))
    E_rel_error = E_error / E_total * 100  # Relative error in total energy

    total_energies.append(E_total)
    energy_errors.append(E_error)  

    print(f"{label}:")
    print(f"  Mean Power Radiated = {np.mean(P_rad_flux):,.2f} ± {np.mean(P_rel_uncertainty):,.2f} % W/m^2 || Total Energy = {E_total:,.2f} ± {E_rel_error:,.2f} % J/m²\n")


# ── Plot ─────────────────────────────────────────────────────────────────────
# Each material gets its own subplot.
# scatter = light/muted tone; fit line = deep/saturated tone of the same hue.
# (scatter_color, fit_color, title)

styles = {
    'Aluminium': ('#7EB8E8', '#1A5F99'),   # light blue  / deep blue
    'Tape':      ('#FFBA7A', '#C85A00'),   # light orange / deep orange
    'Glass':     ('#7DD17D', '#1E6E1E'),   # light green / deep green
    'Copper':    ("#F460ED", "#6D138B"),   # light violet  / deep violet
}


fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
fig.suptitle("Cooling curves — Aluminium, Tape, Glass, Copper", fontsize=13)

for ax, (t, temp, popt, perr, emissivity, label) in zip(axes.flatten(), datasets):
    scatter_c, fit_c = styles[label]

    ax.scatter(t, temp, s=1, alpha=0.35, color=scatter_c, label='Data')
    if popt is not None:
        x_model = np.linspace(t.min(), t.max(), 500)
        y_model = exponential(x_model, *popt)
        ax.plot(x_model, y_model, color=fit_c, linewidth=2,
                label=f'Fit  (k = {popt[1]:.5f} ± {perr[1]:.5f} s⁻¹)')

    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.legend(markerscale=6, fontsize=8)

axes[0, 0].set_ylabel("Temperature (°C)")
plt.tight_layout()
plt.savefig("cooling_curves.pdf", format="pdf", bbox_inches="tight")
plt.show()


t, temp, popt, sem, emissivity, label = zip(*datasets)

plt.figure(figsize=(8,6))

plt.errorbar(
        emissivity, 
        total_energies, 
        yerr=energy_errors, 
        fmt='o', 
        color='purple', 
        markersize=8, 
        capsize=5,
        label='Experimental Data')


for i in range(4):
    y_offset = -20 if i == 1 else 15
    short_label = label[i].split(' (')[0]
    
    plt.annotate(
        short_label, 
        (emissivity[i], total_energies[i]), 
        textcoords="offset points", 
        xytext=(0, y_offset), 
        ha='center',
        fontsize=10
    )

coeffs = np.polyfit(emissivity, total_energies, 1)
trend_x = np.linspace(0, 1.0, 100)
trend_y = np.polyval(coeffs, trend_x)
plt.plot(trend_x, trend_y, '--', color='gray', alpha=0.5, label='Linear Fit')
plt.title("Relationship between Emissivity and Total Energy Lost")
plt.xlabel("Emissivity ($\epsilon$)")
plt.ylabel("Total Energy Lost ($J/m^2$)")
plt.xlim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

