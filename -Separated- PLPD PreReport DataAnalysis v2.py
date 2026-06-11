import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import ecdf

# ── Physical constants & experimental parameters ─────────────────────────────

sigma      = 5.67e-8        # W m⁻² K⁻⁴  Stefan–Boltzmann constant
T_AMBIENT  = 22.0           # °C   measured room temperature
T_AMB_K    = T_AMBIENT + 273.15

M_WATER    = 0.500          # kg   (500 ml jar)
DM_WATER   = 0.010          # kg   filling uncertainty (~10 ml)
C_WATER    = 4186.0         # J kg⁻¹ K⁻¹
DC_WATER   = 40.0           # J kg⁻¹ K⁻¹ (T-dependence of c over 50–80 °C)


AREA       = 0.28*0.12
DAREA_REL  = 0.05           # 5 % relative uncertainty on A

SIGMA_T_SYS = 0.5           # °C  systematic probe-calibration uncertainty

EMISSIVITY = {    # ε,    Δε
    'Aluminium': (0.04, 0.02),
    'Tape':      (0.95, 0.03),
    'Glass':     (0.90, 0.05),
    'Copper':    (0.02, 0.01),
}

# ── Data import ──────────────────────────────────────────────────────────────

def importer(name):
    csv_path = Path.cwd() / name
    df = pd.read_csv(csv_path, sep=';', decimal=',', header=0, engine='python')
    df.columns = [c.strip() for c in df.columns]
    arrays = {col: pd.to_numeric(df[col], errors='coerce').to_numpy() for col in df.columns}
    return arrays

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
time1_Al   = time1[2000:] - time1[2000]        # trim time to match

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
time6_tape = time6[2000:] - time6[2000]        # trim time to match
tape7 = data_05["Temperature 1 (°C) Run #3"][2000:]
time7_tape = time7[2000:] - time7[2000]        # trim time to match

tape = [tape1, tape2, tape3, tape4, tape5, tape6, tape7]  # for easy iteration

glass1 = data_20["Temperature 3 (°C) Run #1"]
glass2 = data_20["Temperature 1 (°C) Run #3"]
glass3 = data_27["Temperature 1 (°C) Run #1"]
glass4 = data_03["Temperature 3 (°C) Run #2"]
glass5 = data_03["Temperature 1 (°C) Run #2"]

glass = [glass1, glass2, glass3, glass4, glass5]  # for easy iteration

copper1 = data_05["Temperature 1 (°C) Run #2"][3000:]
time61_copper = time6[3000:] - time6[3000]
copper2 = data_05["Temperature 2 (°C) Run #2"][4000:]
time62_copper = time6[4000:] - time6[4000]        # trim time to match
copper3 = data_05["Temperature 2 (°C) Run #3"][3000:]
time7_copper = time7[3000:] - time7[3000]          # trim time to match
copper4 = data_05["Temperature 3 (°C) Run #3"][3000:]

copper = [copper1, copper2, copper3, copper4]  # for easy iteration

# ── Stitch runs onto a continuous timeline ───────────────────────────────────

def finder(target_temp, array):
    for i in range(len(array)):
        if array[i] <= target_temp:
            return i
    return -1

def max_temperature(array):
    first_values = []
    for i in range(len(array)):
        first_values.append(array[i][0])
    winner = first_values.index(max(first_values))
    return winner

def stitch_times(time_full, array_material):
    ref_run_time     = time_full.pop(max_temperature(array_material))
    ref_run_material = array_material.pop(max_temperature(array_material))

    for i, item in enumerate(array_material):
        idx = finder(item[0], ref_run_material)
        if idx == -1:
            raise ValueError(f"Could not find temperature {item[0]:.2f} in reference run data.")
        offset       = ref_run_time[idx]
        time_array   = np.array(time_full[i])
        time_full[i] = time_array + offset
    return time_full, ref_run_time, array_material, ref_run_material

# Aluminium
time_Al = [time1_Al, time2, time3, time4, time5]
time_Al_stitched, ref_time_Al, aluminium_stitched, ref_aluminium = stitch_times(time_Al, aluminium)
time_Al_stitched.append(ref_time_Al)
aluminium_stitched.append(ref_aluminium)
time_Al    = np.concatenate(time_Al_stitched)
aluminium  = np.concatenate(aluminium_stitched)

# Tape
time_tape = [time1, time2, time3, time4, time5, time6_tape, time7_tape]
time_tape_stitched, ref_time_tape, tape_stitched, ref_tape = stitch_times(time_tape, tape)
time_tape_stitched.append(ref_time_tape)
tape_stitched.append(ref_tape)
time_tape  = np.concatenate(time_tape_stitched)
tape       = np.concatenate(tape_stitched)

# Glass
time_glass = [time1, time2, time3, time4, time5]
time_glass_stitched, ref_time_glass, glass_stitched, ref_glass = stitch_times(time_glass, glass)
time_glass_stitched.append(ref_time_glass)
glass_stitched.append(ref_glass)
time_glass = np.concatenate(time_glass_stitched)
glass      = np.concatenate(glass_stitched)

# Copper
time_copper = [time61_copper, time62_copper, time7_copper, time7_copper]
time_copper_stitched, ref_time_copper, copper_stitched, ref_copper = stitch_times(time_copper, copper)
time_copper_stitched.append(ref_time_copper)
copper_stitched.append(ref_copper)
time_copper = np.concatenate(time_copper_stitched)
copper      = np.concatenate(copper_stitched)

# ── Clean (remove NaN / Inf) ─────────────────────────────────────────────────

def clean(time_arr, temp_arr):
    mask = ~(np.isnan(temp_arr) | np.isinf(temp_arr) |
             np.isnan(time_arr) | np.isinf(time_arr))
    return time_arr[mask], temp_arr[mask]

time_Al_c,     al_c      = clean(time_Al,     aluminium)
time_tape_c,   tape_c    = clean(time_tape,   tape)
time_glass_c,  glass_c   = clean(time_glass,  glass)
time_copper_c, copper_c  = clean(time_copper, copper)

# ── Newton-cooling fits (per run → mean ± SEM) ───────────────────────────────

def linear_model(x, a, b):
    return a * x + b

def error_sigma(temp):
    return 0.5/np.absolute(temp - T_AMBIENT)

def goodness_of_fit(time, temp, popt):
    linear_temp = np.log(temp)
    y_pred    = linear_model(time, *popt)
    residuals = linear_temp - y_pred
    residual_norm = residuals/error_sigma(temp)
    ss_res    = np.sum(residuals ** 2)
    ss_tot    = np.sum((linear_temp - np.mean(linear_temp)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    chi2 = np.sum(residual_norm ** 2)
    #chi2      = np.sum((residuals / probe_error) ** 2)
    dof       = len(temp) - len(popt)
    chi2_red  = chi2 / dof
    return {'R2': r_squared, 'chi2': chi2, 'chi2_red': chi2_red, 'dof': dof, 'R_n': residual_norm}

def fit_single_run(time, temp):
    linear_temp = np.log(temp)
    try:
        popt, pcov = curve_fit(linear_model, time, linear_temp,
                            p0=(-0.001, np.log(70)), maxfev=10000, sigma = 0.5/np.absolute(temp - T_AMBIENT), absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov)) # Standard deviation error in parameters
        return popt, perr 
    except RuntimeError:
        return None

def fit_per_run_pairs(times, temps, label):
    results = []
    results_errors = []
    for i, (t, temp) in enumerate(zip(times, temps), 1):
        t_c, temp_c = clean(np.asarray(t), np.asarray(temp))
        popt, perr = fit_single_run(t_c, temp_c)
        if popt is not None:
            gof = goodness_of_fit(t_c, temp_c, popt)
            results.append(popt)
            results_errors.append(perr)
            print(f"  Run {i}: a = {popt[0]:.5f} +/- {perr[0]:.4g},  b = {popt[1]:.5f} +/- {perr[1]:.4g} | "
                  f"R² = {gof['R2']:.4f},  χ²_red = {gof['chi2_red']:.4f}  (dof = {gof['dof']})")
        else:
            print(f"  Run {i}: fit failed")
    if not results:
        print(f"{label}: all runs failed.\n")
        return None
    results = np.array(results)
    mean_results = results.mean(axis=0)
    results_errors = np.array(results_errors)
    mean_results_errors = results_errors.mean(axis=0)
    print(f"{label} summary — a = {mean_results[0]:.4f} ± {mean_results_errors[0]:.2g},  b = {mean_results[1]:.4f} ± {mean_results_errors[1]:.2g}\n")
    return mean_results, mean_results_errors

print("Linear fit  T(t) = a·t + b,  per-run results:\n")

popt_Al,     perr_Al,    = fit_per_run_pairs(time_Al_stitched,     aluminium_stitched, "Aluminium")
popt_tape,   perr_tape,  = fit_per_run_pairs(time_tape_stitched,   tape_stitched,      "Tape")
popt_glass,  perr_glass,  = fit_per_run_pairs(time_glass_stitched,  glass_stitched,     "Glass")
popt_copper, perr_copper, = fit_per_run_pairs(time_copper_stitched, copper_stitched,    "Copper")

datasets = [
    (time_Al_c,     al_c,     popt_Al,     perr_Al,     0.04, 'Aluminium'),
    (time_tape_c,   tape_c,   popt_tape,   perr_tape,   0.95, 'Tape'),
    (time_glass_c,  glass_c,  popt_glass,  perr_glass,  0.90, 'Glass'),
    (time_copper_c, copper_c, popt_copper, perr_copper, 0.02, 'Copper'),
]

print("Global goodness-of-fit  —  mean fit vs. full concatenated dataset:\n")
for t, temp, popt, perr, emissivity, label in datasets:
    gof = goodness_of_fit(t, temp, popt)
    print(f"  {label:9s}: k = {popt[0]:.5f} ± {perr[0]:.3g} s⁻¹ | "
          f"R² = {gof['R2']:.4f},  χ²_red = {gof['chi2_red']:.4f}  (dof = {gof['dof']})")
print()

# ── PLOT 1 — cooling curves + Newton fits ────────────────────────────────────

styles = {
    'Aluminium': ('#7EB8E8', '#1A5F99'),
    'Tape':      ('#FFBA7A', '#C85A00'),
    'Glass':     ('#7DD17D', '#1E6E1E'),
    'Copper':    ('#F490ED', '#6D138B'),
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
fig.suptitle("Cooling curves with Newton-law fits", fontsize=13)
for ax, (t, temp, popt, perr, emissivity, label) in zip(axes.flatten(), datasets):
    scatter_c, fit_c = styles[label]
    ax.scatter(t[::20], np.log(temp[::20]), s=1, alpha=0.35, color=scatter_c, label='Data')
    x_model = np.linspace(t.min(), t.max(), 500)
    y_model = linear_model(x_model, *popt)
    ax.plot(x_model, y_model, color=fit_c, linewidth=2,
            label=f'Fit  (k = {popt[0]:.5f} ± {perr[0]:.5f} s⁻¹)')
    #ax.axhline(np.log(T_AMBIENT), color='gray', ls=':', lw=1)
    ax.set_title(label)
    ax.set_xlabel("Time (s)")
    ax.legend(markerscale=6, fontsize=8)
axes[0, 0].set_ylabel("Log Temperature ln(°C)")
axes[1, 0].set_ylabel("Log Temperature ln(°C)")
plt.tight_layout()
plt.show()

# ── PLOT 1.2 — comparison of empiracal CDF to normalized Gaussian CDF ────────

def residuals(t, temp, pcov):
    df = pd.DataFrame({'t': t, 'temp': temp})
    result = df.groupby('t', as_index=False)['temp'].mean()
    unique_t = result['t'].to_numpy()
    avg_temp = result['temp'].to_numpy()
    gof = goodness_of_fit(unique_t, avg_temp, pcov)
    R_sorted = np.sort(gof['R_n'])
    N = len(R_sorted)
    ecdf = np.arange(1, N+1) / N
    return R_sorted, ecdf

plt.figure(figsize=(6,5))
colors = {'Aluminium': 'r','Tape': 'g','Glass': 'b','Copper': 'y'}
for t, temp, popt, perr, emissivity, labels in datasets:
    R_n, e_cdf = residuals(t, temp, popt)
    plt.step(R_n, e_cdf, where='post', label='Empirical CDF', color=colors[labels])

plt.show()

# ── PLOT 2 — f_rad = P_rad / P_total vs time ─────────────────────────────────

BIN_DT = 60.0   # s — bin width for averaging before numerical differentiation

def bin_average(t, y, dt=BIN_DT):
    edges = np.arange(t.min(), t.max() + dt, dt)
    idx   = np.digitize(t, edges)
    tb, yb, ysem = [], [], []
    for k in range(1, len(edges)):
        m = idx == k
        if m.sum() < 10:
            continue
        tb.append(t[m].mean())
        yb.append(y[m].mean())
        ysem.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(tb), np.array(yb), np.array(ysem)

def frad_from_data(t_run, T_run, eps, deps):
    t, T, T_sem = bin_average(*clean(t_run, T_run))
    dTdt  = np.gradient(T, t)
    T_K   = T + 273.15
    P_rad = eps * sigma * AREA * (T_K**4 - T_AMB_K**4)
    P_tot = -M_WATER * C_WATER * dTdt
    dPrad_rel = np.sqrt((deps/eps)**2 + DAREA_REL**2 +
                        (4*T_K**3 * SIGMA_T_SYS / (T_K**4 - T_AMB_K**4))**2)
    sdTdt     = np.sqrt(2) * T_sem / (2 * BIN_DT)
    dPtot_rel = np.sqrt((DM_WATER/M_WATER)**2 + (DC_WATER/C_WATER)**2 +
                        (sdTdt / np.abs(dTdt))**2)
    mask  = P_tot > 1.0
    f     = P_rad[mask] / P_tot[mask]
    f_err = f * np.sqrt(dPrad_rel[mask]**2 + dPtot_rel[mask]**2)
    return t[mask], f, f_err

# reference run per material (returned separately by stitch_times)
plot2_data = [
    (ref_time_Al,     ref_aluminium, 'Aluminium'),
    (ref_time_tape,   ref_tape,      'Tape'),
    (ref_time_glass,  ref_glass,     'Glass'),
    (ref_time_copper, ref_copper,    'Copper'),
]

fig2, ax2 = plt.subplots(figsize=(9, 6))
for t_ref, T_ref, label in plot2_data:
    eps, deps = EMISSIVITY[label]
    tf, f, f_err = frad_from_data(t_ref, T_ref, eps, deps)
    _, fc = styles[label]
    ax2.plot(tf, f, color=fc, lw=1.8,
             label=f"{label} (ε = {eps:.2f}, f̄ = {f.mean():.2f})")
    ax2.fill_between(tf, f - f_err, f + f_err, color=fc, alpha=0.18)
    print(f"  {label:9s}: mean f_rad = {f.mean():.3f} ± {np.mean(f_err):.3f}")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel(r"$f_\mathrm{rad} = P_\mathrm{rad}\,/\,P_\mathrm{total}$")
ax2.set_title("Radiative fraction of the heat loss — computed directly from T(t) data")
ax2.set_ylim(bottom=0)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=9)
plt.tight_layout()
plt.show()

# ── PLOT 3 — total radiated energy vs emissivity ─────────────────────────────

T_HI, T_LO = 75.0, 58.0
E_TOTAL_WINDOW = M_WATER * C_WATER * (T_HI - T_LO)

print("\n" + "═" * 70)
print(f"RADIATED ENERGY over the common window {T_HI:.0f} → {T_LO:.0f} °C "
      f"(total heat lost = {E_TOTAL_WINDOW/1e3:.1f} kJ per jar)")
print("═" * 70)

# full stitched dataset per material
plot3_data = [
    (time_Al_c,     al_c,     'Aluminium'),
    (time_tape_c,   tape_c,   'Tape'),
    (time_glass_c,  glass_c,  'Glass'),
    (time_copper_c, copper_c, 'Copper'),
]

eps_list, deps_list, E_list, dE_list, labels = [], [], [], [], []
for t_all, y_all, label in plot3_data:
    eps, deps = EMISSIVITY[label]
    order = np.argsort(t_all)
    tb, Tb, _ = bin_average(t_all[order], y_all[order], dt=30.0)
    win = (Tb <= T_HI) & (Tb >= T_LO)
    tw, Tw = tb[win], Tb[win]
    T_K  = Tw + 273.15
    Pr   = eps * sigma * AREA * (T_K**4 - T_AMB_K**4)
    E    = np.trapezoid(Pr, tw)
    rel  = np.sqrt((deps/eps)**2 + DAREA_REL**2 +
                   (np.mean(4*T_K**3) * SIGMA_T_SYS /
                    np.mean(T_K**4 - T_AMB_K**4))**2)
    dE   = E * rel
    eps_list.append(eps); deps_list.append(deps)
    E_list.append(E);     dE_list.append(dE);     labels.append(label)
    print(f"  {label:9s}: ε = {eps:.2f} ± {deps:.2f} | "
          f"E_rad = {E:8.1f} ± {dE:6.1f} J "
          f"({E/E_TOTAL_WINDOW*100:5.1f} % of total) | "
          f"cooling time {tw.max()-tw.min():6.0f} s")

eps_a, E_a, dE_a = map(np.array, (eps_list, E_list, dE_list))

w = 1 / dE_a**2
coeffs, cov = np.polyfit(eps_a, E_a, 1, w=np.sqrt(w), cov=True)
slope, intercept = coeffs
dslope = np.sqrt(cov[0, 0])
print(f"\n  Weighted linear fit: E = ({slope:.0f} ± {dslope:.0f})·ε "
      f"+ {intercept:.0f} J")

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.errorbar(eps_a, E_a, yerr=dE_a, xerr=deps_list, fmt='o', color='purple',
             ms=8, capsize=5, label='Experimental data', zorder=3)
for i, lab in enumerate(labels):
    dx, dy = (0, 12)
    if lab == 'Copper':    dx, dy = (10, -20)
    if lab == 'Aluminium': dx, dy = (35, 8)
    ax3.annotate(lab, (eps_a[i], E_a[i]), textcoords="offset points",
                 xytext=(dx, dy), ha='center', fontsize=10)
xs = np.linspace(0, 1.0, 100)
ax3.plot(xs, np.polyval(coeffs, xs), '--', color='gray', alpha=0.7,
         label=f'Weighted linear fit (slope = {slope:.0f} ± {dslope:.0f} J)')
ax3.set_xlabel(r"Emissivity $\varepsilon$")
ax3.set_ylabel(r"Radiated energy over 75→58 °C window  $E_\mathrm{rad}$  (J)")
ax3.set_title("Total radiated energy vs emissivity")
ax3.set_xlim(-0.05, 1.05); ax3.grid(alpha=0.3); ax3.legend()
plt.tight_layout()
plt.show()
