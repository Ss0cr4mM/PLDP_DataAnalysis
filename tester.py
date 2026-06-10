import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# ── Physical constants & experimental parameters ─────────────────────────────

SIGMA      = 5.67e-8        # W m⁻² K⁻⁴  Stefan–Boltzmann constant
T_AMBIENT  = 22.0           # °C   measured room temperature
T_AMB_K    = T_AMBIENT + 273.15

M_WATER    = 0.500          # kg   (500 ml jar)
DM_WATER   = 0.010          # kg   filling uncertainty (~10 ml)
C_WATER    = 4186.0         # J kg⁻¹ K⁻¹
DC_WATER   = 40.0           # J kg⁻¹ K⁻¹ (T-dependence of c over 50–80 °C)

# Jar geometry → radiating area (lateral surface). REPLACE with your measured
# jar dimensions if different!
JAR_D      = 0.085          # m, outer diameter
JAR_H      = 0.105          # m, wetted height
AREA       = np.pi * JAR_D * JAR_H     # ≈ 0.028 m²
DAREA_REL  = 0.05           # 5 % relative uncertainty on A

SIGMA_T_SYS   = 0.5          # °C  systematic probe-calibration uncertainty


EMISSIVITY = {    # ε,    Δε
    'Aluminium': (0.04, 0.02),
    'Tape':      (0.95, 0.03),
    'Glass':     (0.90, 0.05),
    'Copper':    (0.02, 0.01),  
}

# ── Data import ──────────────────────────────────────────────────────────────

def importer(name):
    """Reads a PASCO Capstone export (';' separated, ',' decimals)."""
    candidates = [Path('/mnt/project') / name, Path.cwd() / name]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(name)
    df = pd.read_csv(csv_path, sep=';', decimal=',', header=0, engine='python')
    df.columns = [c.strip().lstrip('\ufeff').strip('"') for c in df.columns]
    return {col: pd.to_numeric(df[col], errors='coerce').to_numpy()
            for col in df.columns}

data_20 = importer("Temp_Measurements_200526.csv")
data_27 = importer("Temp_Measurements_270526.csv")
data_03 = importer("Temp_Measurements_030626.csv")
data_05 = importer("Temp_Measurements_050626.csv")

time1 = data_20["Time (s) Run #1"]
time2 = data_20["Time (s) Run #3"]
time3 = data_27["Time (s) Run #1"]
time4 = data_03["Time (s) Run #1"]
time5 = data_03["Time (s) Run #2"]
time6 = data_05["Time (s) Run #2"]
time7 = data_05["Time (s) Run #3"]

# Aluminium (probe late in run 1 → drop first 2000 samples)
aluminium1 = data_20["Temperature 1 (°C) Run #1"][2000:]
time1_Al   = time1[2000:] - time1[2000]
aluminium2 = data_20["Temperature 3 (°C) Run #3"]
aluminium3 = data_27["Temperature 3 (°C) Run #1"]
aluminium4 = data_03["Temperature 2 (°C) Run #1"]
aluminium5 = data_03["Temperature 2 (°C) Run #2"]
aluminium  = [aluminium1, aluminium2, aluminium3, aluminium4, aluminium5]

tape1 = data_20["Temperature 2 (°C) Run #1"]
tape2 = data_20["Temperature 2 (°C) Run #3"]
tape3 = data_27["Temperature 2 (°C) Run #1"]
tape4 = data_03["Temperature 1 (°C) Run #1"]
tape5 = data_03["Temperature 3 (°C) Run #2"]
tape6 = data_05["Temperature 3 (°C) Run #2"][2000:]
time6_tape = time6[2000:] - time6[2000]
tape7 = data_05["Temperature 1 (°C) Run #3"][2000:]
time7_tape = time7[2000:] - time7[2000]
tape  = [tape1, tape2, tape3, tape4, tape5, tape6, tape7]

glass1 = data_20["Temperature 3 (°C) Run #1"]
glass2 = data_20["Temperature 1 (°C) Run #3"]
glass3 = data_27["Temperature 1 (°C) Run #1"]
glass4 = data_03["Temperature 3 (°C) Run #2"]
glass5 = data_03["Temperature 1 (°C) Run #2"]
glass  = [glass1, glass2, glass3, glass4, glass5]

copper1 = data_05["Temperature 1 (°C) Run #2"][3000:]
time61_copper = time6[3000:] - time6[3000]
copper2 = data_05["Temperature 2 (°C) Run #2"][4000:]
time62_copper = time6[4000:] - time6[4000]
copper3 = data_05["Temperature 2 (°C) Run #3"][3000:]
time7_copper  = time7[3000:] - time7[3000]
copper4 = data_05["Temperature 3 (°C) Run #3"][3000:]
copper  = [copper1, copper2, copper3, copper4]

# ── Stitch runs onto a continuous timeline ───────────────────────────────────
# The run with the highest initial T is the reference; every other run is
# shifted so that its starting temperature lands on the reference curve.

def finder(target_temp, array):
    """Index of first value <= target_temp, or -1."""
    idx = np.argmax(array <= target_temp)
    if array[idx] <= target_temp:
        return idx
    return -1

def stitch_times(time_list, temp_list):
    """Returns (times, temps) lists with all runs offset onto the reference
    timeline. The reference run is placed LAST in the returned lists.
    Does not duplicate the reference (bug in v2)."""
    time_list = list(time_list)
    temp_list = [np.asarray(t) for t in temp_list]
    ref_i     = int(np.argmax([t[0] for t in temp_list]))
    ref_time  = np.asarray(time_list.pop(ref_i))
    ref_temp  = temp_list.pop(ref_i)

    out_times, out_temps = [], []
    for t, temp in zip(time_list, temp_list):
        idx = finder(temp[0], ref_temp)
        if idx == -1:
            raise ValueError(f"Start temp {temp[0]:.2f} °C not found in reference run.")
        offset = ref_time[idx]
        out_times.append(np.asarray(t) + offset)
        out_temps.append(temp)
    out_times.append(ref_time)
    out_temps.append(ref_temp)
    return out_times, out_temps

time_Al_runs,   al_runs    = stitch_times([time1_Al, time2, time3, time4, time5], aluminium)
time_tape_runs, tape_runs  = stitch_times([time1, time2, time3, time4, time5, time6_tape, time7_tape], tape)
time_gl_runs,   glass_runs = stitch_times([time1, time2, time3, time4, time5], glass)
time_cu_runs,   cu_runs    = stitch_times([time61_copper, time62_copper, time7_copper, time7_copper], copper)

def clean(t, y):
    t, y = np.asarray(t), np.asarray(y)
    m = ~(np.isnan(y) | np.isinf(y) | np.isnan(t) | np.isinf(t))
    return t[m], y[m]

def concat_runs(times, temps):
    t = np.concatenate(times); y = np.concatenate(temps)
    return clean(t, y)

time_Al_c,   al_c    = concat_runs(time_Al_runs,   al_runs)
time_tape_c, tape_c  = concat_runs(time_tape_runs, tape_runs)
time_gl_c,   glass_c = concat_runs(time_gl_runs,   glass_runs)
time_cu_c,   cu_c    = concat_runs(time_cu_runs,   cu_runs)

# ── Newton-cooling fits (per run → mean ± SEM) ───────────────────────────────

def exponential(t, a, b):
    """Newton's law of cooling: T(t) = a·exp(b·t) + T_ambient."""
    return a * np.exp(b * t) + T_AMBIENT

def goodness_of_fit(t, temp, popt, probe_error=0.5):
    res   = temp - exponential(t, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((temp - temp.mean())**2)
    chi2   = np.sum((res / probe_error)**2)
    dof    = len(temp) - len(popt)
    rms    = np.sqrt(np.mean(res**2)) 
                                        
    return {'R2': 1 - ss_res/ss_tot, 'chi2': chi2,
            'chi2_red': chi2/dof, 'dof': dof, 'rms': rms}
def fit_material(times, temps, label):
    """Fits each run independently; returns (mean(a,b), sem(a,b))."""
    results = []
    print(f"{label}:")
    for i, (t, temp) in enumerate(zip(times, temps), 1):
        t_c, y_c = clean(t, temp)
        try:
            popt, _ = curve_fit(exponential, t_c, y_c,
                                p0=(y_c[0] - T_AMBIENT, -1e-4), maxfev=20000)
        except RuntimeError:
            print(f"  Run {i}: fit failed"); continue
        gof = goodness_of_fit(t_c, y_c, popt)
        results.append(popt)
        print(f"  Run {i}: a = {popt[0]:7.3f} °C,  k = {-popt[1]:.3e} s⁻¹ | "
              f"R² = {gof['R2']:.4f},  χ²_red = {gof['chi2_red']:.2f}")
    results = np.array(results)
    mean = results.mean(axis=0)
    if len(results) > 1:
        sem = results.std(axis=0, ddof=1) / np.sqrt(len(results))
    else:
        sem = np.zeros(2)
    print(f"  → mean: a = {mean[0]:.3f} ± {sem[0]:.3f} °C,  "
          f"k = {-mean[1]:.3e} ± {sem[1]:.3e} s⁻¹\n")
    return mean, sem

print("═" * 70)
print("NEWTON-COOLING FITS  T(t) = a·e^{-kt} + T_amb   (per-run → mean ± SEM)")
print("═" * 70)
popt_Al,   sem_Al   = fit_material(time_Al_runs,   al_runs,    "Aluminium")
popt_tape, sem_tape = fit_material(time_tape_runs, tape_runs,  "Tape")
popt_gl,   sem_gl   = fit_material(time_gl_runs,   glass_runs, "Glass")
popt_cu,   sem_cu   = fit_material(time_cu_runs,   cu_runs,    "Copper")

datasets = [
    # (stitched t, stitched T, per-run t, per-run T, popt, sem, label)
    (time_Al_c,   al_c,    time_Al_runs,   al_runs,    popt_Al,   sem_Al,   'Aluminium'),
    (time_tape_c, tape_c,  time_tape_runs, tape_runs,  popt_tape, sem_tape, 'Tape'),
    (time_gl_c,   glass_c, time_gl_runs,   glass_runs, popt_gl,   sem_gl,   'Glass'),
    (time_cu_c,   cu_c,    time_cu_runs,   cu_runs,    popt_cu,   sem_cu,   'Copper'),
]

print("Global goodness of fit (mean fit vs full stitched dataset):")
for t, y, *_, popt, sem, label in [(d[0], d[1], d[4], d[5], d[6]) for d in datasets]:
    pass  # (kept simple below)
for d in datasets:
    t, y, _, _, popt, sem, label = d
    gof = goodness_of_fit(t, y, popt)
    print(f"  {label:9s}: k = {-popt[1]:.3e} ± {sem[1]:.3e} s⁻¹ | "
          f"R² = {gof['R2']:.4f},  χ²_red = {gof['chi2_red']:.2f}")
print()

# ── PLOT 1 — cooling curves + Newton fits ────────────────────────────────────

styles = {'Aluminium': ('#7EB8E8', '#1A5F99'),
          'Tape':      ('#FFBA7A', '#C85A00'),
          'Glass':     ('#7DD17D', '#1E6E1E'),
          'Copper':    ('#F490ED', '#6D138B')}

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
fig.suptitle("Cooling curves with Newton-law fits", fontsize=13)
for ax, d in zip(axes.flat, datasets):
    t, y, _, _, popt, sem, label = d
    sc, fc = styles[label]
    # decimate scatter for a lighter figure (every 20th point ≈ 1 Hz)
    ax.scatter(t[::20], y[::20], s=1, alpha=0.3, color=sc, label='Data')
    xm = np.linspace(t.min(), t.max(), 500)
    ax.plot(xm, exponential(xm, *popt), color=fc, lw=2,
            label=f'Newton fit\nk = ({-popt[1]*1e4:.2f} ± {sem[1]*1e4:.2f})·10⁻⁴ s⁻¹')
    ax.axhline(T_AMBIENT, color='gray', ls=':', lw=1)
    ax.set_title(label); ax.set_xlabel("Time (s)")
    ax.legend(markerscale=6, fontsize=8, loc='upper right')
axes[0, 0].set_ylabel("Temperature (°C)"); axes[1, 0].set_ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# ── PLOT 2 — f_rad = P_rad / P_total vs time, straight from the data ─────────
# P_total = -m·c·dT/dt  with dT/dt obtained numerically from BIN-AVERAGED raw
# data (no fit parameters anywhere). P_rad = ε·σ·A·(T⁴ - T_amb⁴) from raw T.

BIN_DT = 60.0   # s — bin width; the 20 Hz raw data is far too noisy to
                #     differentiate point-by-point, so we average first.

def bin_average(t, y, dt=BIN_DT):
    """Bin-averages (t, y); returns bin centres, means and SEM of y per bin."""
    edges  = np.arange(t.min(), t.max() + dt, dt)
    idx    = np.digitize(t, edges)
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
    """Radiative fraction vs time computed purely from the measured T(t).
    Returns t, f_rad, σ_f."""
    t, T, T_sem = bin_average(*clean(t_run, T_run))
    dTdt = np.gradient(T, t)                       # °C/s, from data
    # --- powers -------------------------------------------------------------
    T_K   = T + 273.15
    P_rad = eps * SIGMA * AREA * (T_K**4 - T_AMB_K**4)
    P_tot = -M_WATER * C_WATER * dTdt
    # --- uncertainties ------------------------------------------------------
    # P_rad: ε, A and the SYSTEMATIC ±2 °C calibration offset on T
    dPrad_rel = np.sqrt((deps/eps)**2 + DAREA_REL**2 +
                        (4*T_K**3 * SIGMA_T_SYS / (T_K**4 - T_AMB_K**4))**2)
    # P_tot: m, c and the bin-SEM on the derivative.
    # gradient over neighbouring bins: σ(dT/dt) ≈ √2 · σ_T,bin / (2·Δt_bin)
    sdTdt     = np.sqrt(2) * T_sem / (2 * BIN_DT)
    dPtot_rel = np.sqrt((DM_WATER/M_WATER)**2 + (DC_WATER/C_WATER)**2 +
                        (sdTdt / np.abs(dTdt))**2)
    # --- fraction -----------------------------------------------------------
    mask  = P_tot > 1.0          # drop bins where dT/dt is noise-dominated
    f     = P_rad[mask] / P_tot[mask]
    f_err = f * np.sqrt(dPrad_rel[mask]**2 + dPtot_rel[mask]**2)
    return t[mask], f, f_err

fig2, ax2 = plt.subplots(figsize=(9, 6))
for d in datasets:
    _, _, t_runs, T_runs, popt, sem, label = d
    eps, deps = EMISSIVITY[label]
    # use the reference run (last in list = highest T₀, longest curve)
    tf, f, f_err = frad_from_data(t_runs[-1], T_runs[-1], eps, deps)
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
# Fair comparison: integrate P_rad over the SAME temperature window for every
# jar (75 → 58 °C, covered by all stitched curves). Over that window every jar
# loses the same total energy m·c·ΔT ≈ 35.6 kJ — only the radiated share
# differs, which is exactly what emissivity controls.

T_HI, T_LO = 75.0, 58.0
E_TOTAL_WINDOW = M_WATER * C_WATER * (T_HI - T_LO)   # J, same for all jars

print("\n" + "═" * 70)
print(f"RADIATED ENERGY over the common window {T_HI:.0f} → {T_LO:.0f} °C "
      f"(total heat lost = {E_TOTAL_WINDOW/1e3:.1f} kJ per jar)")
print("═" * 70)

eps_list, deps_list, E_list, dE_list, labels = [], [], [], [], []
for d in datasets:
    t_all, y_all, *_ , label = d[0], d[1], d[6]
    label = d[6]
    eps, deps = EMISSIVITY[label]
    # bin-average the full stitched dataset (averages overlapping runs too)
    order = np.argsort(t_all)
    tb, Tb, _ = bin_average(t_all[order], y_all[order], dt=30.0)
    win = (Tb <= T_HI) & (Tb >= T_LO)
    tw, Tw = tb[win], Tb[win]
    T_K  = Tw + 273.15
    Pr   = eps * SIGMA * AREA * (T_K**4 - T_AMB_K**4)
    E    = np.trapezoid(Pr, tw)                   # J — integral of data
    # uncertainty: Δε, ΔA and systematic ΔT (fully correlated in time →
    # propagate on the mean, not in quadrature point-by-point)
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

# Weighted linear fit E = m·ε + b  (Stefan–Boltzmann predicts E ∝ ε if all
# jars spent identical time in the window — they don't, hence deviations)
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
