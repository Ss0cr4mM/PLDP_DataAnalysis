import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

try:
    df1 = pd.read_csv('labs 2.xlsx - Sheet1.csv')
except FileNotFoundError:
    df1 = pd.read_excel('labs 2.xlsx')

try:
    df2 = pd.read_csv('labs.xlsx - Sheet1.csv')
except FileNotFoundError:
    df2 = pd.read_excel('labs.xlsx')

try:
    df_c = pd.read_csv('copper and black tape.xlsx - Sheet1.csv')
except FileNotFoundError:
    df_c = pd.read_excel('copper and black tape.xlsx')

col_pairs = [
    ('Temperature 1 (°C) Run #1', 'Temperature 1 (°C) Run #2'),
    ('Temperature 2 (°C) Run #1', 'Temperature 2 (°C) Run #2'),
    ('Temperature 3 (°C) Run #1', 'Temperature 3 (°C) Run #2')
]

copper_cols = [
    'Copper foil Temperature 1 (°C) Run #1',
    'Copper foil Temperature 1 (°C) Run #2',
    'Copper foil Temperature 2 (°C) Run #1',
    'Copper foil Temperature 3 (°C) Run #2'
]

colors = ['blue', 'green', 'red', '#d97726']
emissivities = [0.95, 0.04, 0.9, 0.02] 
T_env = 22.0
sigma = 5.67e-8
T_env_K = T_env + 273.15

dt = (45 * 60) / len(df2)

def newtons_cooling_fixed_env(t, T0, k):
    return T_env + (T0 - T_env) * np.exp(-k * t)


lengths = []
aligned_data = {}

for i, (col1, col2) in enumerate(col_pairs):
    T_run1 = df1[col1].values
    T_run2 = df2[col2].values
    
    idx_1 = np.where(T_run1 <= 75.0)[0]
    start_1 = int(idx_1[0]) if idx_1.size > 0 else 0
    idx_2 = np.where(T_run2 <= 75.0)[0]
    start_2 = int(idx_2[0]) if idx_2.size > 0 else 0
    
    lengths.extend([len(T_run1) - start_1, len(T_run2) - start_2])
    aligned_data[i] = [T_run1[start_1:], T_run2[start_2:]]

copper_slices = []
for col in copper_cols:
    T_run = df_c[col].dropna().values
    idx = np.where(T_run <= 75.0)[0]
    start = int(idx[0]) if idx.size > 0 else 0
    lengths.append(len(T_run) - start)
    copper_slices.append(T_run[start:])

aligned_data[3] = copper_slices

min_length = min(lengths) if lengths else 0
time_sec = np.arange(min_length) * dt
time_min = time_sec / 60

processed_means = []
processed_sems = []
jar_labels = ['Black tape (ε=0.95)', 'Aluminum foil (ε=0.04)', 'Control(bare glass) (ε=0.90)', 'Copper Foil (ε=0.02)']

for i in range(4):
    runs = [r[:min_length] for r in aligned_data[i]]
    T_mean = np.mean(runs, axis=0)
    T_sem = np.std(runs, axis=0, ddof=1) / np.sqrt(len(runs))
    processed_means.append(T_mean)
    processed_sems.append(T_sem)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Newton's Law of Cooling Fits by Material", fontsize=16)

axs = axs.flatten()

print("\n--- NEWTON'S LAW OF COOLING FIT PARAMETERS ---")
for i in range(4):
    T_mean = processed_means[i]
    T_sem = processed_sems[i]
    step = max(1, len(time_sec) // 100)
    
    try:
        popt, pcov = curve_fit(newtons_cooling_fixed_env, time_sec[::step], T_mean[::step], p0=[float(T_mean[0]), 1e-4], maxfev=10000)
        T_pred = newtons_cooling_fixed_env(time_sec, *popt)
        r2 = r2_score(T_mean, T_pred)
        fit_label = f'Fit (k={popt[1]:.2e} s⁻¹, R²={r2:.4f})'
        
        print(f"{jar_labels[i]}:")
        print(f"  T_0 (Initial Temp)  = {popt[0]:.2f} °C")
        print(f"  k   (Cooling Const) = {popt[1]:.3e} s⁻¹")
        print(f"  R²  (Goodness fit)  = {r2:.4f}\n")
    except Exception:
        T_pred = np.full_like(time_sec, T_mean[0], dtype=float)
        fit_label = 'Fit Failed'
        
    axs[i].plot(time_min, T_mean, label='Mean Data', color=colors[i], alpha=0.8)
    axs[i].fill_between(time_min, T_mean - T_sem, T_mean + T_sem, color=colors[i], alpha=0.3, label='±SEM')
    axs[i].plot(time_min, T_pred, '--', label=fit_label, color='black', linewidth=1.5) 
    
    axs[i].set_title(jar_labels[i])
    axs[i].set_xlabel("Time (minutes)")
    axs[i].set_ylabel("Temperature (°C)")
    axs[i].legend()
    axs[i].grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show()


plt.figure(figsize=(10, 6))
for i in range(4):
    T_mean_K = processed_means[i] + 273.15
    T_sem = processed_sems[i]
    
    P_rad_flux = sigma * emissivities[i] * (T_mean_K**4 - T_env_K**4)
    P_sem = 4 * sigma * emissivities[i] * (T_mean_K**3) * T_sem
    
    plt.plot(time_min, P_rad_flux, label=jar_labels[i], color=colors[i])
    plt.fill_between(time_min, P_rad_flux - P_sem, P_rad_flux + P_sem, color=colors[i], alpha=0.3, edgecolor=None)

plt.title("Radiative power density versus Time ($\pm$ SEM)")
plt.xlabel("Time since reaching 75°C (minutes)")
plt.ylabel("Radiative Heat Flux ($W/m^2$)") # Fixed y-axis label
plt.legend()
plt.grid(True)
plt.show()


total_energies = []
energy_errors = []

for i in range(4):
    T_mean_K = processed_means[i] + 273.15
    T_sem = processed_sems[i]
    
    P_rad_flux = sigma * emissivities[i] * (T_mean_K*4 - T_env_K*4) # Shouldn't it be **4 instead of *4?
    P_sem = 4 * sigma * emissivities[i] * (T_mean_K**3) * T_sem
    
    E_total = np.sum(P_rad_flux * dt)
    E_error = np.sqrt(np.sum((P_sem * dt)**2))
    
    total_energies.append(E_total)
    energy_errors.append(E_error)


print("--- TOTAL RADIATIVE ENERGY LOST ---")
for i in range(4):
    print(f"{jar_labels[i]}:")
    print(f"  Total Energy = {total_energies[i]:,.2f} ± {energy_errors[i]:,.2f} J/m²\n")

plt.figure(figsize=(8, 6))

plt.errorbar(
    emissivities, 
    total_energies, 
    yerr=energy_errors, 
    fmt='o', 
    color='purple', 
    markersize=8, 
    capsize=5, 
    label='Experimental Data'
)

for i in range(4):
    y_offset = -20 if i == 1 else 15
    short_label = jar_labels[i].split(' (')[0]
    
    plt.annotate(
        short_label, 
        (emissivities[i], total_energies[i]), 
        textcoords="offset points", 
        xytext=(0, y_offset), 
        ha='center',
        fontsize=10
    )

coeffs = np.polyfit(emissivities, total_energies, 1)
trend_x = np.linspace(0, 1.0, 100)
trend_y = np.polyval(coeffs, trend_x)
plt.plot(trend_x, trend_y, '--', color='gray', alpha=0.5, label='Linear Fit')


print("--- LINEAR FIT: EMISSIVITY VS. ENERGY ---")
print(f"Slope (m): {coeffs[0]:,.2f} J/m² per unit emissivity")
print(f"Y-Intercept (b): {coeffs[1]:,.2f} J/m²\n")


plt.title("Relationship between Emissivity and Total Energy Lost")
plt.xlabel("Emissivity ($\epsilon$)")
plt.ylabel("Total Energy Lost ($J/m^2$)")
plt.xlim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()