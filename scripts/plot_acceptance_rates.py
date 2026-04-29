import csv
import matplotlib.pyplot as plt
import sys
from collections import defaultdict

x_label_dict = {"eps": r"$\epsilon$",
                "eps_2": r"$\epsilon^2$"}
y_label_dict = {"acceptance_rate": "Acceptance Rate",
                "delta_H_abs_mean": r"$\langle |\Delta H| \rangle$",
                "delta_H_abs_std": r"Std Dev of $|\Delta H|$"}
plot_label = r"Acceptance Rate vs Step Size in 2D $\phi^4$ HMC"

data_path = "results/phi4_acceptance_rates.csv"
out_path = "results/phi4_acceptance_rate_plots/acceptance_rate_vs_eps"

kappa_values = [0.85, 0.44, 0.28, 0.22]
lambda_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
lattice_sizes = [8,16,32,64]
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
def plot_acceptance_data(csv_path: str,
                         kappa: float,
                         x_key: str,
                         y_key: str,
                         z_key: str,
                         out_path: str):
    data_by_L_and_z = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            L = int(row['L'])
            if float(row['kappa']) == kappa:
                if x_key == "eps_2":
                    x_val = float(row['eps_2'])
                if y_key == "acceptance_rate":
                    y_val = float(row['acceptance_rate'])
                if z_key == "lambda":
                    z_val = float(row['lambda'])
            
                # group data by both L and z (lambda) to plot separate curves for each lambda
                # and separate figures for each L
                data_by_L_and_z[(L, z_val)].append((x_val, y_val))
    # Now we have data grouped by (L, lambda), we can plot
    for L in sorted(set(k for k, _ in data_by_L_and_z.keys())):
        plt.figure(figsize=(8, 6))
        for z_val in sorted(set(z for (L_key, z) in data_by_L_and_z.keys() if L_key == L)):
            points = sorted(data_by_L_and_z[(L, z_val)], key=lambda p: p[0])
            x_vals, y_vals = zip(*points)
            plt.plot(x_vals, y_vals,
                        marker=markers.pop(0) if markers else 'o',
                        linestyle=':',
                        label=rf"$\lambda={z_val}$",
                        alpha=0.5)
        
        plt.xlabel(x_label_dict.get(x_key, x_key))
        plt.ylabel(y_label_dict.get(y_key, y_key))
        plt.title(f"{plot_label} for $L={L}$, kappa={kappa}")
        plt.legend()
        plt.xscale('log')
        plt.savefig(f"{out_path}_L{L}.png")

plot_acceptance_data(data_path,
                     kappa=0.44,
                     x_key="eps_2",
                     y_key="acceptance_rate",
                     z_key="lambda",
                     out_path=out_path)