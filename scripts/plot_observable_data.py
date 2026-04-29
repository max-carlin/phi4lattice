import csv
import matplotlib.pyplot as plt
import sys
from collections import defaultdict


x_label_dict = {"kappa": r"$\kappa$",
                "kT_over_J": r"$k_B T/J = 1/\kappa$",
                "L": r"$L$"}
y_label_dict = {"abs_magnetization": r"$\langle |m| \rangle$",
                # "magnetization": r"$\langle m \rangle$",
                # "magnetization_squared": r"$\langle m^2 \rangle$",
                "action": r"$\langle S \rangle$",
                # "action_squared": r"$\langle S^2 \rangle$",
                "susceptibility": r"$\chi$",
                "binder_cumulant": r"$U_4$",
                "C_per_Nk": r"$C/(N k_B)$"}
plot_label = "2D Ising"

data_path = "results/ising_observables_results.csv"

def plot_observable_data(csv_path: str,
                            x_key: str,
                            y_key: str,
                            out_path: str):
    # make a dict of L to list of (x,y) pairs
    data_by_L = defaultdict(list)
    marker_dict = {8: 'o', 16: 's', 32: '^', 64: 'D'}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            L = int(row['L'])
            
            if x_key == "kT_over_J":
                x_val = 1/float(row['kappa'])
            else:
                x_val = float(row[x_key])
            
            y_val = float(row[y_key])
            
            data_by_L[L].append((x_val, y_val))

    plt.figure(figsize=(8, 6))

    for L in sorted(data_by_L.keys()):
        # Sort the (x,y) pairs by x value to ensure proper plotting
        points = sorted(data_by_L[L], key=lambda p: p[0])
        x_vals, y_vals = zip(*points)

        plt.plot(x_vals, y_vals,
                    marker=marker_dict.get(L, 'o'),
                    linestyle = '-.',
                    label = f"$L={L}$",
                    alpha=0.7)
    
    plt.xlabel(x_label_dict.get(x_key, x_key))
    plt.ylabel(y_label_dict.get(y_key, y_key))
    plt.title(f"{plot_label}: {y_label_dict.get(y_key, y_key)} vs {x_label_dict.get(x_key, x_key)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Plot saved to {out_path}")

for y_key in y_label_dict.keys():
    plot_observable_data(csv_path=data_path,
                        x_key="kT_over_J",
                        y_key=y_key,
                        out_path=f"results/plots/ising_{y_key}_vs_kT_over_J.png")
