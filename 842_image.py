import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- Configuration ---
np.random.seed(2025)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "boxplot_stimuli")
os.makedirs(output_dir, exist_ok=True)

conditions = [
    ("Jitter On", "Tukey", "Points On"),
    ("Jitter On", "Tukey", "Points Off"),
    ("Jitter On", "MinMax", "Points On"),
    ("Jitter On", "MinMax", "Points Off"),
    ("Jitter Off", "Tukey", "Points On"),
    ("Jitter Off", "Tukey", "Points Off"),
    ("Jitter Off", "MinMax", "Points On"),
    ("Jitter Off", "MinMax", "Points Off")
]

trial_types = ["Different SDs", "Outlier vs No-Outlier", "Skew vs Symmetric", "Unequal Sample Size"]

# --- Function to generate data ---
def generate_trial_data(trial_type):
    mu = 5
    if trial_type == "Different SDs":
        left = np.random.normal(mu, 1.0, 50)
        right = np.random.normal(mu, 2.0, 50)
    elif trial_type == "Outlier vs No-Outlier":
        left = np.random.normal(mu, 1.0, 50)
        left = np.append(left, [mu+8, mu+10, mu-8, mu-10, mu+12])  # extreme outliers
        right = np.random.normal(mu, 1.2, 50)
    elif trial_type == "Skew vs Symmetric":
        left = np.random.gamma(shape=2.0, scale=1.0, size=50) + mu - 2
        right = np.random.normal(mu, 1.0, 50)
    elif trial_type == "Unequal Sample Size":
        left = np.random.normal(mu, 1.0, 80)
        right = np.random.normal(mu, 1.2, 20)
    return left, right

# --- Function to plot boxplots ---
def plot_trial(left, right, condition, trial_type, trial_idx):
    jitter, whisker, points = condition
    plt.figure(figsize=(6,4))
    
    positions = [1, 2]
    whis_value = 1.5 if whisker == "Tukey" else (0, 100) 
    
    # Plot boxplots first
    plt.boxplot([left, right],
                positions=positions,
                patch_artist=True,
                widths=0.6,
                flierprops=dict(marker='o', color='red', alpha=0.5),
                whiskerprops=dict(color='black'),
                showmeans=False,
                whis=whis_value,
                zorder=1)  
    
    # Add scatter points only if Points On
    if points == "Points On":
        point_color = "seagreen"
        for i, data in enumerate([left, right]):
            if jitter == "Jitter On":
                x = np.random.normal(positions[i], 0.05, size=len(data))
            else:
                x = np.full(len(data), positions[i]) 
            plt.scatter(x, data, alpha=0.6, color=point_color, s=15, zorder=2) 
    
    plt.ylabel("Value")
    plt.xticks(positions, ["Left", "Right"])
    plt.title(f"Trial {trial_idx} | {jitter} x {whisker} x {points}")
    
    filename = f"{trial_idx}_{trial_type.replace(' ', '_')}_{jitter.replace(' ','')}_{whisker}_{points.replace(' ', '')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()

# --- Generate data sets first ---
num_instances_per_trial_type = 4  # number of different data sets per trial type
data_sets = {}  # Store data sets: {(trial_type, instance_idx): (left, right)}

print("Generating data sets...")
for trial_type in trial_types:
    for instance in range(num_instances_per_trial_type):
        left, right = generate_trial_data(trial_type)
        # Randomly swap left/right for some instances
        if np.random.rand() > 0.5:
            left, right = right, left
        data_sets[(trial_type, instance)] = (left.copy(), right.copy())

print(f"Generated {len(data_sets)} data sets ({num_instances_per_trial_type} per trial type)")

# --- CSV log file ---
csv_path = os.path.join(output_dir, "trial_sd_log.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["TrialIdx", "TrialType", "Instance", "Jitter", "Whisker", "Points", "Left_SD", "Right_SD", "More_Variable"])
    
    trial_idx = 1
    # For each data set, create all condition combinations
    for trial_type in trial_types:
        for instance in range(num_instances_per_trial_type):
            left, right = data_sets[(trial_type, instance)]
            
            # Calculate SDs (same for all conditions using this data)
            left_sd = np.std(left, ddof=1)
            right_sd = np.std(right, ddof=1)
            
            if left_sd > right_sd:
                more_var = "Left"
            elif right_sd > left_sd:
                more_var = "Right"
            else:
                more_var = "Equal"
            
            # Generate plots for all conditions using the same data
            for condition in conditions:
                # log one row
                writer.writerow([
                    trial_idx, trial_type, instance, condition[0], condition[1], condition[2],
                    f"{left_sd:.3f}", f"{right_sd:.3f}", more_var
                ])
                
                # save figure
                plot_trial(left, right, condition, trial_type, trial_idx)
                trial_idx += 1

print(f"All boxplots with trial indices saved in {output_dir}")
print(f"SD log saved to {csv_path}")
