import matplotlib.pyplot as plt
import numpy as np

# Data from the provided dictionary
yolos = {
    'map50': {
        'baseline': 43.979,
        'concat': {
            'trainStd_testStd': 45.58,
            'trainOnes_testOnes': 44.94,
            'trainRand_testRand': 44.65,
            'trainStd_testOnes': 40.36,
            'trainStd_testRand': 34.55,
        },
        'affine': {
            'trainStd_testStd': 43.85,
            'trainStd_testOnes': 41.82,
            'trainStd_testRand': 41.01,
            'trainRand_testRand': 6.07,
            'trainOnes_testOnes': 5.51,
        },
    }
}

detr = {
    'map50': {
        'baseline': 46.87,
        'concat': {
            'trainStd_testStd': 48.80,
            'trainOnes_testOnes': 48.16,
            'trainRand_testRand': 47.54,
            'trainStd_testOnes': 46.89,
            'trainStd_testRand': 39.49,
        },
        'affine': {
            'trainStd_testStd': 47.05,
            'trainOnes_testOnes': 47.27,
            'trainRand_testRand': 44.12,
            'trainStd_testOnes': 8.69,
            'trainStd_testRand': 4.02,
        },
    }
}

data = yolos

# Extracting data for plotting
baseline_value = data['map50']['baseline']
concat_values = list(data['map50']['concat'].values())
affine_values = list(data['map50']['affine'].values())

# X positions for groups
x_baseline = [0]  # Position for the baseline group
x_concat = np.arange(1.6, 6.6) * .9  # Positions for the concat group
x_affine = np.arange(7.2, 12.2) * .9 # Positions for the affine group

# Calculating positions for group labels
group_labels_positions = [
    np.mean(x_baseline),  # Center position for 'Baseline'
    np.mean(x_concat),    # Center position for 'Concat'
    np.mean(x_affine)     # Center position for 'Affine'
]

# Colors
baseline_color = 'tab:blue'  # Distinct color for the baseline
colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']  # Colors for the other groups

size = 14
plt.rc('font', size=size)  # Font size
plt.rc('axes', labelsize=size)  # Font size for x and y labels
plt.rc('xtick', labelsize=size)  # Font size for x-ticks
plt.rc('ytick', labelsize=size)  # Font size for y-ticks

# Creating the plot
fig, ax = plt.subplots(figsize=(6.57, 5.1))


# set y-axis limits
ax.set_ylim(0, 69.9)

# Plotting each group
w = 0.55  # Width of the bars
a = 0.8  # Alpha value for the bars
bars_baseline = ax.bar(x_baseline, [baseline_value], width=w, label='Baseline', color=baseline_color, alpha=a)
bars_concat = ax.bar(x_concat, concat_values, width=w, label='Concat', color=colors, alpha=a)
bars_affine = ax.bar(x_affine, affine_values, width=w, label='Affine', color=colors, alpha=a)



# Adding value labels on top of each bar
def add_value_labels(bars):
    """Attach a text label above each bar displaying its height."""
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            yval + 1,  # Position the label slightly above the bar
            round(yval, 2), 
            ha='center', 
            va='bottom',
            # Font size and color
            fontsize=10.7,
        )

# Add labels for all groups
add_value_labels(bars_baseline)
add_value_labels(bars_concat)
add_value_labels(bars_affine)

# Adding labels and title
ax.set_xticks(group_labels_positions)
ax.set_xticklabels(['Baseline', 'Concat', 'Affine'])
ax.set_ylabel('mAP@50')

# Optional: Adding grid lines for clarity
#ax.grid(axis='y', linestyle='--', alpha=0.7)


# Adding legend containg 5 entries: BaseLine, trainOnes_testOnes, trainRand_testRand, trainStd_testOnes, trainStd_testRand
# The first entry is the baseline, the next 4 refer to both concat and affine groups
# The colors are set to match the bar colors
legend_labels = ['Baseline', 'trainStd_testStd', 'trainOnes_testOnes', 'trainRand_testRand', 'trainStd_testOnes', 'trainStd_testRand']
legend_colors = [baseline_color] + colors 
legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=a) for color in legend_colors]
ax.legend(legend_patches, legend_labels, loc='upper right', ncol=2)

plt.tight_layout()
plt.show()
