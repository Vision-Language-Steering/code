import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from PIL import Image

# Set serif font (Liberation Serif as Cambria alternative)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Liberation Serif', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

# ============================================================
# 1) Data from the table
# ============================================================
methods = ["π₀.₅", "VLS (Ours)"]

# Colors: purple for pi, orange for VLS
colors = {
    "π₀.₅": "#5B5EA6",      # purple
    "VLS (Ours)": "#D35400",        # orange
}

# In-Distribution tasks (A-F + Average)
in_dist_tasks = ["A", "B", "C", "D", "E", "F", "Average"]
in_dist_data = {
    "π₀.₅": [0.75, 0.58, 0.90, 0.53, 0.05, 0.20, 0.50],
    "VLS (Ours)":  [0.75, 0.75, 0.84, 0.58, 0.58, 0.65, 0.69],
}

# Out-of-Distribution tasks (G-G, G-R, H-G, H-R, I + Average)
ood_tasks = ["G-G", "G-R", "H-G", "H-R", "I", "Average"]
ood_data = {
    "π₀.₅": [0.78, 0.45, 0.40, 0.48, 0.00, 0.42],
    "VLS (Ours)":  [0.85, 0.80, 0.88, 0.53, 0.40, 0.69],
}

# ============================================================
# 2) Plotting function (matching CALVIN style)
# ============================================================
def grouped_barplot(ax, categories, methods_list, data_dict, title, ylim, highlight_last=True):
    x = np.arange(len(categories))
    n = len(methods_list)
    width = 0.35

    # Gray highlight for the last column (Average)
    if highlight_last:
        ax.axvspan(len(categories) - 1 - 0.5, len(categories) - 1 + 0.5, color="#EFEFEF", zorder=0)

    for i, method in enumerate(methods_list):
        vals = np.array(data_dict[method], dtype=float)
        offset = (i - (n - 1) / 2) * width

        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=colors.get(method),
            edgecolor="none",
            zorder=2,
        )

        # Value labels on top of bars (extra bold, sans-serif for better bold)
        for xi, yi in zip(x + offset, vals):
            ax.text(
                xi, yi + 0.02,
                f"{yi:.2f}",
                ha="center", va="bottom",
                fontsize=10,
                fontweight='bold',
                fontfamily='sans-serif'
            )

    ax.set_ylabel("Success Rate", fontsize=16, fontweight='bold')
    ax.set_title(title, fontsize=22, fontweight='black', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=15)
    ax.set_ylim(*ylim)
    # Keep all spines visible for the box border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    ax.tick_params(axis="y", labelsize=12)

# ============================================================
# 3) Create the figure with images and plots
# ============================================================
# Load images
img_in1 = Image.open("/home/ishneet/Desktop/expts/VLS/in1.png")
img_in2 = Image.open("/home/ishneet/Desktop/expts/VLS/in2.png")
img_out1 = Image.open("/home/ishneet/Desktop/expts/VLS/out1.png")
img_out2 = Image.open("/home/ishneet/Desktop/expts/VLS/out2.png")
img_out3 = Image.open("/home/ishneet/Desktop/expts/VLS/out3.png")

# Create figure with custom layout
fig = plt.figure(figsize=(24, 7))

# Main GridSpec: 3 rows, 6 columns
# col 0: in-dist images, col 1: spacing, col 2: in-dist plot, col 3: out-dist images, col 4: spacing, col 5: out-dist plot
gs = GridSpec(3, 6, width_ratios=[1.4, 0.08, 2.2, 1.0, 0.08, 2.2], height_ratios=[1, 1, 1],
              wspace=0.01, hspace=0.03)

# In-distribution images (2 images stacked, equal size)
# Use nested gridspec for equal sizing
from matplotlib.gridspec import GridSpecFromSubplotSpec
gs_in = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 0], hspace=0.03)
ax_in1 = fig.add_subplot(gs_in[0, 0])
ax_in2 = fig.add_subplot(gs_in[1, 0])

# In-distribution bar plot (spans all rows) - col 2 (after spacing col 1)
ax_plot1 = fig.add_subplot(gs[:, 2])

# Out-of-distribution: G, H, I stacked vertically - col 3 (right next to in-dist plot)
ax_out_g = fig.add_subplot(gs[0, 3])
ax_out_h = fig.add_subplot(gs[1, 3])
ax_out_i = fig.add_subplot(gs[2, 3])

# Out-of-distribution bar plot (spans all rows) - col 5 (after spacing col 4)
ax_plot2 = fig.add_subplot(gs[:, 5])

# Display in-distribution images
ax_in1.imshow(img_in1)
ax_in1.axis('off')
ax_in2.imshow(img_in2)
ax_in2.axis('off')

# Display out-of-distribution images: G, H, I stacked vertically
ax_out_g.imshow(img_out1)
ax_out_g.axis('off')
ax_out_h.imshow(img_out2)
ax_out_h.axis('off')
ax_out_i.imshow(img_out3)
ax_out_i.axis('off')

# Create the bar plots
plot_order = ["π₀.₅", "VLS (Ours)"]

grouped_barplot(
    ax_plot1,
    categories=in_dist_tasks,
    methods_list=plot_order,
    data_dict=in_dist_data,
    title="In-Distribution Performance",
    ylim=(0.0, 1.05),
    highlight_last=True
)

grouped_barplot(
    ax_plot2,
    categories=ood_tasks,
    methods_list=plot_order,
    data_dict=ood_data,
    title="Out-of-Distribution Performance",
    ylim=(0.0, 1.05),
    highlight_last=True
)

# Shared legend at the bottom
handles = [Patch(color=colors[m], label=m) for m in plot_order]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03),
    prop={'size': 20},
    columnspacing=1.5,
    handletextpad=0.5
)

plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10, wspace=0.05, hspace=0.05)
plt.savefig("/home/ishneet/Desktop/expts/VLS/pi05_results.png", dpi=300, bbox_inches='tight')
plt.savefig("/home/ishneet/Desktop/expts/VLS/pi05_results.svg", format='svg', bbox_inches='tight')
plt.savefig("/home/ishneet/Desktop/expts/VLS/pi05_results.pdf", format='pdf', bbox_inches='tight')
plt.show()
