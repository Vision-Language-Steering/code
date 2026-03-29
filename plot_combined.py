import matplotlib.pyplot as plt
import numpy as np

# Ablation Data (VLS at end with orange to match line graphs)
methods = ['w/o\nGrad', 'w/o\nFKD', 'w/o\nRBF', 'VLS']
abl_success_rates = [17.3, 85.3, 86.0, 88]
abl_episode_lengths = [178.6, 64.7, 69.3, 61]
abl_inference_times = [979.3, 1170.7, 1246.0, 1189.3]

# Runtime Scaling Data
k_values = [1, 2, 5, 10]
rt_success_rates = [88, 88, 90, 90]
rt_episode_lengths = [73.0, 58.0, 57.8, 55.9]
rt_inference_times = [665, 799, 953, 1239]

# Color scheme for ablation bars (one color per method)
bar_colors = ['#E8C547', '#7FB3D5', '#5B5EA6', '#D35400']  # VLS, w/o FKD, w/o RBF, w/o Grad
line_color = '#D35400'  # Orange from palette for runtime scaling line plots

# MAXIMUM CLARITY SETTINGS
plt.rcParams.update({
    # Font settings - MAXIMUM SIZE
    'font.family': 'serif',
    'font.serif': ['Cambria', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 24,
    'font.weight': 'bold',
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,

    # Line settings - THICKER
    'axes.linewidth': 3,
    'lines.linewidth': 4,
    'lines.markersize': 16,

    # MAXIMUM QUALITY RENDERING
    'figure.dpi': 300,
    'savefig.dpi': 1200,  # Ultra high DPI
    'text.antialiased': True,
    'lines.antialiased': True,

    # Vector quality for PDF/SVG
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

# Set up the figure - LARGER
fig, axes = plt.subplots(1, 6, figsize=(48, 8))

# X positions for categorical data
x_positions = np.arange(len(methods))

# ============ ABLATION PLOTS (first 3) - BAR GRAPHS ============

# Thin bars touching each other (like reference image)
bar_width = 0.18
method_labels = ['w/o Grad', 'w/o FKD', 'w/o RBF', 'VLS']  # Clean labels for legend
# Positions so bars touch: 0, 0.18, 0.36, 0.54 (centered around 0.27)
x_bar = np.array([i * bar_width for i in range(4)])

# Plot 1: Ablation Success Rate
ax1 = axes[0]
for i in range(4):
    ax1.bar(x_bar[i], abl_success_rates[i], width=bar_width, color=bar_colors[i], edgecolor='black', linewidth=1.5)
for i, (x, y) in enumerate(zip(x_bar, abl_success_rates)):
    ax1.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=16, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', fontsize=26, fontweight='bold')
ax1.set_title('Ablation\nSuccess Rate (%)', fontsize=28, fontweight='bold', pad=25)
ax1.set_xticks([])
ax1.set_xlim([-0.15, 0.85])
ax1.set_ylim([0, 100])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(3)
ax1.spines['bottom'].set_linewidth(3)
ax1.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

# Plot 2: Ablation Episode Length
ax2 = axes[1]
for i in range(4):
    ax2.bar(x_bar[i], abl_episode_lengths[i], width=bar_width, color=bar_colors[i], edgecolor='black', linewidth=1.5)
for i, (x, y) in enumerate(zip(x_bar, abl_episode_lengths)):
    ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=16, fontweight='bold')
ax2.set_ylabel('Avg. Episode\nLength', fontsize=26, fontweight='bold')
ax2.set_title('Ablation\nEpisode Length (# steps)', fontsize=28, fontweight='bold', pad=25)
ax2.set_xticks([])
ax2.set_xlim([-0.15, 0.85])
ax2.set_ylim([0, 200])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(3)
ax2.spines['bottom'].set_linewidth(3)
ax2.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

# Plot 3: Ablation Inference Time
ax3 = axes[2]
for i in range(4):
    ax3.bar(x_bar[i], abl_inference_times[i], width=bar_width, color=bar_colors[i], edgecolor='black', linewidth=1.5)
for i, (x, y) in enumerate(zip(x_bar, abl_inference_times)):
    ax3.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=16, fontweight='bold')
ax3.set_ylabel('Avg. Inference\nTime (ms)', fontsize=26, fontweight='bold')
ax3.set_title('Ablation\nInference Time (ms)', fontsize=28, fontweight='bold', pad=25)
ax3.set_xticks([])
ax3.set_xlim([-0.15, 0.85])
ax3.set_ylim([0, 1400])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(3)
ax3.spines['bottom'].set_linewidth(3)
ax3.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

# Legend will be added after tight_layout
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=bar_colors[i], edgecolor='black', linewidth=2, label=method_labels[i]) for i in range(4)]

# ============ RUNTIME SCALING PLOTS (last 3) ============

# Plot 4: Runtime Success Rate
ax4 = axes[3]
ax4.plot(k_values, rt_success_rates, marker='o', color=line_color, linewidth=4, markersize=16, markerfacecolor=line_color, markeredgewidth=0)
for i, (x, y) in enumerate(zip(k_values, rt_success_rates)):
    ax4.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0, 18), ha='center', fontsize=18, fontweight='bold')
ax4.set_xlabel('Number of Samples', fontsize=26, fontweight='bold')
ax4.set_ylabel('Success Rate (%)', fontsize=26, fontweight='bold')
ax4.set_title('Runtime Scaling\nSuccess Rate (%)', fontsize=28, fontweight='bold', pad=25)
ax4.set_xticks(k_values)
ax4.set_xticklabels([str(k) for k in k_values], fontsize=20, fontweight='bold')
ax4.set_ylim([0, 100])
ax4.set_xlim([0, 11])
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_linewidth(3)
ax4.spines['bottom'].set_linewidth(3)
ax4.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

# Plot 5: Runtime Episode Length
ax5 = axes[4]
ax5.plot(k_values, rt_episode_lengths, marker='o', color=line_color, linewidth=4, markersize=16, markerfacecolor=line_color, markeredgewidth=0)
for i, (x, y) in enumerate(zip(k_values, rt_episode_lengths)):
    ax5.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0, 18), ha='center', fontsize=18, fontweight='bold')
ax5.set_xlabel('Number of Samples', fontsize=26, fontweight='bold')
ax5.set_ylabel('Avg. Episode\nLength', fontsize=26, fontweight='bold')
ax5.set_title('Runtime Scaling\nEpisode Length (# steps)', fontsize=28, fontweight='bold', pad=25)
ax5.set_xticks(k_values)
ax5.set_xticklabels([str(k) for k in k_values], fontsize=20, fontweight='bold')
ax5.set_ylim([50, 80])
ax5.set_xlim([0, 11])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_linewidth(3)
ax5.spines['bottom'].set_linewidth(3)
ax5.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

# Plot 6: Runtime Inference Time
ax6 = axes[5]
ax6.plot(k_values, rt_inference_times, marker='o', color=line_color, linewidth=4, markersize=16, markerfacecolor=line_color, markeredgewidth=0)
for i, (x, y) in enumerate(zip(k_values, rt_inference_times)):
    ax6.annotate(f'{int(y)}', (x, y), textcoords="offset points", xytext=(0, 18), ha='center', fontsize=18, fontweight='bold')
ax6.set_xlabel('Number of Samples', fontsize=26, fontweight='bold')
ax6.set_ylabel('Avg. Inference\nTime (ms)', fontsize=26, fontweight='bold')
ax6.set_title('Runtime Scaling\nInference Time (ms)', fontsize=28, fontweight='bold', pad=25)
ax6.set_xticks(k_values)
ax6.set_xticklabels([str(k) for k in k_values], fontsize=20, fontweight='bold')
ax6.set_ylim([0, 1400])
ax6.set_xlim([0, 11])
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_linewidth(3)
ax6.spines['bottom'].set_linewidth(3)
ax6.tick_params(axis='both', which='major', labelsize=22, width=3, length=8)

plt.tight_layout(pad=4.0)

# Add legend for ablation plots AFTER tight_layout (aligned with x-axis labels)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=20, frameon=False,
           bbox_to_anchor=(0.25, 0.14), handlelength=2, handleheight=1.5)

# Save in multiple formats for maximum clarity
plt.savefig('/home/ishneet/Desktop/expts/VLS/combined_results.png', dpi=1200, bbox_inches='tight', facecolor='white', edgecolor='none', transparent=False)
plt.savefig('/home/ishneet/Desktop/expts/VLS/combined_results.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/home/ishneet/Desktop/expts/VLS/combined_results.svg', bbox_inches='tight', facecolor='white', edgecolor='none')
print("Saved plots to combined_results.png (1200 DPI), .pdf, and .svg")
plt.show()
