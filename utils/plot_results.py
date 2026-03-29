"""
Plot experiment results as grouped bar charts with error bars.

Usage:
    from utils.plot_results import plot_comparison_bars, plot_single_category
    
    # Single category plot
    plot_single_category(
        data={
            'Button On': {'Ours': (0.7, 0.05), 'Baseline': (0.4, 0.03)},
            'Switch On': {'Ours': (0.9, 0.02), 'Baseline': (0.5, 0.04)},
        },
        title='Articulated Parts',
        save_path='results.png'
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os


# Default color scheme (similar to the reference image)
PASTEL_TECH = {
    'Ours': '#4A6FA5',            # Steel Blue
    'Dynaguide': '#89B4E8', # Soft Sky Blue
    'ITPS': '#F2A97C', # Warm Peach
    'Base': '#D9D9D9', # Neutral Light Gray
}



def plot_single_category(
    data: Dict[str, Dict[str, Tuple[float, float]]],
    title: str = 'Results',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    colors: Optional[Dict[str, str]] = None,
    ylabel: str = 'Success Rate',
    show_values: bool = True,
    show_boost: bool = False,
    boost_baseline: str = None,
    boost_target: str = None,
):
    """
    Plot a single category of results as grouped bar chart.
    
    Args:
        data: Dict mapping task_name -> {method_name: (mean, std)}
              Example: {'Button On': {'Ours': (0.7, 0.05), 'Baseline': (0.4, 0.03)}}
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        colors: Dict mapping method names to colors
        ylabel: Y-axis label
        show_values: Whether to show values on top of bars
        show_boost: Whether to show boost annotation on Average bar
        boost_baseline: Method name to use as baseline for boost calculation
        boost_target: Method name to show boost for
    
    Returns:
        matplotlib figure
    """
    if colors is None:
        colors = PASTEL_TECH
    
    tasks = list(data.keys())
    methods = list(data[tasks[0]].keys())
    n_tasks = len(tasks)
    n_methods = len(methods)
    
    # Bar positions
    x = np.arange(n_tasks)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot bars for each method
    bars_dict = {}
    for i, method in enumerate(methods):
        means = [data[task][method][0] for task in tasks]
        stds = [data[task][method][1] for task in tasks]
        
        color = colors.get(method, f'C{i}')
        offset = (i - n_methods/2 + 0.5) * width
        
        bars = ax.bar(
            x + offset, means, width,
            yerr=stds, capsize=3,
            label=method, color=color,
            edgecolor='black', linewidth=0.5,
            error_kw={'elinewidth': 1.5, 'capthick': 1.5}
        )
        bars_dict[method] = bars
        
        # Add value labels
        if show_values:
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(
                    f'{mean:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )
    
    # Show boost annotation if requested
    if show_boost and boost_baseline and boost_target and 'Average' in tasks:
        avg_idx = tasks.index('Average')
        baseline_val = data['Average'][boost_baseline][0]
        target_val = data['Average'][boost_target][0]
        if baseline_val > 0:
            boost = target_val / baseline_val
            
            # Add shaded rectangle for Average
            ax.axvspan(avg_idx - 0.5, avg_idx + 0.5, alpha=0.1, color='gray')
            
            # Add boost text
            ax.annotate(
                f'{boost:.1f}x\nboost',
                xy=(avg_idx + 0.3, (target_val + baseline_val) / 2),
                fontsize=10, color='#1f4e79', fontweight='bold',
                ha='left'
            )
    
    # Styling
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, min(1.15, ax.get_ylim()[1] * 1.15))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def plot_comparison_bars(
    results: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
    colors: Optional[Dict[str, str]] = None,
):
    """
    Plot multiple categories of results in subplots.
    
    Args:
        results: Dict mapping category_name -> task_data
                 where task_data is Dict[task_name, Dict[method_name, (mean, std)]]
        save_path: Path to save the figure
        figsize: Figure size
        colors: Dict mapping method names to colors
    
    Returns:
        matplotlib figure
    """
    if colors is None:
        colors = PASTEL_TECH
    
    n_categories = len(results)
    
    # Determine subplot layout
    if n_categories <= 2:
        nrows, ncols = 1, n_categories
    elif n_categories <= 4:
        nrows, ncols = 2, 2
    else:
        nrows = (n_categories + 2) // 3
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_categories == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (category, data) in enumerate(results.items()):
        ax = axes[idx]
        
        tasks = list(data.keys())
        methods = list(data[tasks[0]].keys())
        n_tasks = len(tasks)
        n_methods = len(methods)
        
        x = np.arange(n_tasks)
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            means = [data[task][method][0] for task in tasks]
            stds = [data[task][method][1] for task in tasks]
            
            color = colors.get(method, f'C{i}')
            offset = (i - n_methods/2 + 0.5) * width
            
            bars = ax.bar(
                x + offset, means, width,
                yerr=stds, capsize=2,
                label=method, color=color,
                edgecolor='black', linewidth=0.3,
                error_kw={'elinewidth': 1, 'capthick': 1}
            )
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.annotate(
                    f'{mean:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )
        
        ax.set_title(category, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right', fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=11)
        
        if idx == 0:
            ax.set_ylabel('Success Rate', fontsize=14)
    
    # Hide empty subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return fig


def create_example_data():
    """Create example data structure for testing."""
    return {
        'Articulated Parts': {
            'Button On': {'Ours': (0.64, 0.0), 'Dynaguide': (0.7, 0.0), 'ITPS': (0.4, 0.0), 'Base': (0.1, 0.0)},
            'Button Off': {'Ours': (0.64, 0.0), 'Dynaguide': (0.7, 0.0), 'ITPS': (0.5, 0.0), 'Base': (0.1, 0.0)},
            'Switch On': {'Ours': (0.8, 0.0), 'Dynaguide': (0.7, 0.0), 'ITPS': (0.4, 0.0), 'Base': (0.1, 0.0)},
            'Switch Off': {'Ours': (0.59, 0.0), 'Dynaguide': (0.8, 0.0), 'ITPS': (0.5, 0.0), 'Base': (0.1, 0.0)},
            'Drawer Open': {'Ours': (0.54, 0.0), 'Dynaguide': (0.7, 0.0), 'ITPS': (0.4, 0.0), 'Base': (0.1, 0.0)},
            'Drawer Close': {'Ours': (0.4, 0.0), 'Dynaguide': (0.8, 0.0), 'ITPS': (0.5, 0.0), 'Base': (0.1, 0.0)},
            'Door Left': {'Ours': (0.49, 0.0), 'Dynaguide': (0.7, 0.0), 'ITPS': (0.4, 0.0), 'Base': (0.1, 0.0)},
            'Door Right': {'Ours': (0.5, 0.0), 'Dynaguide': (0.8, 0.0), 'ITPS': (0.5, 0.0), 'Base': (0.1, 0.0)},
        },
        'Movable Objects': {
            'Red Block': {'Ours': (0.82, 0.0), 'Dynaguide': (0.3, 0.0), 'Base': (0.11, 0.03)},
            'Blue Block': {'Ours': (0.85, 0.0), 'Dynaguide': (0.24, 0.0), 'Base': (0.13, 0.03)},
            'Pink Block': {'Ours': (0.83, 0.0), 'Dynaguide': (0.24, 0.0), 'Base': (0.14, 0.03)},
        },
    }

def calculate_avg(data: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]):
    """
    Calculate the average for each category and add as 'Average' entry.
    
    Args:
        data: Dict mapping category -> {task: {method: (mean, std)}}
    
    Returns:
        Updated data with 'Average' added to each category
    """
    for category, tasks in data.items():
        # Get all methods from the first task
        first_task = list(tasks.values())[0]
        methods = list(first_task.keys())
        
        # Calculate average for each method
        avg_dict = {}
        for method in methods:
            means = []
            for task_name, task_data in tasks.items():
                if task_name == 'Average':  # Skip if already exists
                    continue
                if method in task_data:
                    means.append(task_data[method][0])
            
            if means:
                avg_mean = np.mean(means)
                # Set std to 0 for average (not real experimental variance)
                avg_dict[method] = (round(avg_mean, 2), 0.0)
        
        # Add Average to the category
        data[category]['Average'] = avg_dict
    
    return data

if __name__ == "__main__":
    # Example usage
    example_data = create_example_data()
    
    # Calculate and add Average for each category
    example_data = calculate_avg(example_data)
    
    # Print calculated averages
    for category, tasks in example_data.items():
        if 'Average' in tasks:
            print(f"\n{category} - Average:")
            for method, (mean, std) in tasks['Average'].items():
                print(f"  {method}: {mean:.2f} ± {std:.3f}")

    # Plot single category
    plot_single_category(
        example_data['Articulated Parts'],
        title='Articulated Parts',
        save_path="example_single_articulated.png",
        show_boost=True,
        boost_baseline='Base',
        boost_target='Ours'
    )
    
    plot_single_category(
        example_data['Movable Objects'],
        title='Movable Objects',
        save_path="example_single_movable.png",
        show_boost=True,
        boost_baseline='Base',
        boost_target='Ours'
    )

    plt.show()

