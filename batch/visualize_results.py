"""
Visualize Emotion Distribution by Category
Creates a stacked percentage bar chart showing emotion distribution across categories
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# Emotion colors matching the server
EMOTION_COLORS = {
    "love": "#FACC15",
    "joy": "#22C55E",
    "surprise": "#06B6D4",
    "sadness": "#3B82F6",
    "anger": "#D946EF",
    "fear": "#EF4444",
    "disgust": "#84CC16",
}


def load_results(results_file: str) -> List[Dict]:
    """Load evaluation results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def compute_category_emotion_percentages(results: List[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compute percentage of total emotion scores for each emotion within each category.

    For each category, we:
    1. Sum up all emotion scores across all prompts and all tokens
    2. Take absolute values (since cosine similarity can be negative)
    3. Calculate what percentage each emotion contributes to the total

    Returns:
        Dict mapping category -> emotion -> percentage (0-100)
    """
    # Group by category
    category_data = {}
    for result in results:
        category = result["category"]
        if category not in category_data:
            category_data[category] = []
        category_data[category].append(result)

    # Compute percentages for each category
    category_percentages = {}
    for category, prompts in category_data.items():
        # Sum all emotion scores across all prompts in this category
        emotion_totals = {}

        for prompt_result in prompts:
            # Get average scores for this prompt (already computed in results)
            avg_scores = prompt_result["avg_scores"]
            for emotion, score in avg_scores.items():
                # Take absolute value to handle negative scores from cosine similarity
                emotion_totals[emotion] = emotion_totals.get(emotion, 0.0) + abs(score)

        # Convert to percentages
        total_score = sum(emotion_totals.values())
        if total_score > 0:
            emotion_percentages = {
                emotion: (score / total_score) * 100
                for emotion, score in emotion_totals.items()
            }
        else:
            # If no scores, distribute evenly
            num_emotions = len(emotion_totals)
            emotion_percentages = {
                emotion: 100.0 / num_emotions if num_emotions > 0 else 0.0
                for emotion in emotion_totals.keys()
            }

        category_percentages[category] = emotion_percentages

    return category_percentages


def create_stacked_bar_chart(category_percentages: Dict[str, Dict[str, float]], output_file: str = None):
    """
    Create a stacked percentage bar chart.

    Args:
        category_percentages: Dict mapping category -> emotion -> percentage
        output_file: Path to save the figure (optional)
    """
    # Get categories and emotions (preserve order from first category)
    categories = sorted(category_percentages.keys())
    emotions = list(EMOTION_COLORS.keys())

    # Filter to only emotions that exist in the data
    first_category = categories[0]
    emotions = [e for e in emotions if e in category_percentages[first_category]]

    # Prepare data for stacked bar chart
    data = np.zeros((len(emotions), len(categories)))
    for i, emotion in enumerate(emotions):
        for j, category in enumerate(categories):
            data[i, j] = category_percentages[category].get(emotion, 0.0)

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(10, 5))

    # Create stacked bars
    bottom = np.zeros(len(categories))
    bars = []
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, "#666666")
        bar = ax.bar(
            categories,
            data[i],
            bottom=bottom,
            label=emotion.capitalize(),
            color=color,
            edgecolor="white",
            linewidth=1,
        )
        bars.append(bar)
        bottom += data[i]

    # Customize chart
    ax.set_ylabel("Percentage of Total Emotion Score (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Category", fontsize=11, fontweight="bold")
    ax.set_title("Emotion Distribution by Category", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 100)

    # Add grid for readability
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Customize legend - place to the right of the plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=9,
    )

    # Rotate x-axis labels if needed
    plt.xticks(rotation=0, ha="center")

    # Add percentage labels on bars (only for segments > 5%)
    for i, emotion in enumerate(emotions):
        for j, category in enumerate(categories):
            percentage = data[i, j]
            if percentage > 5:  # Only show label if segment is large enough
                # Calculate y position (middle of the segment)
                y_pos = bottom[j] - data[i, j] / 2
                # Recalculate bottom for this iteration
                segment_bottom = sum(data[k, j] for k in range(i))
                y_pos = segment_bottom + data[i, j] / 2

                ax.text(
                    j,
                    y_pos,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white" if percentage > 15 else "black",
                )

    # Use tight_layout to minimize whitespace
    plt.tight_layout()

    # Save or show
    if output_file:
        # bbox_inches='tight' removes extra whitespace
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

    plt.close()


def print_summary_table(category_percentages: Dict[str, Dict[str, float]]):
    """Print a text summary table of the percentages."""
    print("\n" + "=" * 80)
    print("Emotion Distribution by Category (Percentage of Total)")
    print("=" * 80)

    # Get emotions
    emotions = list(EMOTION_COLORS.keys())
    categories = sorted(category_percentages.keys())

    # Filter to emotions that exist in data
    first_category = categories[0]
    emotions = [e for e in emotions if e in category_percentages[first_category]]

    # Print header
    print(f"\n{'Category':<15}", end="")
    for emotion in emotions:
        print(f"{emotion.capitalize():>10}", end="")
    print()
    print("-" * (15 + 10 * len(emotions)))

    # Print data
    for category in categories:
        print(f"{category:<15}", end="")
        for emotion in emotions:
            percentage = category_percentages[category].get(emotion, 0.0)
            print(f"{percentage:>9.1f}%", end="")
        print()

    print()


def main():
    # Get results file from command line or use default
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = Path(__file__).parent / "evaluation_results.json"

    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} [results_file.json]")
        print(f"\nDefault: {results_file}")
        sys.exit(1)

    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} results")

    # Compute percentages
    print("\nComputing emotion percentages by category...")
    category_percentages = compute_category_emotion_percentages(results)

    # Print summary table
    print_summary_table(category_percentages)

    # Create visualization
    output_file = Path(results_file).parent / "emotion_distribution.png"
    print(f"\nGenerating chart...")
    create_stacked_bar_chart(category_percentages, str(output_file))

    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()
