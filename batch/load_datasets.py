"""
Load prompts from HuggingFace datasets
Generates JSON files with prompts from GSM8K and JailbreakBench datasets
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)


def load_gsm8k_prompts(num_prompts: int = 100) -> list:
    """
    Load prompts from OpenAI GSM8K dataset.

    Args:
        num_prompts: Number of prompts to load

    Returns:
        List of dicts with 'category' and 'prompt' fields
    """
    print(f"Loading GSM8K dataset...")

    # Load the test split of GSM8K
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    # Sample prompts
    if len(dataset) > num_prompts:
        indices = random.sample(range(len(dataset)), num_prompts)
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)

    # Extract questions
    prompts = []
    for sample in samples:
        prompts.append({
            "category": "gsm8k",
            "prompt": sample["question"]
        })

    print(f"  Loaded {len(prompts)} prompts from GSM8K")
    return prompts


def load_jailbreakbench_prompts(num_prompts: int = 100) -> list:
    """
    Load prompts from JailbreakBench/JBB-Behaviors dataset.

    Args:
        num_prompts: Number of prompts to load

    Returns:
        List of dicts with 'category' and 'prompt' fields
    """
    print(f"Loading JailbreakBench dataset...")

    # Load the JailbreakBench dataset
    dataset = load_dataset("JailbreakBench/JBB-Behaviors",
                           'behaviors', split="harmful")

    # Sample prompts
    if len(dataset) > num_prompts:
        indices = random.sample(range(len(dataset)), num_prompts)
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)

    # Extract goals
    prompts = []
    for sample in samples:
        prompts.append({
            "category": "jailbreakbench",
            "prompt": sample["Goal"]
        })

    print(f"  Loaded {len(prompts)} prompts from JailbreakBench")
    return prompts


def main():
    print("=" * 80)
    print("Loading Prompts from HuggingFace Datasets")
    print("=" * 80)
    print()

    base_dir = Path(__file__).parent

    # Load GSM8K
    print("\n1. GSM8K Dataset")
    print("-" * 40)
    gsm8k_prompts = load_gsm8k_prompts(num_prompts=100)

    # Save GSM8K
    gsm8k_file = base_dir / "gsm8k_prompts.json"
    with open(gsm8k_file, "w") as f:
        json.dump(gsm8k_prompts, f, indent=2)
    print(f"  Saved to {gsm8k_file}")

    # Load JailbreakBench
    print("\n2. JailbreakBench Dataset")
    print("-" * 40)
    jbb_prompts = load_jailbreakbench_prompts(num_prompts=100)

    # Save JailbreakBench
    jbb_file = base_dir / "jailbreakbench_prompts.json"
    with open(jbb_file, "w") as f:
        json.dump(jbb_prompts, f, indent=2)
    print(f"  Saved to {jbb_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"GSM8K:          {len(gsm8k_prompts)} prompts -> {gsm8k_file}")
    print(f"JailbreakBench: {len(jbb_prompts)} prompts -> {jbb_file}")
    print(f"\nTotal: {len(gsm8k_prompts) + len(jbb_prompts)} prompts")
    print("\n✓ Dataset loading complete!")

    # Show examples
    print("\n" + "=" * 80)
    print("Examples")
    print("=" * 80)

    print("\nGSM8K (first 3):")
    for i, prompt in enumerate(gsm8k_prompts[:3], 1):
        print(f"\n{i}. {prompt['prompt'][:100]}...")

    print("\n\nJailbreakBench (first 3):")
    for i, prompt in enumerate(jbb_prompts[:3], 1):
        print(f"\n{i}. {prompt['prompt'][:100]}...")


if __name__ == "__main__":
    main()
