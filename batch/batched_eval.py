"""
Batched Emotion Evaluation Script
Processes multiple prompts in batch mode and captures emotion scores
"""

import os
import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "model_name": "Qwen/Qwen3-14B",
    "vector_dir": os.environ.get("VECTOR_DIR", "vectors/Qwen3-14B"),
    "emotions": os.environ.get("EMOTIONS", "joy,love,sadness,surprise,disgust,fear,anger").split(","),
    "orthogonalize": os.environ.get("ORTHOGONALIZE", "residual"),
    "projection": os.environ.get("PROJECTION", "cos"),
    "device": os.environ.get("DEVICE", "cuda:0"),
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.8,
    "do_sample": True,
}

device = torch.device(CONFIG["device"])

# ============================================================================
# Vector Loading & Orthogonalization
# ============================================================================


def load_vectors(vector_dir: str, emotions: List[str]) -> Dict[str, torch.Tensor]:
    """Load emotion vectors for specified emotions."""
    vectors = {}
    for emotion in emotions:
        path = Path(vector_dir) / f"{emotion}_response_avg_diff.pt"
        if path.exists():
            vec = torch.load(path, weights_only=True, map_location="cpu")
            vectors[emotion] = vec.to(device)
            print(f"  Loaded {emotion}: shape {vec.shape}")
        else:
            print(f"  Warning: {path} not found")
    return vectors


def orthogonalize_vectors(vectors: Dict[str, torch.Tensor], method: str) -> Dict[str, torch.Tensor]:
    """Orthogonalize vectors to reduce correlation."""
    if method == "none" or len(vectors) == 0:
        return vectors

    emotions = list(vectors.keys())
    first_vec = next(iter(vectors.values()))
    num_layers = first_vec.shape[0]

    result = {e: torch.zeros_like(vectors[e]) for e in emotions}

    for layer_idx in range(num_layers):
        V = torch.stack([vectors[e][layer_idx] for e in emotions])

        if method == "center":
            mean_vec = V.mean(dim=0)
            V_orth = V - mean_vec
        elif method == "residual":
            V_orth = torch.zeros_like(V)
            for i in range(len(emotions)):
                target = V[i]
                others_idx = [j for j in range(len(emotions)) if j != i]
                if len(others_idx) == 0:
                    V_orth[i] = target
                    continue
                others = V[others_idx]
                Q, _ = torch.linalg.qr(others.T)
                projection = Q @ (Q.T @ target)
                V_orth[i] = target - projection
        else:
            V_orth = V

        for i, e in enumerate(emotions):
            result[e][layer_idx] = V_orth[i]

    print(f"  Applied '{method}' orthogonalization")
    return result


# ============================================================================
# Activation Capture for Batched Generation
# ============================================================================


class BatchedActivationCapturer:
    """Hook to capture activations during batched generation."""

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        # Store all activations across generation steps
        # {layer_idx: list of tensors (batch_size, hidden_dim) for each token}
        self.all_activations: Dict[int, List[torch.Tensor]] = {
            idx: [] for idx in layer_indices}
        self._handles = []

    def _hook_fn_factory(self, layer_idx):
        def _hook_fn(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Capture last token for each batch item: (batch_size, hidden_dim)
            self.all_activations[layer_idx].append(
                hidden[:, -1, :].detach().clone())

        return _hook_fn

    def _locate_layer(self, layer_idx):
        for path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    return cur[layer_idx]
        raise ValueError(f"Could not locate layer {layer_idx}")

    def __enter__(self):
        for layer_idx in self.layer_indices:
            layer = self._locate_layer(layer_idx)
            handle = layer.register_forward_hook(
                self._hook_fn_factory(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_token_activations(self, token_idx: int) -> Dict[int, torch.Tensor]:
        """Get activations for a specific token position across all layers."""
        return {layer_idx: acts[token_idx] for layer_idx, acts in self.all_activations.items()}


# ============================================================================
# Emotion Scoring
# ============================================================================


def compute_scores_batched(
    activations: torch.Tensor,
    vectors: Dict[str, torch.Tensor],
    best_layers: Dict[str, int],
    projection: str = "cos",
) -> Dict[str, List[float]]:
    """
    Compute emotion scores for batched activations.

    Args:
        activations: Dict mapping layer_idx to tensor of shape (batch_size, hidden_dim)
        vectors: Emotion vectors
        best_layers: Best layer for each emotion
        projection: Projection method ('cos' or 'dot')

    Returns:
        Dict mapping emotion to list of scores (one per batch item)
    """
    batch_size = next(iter(activations.values())).shape[0]
    scores = {emotion: [] for emotion in vectors.keys()}

    for emotion, vector in vectors.items():
        layer_idx = best_layers[emotion]
        if layer_idx not in activations:
            scores[emotion] = [0.0] * batch_size
            continue

        activation = activations[layer_idx].float()  # (batch_size, hidden_dim)
        emotion_vec = vector[layer_idx].float()  # (hidden_dim,)

        if projection == "cos":
            # Cosine similarity for each batch item
            dot = (activation * emotion_vec.unsqueeze(0)).sum(dim=1)
            act_norm = activation.norm(dim=1)
            vec_norm = emotion_vec.norm()
            score = (dot / (act_norm * vec_norm + 1e-8)).cpu().tolist()
        else:
            # Dot product projection
            emotion_vec_norm = emotion_vec / (emotion_vec.norm() + 1e-8)
            score = (activation * emotion_vec_norm.unsqueeze(0)
                     ).sum(dim=1).cpu().tolist()

        scores[emotion] = score

    return scores


def get_best_layers(vectors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Find best layer for each emotion based on vector norm."""
    best_layers = {}
    for emotion, vector in vectors.items():
        norms = torch.norm(vector, dim=1)
        best_layers[emotion] = torch.argmax(norms).item()
    return best_layers


def get_dominant_emotion(scores: Dict[str, float]) -> Tuple[str, float]:
    """Get the dominant emotion and its score."""
    dominant = max(scores, key=scores.get)
    return dominant, scores[dominant]


# ============================================================================
# Batched Evaluation
# ============================================================================


def load_prompts(prompts_file: str) -> List[Dict]:
    """Load prompts from JSON file."""
    with open(prompts_file, "r") as f:
        return json.load(f)


def load_all_prompts(base_dir: Path, categories: List[str] = None, max_prompts: int = None) -> List[Dict]:
    """
    Load prompts from multiple category files.

    Args:
        base_dir: Directory containing prompt files
        categories: List of categories to load (e.g., ['math', 'coding'])
                   If None, loads all available categories
        max_prompts: Maximum number of prompts to load per category
                    If None, loads all prompts from each category

    Returns:
        List of prompt dicts with 'category' and 'prompt' fields
    """
    available_files = {
        "gsm8k": "gsm8k_prompts.json",
        "jailbreakbench": "jailbreakbench_prompts.json",
        "math": "math_prompts.json",
        "coding": "coding_prompts.json",
        "poetry": "poetry_prompts.json",
    }

    if categories is None:
        categories = list(available_files.keys())

    all_prompts = []
    for category in categories:
        if category not in available_files:
            print(f"Warning: Unknown category '{category}', skipping")
            continue

        file_path = base_dir / available_files[category]
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        prompts = load_prompts(file_path)

        # Limit number of prompts if max_prompts is specified
        if max_prompts is not None and max_prompts > 0:
            prompts = prompts[:max_prompts]
            print(
                f"  Loaded {len(prompts)} prompts from {category} (limited to {max_prompts})")
        else:
            print(f"  Loaded {len(prompts)} prompts from {category}")

        all_prompts.extend(prompts)

    return all_prompts


def evaluate_batch(
    prompt_dicts: List[Dict],
    model,
    tokenizer,
    emotion_vectors: Dict[str, torch.Tensor],
    best_layers: Dict[str, int],
    config: Dict,
) -> List[Dict]:
    """
    Evaluate a batch of prompts and return emotion scores for each token.

    Args:
        prompt_dicts: List of dicts with 'category' and 'prompt' fields

    Returns:
        List of dicts, one per prompt, each containing:
        - category: prompt category
        - prompt: the input prompt
        - response: generated text
        - token_scores: list of emotion scores per token
        - dominant_emotions: list of dominant emotions per token
        - avg_scores: average emotion scores across all tokens
        - overall_dominant: the overall dominant emotion
    """
    target_layers = list(set(best_layers.values()))

    # Prepare prompts with chat template
    formatted_prompts = []
    for item in prompt_dicts:
        prompt = item["prompt"]
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        formatted_prompts.append(formatted)

    # Tokenize batch
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    results = []
    total_time = 0.0
    total_tokens = 0

    # Generate for each prompt individually to capture per-token emotions
    pbar = tqdm(enumerate(prompt_dicts), total=len(
        prompt_dicts), desc="Evaluating prompts")

    for i, item in pbar:
        prompt = item["prompt"]
        category = item.get("category", "unknown")

        # Update progress bar description
        pbar.set_description(f"[{category}] {prompt[:30]}...")

        # Get single prompt inputs
        single_input = {
            "input_ids": inputs.input_ids[i:i+1],
            "attention_mask": inputs.attention_mask[i:i+1],
        }

        token_scores = []
        dominant_emotions = []
        generated_tokens = []

        # Time this iteration
        iter_start = time.time()

        with BatchedActivationCapturer(model, target_layers) as capturer:
            with torch.no_grad():
                outputs = model.generate(
                    **single_input,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    do_sample=config["do_sample"],
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode generated text
            generated_ids = outputs.sequences[0][single_input["input_ids"].shape[1]:]
            response = tokenizer.decode(
                generated_ids, skip_special_tokens=True)

            # Compute scores for each generated token
            num_tokens = len(capturer.all_activations[target_layers[0]])

            for token_idx in range(num_tokens):
                token_activations = capturer.get_token_activations(token_idx)
                scores = compute_scores_batched(
                    token_activations,
                    emotion_vectors,
                    best_layers,
                    config["projection"],
                )
                # Extract single item from batch
                token_score = {emotion: scores[emotion][0]
                               for emotion in scores}
                token_scores.append(token_score)

                dominant, _ = get_dominant_emotion(token_score)
                dominant_emotions.append(dominant)

        # Compute average scores
        if token_scores:
            avg_scores = {
                emotion: sum(ts[emotion]
                             for ts in token_scores) / len(token_scores)
                for emotion in emotion_vectors.keys()
            }
            overall_dominant, _ = get_dominant_emotion(avg_scores)
        else:
            avg_scores = {emotion: 0.0 for emotion in emotion_vectors.keys()}
            overall_dominant = "none"

        # Calculate iteration time
        iter_time = time.time() - iter_start
        total_time += iter_time
        total_tokens += len(token_scores)

        # Calculate average time per prompt and tokens/sec
        avg_time_per_prompt = total_time / (i + 1)
        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

        results.append({
            "category": category,
            "prompt": prompt,
            "response": response,
            "token_scores": token_scores,
            "dominant_emotions": dominant_emotions,
            "avg_scores": avg_scores,
            "overall_dominant": overall_dominant,
            "num_tokens": len(token_scores),
            "time_seconds": iter_time,
        })

        # Update progress bar with timing stats
        pbar.set_postfix({
            'tokens': len(token_scores),
            'iter_time': f"{iter_time:.2f}s",
            'avg_time': f"{avg_time_per_prompt:.2f}s",
            'tok/s': f"{tokens_per_sec:.1f}"
        })

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 80)
    print("Batched Emotion Evaluation")
    print("=" * 80)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Emotions: {CONFIG['emotions']}")
    print(f"Orthogonalize: {CONFIG['orthogonalize']}")
    print(f"Projection: {CONFIG['projection']}")
    print()

    # Load vectors
    print("Loading emotion vectors...")
    emotion_vectors = load_vectors(CONFIG["vector_dir"], CONFIG["emotions"])
    emotion_vectors = orthogonalize_vectors(
        emotion_vectors, CONFIG["orthogonalize"])

    # Load model
    print(f"\nLoading model {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=CONFIG["device"],
        trust_remote_code=True,
    )
    model.eval()

    best_layers = get_best_layers(emotion_vectors)
    print(f"\nBest layers: {best_layers}")

    # Load prompts from all categories
    base_dir = Path(__file__).parent
    print(f"\nLoading prompts from {base_dir}...")

    # Get categories from environment or use defaults
    categories_str = os.environ.get("CATEGORIES", None)
    if categories_str:
        categories = categories_str.split(",")
        print(f"Loading categories: {categories}")
    else:
        categories = None  # Load all
        print("Loading all available categories")

    # Get max_prompts from environment
    max_prompts_str = os.environ.get("MAX_PROMPTS", None)
    if max_prompts_str:
        try:
            max_prompts = int(max_prompts_str)
            print(f"Limiting to {max_prompts} prompts per category")
        except ValueError:
            print(
                f"Warning: Invalid MAX_PROMPTS value '{max_prompts_str}', ignoring")
            max_prompts = None
    else:
        max_prompts = None

    prompts_data = load_all_prompts(base_dir, categories, max_prompts)
    print(f"\nTotal prompts loaded: {len(prompts_data)}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    results = evaluate_batch(
        prompts_data,
        model,
        tokenizer,
        emotion_vectors,
        best_layers,
        CONFIG,
    )

    # Save results
    output_file = Path(__file__).parent / "evaluation_results.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    # Count by category
    category_counts = {}
    for result in results:
        cat = result['category']
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(
        f"\nProcessed {len(results)} prompts across {len(category_counts)} categories:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} prompts")

    # Timing statistics
    total_eval_time = sum(r['time_seconds'] for r in results)
    total_generated_tokens = sum(r['num_tokens'] for r in results)
    avg_time_per_prompt = total_eval_time / len(results) if results else 0
    avg_tokens_per_prompt = total_generated_tokens / \
        len(results) if results else 0
    tokens_per_second = total_generated_tokens / \
        total_eval_time if total_eval_time > 0 else 0

    print("\nTiming Statistics:")
    print(
        f"  Total evaluation time: {total_eval_time:.2f}s ({total_eval_time/60:.1f} min)")
    print(f"  Average time per prompt: {avg_time_per_prompt:.2f}s")
    print(f"  Total tokens generated: {total_generated_tokens}")
    print(f"  Average tokens per prompt: {avg_tokens_per_prompt:.1f}")
    print(f"  Generation speed: {tokens_per_second:.2f} tokens/sec")

    # Show first few results
    print("\nFirst 5 results:")
    for i, result in enumerate(results[:5]):
        print(f"\n[{result['category']}] Prompt {i+1}: {result['prompt'][:60]}...")
        print(f"  Response: {result['response'][:100]}...")
        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Overall dominant: {result['overall_dominant']}")

    print(f"\n✓ Evaluation complete! Results saved to {output_file}")


if __name__ == "__main__":
    main()
