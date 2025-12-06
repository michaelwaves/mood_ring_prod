# Batched Emotion Evaluation

This directory contains tools for running batched emotion evaluations on multiple prompts across different task categories.

## Files

### Main Script
- `batched_eval.py` - Main evaluation script that processes prompts in batch mode

### Prompt Files (100 prompts each)
- `gsm8k_prompts.json` - Grade school math word problems (from OpenAI GSM8K dataset)
- `harmbench_prompts.json` - Harmful/adversarial prompts (from JailbreakBench)
- `load_datasets.py` - Script to regenerate GSM8K and HarmBench prompts from HuggingFace
- `math_prompts.json` - Simple arithmetic and algebra problems
- `coding_prompts.json` - Programming task prompts
- `poetry_prompts.json` - Creative writing prompts for poetry

### Output
- `evaluation_results.json` - Generated results with per-token emotion scores (created after running)
- `emotion_distribution.png` - Visualization chart (created by visualize_results.py)

### Visualization Script
- `visualize_results.py` - Creates stacked bar charts showing emotion distribution by category

## Features

The batched evaluation script:

- Captures activations during generation using hooks (similar to /ws/chat endpoint)
- Computes emotion scores for each generated token
- Tracks dominant emotions across the response
- Calculates average emotion scores
- Saves detailed results to JSON

## Loading Datasets from HuggingFace

The GSM8K and HarmBench prompts can be regenerated from HuggingFace datasets:

```bash
# Install datasets library if needed
pip install datasets

# Load 100 prompts from each dataset
python3 load_datasets.py
```

This will:
- Load 100 random samples from `openai/gsm8k` (test split, column: `question`)
- Load 100 random samples from `JailbreakBench/JBB-Behaviors` (train split, column: `Goal`)
- Save them to `gsm8k_prompts.json` and `harmbench_prompts.json`
- Use random seed 42 for reproducibility

## Usage

### Basic Usage - All Categories

By default, evaluates all prompts from all categories (320 prompts total):

```bash
cd /mnt/nw/home/m.yu/repos/emotional-llms/mood_ring/batch
python3 batched_eval.py
```

### Select Specific Categories

Use the `CATEGORIES` environment variable to select which categories to evaluate:

```bash
# Evaluate only math and coding prompts
export CATEGORIES="math,coding"
python3 batched_eval.py

# Evaluate only GSM8K prompts
export CATEGORIES="gsm8k"
python3 batched_eval.py

# Available categories: gsm8k, harmbench, math, coding, poetry
```

### Limit Prompts Per Category

Use the `MAX_PROMPTS` environment variable to limit how many prompts are loaded from each category:

```bash
# Load only 10 prompts from each category
export MAX_PROMPTS=10
python3 batched_eval.py

# Load 5 prompts from math and coding categories
export CATEGORIES="math,coding"
export MAX_PROMPTS=5
python3 batched_eval.py

# Quick test with 1 prompt per category
MAX_PROMPTS=1 python3 batched_eval.py
```

### Configuration Environment Variables

Configure the evaluation using these environment variables:

```bash
# Prompt selection
export CATEGORIES="math,coding,poetry"  # Which categories to evaluate (default: all)
export MAX_PROMPTS=10                   # Max prompts per category (default: all)

# Model configuration
export MODEL_NAME="Qwen/Qwen3-14B"
export VECTOR_DIR="../deploy/vectors/Qwen3-14B"

# Emotion tracking
export EMOTIONS="joy,love,sadness,surprise,disgust,fear,anger"

# Orthogonalization method: none, center, residual
export ORTHOGONALIZE="residual"

# Projection method: cos, dot
export PROJECTION="cos"

# GPU device
export DEVICE="cuda:0"

# Run evaluation
python3 batched_eval.py
```

### Quick Examples

```bash
# Fast test: 1 prompt per category
MAX_PROMPTS=1 python3 batched_eval.py

# Medium test: 10 prompts from math and coding
CATEGORIES=math,coding MAX_PROMPTS=10 python3 batched_eval.py

# Full evaluation of all categories
python3 batched_eval.py
```

### Adding Custom Prompts

Create a new JSON file following this format:

```json
[
  {
    "category": "your_category",
    "prompt": "Your prompt text here"
  }
]
```

Then modify `batched_eval.py` to add your category to the `available_files` dict in `load_all_prompts()`.

## Output Format

The script generates `evaluation_results.json` with the following structure:

```json
[
  {
    "category": "math",
    "prompt": "What is 15 + 27?",
    "response": "Generated response text...",
    "token_scores": [
      {"joy": 0.12, "love": 0.08, "sadness": 0.03, ...},
      {"joy": 0.15, "love": 0.06, "sadness": 0.02, ...}
    ],
    "dominant_emotions": ["joy", "joy", "love", ...],
    "avg_scores": {
      "joy": 0.13,
      "love": 0.07,
      "sadness": 0.02,
      ...
    },
    "overall_dominant": "joy",
    "num_tokens": 45
  }
]
```

Results are organized by category for easy analysis and comparison across task types.

## Visualizing Results

After running the evaluation, visualize the emotion distribution across categories:

```bash
# Visualize results from the default file
python3 visualize_results.py

# Visualize results from a specific file
python3 visualize_results.py path/to/evaluation_results.json
```

The visualization script:
- Creates a **stacked percentage bar chart** showing emotion distribution by category
- Each bar represents 100% of the total emotion score for that category
- Uses the official emotion colors from the Mood Ring server
- Saves the chart as `emotion_distribution.png`
- Prints a summary table to the console

### Example Output

The chart shows:
- **Y-axis**: Percentage of total emotion score (0-100%)
- **X-axis**: Category names (gsm8k, harmbench, math, coding, poetry)
- **Stacked bars**: Each emotion's contribution to the total, colored according to the Mood Ring color scheme

This makes it easy to compare emotional patterns across different task types. For example:
- Math problems might show more neutral/analytical emotions
- Poetry prompts might show more diverse emotional content
- Harmful prompts might trigger different emotional responses

## Implementation Details

### Activation Capture

The script uses `BatchedActivationCapturer` which:
- Registers forward hooks on specified model layers
- Captures the last token's hidden states for each generation step
- Stores activations across all tokens for post-processing

### Emotion Scoring

For each token:
1. Get activations from the best layer for each emotion
2. Compute similarity score using cosine similarity or dot product
3. Determine dominant emotion (max score)
4. Calculate average scores across all tokens

### Comparison with /ws/chat

The batched version:
- Processes prompts sequentially (but could be parallelized)
- Captures all activations during generation
- Computes scores after generation completes
- Saves results to file instead of streaming

The /ws/chat websocket:
- Processes one prompt at a time
- Streams tokens as they're generated
- Computes scores in real-time
- Sends results via websocket

## Requirements

Same as the main Mood Ring server:
- PyTorch
- Transformers
- Emotion vectors in the specified directory
