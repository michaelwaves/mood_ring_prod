"""
Mood Ring - Production Server
Real-time emotion visualization for LLM responses
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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
}

device = torch.device(CONFIG["device"])

EMOTION_COLORS = {
    "love": "#FACC15",
    "joy": "#22C55E",
    "surprise": "#06B6D4",
    "sadness": "#3B82F6",
    "anger": "#D946EF",
    "fear": "#EF4444",
    "disgust": "#84CC16",
}

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Mood Ring")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Global state
model = None
tokenizer = None
emotion_vectors: Dict[str, torch.Tensor] = {}

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
# Activation Capture
# ============================================================================


class ActivationCapturer:
    """Hook to capture activations during generation from multiple layers."""

    def __init__(self, model, layer_indices: List[int]):
        self.model = model
        self.layer_indices = layer_indices
        self.captured_activations: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook_fn_factory(self, layer_idx):
        def _hook_fn(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            self.captured_activations[layer_idx] = hidden[:, -1, :].detach().clone()

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
            handle = layer.register_forward_hook(self._hook_fn_factory(layer_idx))
            self._handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self._handles:
            handle.remove()
        self._handles = []


# ============================================================================
# Emotion Scoring
# ============================================================================


def compute_scores(
    activations: Dict[int, torch.Tensor],
    vectors: Dict[str, torch.Tensor],
    best_layers: Dict[str, int],
    projection: str = "cos",
) -> Dict[str, float]:
    """Compute emotion scores for activations."""
    scores = {}

    for emotion, vector in vectors.items():
        layer_idx = best_layers[emotion]
        if layer_idx not in activations:
            continue

        activation = activations[layer_idx].float().squeeze()
        emotion_vec = vector[layer_idx].float()

        if projection == "cos":
            dot = (activation * emotion_vec).sum()
            score = (dot / (activation.norm() * emotion_vec.norm() + 1e-8)).item()
        else:
            emotion_vec_norm = emotion_vec / (emotion_vec.norm() + 1e-8)
            score = (activation * emotion_vec_norm).sum().item()

        scores[emotion] = score

    return scores


def get_best_layers(vectors: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Find best layer for each emotion based on vector norm."""
    best_layers = {}
    for emotion, vector in vectors.items():
        norms = torch.norm(vector, dim=1)
        best_layers[emotion] = torch.argmax(norms).item()
    return best_layers


# ============================================================================
# Startup
# ============================================================================


@app.on_event("startup")
async def startup():
    global model, tokenizer, emotion_vectors

    print("=" * 60)
    print("Mood Ring - Starting up")
    print("=" * 60)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Emotions: {CONFIG['emotions']}")
    print(f"Orthogonalize: {CONFIG['orthogonalize']}")
    print(f"Projection: {CONFIG['projection']}")
    print()

    # Load vectors
    print("Loading emotion vectors...")
    emotion_vectors = load_vectors(CONFIG["vector_dir"], CONFIG["emotions"])
    emotion_vectors = orthogonalize_vectors(emotion_vectors, CONFIG["orthogonalize"])

    # Load model
    print(f"\nLoading model {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.bfloat16,
        device_map=CONFIG["device"],
        trust_remote_code=True,
    )
    model.eval()

    print("\n" + "=" * 60)
    print("Mood Ring ready!")
    print("=" * 60)


# ============================================================================
# Routes
# ============================================================================

# Serve static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")


@app.get("/config")
async def get_config():
    """Return configuration for frontend."""
    return {
        "model": CONFIG["model_name"],
        "emotions": CONFIG["emotions"],
        "orthogonalize": CONFIG["orthogonalize"],
        "projection": CONFIG["projection"],
        "colors": {e: EMOTION_COLORS.get(e, "#666") for e in CONFIG["emotions"]},
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat with emotion scores."""
    await websocket.accept()

    best_layers = get_best_layers(emotion_vectors)
    # Get all unique layers we need to monitor
    target_layers = list(set(best_layers.values()))

    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            history = data.get("history", [])

            if not user_message.strip():
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Build messages
            messages = []
            for h in history:
                messages.append({"role": "user", "content": h["user"]})
                messages.append({"role": "assistant", "content": h["assistant"]})
            messages.append({"role": "user", "content": user_message})

            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            gen_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.8,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": tokenizer.pad_token_id,
            }

            token_count = 0

            with ActivationCapturer(model, target_layers) as capturer:

                def generate():
                    with torch.no_grad():
                        model.generate(**gen_kwargs)

                thread = Thread(target=generate)
                thread.start()

                try:
                    for token_text in streamer:
                        if not token_text:
                            continue

                        if capturer.captured_activations:
                            token_count += 1
                            scores = compute_scores(
                                capturer.captured_activations, emotion_vectors, best_layers, CONFIG["projection"]
                            )

                            dominant = max(scores, key=scores.get)

                            await websocket.send_json(
                                {
                                    "type": "token",
                                    "token": token_text,
                                    "scores": scores,
                                    "dominant_emotion": dominant,
                                    "color": EMOTION_COLORS.get(dominant, "#666"),
                                    "token_count": token_count,
                                }
                            )
                        else:
                            await websocket.send_json(
                                {
                                    "type": "token",
                                    "token": token_text,
                                    "scores": {e: 0.0 for e in CONFIG["emotions"]},
                                    "token_count": 0,
                                }
                            )

                        await asyncio.sleep(0.01)

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

                thread.join()

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8080)
