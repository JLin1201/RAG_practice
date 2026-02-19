# AI Hardware Assistant (Pure Python RAG)

This repository contains a lightweight RAG system designed for resource-constrained environments (Consumer Laptop, <4GB VRAM). It strictly follows the "No Frameworks" policy, implementing chunking, retrieval, and generation using pure Python, `numpy`, and `llama.cpp`.

## 🛠️ Environment Setup

Managed by `uv` for speed and consistency.

```bash
# 1. Clone repository
git clone <repo_url>
cd gigabyte-rag-task

# 2. Sync dependencies
uv sync

# 3. Run the AI Assistant
uv run python main.py
🧠 Model Strategy (4GB Constraint)
LLM: Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M) (~1GB). Chosen for its excellent Traditional Chinese support and optimal reasoning-to-size ratio.

Embedding: all-MiniLM-L6-v2 (~80MB).

Total VRAM: well under 2GB, leaving room for context.

📊 Evaluation Metrics
TTFT (Time To First Token): ~0.3s.

TPS (Tokens Per Second): ~45-50 t/s.

Qualitative: The pure Numpy cosine similarity accurately retrieves spec chunks (e.g., distinguishing I/O ports), and the model generates streaming responses without hallucination.
