# Edge AI Hardware Assistant (Pure Python RAG)

This repository contains a lightweight Retrieval-Augmented Generation (RAG) system designed for resource-constrained environments (e.g., Consumer Laptops, <4GB VRAM). It strictly follows a "No Frameworks" approach, implementing chunking, retrieval, and generation using pure Python, `numpy`, and `llama.cpp`.

## 🛠️ Environment Setup

Managed by `uv` for speed and consistency.

```bash
# 1. Clone repository
git clone <repo_url>
cd edge-rag-assistant

# 2. Sync dependencies
uv sync

# 3. Run the AI Assistant
uv run python main.py
```

## 🧠 Model Selection Strategy (<4GB VRAM Constraint)

* **LLM**: `Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)` (~1.1GB). Chosen for its excellent Traditional Chinese support and optimal reasoning-to-size ratio.
* **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (~470MB). Selected specifically to handle mixed Traditional Chinese and English queries.
* **Total VRAM**: Estimated usage is **< 2GB**, leaving ample headroom for the context window and OS overhead, perfectly adhering to edge constraints.

## 📊 Evaluation Metrics

Tested on a local GPU environment simulating edge constraints:

* **TTFT (Time To First Token)**: `~0.23s`. Extremely fast prefilling due to the lightweight quantized model.
* **TPS (Tokens Per Second)**: `~39 t/s`. Smooth streaming generation experience.

### Qualitative Analysis
* **Retrieval Accuracy**: The manual pure Numpy cosine similarity accurately retrieves and differentiates specific technical spec chunks (e.g., successfully distinguishing between Left and Right I/O ports on a device).
* **Bilingual Capability**: Thanks to the multilingual embedding model, the system successfully understands Traditional Chinese queries (e.g., "電池容量與變壓器瓦數") and matches them with English spec sheets (e.g., "99Wh" and "240W") without missing context.
* **Generation**: The model generates streaming responses strictly based on the provided hardware context without hallucination.
