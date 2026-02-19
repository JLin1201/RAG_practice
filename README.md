# GIGABYTE AORUS AI Assistant (Pure Python RAG)

This repository contains a lightweight Retrieval-Augmented Generation (RAG) system specifically designed for the **GIGABYTE AORUS MASTER 16 AM6H**. It is built for a resource-constrained consumer laptop environment (<4GB VRAM) and strictly follows the "No Frameworks" policy, implementing chunking, retrieval, and streaming generation using pure Python, `numpy`, and `llama.cpp`.

## 🛠️ Environment Setup (uv)

Managed by `uv` for strict environment control and fast dependency resolution.

```bash
# 1. Clone repository
git clone <repo_url>
cd gigabyte-aorus-rag

# 2. Sync dependencies
uv sync

# 3. Run the AI Assistant
uv run python main.py
```

## 🧠 Model Selection Strategy (<4GB VRAM Constraint)

To strictly adhere to the 4GB VRAM limit while maintaining high-quality bilingual (TC/EN) support:

* **LLM**: `Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)` (~1.1GB). 
  * *Reason*: Qwen series demonstrates superior performance in Traditional Chinese compared to Llama models of similar size. The 4-bit quantization keeps the memory footprint extremely low while retaining sufficient reasoning capability for spec extraction.
* **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` (~470MB). 
  * *Reason*: Selected specifically to handle the mixed Traditional Chinese and English queries effectively.
* **Total VRAM**: Estimated usage is **< 2GB**, leaving ample headroom for the 2048 context window and OS overhead.

## 📊 System Evaluation (System Evaluation)

Tested on a local GPU environment simulating edge constraints:

### Quantitative Metrics
* **TTFT (Time To First Token)**: `~0.23s`. Extremely fast prefilling.
* **TPS (Tokens Per Second)**: `~39 t/s`. Smooth streaming generation experience.

### Qualitative Analysis (Benchmark)
* **Data Parsing & Retrieval**: The manual pure Numpy cosine similarity accurately retrieves and differentiates specific structural specs (e.g., successfully distinguishing between Left and Right I/O ports of the AORUS MASTER 16).
* **Bilingual Capability**: The system successfully understands Traditional Chinese queries (e.g., "電池容量與變壓器瓦數") and matches them with English spec sheets (e.g., "99Wh" and "240W").
* **Generation Quality**: Responses are streamed directly from the retrieved context, successfully preventing hallucinations regarding hardware specifications.
