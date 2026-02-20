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

## ⚙️ RAG Pipeline Implementation (No-Framework Approach)

To strictly adhere to the "No Frameworks" rule and edge device constraints, this system implements a custom, lightweight Dense Retrieval RAG pipeline:

1. **Structured Chunking Strategy**: Instead of arbitrary character-level splitting, the knowledge base (`data.txt`) is pre-processed into structurally meaningful chunks (e.g., `[Common Specs]`, `[Model Specific]`). This guarantees that semantic boundaries are preserved, drastically reducing retrieval noise.
2. **Dense Vector Retrieval**: Utilizes `SentenceTransformer` to encode queries and documents into dense vectors.
3. **Pure Numpy Cosine Similarity**: Replaced heavy vector databases (like ChromaDB/FAISS) with a pure `numpy` exact nearest-neighbor (KNN) search. This ensures O(N) retrieval complexity with virtually zero memory overhead, perfect for edge environments.
4. **Context-Augmented Streaming Generation**: The retrieved Top-K chunks are injected into a highly constrained system prompt, guiding the `llama-cpp-python` engine to generate streaming responses (enhancing TTFT UX) while strictly grounding answers to the provided context.

## 📊 System Evaluation (System Evaluation)

Tested on a local GPU environment simulating edge constraints:

### Quantitative Metrics
* **TTFT (Time To First Token)**: `~0.23s`. Extremely fast prefilling.
* **TPS (Tokens Per Second)**: `~39 t/s`. Smooth streaming generation experience.

### Qualitative Analysis (Benchmark)
* **Complex Data Parsing & Anti-Hallucination**: The knowledge base was strategically structured into "Common Specs" and "Model Specific" tiers to handle the AM6H series' internal GPU variants (BZH, BYH, BXH). The manual Numpy cosine similarity successfully differentiates between these variants without cross-contaminating specifications.
  * **Test Query:** "請問頂規的 BZH 和入門的 BXH，這兩款的顯示卡 (GPU) 分別是配哪一張？它們的 Maximum Graphics Power 是一樣的嗎？"
  * **System Response:** "頂規的 BZH (RTX 5090) 和入門的 BXH (RTX 5070 Ti) 的 GPU 分別是配 RTX 5090 和 RTX 5070 Ti。兩者的 Maximum Graphics Power 是不同的，BZH 是 175W，而 BXH 是 140W。"
  * **Result:** Perfect extraction and differentiation under 4GB VRAM constraint.
* **Precision Retrieval**: Accurately retrieves and differentiates spatial hardware details (e.g., distinguishing between Left and Right I/O ports, including Thunderbolt 4 vs 5 placements).
* **Bilingual Capability**: The multilingual embedding effectively maps Traditional Chinese queries to the English spec sheet.
