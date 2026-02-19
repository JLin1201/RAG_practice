
import time, numpy as np
from typing import List, Generator
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

class SimpleRAG:
    def __init__(self, model_path: str, data_path: str):
        # 【修正 1】換成支援中文與多國語言的 Embedding 模型 (約 470MB，依然很輕量)
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        self.documents = [line.strip() for line in open(data_path, 'r').read().split('\n') if line.strip()]
        self.index = self.embedder.encode(self.documents)

    # 【修正 2】把 top_k 提高到 5，確保重要規格不會被漏掉
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        q_vec = self.embedder.encode([query])[0]
        scores = np.dot(self.index, q_vec) / (np.linalg.norm(self.index, axis=1) * np.linalg.norm(q_vec))
        return [self.documents[i] for i in np.argsort(scores)[::-1][:top_k]]

    def generate_stream(self, query: str) -> Generator[str, None, None]:
        context = "\n".join([f"- {c}" for c in self.retrieve(query)])
        messages = [
            {"role": "system", "content": "You are a GIGABYTE AI assistant. Use context strictly to answer the user's question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
        ]
        
        start_time = time.time()
        first_token_time, token_count = None, 0
        
        for chunk in self.llm.create_chat_completion(messages=messages, max_tokens=256, stream=True, temperature=0.3):
            if 'content' in chunk['choices'][0]['delta']:
                if not first_token_time:
                    first_token_time = time.time()
                    yield f"[METRICS] TTFT: {first_token_time - start_time:.4f}s\n"
                token_count += 1
                yield chunk['choices'][0]['delta']['content']
                
        if token_count > 0:
            yield f"\n\n[METRICS] TPS: {token_count / (time.time() - start_time):.2f} tokens/sec"
