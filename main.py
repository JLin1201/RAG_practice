
import sys
from rag_engine import SimpleRAG

def main():
    print("🚀 啟動 AI Assistant (4GB VRAM Edition)...")
    
    # 初始化 RAG
    rag = SimpleRAG(
        model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf", 
        data_path="data.txt"
    )
    
    print("\n💡 互動模式 (輸入 'exit' 離開)")
    print("提示: 支援繁體中文與英文提問。")
    
    # 模擬互動迴圈
    # 注意：若要自動化測試可修改此處，這裡保留互動功能
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
                
            print("AI: ", end="", flush=True)
            metrics_log = []
            
            # 接收串流輸出
            for token in rag.generate_stream(query):
                if "[METRICS]" in token:
                    metrics_log.append(token.strip())
                else:
                    print(token, end="", flush=True)
            
            # 顯示效能數據 (TTFT & TPS)
            print("\n\n" + "="*20 + " Performance " + "="*20)
            for m in metrics_log:
                print(m)
            print("="*53)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
