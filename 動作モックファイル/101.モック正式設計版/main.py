# main.py (v9.0: 無限対話ループ & SRAG統合版)

import os
import numpy as np

# --- 環境変数設定 ---
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"  # Gemini APIキーを設定してください
os.environ["GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
os.environ["GEMINI_LLM_MODEL"] = "gemini-3.5-flash"

# --- APIコールウェイトタイム ---
os.environ["API_CALL_WAIT_TIME"] = "0.3"  # 秒単位で指定。Gemini APIのレート制限対策。

# --- SRAG検索結果の「葛藤検知」スイッチ ---
# boolean値を文字列で指定。TrueならAnalyzeEngineで葛藤検知を行い、LLMに通知する。
os.environ["CONFLICT_DETECT_SWITCH"] = "True"


# --- モジュールのインポート ---
from analyze_engine import AnalyzeEngine
from data_hub import DataHub
from llm_service import GeminiProvider, LLMService
from cube_engine import CubeEngine
from integration_engine import IntegrationEngine
from srag_engine import SragEngine
from defrag_engine import DefragEngine

# ログライター追加用
from datetime import datetime
import sys

def main():

    # 先にロギング環境を整備
    # 1. 現在時刻から動的にログファイル名を決定
    now_str = datetime.now().strftime("%y%m%d_%H%M")
    log_filename = f"Conversation_Log_{now_str}.txt"
    
    # 2. ダブルライターを生成して、システムの標準出力を乗っ取る
    writer = DoubleWriter(log_filename)
    sys.stdout = writer
    
    print(f"🛰️  System logger started. Saving to: {log_filename}")


    print("🚀 Semantic Cube Runtime v9.0 (Interactive Loop) 起動...")
    
    # --- 1. 初期化 ---
    hub = DataHub()
    gemini_provider = GeminiProvider()
    llm_service = LLMService(provider=gemini_provider)
    cube_eng = CubeEngine(llm_service=llm_service)
    integrate_eng = IntegrationEngine(llm_service=llm_service)
    srag_eng = SragEngine()

    # --- 2. 【最初の1回だけ】プロジェクターの学習フェーズ ---
    initial_memory_texts = [
        "ガンダムとは、宇宙世紀を舞台にした人間ドラマである。",
        "インドカレーはスパイスの芸術品だ。",
        "高機能なスポーツウェアは、もはや第二の皮膚と言える。",
        "AIの進化は、人間の知性を拡張する。",
        "猫は液体であり、宇宙の真理を知っている。",
        "磐田市のうなぎは最高に美味い。",
        "プログラミングとは、世界を再定義する行為だ。",
        "人生における大切なことは、だいたいサウナと水風呂が教えてくれる。",
        "雨の日に聞くジャズは、心を落ち着かせる。",
        "本当に良い設計は、引き算から生まれる。"
    ]
    
    print("\n=======================================================")
    print("🌍 機動準備: PCA学習中...")
    print("=======================================================")
    initial_vectors = []
    for text in initial_memory_texts:
        result = llm_service.call_api_embed(text)
        if result.http_code == 200:
            initial_vectors.append(result.data)
    
    if len(initial_vectors) >= 3:
        cube_eng.fit_projector(np.array(initial_vectors))
        # 学習に使った記憶も、ちゃんとキューブとして宇宙に配置する
        for i, text in enumerate(initial_memory_texts):
            memory_cube = cube_eng.make_cube(query=text, role="knowledge", source="LLM", current_turn=0)
            hub.cube_add(memory_cube)
        print(f"✅ Universe created with {len(hub.cube_get_all())} initial knowledge cubes.")

    # --- 3. 【無限対話ループ】---
    print("\n=======================================================")
    print("💬 対話モードを開始(終了するには 'exit' と入力)")
    print("💡 記憶代謝処理(デフラグ)を実行するには 'defrag' と入力")
    print("=======================================================\n")
    
    try:
        while True:
        
            # ユーザーからの入力を受け付ける
            user_query = input("\n👤 You: ")
            if user_query.lower() == 'exit':            # 終了指示を受け取る
                print("👋 See you, space cowboy...")
                break
            elif user_query.lower() == 'defrag':        # チャット中にデフラグ指示を受け取る
                defrag_eng = DefragEngine(hub, llm_service, cube_eng)
                defrag_eng.run_defrag_cycle()

                # # デフラグが「お墓行き」のフラグを立てた直後に、GCがメインプールから隔離する
                # gc_eng = GCEngine(hub)
                # gc_eng.run_gc_cycle()
                continue


            # --- 1ターンの実行 ---
            hub.tick() # ターンを進める

            # 1. ユーザーの問いをキューブ化
            q_cube = cube_eng.make_cube(query=user_query, role="history", source="User", current_turn=hub.global_turn)
            hub.cube_add(q_cube)

            # プロンプト構成のため、ここでは追加しない
            # hub.history_push(q_cube)
            
            # 2. SRAGで過去の記憶を検索！
            search_results = srag_eng.search(q_cube, hub, mode="exploratory")  # "standard" or "exploratory"
            
            # --- LLM応答を取得 ---
            answer = ""
            if not search_results:
                # 【ルート1】SRAGが0件なら、履歴だけを頼りに single を呼ぶ
                answer = integrate_eng.single(q_cube, hub.history_get_all())
            else:
                # 【ルート2】SRAGが1件以上なら、結果をブレンドする integration を呼ぶ
                conflict_detect_switch = os.environ.get("CONFLICT_DETECT_SWITCH", "True").lower() == "true"
                tension_meta = None
                if conflict_detect_switch:
                    # 葛藤検知スイッチがONなら、AnalyzeEngineで葛藤を検知してから integration を呼ぶ
                    analyze_eng = AnalyzeEngine(tension_threshold=0.5)
                    tension_meta = analyze_eng.detect_tension(search_results)
                    answer = integrate_eng.integration(q_cube, search_results, hub.history_get_all(), tension_meta=tension_meta, conflict_detect_switch=True)
                else:
                    # 葛藤検知スイッチがOFFなら、単純に integration を呼ぶ
                    answer = integrate_eng.integration(q_cube, search_results, hub.history_get_all(), tension_meta=None, conflict_detect_switch=False)


            # 4. ログと応答の表示
            if not search_results:
                # print("  --- 該当する過去の記憶は見つかりませんでした ---")
                pass
            else:
                for i, result in enumerate(search_results):
                    print(f"  [Rank {i+1}] '{result.cube.summary[:20]}...' (Score: {result.final_score:.3f})")
                    for match in result.matched_minicubes:
                        print(f"    - Matched: '{match.query_phrase}' -> '{match.target_phrase}'")

            print("-" * 32 + "\n")
            print(f"\n🤖 LLM: {answer}")
            print("-" * 32)

            # プロンプト構築の整合性のため、ＬＬＭ回答を取得した後にQ-cubeを履歴に追加する
            hub.history_push(q_cube)

            # 5. AIの応答もキューブ化して、因果を紐付ける
            a_cube = cube_eng.make_cube(query=answer, role="history", source="LLM", current_turn=hub.global_turn)
            a_cube.reactionLink_ID = q_cube.cube_id
            q_cube.reactionLink_ID = a_cube.cube_id
            hub.cube_add(a_cube)
            hub.history_push(a_cube)
            
            print(f"\n[System] Turn {hub.global_turn} complete. Total cubes: {len(hub.cube_pool)}.")

    except KeyboardInterrupt:
        print("\n👋 System interrupted. Shutting down.")
    except Exception as e:
        print(f"🚨 An unexpected error occurred: {e}")
    finally:
        # ─── 最後に忘れずにファイルを閉じる ───
        sys.stdout = writer.terminal  # システム出力を元に戻す
        writer.close()
        print(f"💾 Conversation log saved successfully: {log_filename}")


# --- ログ出力をコンソールとファイルの両方に同時に出力するためのマニュピレータ ---
class DoubleWriter:
    """コンソール(stdout)とファイルの両方に同時に出力するマニュピレータ"""
    def __init__(self, file_path: str):
        self.terminal = sys.stdout  # 元のコンソール出力を保持
        self.log_file = open(file_path, "w", encoding="utf-8") # ログファイルをOpen

    def write(self, message):
        self.terminal.write(message)      # 画面に出力
        self.log_file.write(message)      # ファイルに書き込み
        self.log_file.flush()             # リアルタイムにディスクに書き込みを反映

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# --- エントリーポイント ---
if __name__ == "__main__":
    main()
