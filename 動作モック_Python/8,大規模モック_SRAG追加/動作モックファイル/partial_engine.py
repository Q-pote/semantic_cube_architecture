"""
partial_engine.py
Pure function: 単一の根拠（キューブ）から部分回答（Partial Response）を生成する。
※ v3.1: 語義DNA(MiniCube) ＋ 時間/生命的メタデータ(Trust)の3点セット注入仕様
"""

import time
import google.generativeai as genai
from common_types import SemanticCube

def call_llm(prompt: str, model_name: str) -> str:
    """共通のLLM呼び出し関数（エラーハンドリング付き）"""
    
    # # デバッグ
    # print( "="*50 + "\n") # 区切り線
    # print(f"🔍 [LLM Prompt]\n{prompt}\n")
    # print( "="*50 + "\n") # 区切り線

    try:
        time.sleep(1) # API制限回避用
        model = genai.GenerativeModel(model_name)
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        print(f"⚠️ [API Error] LLM Call failed: {e}")
        return f"[Mock LLM Response] LLM呼び出しに失敗しました。"

def make_partial(query: str, cube: SemanticCube, score: float, model_name: str) -> str:
    """
    【Copilot 指示書2: プロンプトシード 3-1 ＆ v3.1 3点セット注入仕様反映】
    1つのキューブのみを絶対的な前提とし、語義DNAと時間メタデータを用いて
    超高精度にユーザーの質問に部分応答する。
    """
    # 1. 時間・生命的メタデータの抽出
    created_at = cube.trust.created_at_time
    ref_count = cube.trust.ref_count
    gravity = cube.trust.gravity
    
    # 2. 語義DNA (MiniCubeフレーズ) の抽出
    keyphrases = [mc.phrase for mc in cube.mini_cubes]
    kp_text = ", ".join(keyphrases) if keyphrases else "None (語義フレーズなし)"

    # 3. 3点セット（Summary + Keyphrases + Temporal Meta）をヘッダに持つ極上プロンプト
    prompt = f"""
オーダー：
【回答根拠を補助としてユーザーからの質問に回答してください。
非常に抽象化された回答や無意味なポエムは禁止とします。
（ただし、ユーザーがそれら抽象化やポエム回答を望む場合は許可します）】

ユーザーからの質問：
【{query}】

【根拠の補助となる情報】
- 根拠ID: {cube.cube_id}
- 話題コンテキスト（Summary）: "{cube.summary}"
- 抽出キーフレーズ（Keyphrases）: [{kp_text}]
- メタデータ（Metadata）:
  * 参照頻度 (Ref Count): {ref_count:.2f}
  * 生成ターン (Created time): {created_at}
- 質問との類似スコア (Hybrid Trust): {score:.3f}
"""
    
    # デバッグ用プロンプト表示    
    print("="*60)
    print(f"  🧩 [Partial Prompt]\n{prompt}\n")
    print("="*60)

    llm_response = call_llm(prompt, model_name)
    # print(f"  👉 LLM回答:\n【{llm_response}】\n")
    return llm_response
