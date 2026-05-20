"""
partial_engine.py
Pure function: 単一の根拠（キューブ）から部分回答（Partial Response）を生成する。
"""

import time
import google.generativeai as genai
from common_types import SemanticCube

def call_llm(prompt: str, model_name: str) -> str:
    """共通のLLM呼び出し関数（エラーハンドリング付き）"""
    try:
        time.sleep(1) # API制限回避用
        model = genai.GenerativeModel(model_name)
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        print(f"⚠️ [API Error] LLM Call failed: {e}")
        return f"[Mock LLM Response] LLM呼び出しに失敗しました。"

def make_partial(query: str, cube: SemanticCube, score: float, model_name: str) -> str:
    """
    【Copilot 指示書2: プロンプトシード 3-1 反映】
    1つのキューブのみを絶対的な前提として、部分回答を生成する。
    """
    prompt = f"""あなたは参謀です。
以下の【1つの根拠】だけを絶対的な前提として、ユーザーの質問に部分的に答えてください。
- 他の視点には触れない
- 新しい主張を追加しない
- この根拠の視点だけで答える

【根拠】
- [{cube.role}] {cube.summary} (Hybrid Trust: {score:.3f})

【質問】
{query}
"""
    return call_llm(prompt, model_name)
