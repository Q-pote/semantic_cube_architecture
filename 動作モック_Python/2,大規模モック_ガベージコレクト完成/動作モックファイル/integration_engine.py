"""
integration_engine.py
Pure function: 葛藤（矛盾）がないと判定された部分回答群を、自然に1つの提案に統合する。
"""

from typing import List, Tuple
from partial_engine import call_llm

def integrate_responses(partials: List[Tuple[str, str, float]], model_name: str) -> str:
    """
    【Copilot 指示書2: プロンプトシード 3-2 反映】
    部分回答群に矛盾がないことを前提とし、一貫した最終提案を生成する。
    """
    p_text = "\n\n".join([f"[{i+1}] (Trust: {score:.3f})\n{resp}" for i, (cid, resp, score) in enumerate(partials)])
    
    prompt = f"""以下の部分回答は互いに矛盾がなく、同じ方向性を補完しています。
これらを自然に統合し、一貫した最終提案を作成してください。

- 新しい主張は追加しない
- 部分回答の内容だけで統合する
- 方向性の一貫性を保つ

【部分回答】
{p_text}
"""
    return call_llm(prompt, model_name)
