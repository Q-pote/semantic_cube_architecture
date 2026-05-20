"""
conflict_engine.py
Pure function: 物理的に葛藤（方位差が大きい）と判定された場合、
そのトラスト分布から「対立可視化（高トラスト）」か「採用却下（低トラスト）」かをLLMに判断させる。
"""

from typing import List, Tuple
from common_types import SemanticCube
from trust_evaluator import angular_distance
from partial_engine import call_llm

def determine_route_by_physics(cubes: List[SemanticCube]) -> bool:
    """
    【物理法則によるルート分岐】
    抽出されたキューブ群の最大方位差を計算し、0.3以上（意味方向が異なる）なら葛藤ルート(True)へ投げる。
    """
    if len(cubes) < 2: return False
    
    max_dist = 0.0
    for i in range(len(cubes)):
        for j in range(i+1, len(cubes)):
            dist = angular_distance(cubes[i].trust.orientation, cubes[j].trust.orientation)
            if dist > max_dist:
                max_dist = dist
                
    return max_dist >= 0.3

def analyze_conflict(partials: List[Tuple[str, str, float]], model_name: str) -> str:
    """
    【Copilot 指示書2: プロンプトシード 3-3 & 3-4 反映（閾値レス統合）】
    Trustスコアの分布をLLMに提示し、メタ判断（採用/保留）を委ねる。
    """
    p_text = "\n\n".join([f"[{i+1}] (Trust: {score:.3f})\n{resp}" for i, (cid, resp, score) in enumerate(partials)])
    
    prompt = f"""以下の部分回答には、方向性の異なる主張が存在します。
各回答には信頼性(Trustスコア)が付与されています。これらを総合的に分析し、以下の方針の【どちらか適切と思われる方】で回答してください。

■ 方針A（全体的にTrustスコアが低いノイズ群であると判断した場合）
- なぜ採用できないのか（信頼性の低さ）を説明する
- 追加の情報やより信頼できる根拠が必要であることを示す
- 結論は保留とし、判断を急がないよう助言する

■ 方針B（十分なTrustスコアを持つ対立意見であると判断した場合）
- 対立点と共通点を整理する
- それぞれの視点が成立する条件を説明する
- 統合は行わず、選択肢として提示する
- 最後に判断軸を示す

【部分回答】
{p_text}
"""
    return call_llm(prompt, model_name)
