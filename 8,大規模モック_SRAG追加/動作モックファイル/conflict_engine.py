"""
conflict_engine.py
※ v3.1: 引数に SemanticCube を直接受け取り、語義DNAとメタデータをプロンプトに対比注入する仕様へ変更
"""
from typing import List, Tuple
from common_types import SemanticCube
from trust_evaluator import angular_distance
from partial_engine import call_llm
import time

def determine_route_by_physics(cubes: List[SemanticCube]) -> bool:
    """物理的葛藤検知：最大方位角差が0.3以上なら葛藤ルートへ"""
    if len(cubes) < 2: return False
    max_dist = 0.0
    for i in range(len(cubes)):
        for j in range(i+1, len(cubes)):
            dist = angular_distance(cubes[i].trust.orientation, cubes[j].trust.orientation)
            if dist > max_dist:
                max_dist = dist
    return max_dist >= 0.3

# ✨ 変更: `Tuple[str, str, float]` から `Tuple[SemanticCube, str, float]` へ！
def analyze_conflict(partials: List[Tuple[SemanticCube, str, float]], model_name: str) -> str:
    blocks = []
    for i, (cube, resp, score) in enumerate(partials):
        kp = ", ".join([mc.phrase for mc in cube.mini_cubes]) if cube.mini_cubes else "None"
        block = f"[{i+1}] (根拠ID: {cube.cube_id})\n"
        block += f"  - LLMへの過去の質問=【{cube.summary}】\n"
        block += f"  - 抽出キーフレーズ (Keyphrases): [{kp}]\n"
        block += f"  - メタデータ: 参照頻度={cube.trust.ref_count:.1f}, 生成ターン={cube.trust.created_at_time}, 関連性(Trust)={score:.3f}\n"
        block += f"  - 質問に対するLLMからの回答=【{cube.response_and_answer}】\n"
        blocks.append(block)
        
    p_text = "\n\n".join(blocks)
    
    prompt = f"""以下の回答根拠の中には、方向性の異なる意見の葛藤が存在します。
各回答には、抽出したキーフレーズ、参照頻度や質問との関連性が付与されているので参考にしてください。
これらを総合的に分析し適切と思われる回答を行ってください。

■ 方針A（全体的にTrustスコアが低いノイズ群であると判断した場合）
- なぜ採用できないのかを説明してください。（根拠データに基づく判断内容など）
- 結論は保留とし、判断を急がないよう助言してください。

■ 方針B（十分なTrustスコアを持つ対立意見であると判断した場合）
- 対立点と共通点を整理する（各回答の抽出キーフレーズの違い等から対立軸を明確にしてください）
- 意見の統合は行わず、どちらの回答が適しているか選択肢として提示するようにしてください。

==【分析対象の根拠群】==
{p_text}
"""
    return call_llm(prompt, model_name)
