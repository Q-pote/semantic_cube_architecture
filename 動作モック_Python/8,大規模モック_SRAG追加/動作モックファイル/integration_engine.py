"""
integration_engine.py
※ v3.1: 引数に SemanticCube を直接受け取り、語義DNAとメタデータを調和の軸としてプロンプトに注入
"""
from typing import List, Tuple
from common_types import SemanticCube
from partial_engine import call_llm

# ✨ 変更: `Tuple[str, str, float]` から `Tuple[SemanticCube, str, float]` へ！
def integrate_responses(partials: List[Tuple[SemanticCube, str, float]], model_name: str) -> str:
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
    
    prompt = f"""以下の根拠を統合して適切な文章を作成してください。
各根拠には、抽出キーフレーズ、参照頻度や関連性(Trustスコア)が付与されています。
これらを自然に統合し、一貫した提案を作成してください。

【ルール】
- 根拠に基づき判断を行うことを優先してください。
- 抽象的な表現のみの文章は作成しないでください（ポエムは禁止）
- 関連性や参照頻度が高い回答のニュアンスをやや優先しつつ、方向性を保つようにしてください。

==【分析対象の根拠群】==
{p_text}
"""
    return call_llm(prompt, model_name)
