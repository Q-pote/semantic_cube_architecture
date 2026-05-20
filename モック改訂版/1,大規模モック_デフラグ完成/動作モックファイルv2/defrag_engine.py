"""
defrag_engine.py
Pure function: 意味空間の自然なクラスタを検知し、重力（Trust）の重心を計算。
そこから抽象度の高い新しい「Origin」を再生成する。
※ v2.0: 監査ログ出力を強化。MiniCubeの恩恵を受けたハイブリッドスコアでクラスタを形成。
"""

import math
from typing import List
from common_types import SemanticCube
from trust_evaluator import calculate_hybrid_trust
from cube_factory import make_cube
from partial_engine import call_llm

def defrag_cluster(seed_cube: SemanticCube, all_cubes: List[SemanticCube], current_turn: int, model_name: str) -> SemanticCube:
    """
    【Copilot 指示書3: 二段階クラスタリング アルゴリズム 反映】
    """
    print(f"\n🌀 [DefragEngine] 起動: Seed Cube [{seed_cube.cube_id[:4]}] ({seed_cube.summary[:15]}...)")
    
    # --- Step 2: seed の近傍クラスタを検知（局所クラスタリング） ---
    # MiniCubeの類似度を加味した Hybrid Trust で近傍を抽出する
    scored = [(c, calculate_hybrid_trust(c, seed_cube, current_turn)) for c in all_cubes]
    scored.sort(key=lambda x: x[1], reverse=True)
    cluster = [c for c, s in scored[:5]] # Top5 を抽出
    
    print(f"  ├ 局所クラスタ抽出: {len(cluster)} キューブ")
    
    # --- Step 3: クラスタのベクトル重心を計算 ---
    weights = []
    vectors = []
    for c in cluster:
        # 重み = log(1 + ref_count) * trust_score
        trust_w = calculate_hybrid_trust(c, seed_cube, current_turn)
        w = math.log1p(c.trust.ref_count) * trust_w
        weights.append(w)
        vectors.append(c.vector)
        
    weights_sum = sum(weights) + 1e-9
    center_vec = sum(w * v for w, v in zip(weights, vectors)) / weights_sum
    
    print(f"  ├ 重心ベクトル算出完了")

    # --- 監査ログ出力: LLMに渡すクラスタ内の要約 ---
    print(f"\n  👀 [監査ログ] デフラグ入力（抽出された近傍キューブ群）:")
    for i, c in enumerate(cluster):
        print(f"      [{i+1}] (Hybrid Trust: {scored[i][1]:.3f}) {c.summary[:40]}...")

    # --- Step 5: クラスタの代表文を LLM に渡し、抽象化させる ---
    texts = "\n".join([f"- {c.summary}" for c in cluster])
    prompt = f"""以下は同じ方向性を持つ意見のクラスタです。
これらの共通点を抽象化し、新しい「Origin」としてふさわしい1文を要約して生成してください。
文章が短くなりすぎないように注意してください。

- 1文で、他の文の土台になれるように構造化する

【意見クラスタ】
{texts}"""
    
    print(f"\n  ├ LLM による抽象化処理中...")
    new_origin_text = call_llm(prompt, model_name)
    
    # --- 監査ログ出力: LLMが生成した新しいOrigin ---
    print(f"\n  👀 [監査ログ] デフラグ出力（LLMが生成した新Origin）:")
    print(f"      {new_origin_text}\n")

    # --- Step 6: 新しい Origin の生成 ---
    # ※生成時に内部で extract_keyphrases が走り、MiniCube も自動生成される
    new_origin = make_cube(new_origin_text, "origin", turn=current_turn, ref_count=50, llm_model_name=model_name)
    
    return new_origin
