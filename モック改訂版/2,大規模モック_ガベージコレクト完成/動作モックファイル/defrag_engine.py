"""
defrag_engine.py
Pure function: 新8ステップのデフラグアルゴリズム。
※ v2.2: 「デフラグ強度（積極的・標準・厳密）」の概念を導入し、クラスタ抽出の閾値を可変にした。
"""

import math
from typing import List, Optional
from common_types import SemanticCube
from trust_evaluator import calculate_hybrid_trust, update_alpha
from cube_factory import make_cube
from partial_engine import call_llm

def select_defrag_seed(all_cubes: List[SemanticCube]) -> Optional[SemanticCube]:
    """ref_count（αの源泉）が最も大きいhistoryキューブをシードに選ぶ"""
    history_cubes = [c for c in all_cubes if c.role == "history"]
    if not history_cubes:
        return None
    history_cubes.sort(key=lambda c: c.trust.ref_count, reverse=True)
    return history_cubes[0]

def defrag_cluster(all_cubes: List[SemanticCube], current_turn: int, model_name: str, strictness: str = "standard") -> Optional[SemanticCube]:
    """
    【v2.2: デフラグ強度（strictness）の導入】
    "aggressive" (積極的) = 0.3
    "standard"   (標準)   = 0.5
    "strict"     (厳密)   = 0.75
    """
    # 閾値のマッピング
    threshold_map = {
        "aggressive": 0.3,
        "standard": 0.5,
        "strict": 0.75
    }
    threshold = threshold_map.get(strictness, 0.45)

    seed_cube = select_defrag_seed(all_cubes)
    if not seed_cube:
        print("🌀 [DefragEngine] デフラグ可能なHistoryキューブが存在しません。")
        return None
        
    print(f"\n🌀 [DefragEngine] 起動: Seed Cube [{seed_cube.cube_id[:4]}] (ref_count: {seed_cube.trust.ref_count:.2f})")
    print(f"  ├ デフラグ強度: [{strictness.upper()}] (Threshold: {threshold})")
    
    # --- Step 2: 局所クラスタ抽出（閾値適用） ---
    scored = [(c, calculate_hybrid_trust(c, seed_cube, current_turn)) for c in all_cubes]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # ✨ お前のアイデア: ここで可変閾値を適用する ✨
    local_cluster = [c for c, s in scored[:10] if s >= threshold] 
    
    print(f"  ├ Step 2: 局所クラスタ抽出 ({len(local_cluster)} 個)")

    # Seed自身しか残らなかった場合（周りに仲間がいない）はデフラグスキップ
    if len(local_cluster) < 2:
         print("  └ 閾値を超える類似キューブがないため、デフラグをスキップします。")
         return None

    # --- Step 3: クラスタの重心位置を計算 ---
    weights = []
    vectors = []
    for c in local_cluster:
        trust_w = calculate_hybrid_trust(c, seed_cube, current_turn)
        w = math.log1p(c.trust.ref_count) * trust_w
        weights.append(w)
        vectors.append(c.vector)
        
    weights_sum = sum(weights) + 1e-9
    center_vec = sum(w * v for w, v in zip(weights, vectors)) / weights_sum
    print(f"  ├ Step 3: 重心ベクトル算出完了")

    # --- Step 4 & 5: 重心から近い順＆ref_countの大きい順に再ピックアップしTop10で打ち止め ---
    from trust_evaluator import cosine_similarity
    
    re_scored = []
    for c in local_cluster:
        sim_to_center = cosine_similarity(center_vec, c.vector)
        final_score = sim_to_center * (1.0 + math.log1p(c.trust.ref_count))
        re_scored.append((c, final_score))
        
    re_scored.sort(key=lambda x: x[1], reverse=True)
    final_cluster = [c for c, score in re_scored[:10]]
    
    print(f"  ├ Step 4-5: 重心からの再選抜完了 ({len(final_cluster)} 個)")
    print(f"\n  👀 [監査ログ] デフラグ入力（再選抜されたキューブ群）:")
    for i, c in enumerate(final_cluster):
        print(f"      [{i+1}] (ref_count: {c.trust.ref_count:.2f}) {c.summary[:40]}...")

    # --- Step 6: 意訳（LLMによる抽象化） ---
    texts = "\n".join([f"- {c.summary}" for c in final_cluster])
    prompt = f"""以下は同じ方向性を持つ意見のクラスタです。
これらの共通点を抽象化し、新しい「Origin」としてふさわしい1文を生成してください。

- 具体的すぎる要素は排除する
- 価値観・方向性の核を抽象化する
- 1文で、他の文の土台になれるようにする

【意見クラスタ】
{texts}"""
    
    print(f"\n  ├ Step 6: LLM による意訳（抽象化）処理中...")
    new_origin_text = call_llm(prompt, model_name)
    print(f"\n  👀 [監査ログ] デフラグ出力（LLMが生成した新Origin）:")
    print(f"      {new_origin_text}\n")
    
    # --- Step 7: 新Origin生成 ---
    new_origin = make_cube(new_origin_text, "origin", turn=current_turn, ref_count=1, llm_model_name=model_name)
    
    for c in final_cluster:
        c.trust.replacement_closeness = 0.9 
        update_alpha(c)

    return new_origin
