"""
trust_evaluator.py
Pure functions: SemanticCube間のHybrid Trustスコアを計算する。
※ v2.1: αチャンネル（記憶の透明度）の計算ロジックを追加。
"""

import math
import numpy as np
from common_types import TrustStruct, Orientation, SemanticCube

LAYER_MAT = {
    ("origin", "origin"): 0.0, ("history", "history"): 0.0,
    ("origin", "history"): 0.3, ("history", "origin"): 0.3,
}

def angular_distance(ori1: Orientation, ori2: Orientation) -> float:
    dot = math.cos(ori1.elevation)*math.cos(ori2.elevation)*math.cos(ori1.azimuth-ori2.azimuth) + math.sin(ori1.elevation)*math.sin(ori2.elevation)
    dot = max(-1.0, min(1.0, dot))
    return min(1.0, (math.acos(dot) / math.pi)) # 0.0（完全一致）〜 1.0（正反対）

def calculate_base_trust(target: TrustStruct, focus: TrustStruct, current_turn: int) -> float:
    dx = min(abs(target.grid_index[0] - focus.grid_index[0]), 100 - abs(target.grid_index[0] - focus.grid_index[0]))
    dy = min(abs(target.grid_index[1] - focus.grid_index[1]), 100 - abs(target.grid_index[1] - focus.grid_index[1]))
    dz = min(abs(target.grid_index[2] - focus.grid_index[2]), 100 - abs(target.grid_index[2] - focus.grid_index[2]))
    d_grid = min(1.0, math.sqrt(dx*dx + dy*dy + dz*dz) / 100.0)
    
    d_layer = LAYER_MAT.get((target.role, focus.role), 0.5)
    d_angle = angular_distance(target.orientation, focus.orientation)
    
    space_dist = math.sqrt(0.4*(d_grid**2) + 0.3*(d_layer**2) + 0.3*(d_angle**2))
    semantic_trust = math.exp(-2.0 * space_dist) * target.gravity
    return min(1.0, semantic_trust)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

# trust_evaluator.py 修正箇所 (既存の calculate_hybrid_trust をこれらに置き換え/追加)

def map_diff_to_7bucket(score_diff: float) -> tuple[int, float]:
    """
    クエリと候補の「スコア差分(Delta)」を7段階のバケット(0-6)に分類し、
    (バケット番号, 引力係数) を返す。
    ※ Bucket 3 (差分ほぼゼロ) を中心とした対称的な重力構造。
    """
    if score_diff < -0.6: return (0, 0.0) # 離れている（引力0） -> 非表示
    if score_diff < -0.3: return (1, 0.0) # やや低い（引力0） -> 非表示
    if score_diff < -0.1: return (2, 0.5) # 近いが少し低い（引力0.5）
        
    if score_diff <= 0.1: return (3, 1.0) # ほぼ等しい（引力1.0） -> 最優先
        
    if score_diff <= 0.3: return (4, 0.5) # 近いが少し高い（引力0.5）
    if score_diff <= 0.6: return (5, 0.0) # やや高い（引力0） -> 非表示
    
    return (6, 0.0)                       # 離れている（引力0） -> 非表示 (葛藤エンジンへ)

def calculate_hybrid_trust(target: SemanticCube, q_cube: SemanticCube, current_turn: int) -> float:
    """Base Trust に MiniCube の語義類似度（最大Cosine）を合成する"""
    base_trust = calculate_base_trust(target.trust, q_cube.trust, current_turn)
    
    if not q_cube.mini_cubes or not target.mini_cubes:
        return base_trust
        
    max_sim = 0.0
    for qa in q_cube.mini_cubes:
        for ta in target.mini_cubes:
            sim = cosine_similarity(qa.embedding, ta.embedding)
            sim = sim * qa.confidence * ta.confidence
            if sim > max_sim:
                max_sim = sim
                
    hybrid_score = (0.6 * max_sim) + (0.4 * base_trust)
    return min(1.0, hybrid_score)

def calculate_exact_hybrid_score(q_cube: SemanticCube, target_cube: SemanticCube, mc_threshold: float = 0.7) -> float:
    """
    【第二段階用】生ベクトルとMiniCubeを用いた重い精査（マクロ＋ミクロ）
    """
    base_sim = cosine_similarity(q_cube.vector, target_cube.vector)
    
    if not q_cube.mini_cubes or not target_cube.mini_cubes:
        return base_sim
        
    match_count = 0
    total_mc = max(1, len(q_cube.mini_cubes))
    for qa in q_cube.mini_cubes:
        for ta in target_cube.mini_cubes:
            # sim = cosine_similarity(qa.embedding, ta.embedding)
            # sim = sim * qa.confidence * ta.confidence
            # ✨ 修正: Confidenceの掛け算を外し、純粋な「単語のベクトル類似度」だけで判定する！
            sim = cosine_similarity(qa.embedding, ta.embedding)
            
            # ✨ 修正: ハードコード(0.8)を廃止し、引数で動的判定
            if sim > mc_threshold: 
                match_count += 1
                break
                
    mc_alignment = match_count / total_mc
    
    return (0.6 * mc_alignment) + (0.4 * base_sim)


# ✨ v2.1: αチャンネル（記憶の代謝）計算関数 ✨
# もし残すなら、TrustStruct の既存フィールドを使う形に書き換える
def update_alpha(cube: SemanticCube) -> SemanticCube:
    t = cube.trust
    if t.is_privileged:
        t.alpha_base = 1.0
        t.is_archived = False
        return cube
    # 例: 単純に ref_count と last_used_at_turn を使う簡易版
    base_alpha = min(1.0, t.ref_count * 0.1 + 0.5)  # 仮の式
    t.alpha_base = base_alpha
    t.is_archived = (t.alpha_base < 0.01)
    return cube

