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
    return min(1.0, (math.acos(dot) / math.pi))

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

# ✨ v2.1: αチャンネル（記憶の代謝）計算関数 ✨
def update_alpha(cube: SemanticCube) -> SemanticCube:
    """
    ref_count(使用頻度), avg_similarity(参照の強さ), replacement_closeness(代替可能性)
    からα値(透明度)を算出し、一定以下ならGCフラグを立てる。
    """
    t = cube.trust
    # αの基本計算: ref_count * 参照強度 * (1 - 代替可能性)
    # ※originは基準重力源として消えにくくする補正 (+1.0)
    base_alpha = t.ref_count * t.avg_similarity * (1.0 - t.replacement_closeness)
    
    if cube.role == "origin":
        t.alpha = base_alpha + 1.0
    else:
        t.alpha = base_alpha
        
    # α値が 0.5 未満になったhistoryキューブは忘却(GC)の対象とする
    if cube.role == "history" and t.alpha < 0.5:
        t.gc_flag = True
    else:
        t.gc_flag = False
        
    return cube
