"""
trust_evaluator.py
Pure functions: 2つの SemanticCube 間の空間距離・角度差、および
MiniCube間のCosine類似度からハイブリッドTrustスコアを計算する。
"""

import math
import numpy as np
from common_types import TrustStruct, Orientation, SemanticCube

# レイヤー間の距離マトリクス
LAYER_MAT = {
    ("origin", "origin"): 0.0,
    ("history", "history"): 0.0,
    ("origin", "history"): 0.3,
    ("history", "origin"): 0.3,
}

def angular_distance(ori1: Orientation, ori2: Orientation) -> float:
    """2つの方位（Orientation）間の角度距離を計算する (0.0〜1.0)"""
    dot = math.cos(ori1.elevation)*math.cos(ori2.elevation)*math.cos(ori1.azimuth-ori2.azimuth) + math.sin(ori1.elevation)*math.sin(ori2.elevation)
    dot = max(-1.0, min(1.0, dot))
    return min(1.0, (math.acos(dot) / math.pi))

def calculate_base_trust(target: TrustStruct, focus: TrustStruct, current_turn: int) -> float:
    """トーラス空間距離、レイヤー距離、角度差による基本Trustを計算する"""
    # トーラス空間（循環）を考慮したグリッド距離
    dx = min(abs(target.grid_index[0] - focus.grid_index[0]), 100 - abs(target.grid_index[0] - focus.grid_index[0]))
    dy = min(abs(target.grid_index[1] - focus.grid_index[1]), 100 - abs(target.grid_index[1] - focus.grid_index[1]))
    dz = min(abs(target.grid_index[2] - focus.grid_index[2]), 100 - abs(target.grid_index[2] - focus.grid_index[2]))
    d_grid = min(1.0, math.sqrt(dx*dx + dy*dy + dz*dz) / 100.0)
    
    d_layer = LAYER_MAT.get((target.role, focus.role), 0.5)
    d_angle = angular_distance(target.orientation, focus.orientation)
    
    space_dist = math.sqrt(0.4*(d_grid**2) + 0.3*(d_layer**2) + 0.3*(d_angle**2))
    semantic_trust = math.exp(-2.0 * space_dist) * target.gravity
    
    # 時間メタデータと参照回数による補正
    time_mod = max(0.8, min(1.2, (1.0 + math.log1p(target.ref_count)*0.5) * (1.0 / (1.0 + max(0, current_turn - target.last_used_at_turn)))))
    
    return min(1.0, semantic_trust * time_mod)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """ベクトル間のコサイン類似度を計算する"""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: 
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

def calculate_hybrid_trust(target: SemanticCube, q_cube: SemanticCube, current_turn: int) -> float:
    """
    ✨ v2.0: Base Trust に MiniCube の語義類似度（最大Cosine）を合成する ✨
    """
    base_trust = calculate_base_trust(target.trust, q_cube.trust, current_turn)
    
    if not q_cube.mini_cubes or not target.mini_cubes:
        return base_trust
        
    # MiniCube間の最大類似度を探索
    max_sim = 0.0
    for qa in q_cube.mini_cubes:
        for ta in target.mini_cubes:
            sim = cosine_similarity(qa.embedding, ta.embedding)
            # 抽出Confidenceによる重み付け（低い自信度のフレーズは影響を下げる）
            sim = sim * qa.confidence * ta.confidence
            if sim > max_sim:
                max_sim = sim
                
    # 合成 (α=0.6, β=0.4 の比率で合成)
    # ※MiniCubeによる「語義の一致」を強めに評価する
    hybrid_score = (0.6 * max_sim) + (0.4 * base_trust)
    return min(1.0, hybrid_score)
