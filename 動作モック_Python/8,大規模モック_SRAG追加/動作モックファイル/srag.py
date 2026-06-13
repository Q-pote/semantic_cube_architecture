# ==============================================================================
# 2. SRAG v2.5 検索エンジン本体
# ==============================================================================

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Literal
from sklearn.cluster import DBSCAN
from common_types import SemanticCube



@dataclass
class SearchResult:
    """SRAGの検索結果を格納するデータクラス"""
    cube: SemanticCube
    final_score: float
    reason: str
    matched_keywords: List[str] = field(default_factory=list)


# --- ユーティリティ関数 ---
def cosine_similarity(v1, v2):
    if v1 is None or v2 is None: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def manhattan_distance_int(v1, v2):
    """軽量なintベクトルの距離計算"""
    if v1 is None or v2 is None: return float('inf')
    return np.sum(np.abs(v1 - v2))

def srag_search(
    query_cube: SemanticCube,
    all_cubes: List[SemanticCube],
    mode: Literal["standard", "exploratory"] = "standard"
) -> List[SearchResult]:
    """
    SRAG v2.5 検索エンジン
    mode引数によって、標準フローと意味探査フローを切り替える。
    """
    print("\n" + "="*60)
    print(f"🚀 SRAG Engine Start. Mode: '{mode}'")
    print("="*60)

    # 手順1【共通】: 起点候補の選出
    starting_points = _common_step1_select_starting_points(query_cube, all_cubes)
    if not starting_points:
        print("  - Step 1 Result: No starting points found. Search terminated.")
        return []
    
    print(f"  - Step 1 Result: Found {len(starting_points)} starting point candidates.")

    # モードに応じてフローを分岐
    if mode == "standard":
        # 手順2【標準フロー】
        results = _standard_flow_get_direct_answer(query_cube, starting_points)
    elif mode == "exploratory":
        # 手順2【意味探査フロー】
        results = _exploratory_flow_build_context(query_cube, starting_points, all_cubes)
    else:
        raise ValueError("Invalid mode specified. Choose 'standard' or 'exploratory'.")

    print(f"✅ SRAG Engine Finish. Returning {len(results)} results.")
    return results

def _common_step1_select_starting_points(query_cube: SemanticCube, all_cubes: List[SemanticCube]) -> List[Tuple[SemanticCube, float]]:
    """手順1【共通】: MiniCubeをベースに起点候補を絞り込む"""
    # 1. ミニキューブのint3_gridでシード候補を選出
    seed_candidates = []
    for q_mc in query_cube.mini_cubes:
        for db_cube in all_cubes:
            if db_cube.cube_id == query_cube.cube_id: continue
            for db_mc in db_cube.mini_cubes:
                if db_mc.int3_grid == q_mc.int3_grid:
                    seed_candidates.append(db_cube)
                    break 
    
    # 2. ミニキューブのvec_int3で起点候補群をピックアップ
    starting_points_with_dist = []
    for q_mc in query_cube.mini_cubes:
        for db_cube in seed_candidates:
            for db_mc in db_cube.mini_cubes:
                dist = manhattan_distance_int(q_mc.vec_int3, db_mc.vec_int3)
                # 距離が近いものほどスコアが高くなるように変換
                score = 1 / (1 + dist)
                starting_points_with_dist.append((db_cube, score, db_mc.confidence))
    
    # 3&4. trust_scoreを考慮してソート
    # ここでは、(距離スコア * キーワードの信頼度) を簡易的なtrust_scoreとする
    final_candidates = {}
    for cube, dist_score, mc_confidence in starting_points_with_dist:
        trust_score = dist_score * mc_confidence
        if cube.cube_id not in final_candidates or trust_score > final_candidates[cube.cube_id][1]:
            final_candidates[cube.cube_id] = (cube, trust_score)

    sorted_points = sorted(final_candidates.values(), key=lambda x: x[1], reverse=True)
    return sorted_points[:50] # 上位50件を起点候補とする

def _standard_flow_get_direct_answer(query_cube: SemanticCube, starting_points: List[Tuple[SemanticCube, float]]) -> List[SearchResult]:
    scored_results = []
    # 💥【修正点】司令官の指摘通り、MiniCubeのベクトルで比較する
    for cube, initial_trust in starting_points:
        best_mc_score = 0
        for q_mc in query_cube.mini_cubes:
            for c_mc in cube.mini_cubes:
                score = cosine_similarity(q_mc.embedding, c_mc.embedding)
                if score > best_mc_score: best_mc_score = score
        scored_results.append(SearchResult(cube=cube, final_score=best_mc_score, reason="Standard MC Cosine"))
    scored_results.sort(key=lambda x: x.final_score, reverse=True)
    return scored_results[:5]

def _exploratory_flow_build_context(query_cube: SemanticCube, starting_points: List[Tuple[SemanticCube, float]], all_cubes: List[SemanticCube]) -> List[SearchResult]:
    """手順2【意味探査フロー】: クラスタで関連構造を「組む」"""
    print("  - Executing Exploratory Flow...")
    
    # 1. 起点候補のgrid_meshを基にクラスタリング
    # NOTE: ここではDBSCANを使った簡易的な実装
    point_cubes = [c for c, _ in starting_points]
    if not point_cubes: return []
    
    mesh_points = np.array([c.grid_mesh for c in point_cubes])
    # eps: クラスタと見なす最大距離, min_samples: クラスタを形成する最小点数
    clustering = DBSCAN(eps=50, min_samples=3).fit(mesh_points)
    labels = clustering.labels_
    
    clusters: Dict[int, List[SemanticCube]] = {}
    for i, label in enumerate(labels):
        if label != -1: # -1はノイズ
            if label not in clusters: clusters[label] = []
            clusters[label].append(point_cubes[i])
            
    print(f"    - Found {len(clusters)} clusters.")
    if not clusters: return []

    # 2-7. 各クラスタを評価し、最終候補を生成
    final_results = []
    for label, cluster_cubes in clusters.items():
        # 2. 重心計算（ここでは平均grid_meshとする）
        centroid = np.mean([c.grid_mesh for c in cluster_cubes], axis=0)
        
        # 3-5. クラスタ内の評価とソート
        cluster_scored = []
        for cube in cluster_cubes:
            # 簡易的なトラストスコア（クエリとのint3距離）
            trust = 1 / (1 + manhattan_distance_int(query_cube.vector_int3, cube.vector_int3))
            cluster_scored.append((cube, trust))
        
        cluster_scored.sort(key=lambda x: x[1], reverse=True)
        
        # 6. 最終チェックとしてcosine類似度を計算
        top_cube_in_cluster, _ = cluster_scored[0]
        final_score = cosine_similarity(query_cube.vector, top_cube_in_cluster.vector)
        
        # 8-10. 回答候補として整形
        final_results.append(SearchResult(
            cube=top_cube_in_cluster,
            final_score=final_score,
            reason=f"Exploratory Cluster-{label} (size:{len(cluster_cubes)})",
            matched_keywords=[mc.phrase for mc in top_cube_in_cluster.mini_cubes]
        ))
        
    # 7. トラストスコアの高い順に最終ソート
    final_results.sort(key=lambda x: x.final_score, reverse=True)
    return final_results