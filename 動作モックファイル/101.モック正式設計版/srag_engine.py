# srag_engine.py (v2.5 完全版)

import numpy as np
from typing import List, Dict, Tuple, Literal
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN

# 外部のデータ型と、DataHubをインポート
from common_types import SemanticCube, MiniCube, MatchInfo, SearchResult, SearchMode
from data_hub import DataHub

# --- 内部データ構造 ---
# common_typeに定義を移動する。

# --- メインエンジン ---
class SragEngine:
    def __init__(self, 
                 srag_threshold: float = 0.75, 
                 keyword_threshold: float = 0.75, 
                 top_k: int = 5,
                 # 3つのブレンド係数を追加
                 genre_blending: float = 0.4,
                 initial_blending: float = 0.3,
                 similarity_blending: float = 0.3
                 ):
        
        self.srag_threshold = srag_threshold
        self.keyword_threshold = keyword_threshold
        self.top_k = top_k
        self.genre_blending = genre_blending
        self.initial_blending = initial_blending
        self.similarity_blending = similarity_blending

        print(f"✅ SragEngine (v3.0) initialized. SRAG_threshold={srag_threshold}, keyword_threshold={keyword_threshold}, TOP-K={top_k}")
        print(f"    - Blending weights: Genre={genre_blending}, Keyword={initial_blending}, Context={similarity_blending}")

    # --- ユーティリティ関数 ---
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1 is None or v2 is None: return 0.0
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _get_neighbor_grids(self, grid: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """指定されたint3_gridの、周囲27マス（隣接グリッド）の座標を返す"""
        gx, gy, gz = grid
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = gx + dx, gy + dy, gz + dz
                    if 0 <= nx <= 7 and 0 <= ny <= 7 and 0 <= nz <= 7:
                        neighbors.append((nx, ny, nz))
        return neighbors

    # --- 検索フロー ---
    def _select_starting_points(self, query_cube: SemanticCube, all_cubes: List[SemanticCube]) -> List[Tuple[SemanticCube, float, List[MatchInfo]]]:
        if not query_cube.mini_cubes: return []
        
        candidates_with_matches: Dict[str, List[MatchInfo]] = {}

        # 1. DB側のMiniCubeをグリッドでインデックス化
        grid_to_minicubes: Dict[Tuple[int, int, int], List[MiniCube]] = {}
        for db_cube in all_cubes:
            if db_cube.cube_id == query_cube.cube_id: continue
            for db_mc in db_cube.mini_cubes:
                if not hasattr(db_mc, 'parent_cube_id') or not db_mc.parent_cube_id:
                    db_mc.parent_cube_id = db_cube.cube_id 
                if db_mc.int3_grid not in grid_to_minicubes:
                    grid_to_minicubes[db_mc.int3_grid] = []
                grid_to_minicubes[db_mc.int3_grid].append(db_mc)

        # 2. 探索とマッチング
        for q_mc in query_cube.mini_cubes:
            neighbor_grids = self._get_neighbor_grids(q_mc.int3_grid)
            for grid in neighbor_grids:
                for db_mc in grid_to_minicubes.get(grid, []):
                    sim_score = self._cosine_similarity(q_mc.embedding, db_mc.embedding)
                    
                    if sim_score > self.keyword_threshold:
                        match_info = MatchInfo(
                            query_phrase=q_mc.phrase,
                            target_phrase=db_mc.phrase,
                            score=sim_score,
                            parent_cube_id=db_mc.parent_cube_id
                        )
                        parent_id = db_mc.parent_cube_id
                        if parent_id not in candidates_with_matches:
                            candidates_with_matches[parent_id] = []
                        candidates_with_matches[parent_id].append(match_info)

        # 3. 複数マッチを考慮したスコア算出（Copilotの助言を採用！）
        scored_candidates = []
        for parent_id, matches in candidates_with_matches.items():
            # マッチした全スコアの平均に、マッチ数に応じた微小ボーナス(上限あり)を付与して0.0~1.0に収める
            avg_score = sum(m.score for m in matches) / len(matches)
            bonus = min(0.1, (len(matches) - 1) * 0.02) 
            final_initial_score = min(1.0, avg_score + bonus)
            scored_candidates.append((parent_id, final_initial_score, matches))

        # 4. 【修正】IDではなく、算出した final_initial_score (item[1]) でソート！！
        sorted_points = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        
        id_to_cube = {c.cube_id: c for c in all_cubes}
        return [(id_to_cube[cube_id], score, matches) for cube_id, score, matches in sorted_points[:50]]

    def _calculate_genre_similarity(self, query_cube: SemanticCube, target_cube: SemanticCube) -> float:
        """
        【新機能】2つのキューブ間で、role='genre'のMiniCube同士の類似度を計算する。
        """
        query_genres = [mc for mc in query_cube.mini_cubes if mc.role == 'genre']
        target_genres = [mc for mc in target_cube.mini_cubes if mc.role == 'genre']

        # どちらか一方でもジャンル（文化圏アンカー）を持たない場合、一致度は0.0とする
        if not query_genres or not target_genres:
            return 0.0

        max_sim = 0.0
        for q_genre in query_genres:
            for t_genre in target_genres:
                sim = self._cosine_similarity(q_genre.embedding, t_genre.embedding)
                if sim > max_sim:
                    max_sim = sim
        
        return max_sim

    def _run_standard_flow(self, query_cube: SemanticCube, candidates: List[Tuple[SemanticCube, float, List[MatchInfo]]]) -> List[SearchResult]:
        print("    -> Executing Standard Flow...")
        results = []
        for cube, initial_score, matches in candidates:
            similarity_score = self._cosine_similarity(query_cube.embedding_vector, cube.embedding_vector)
            if similarity_score < self.srag_threshold: continue

            genre_score = self._calculate_genre_similarity(query_cube, cube)

            final_score = (genre_score * self.genre_blending) + \
                          (initial_score * self.initial_blending) + \
                          (similarity_score * self.similarity_blending)
            
            # ✨ Copilotリクエストの超詳細ログ
            print(f"      [Debug] Cube: {cube.summary[:20]:<20} | Genre:{genre_score:.3f} | Keyword:{initial_score:.3f} | Context:{similarity_score:.3f} | Final:{final_score:.3f}")
            
            results.append(SearchResult(
                cube=cube, final_score=final_score, 
                reason=f"Standard Flow", matched_minicubes=matches
            ))
            
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:self.top_k]

    def _run_exploratory_flow(self, query_cube: SemanticCube, candidates: List[Tuple[SemanticCube, float, List[MatchInfo]]]) -> List[SearchResult]:
        print("    -> Executing Exploratory Flow...")
        if len(candidates) < 2: return self._run_standard_flow(query_cube, candidates)

        point_cubes = [c for c, _, _ in candidates]
        # initial_score（Keywordスコア）を後で引けるように辞書化しておく
        cube_to_initial_score = {c.cube_id: score for c, score, _ in candidates}
        cube_to_matches = {c.cube_id: matches for c, _, matches in candidates}

        mesh_points = np.array([c.grid_mesh for c in point_cubes])
        clustering = DBSCAN(eps=200, min_samples=2).fit(mesh_points)
        labels = clustering.labels_

        clusters: Dict[int, List[SemanticCube]] = {}
        for i, label in enumerate(labels):
            if label != -1:
                if label not in clusters: clusters[label] = []
                clusters[label].append(point_cubes[i])
        
        if not clusters: return self._run_standard_flow(query_cube, candidates)
        print(f"    - Found {len(clusters)} valid clusters.")

        final_results = []
        for label, cluster_cubes in clusters.items():
            cluster_vectors = [c.embedding_vector for c in cluster_cubes]
            centroid_vector = np.mean(cluster_vectors, axis=0)

            cluster_scored = []
            print(f"\n      --- Evaluating Cluster {label} ---")
            for cube in cluster_cubes:
                internal_score = self._cosine_similarity(cube.embedding_vector, centroid_vector)
                genre_score = self._calculate_genre_similarity(query_cube, cube)
                initial_score = cube_to_initial_score.get(cube.cube_id, 0.0)
                context_score = self._cosine_similarity(query_cube.embedding_vector, cube.embedding_vector)

                cube_final_score = (internal_score * 0.2) + \
                                   (genre_score * self.genre_blending) + \
                                   (initial_score * self.initial_blending) + \
                                   (context_score * self.similarity_blending)

                # ✨ Copilotリクエストの超詳細ログ
                print(f"        [Debug] Cube: {cube.summary[:20]:<20} | Internal:{internal_score:.3f} | Genre:{genre_score:.3f} | Keyword:{initial_score:.3f} | Context:{context_score:.3f} | Final:{cube_final_score:.3f}")

                cluster_scored.append((cube, cube_final_score))
            
            cluster_scored.sort(key=lambda x: x[1], reverse=True)
            top_cube_in_cluster, top_final_score = cluster_scored[0]
            matches = cube_to_matches.get(top_cube_in_cluster.cube_id, [])

            final_results.append(SearchResult(
                cube=top_cube_in_cluster, final_score=top_final_score, 
                reason=f"Exploratory Cluster-{label}", matched_minicubes=matches
            ))
            
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results[:self.top_k]
    

    def search(self, query_cube: SemanticCube, hub: DataHub, mode: SearchMode = "standard") -> List[SearchResult]:
        """SRAGのメインエントリーポイント"""
        print(f"\n🛰️  SRAG Engine: Starting '{mode}' search for '{query_cube.summary[:20]}...'")
        
        all_cubes = hub.cube_get_all()
        starting_points = self._select_starting_points(query_cube, all_cubes)
        
        if not starting_points:
            print("  - No starting points found. Search terminated.")
            return []
            
        print(f"  - Found {len(starting_points)} candidate cubes from neighbors.")

        # 探索モードに応じて、標準フロー or 意味探査フローを実行 
        if mode == "exploratory":
            return self._run_exploratory_flow(query_cube, starting_points)
        else:
            return self._run_standard_flow(query_cube, starting_points)
