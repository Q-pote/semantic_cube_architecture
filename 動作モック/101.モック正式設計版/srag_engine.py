# srag_engine.py (v4.2 / method と ruminate を分離)

import numpy as np
from typing import List, Dict, Tuple, Literal, Any, Set
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN

# 外部のデータ型と DataHub をインポート
# common_types.py に SearchMethod = Literal["standard", "exploratory"] を定義することを推奨
SearchMethod = Literal["standard", "exploratory"]
from common_types import SemanticCube, MiniCube, MatchInfo, SearchResult
from data_hub import DataHub

class SragEngine:
    def __init__(self, 
                 srag_threshold: float = 0.7,               # SRAGの最終スコア(採用意見足きり)の閾値
                 keyword_threshold: float = 0.7,            # ミニキューブ間の類似度スコアの閾値
                 top_k: int = 5,                            # 返却する検索結果の最大件数
                 ruminate_threshold: float = 0.6,           # 反芻ループでの「元のクエリとの類似度」の閾値 (これを下回ると反芻を打ち切る)
                 ruminate_count: int = 5                    # 反芻ループの最大回数
                 ):
        
        self.srag_threshold = srag_threshold
        self.keyword_threshold = keyword_threshold
        self.top_k = top_k
        self.ruminate_threshold = ruminate_threshold
        self.ruminate_count = ruminate_count

        print(f"✅ SragEngine (v4.2 RSRAG) initialized. method/ruminate分離. SRAG_threshold={srag_threshold}, ruminate_threshold={ruminate_threshold}")

    # --- ユーティリティ関数 (変更なし) ---
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1 is None or v2 is None: return 0.0
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

    def _get_neighbor_grids(self, grid: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        gx, gy, gz = grid
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = gx + dx, gy + dy, gz + dz
                    if 0 <= nx <= 7 and 0 <= ny <= 7 and 0 <= nz <= 7:
                        neighbors.append((nx, ny, nz))
        return neighbors

    def _unwrap_candidate(self, item: Any) -> Tuple[SemanticCube, float, List[MatchInfo]]:
        if isinstance(item, tuple):
            return item[0], item[1], item[2]
        return item, 0.0, []

    # --- 検索起点候補の抽出 (変更なし) ---
    def _select_starting_points(self, query_cube: SemanticCube, all_cubes: List[SemanticCube]) -> List[Tuple[SemanticCube, float, List[MatchInfo]]]:
        if not query_cube.mini_cubes: return []
        candidates_with_matches: Dict[str, List[MatchInfo]] = {}
        grid_to_minicubes: Dict[Tuple[int, int, int], List[MiniCube]] = {}
        for db_cube in all_cubes:
            if db_cube.cube_id == query_cube.cube_id: continue
            for db_mc in db_cube.mini_cubes:
                if not hasattr(db_mc, 'parent_cube_id') or not db_mc.parent_cube_id:
                    db_mc.parent_cube_id = db_cube.cube_id
                if db_mc.int3_grid not in grid_to_minicubes:
                    grid_to_minicubes[db_mc.int3_grid] = []
                grid_to_minicubes[db_mc.int3_grid].append(db_mc)
        for q_mc in query_cube.mini_cubes:
            neighbor_grids = self._get_neighbor_grids(q_mc.int3_grid)
            for grid in neighbor_grids:
                for db_mc in grid_to_minicubes.get(grid, []):
                    sim_score = self._cosine_similarity(q_mc.embedding, db_mc.embedding)
                    if sim_score > self.keyword_threshold:
                        match_info = MatchInfo(query_phrase=q_mc.phrase, target_phrase=db_mc.phrase, score=sim_score, parent_cube_id=db_mc.parent_cube_id)
                        parent_id = db_mc.parent_cube_id
                        if parent_id not in candidates_with_matches: candidates_with_matches[parent_id] = []
                        candidates_with_matches[parent_id].append(match_info)
        scored_candidates = []
        for parent_id, matches in candidates_with_matches.items():
            avg_score = sum(m.score for m in matches) / len(matches)
            bonus = min(0.1, (len(matches) - 1) * 0.02)
            scored_candidates.append((parent_id, min(1.0, avg_score + bonus), matches))
        sorted_points = sorted(scored_candidates, key=lambda item: item[1], reverse=True)
        id_to_cube = {c.cube_id: c for c in all_cubes}
        return [(id_to_cube[cube_id], score, matches) for cube_id, score, matches in sorted_points[:50] if cube_id in id_to_cube]

    # --- 類似度 計算関数 (変更なし) ---
    def _calculate_meta_similarity(self, query_cube: SemanticCube, target_cube: SemanticCube) -> float:
        meta_roles = {"genre", "topic", "problem/theme"}
        query_meta = [mc for mc in query_cube.mini_cubes if mc.role in meta_roles and mc.embedding is not None]
        target_meta = [mc for mc in target_cube.mini_cubes if mc.role in meta_roles and mc.embedding is not None]
        if not query_meta or not target_meta: return 0.0
        scores = [max((self._cosine_similarity(q_mc.embedding, t_mc.embedding) for t_mc in target_meta), default=0.0) for q_mc in query_meta]
        return sum(scores) / len(scores) if scores else 0.0

    def _calculate_detail_similarity(self, query_cube: SemanticCube, target_cube: SemanticCube) -> float:
        detail_roles = {"subject", "predicate", "key_phrase"}
        query_detail = [mc for mc in query_cube.mini_cubes if mc.role in detail_roles and mc.embedding is not None]
        target_detail = [mc for mc in target_cube.mini_cubes if mc.role in detail_roles and mc.embedding is not None]
        if not query_detail or not target_detail: return 0.0
        scores = [max((self._cosine_similarity(q_mc.embedding, t_mc.embedding) for t_mc in target_detail), default=0.0) for q_mc in query_detail]
        return sum(scores) / len(scores) if scores else 0.0

    # --- 単発探索フロー ---
    # --- ✨ 標準フロー (Standard Flow) ---
    def _run_standard_flow(self, query_cube: SemanticCube, candidates: List) -> List[SearchResult]:
        print("    -> Executing Standard Flow...")
        results = []
        for item in candidates:
            cube, _, matches = self._unwrap_candidate(item)
            half_score_1 = self._calculate_meta_similarity(query_cube, cube)
            half_score_2 = self._calculate_detail_similarity(query_cube, cube)
            context_sim = self._cosine_similarity(query_cube.embedding_vector, cube.embedding_vector)
            mini_score = (half_score_1 * 0.6) + (half_score_2 * 0.4) if half_score_1 > 0.0 else half_score_2
            final_score = (mini_score * 0.7) + (context_sim * 0.3) if half_score_1 > 0.0 else (half_score_2 * 0.6) + (context_sim * 0.4)
            if final_score >= self.srag_threshold:
                results.append(SearchResult(cube=cube, final_score=final_score, reason="Standard Flow", matched_minicubes=matches))
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:self.top_k]

    # --- ✨ 探索フロー (Exploratory Flow) ---
    def _run_exploratory_flow(self, query_cube: SemanticCube, candidates: List) -> List[SearchResult]:
        print("    -> 🌐 Executing Exploratory Flow...")
        if len(candidates) < 2: return self._run_standard_flow(query_cube, candidates)
        unwrapped = [self._unwrap_candidate(c) for c in candidates]
        point_cubes = [c for c, _, _ in unwrapped]
        cube_to_matches = {c.cube_id: matches for c, _, matches in unwrapped}
        mesh_points = np.array([c.grid_mesh for c in point_cubes])
        clustering = DBSCAN(eps=200, min_samples=2).fit(mesh_points)
        labels = clustering.labels_
        clusters: Dict[int, List[SemanticCube]] = {label: [] for label in set(labels) if label != -1}
        for i, label in enumerate(labels):
            if label != -1: clusters[label].append(point_cubes[i])
        if not clusters: return self._run_standard_flow(query_cube, candidates)
        print(f"    - Found {len(clusters)} valid clusters.")
        final_results = []
        for label, cluster_cubes in clusters.items():
            centroid_vector = np.mean([c.embedding_vector for c in cluster_cubes], axis=0)
            cluster_scored = []
            for cube in cluster_cubes:
                internal_score = self._cosine_similarity(cube.embedding_vector, centroid_vector)
                half_score_1 = self._calculate_meta_similarity(query_cube, cube)
                half_score_2 = self._calculate_detail_similarity(query_cube, cube)
                context_score = self._cosine_similarity(query_cube.embedding_vector, cube.embedding_vector)
                mini_score = (half_score_1 * 0.6) + (half_score_2 * 0.4) if half_score_1 > 0.0 else half_score_2
                cube_final_score = (internal_score * 0.15) + (mini_score * 0.55) + (context_score * 0.30)
                cluster_scored.append((cube, cube_final_score))
            cluster_scored.sort(key=lambda x: x[1], reverse=True)
            top_cube, top_score = cluster_scored[0]
            if top_score >= self.srag_threshold:
                final_results.append(SearchResult(cube=top_cube, final_score=top_score, reason=f"Exploratory Cluster-{label}", matched_minicubes=cube_to_matches.get(top_cube.cube_id, [])))
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        return final_results[:self.top_k]


    # --- ✨ 反芻フロー (RSRAG) ---
    def _run_ruminate_flow(self, origin_query_cube: SemanticCube, hub: DataHub, method: SearchMethod) -> List[SearchResult]:
        """【RSRAGコア】指定されたmethodで連想反芻ループを実行する"""
        print(f"    🌀 Executing RSRAG Flow with '{method}' method...")
        current_query_cube = origin_query_cube
        accumulated_results: Dict[str, SearchResult] = {}
        visited_cube_ids: Set[str] = {origin_query_cube.cube_id}

        for loop_idx in range(self.ruminate_count):
            print(f"\n      🔁 [Rumination Loop {loop_idx + 1}/{self.ruminate_count}] Re-querying with '{method}'...")
            all_cubes = hub.cube_get_all()
            starting_points = self._select_starting_points(current_query_cube, all_cubes)
            if not starting_points:
                print("        - No new starting points. Breaking rumination.")
                break
            
            # ✨ methodに応じて実行フローを切り替え
            pass_results = self._run_exploratory_flow(current_query_cube, starting_points) if method == "exploratory" else self._run_standard_flow(current_query_cube, starting_points)
            
            new_hits = [r for r in pass_results if r.cube.cube_id not in visited_cube_ids]
            if not new_hits:
                print("        - No new unvisited cubes. Rumination converged.")
                break

            loop_promoted_cubes = []
            for res in new_hits:
                origin_sim = self._cosine_similarity(origin_query_cube.embedding_vector, res.cube.embedding_vector)
                current_sim = self._cosine_similarity(current_query_cube.embedding_vector, res.cube.embedding_vector)

                # 将来的にはここ、ブレンドの重みづけを係数化した方がいいかも。
                ruminate_score = (0.5 * current_sim) + (0.3 * origin_sim) + (0.2 * res.final_score)
                
                res.final_score = ruminate_score
                res.reason = f"RSRAG-{method} L{loop_idx+1} (OriginSim: {origin_sim:.3f})"
                visited_cube_ids.add(res.cube.cube_id)
                accumulated_results[res.cube.cube_id] = res
                loop_promoted_cubes.append((res, origin_sim))

            loop_promoted_cubes.sort(key=lambda x: x[0].final_score, reverse=True)
            top_hit, top_origin_sim = loop_promoted_cubes[0]

            if top_origin_sim < self.ruminate_threshold:
                print(f"        🛑 Safety Brake: Origin similarity ({top_origin_sim:.3f}) below threshold. Terminating.")
                break
            current_query_cube = top_hit.cube

        final_sorted = sorted(accumulated_results.values(), key=lambda x: x.final_score, reverse=True)
        print(f"\n    ✅ RSRAG Complete. Total unique memories: {len(final_sorted)}")
        return final_sorted[:self.top_k]


    # --- ✨ メインエントリーポイント ---
    def search(self, query_cube: SemanticCube, hub: DataHub, method: SearchMethod = "standard", ruminate: bool = False) -> List[SearchResult]:
        """
        SRAGのメインエントリーポイント
        
        Args:
            query_cube: 検索クエリとなるSemanticCube
            hub: データハブ
            method (SearchMethod): 'standard'または'exploratory'の探索手法
            ruminate (bool): 反芻ループを実行するかどうか
        """
        print(f"\n🛰️  SRAG Engine: Starting search (method='{method}', ruminate={ruminate}) for '{query_cube.summary[:20]}...'")
        
        if ruminate:
            # 反芻モードの場合は、指定されたmethodで反芻フローを実行
            return self._run_ruminate_flow(query_cube, hub, method)
        else:
            # 単発モードの場合は、起点を見つけてから各フローを実行
            all_cubes = hub.cube_get_all()
            starting_points = self._select_starting_points(query_cube, all_cubes)
            if not starting_points:
                print("  - No starting points found. Search terminated.")
                return []
            print(f"  - Found {len(starting_points)} candidate cubes from neighbors.")

            if method == "exploratory":
                return self._run_exploratory_flow(query_cube, starting_points)
            else: # standard
                return self._run_standard_flow(query_cube, starting_points)

