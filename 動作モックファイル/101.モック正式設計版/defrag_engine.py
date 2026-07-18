# defrag_engine.py (v2.2 Copilotレビュー完全反映版)

import numpy as np
from typing import List, Dict, Tuple, Set
from sklearn.cluster import DBSCAN

from common_types import SemanticCube, CubeRole
from data_hub import DataHub
from llm_service import LLMService
from cube_engine import CubeEngine
from gc_engine import GCEngine

class DefragEngine:
    def __init__(self, hub: DataHub, llm_service: LLMService, cube_engine: CubeEngine,
                 defrag_threshold: float = 0.75,
                 cluster_threshold: int = 3,
                 fuge_cluster_threshold: float = 0.75,
                 newOrigin_threshold: float = 0.3,
                 turn_guard: int = 5):
        
        self.hub = hub
        self.llm_service = llm_service
        self.cube_engine = cube_engine
        
        self.defrag_threshold = defrag_threshold
        self.cluster_threshold = cluster_threshold
        self.fuge_cluster_threshold = fuge_cluster_threshold
        self.newOrigin_threshold = newOrigin_threshold
        self.turn_guard = turn_guard # Origin保護ターン
        print("✅ DefragEngine (v2.2) initialized.")

    # ... (_get_neighbor_grids, _cosine_similarity は変更なし) ...
    def _get_neighbor_grids(self, grid: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """int3_gridの、周囲27マス（隣接グリッド）の座標を返す"""
        gx, gy, gz = grid
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    nx, ny, nz = gx + dx, gy + dy, gz + dz
                    if 0 <= nx <= 7 and 0 <= ny <= 7 and 0 <= nz <= 7:
                        neighbors.append((nx, ny, nz))
        return neighbors
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1 is None or v2 is None: return 0.0
        # np.linalg.normが0になるのを防ぐ
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # --------------------------------

    def _calculate_cluster_centroid(self, cluster: List[SemanticCube]) -> np.ndarray:
        """【手順6】grid_meshを使いクラスタの重心を計算"""
        mesh_points = np.array([c.grid_mesh for c in cluster])
        return np.mean(mesh_points, axis=0)

    def _find_pseudo_query_cube(self, centroid: np.ndarray, cluster: List[SemanticCube]) -> SemanticCube:
        """【手順8】重心に最も近いキューブを「疑似クエリ」として選出"""
        min_dist = float('inf')
        pseudo_query_cube = None
        for cube in cluster:
            dist = np.linalg.norm(np.array(cube.grid_mesh) - centroid)
            if dist < min_dist:
                min_dist = dist
                pseudo_query_cube = cube
        return pseudo_query_cube

    def _refine_cluster_with_pseudo_query(self, pseudo_query: SemanticCube, cluster: List[SemanticCube]) -> List[SemanticCube]:
        """【手順11-12】疑似クエリとの類似度でクラスタを再評価・精製する"""
        refined_cluster = []
        for cube in cluster:
            # vec_int3ではなく、より高次元なembedding_vectorで比較
            similarity = self._cosine_similarity(pseudo_query.embedding_vector, cube.embedding_vector)
            if similarity > self.fuge_cluster_threshold: # (要調整の閾値)
                refined_cluster.append(cube)
        return refined_cluster

    def _check_origin_quality(self, origin_summary: str, cluster: List[SemanticCube]) -> bool:
        """【手順14】生成されたOriginが、元の情報の閾値以上を保持しているかチェック"""
        original_phrases = {mc.phrase for c in cluster for mc in c.mini_cubes}
        if not original_phrases: return True # 元のキーワードがなければ常に合格

        retained_count = sum(1 for p in original_phrases if p in origin_summary)
        retention_rate = retained_count / len(original_phrases)
        
        print(f"    - Origin Quality Check: Retention Rate = {retention_rate:.2f}")
        return retention_rate >= self.newOrigin_threshold

    def _create_origin_from_cluster(self, cluster: List[SemanticCube]) -> Tuple[str, SemanticCube]:
        """
        LLMにクラスタを要約させ、
        「要約テキスト」と「新しいOriginキューブ」のタプルを返す。
        品質チェックは、このメソッドを呼び出す側で行う。
        """
        # 1. LLMに渡すための、クラスタ情報のテキストを生成
        context_text = "\n".join([f"- 「{c.summary}」" for c in cluster])
        
        # 2. LLMに「抽象化・要約」を指示するプロンプト
        prompt = f"""複数の意見の断片的な情報から、その中心にある「本質的な意味（コア・コンセプト）」を抽出しながら要約してください。
抽象化の過程で、元の情報から、重要なニュアンスを示すキーワードを保持するように努めてください。
入力された情報にない、新たな価値観や哲学的な意味づけは、付加しないでください。
以下の【記憶の断片リスト】が解析対象のテキスト群となります。

【回答方法について】
回答については前後のやりとりを除外して、要約文のみを回答してください。

【記憶の断片リスト】
{context_text}
"""
        
        result = self.llm_service.call_api_generate(prompt)
        
        if result.http_code != 200:
            return None, None
        
        new_origin_summary = result.data
        
        # CubeEngineを呼び出して、新しいOriginキューブを生成
        origin_cube = self.cube_engine.make_cube(
            query=new_origin_summary,
            role="origin",
            source="DEFRAG",
            current_turn=self.hub.global_turn
        )
        return new_origin_summary, origin_cube

    def run_defrag_cycle(self):
        """デフラグのメインループ（Copilotの指摘を完全反映）"""
        print("\n" + "="*20 + " Defrag Engine Cycle Start " + "="*20)
        
        self.hub.defrag_turn = self.hub.global_turn

        while True:
            # 0. 処理対象のキューブをフィルタリング
            active_cubes = [
                c for c in self.hub.cube_get_all()
                # ✨【FIX】「defrag_turn」が、今回のサイクルより古いものだけを対象とする！
                if c.defrag_target and c.defrag_turn < self.hub.defrag_turn and
                   (c.role != "origin" or (self.hub.defrag_turn - c.create_turn) > self.turn_guard)
            ]
            
            if len(active_cubes) < self.cluster_threshold:
                print("  - Not enough target cubes. Cycle finished.")
                break

            # 2. 閾値以下のアルファを持つシード候補を探す
            seed_candidates = [c for c in active_cubes if c.get_alpha(self.hub.global_turn) < self.defrag_threshold]
            seed_candidates.sort(key=lambda c: c.get_alpha(self.hub.global_turn))

            if not seed_candidates:
                print("  - No more weak memory seeds. Cycle finished.")
                break

            origin_created_in_this_loop = False
            for seed_cube in seed_candidates:
                if seed_cube.defrag_turn >= self.hub.defrag_turn: continue

                # 3-5. 近傍探査とクラスタ認定
                # (DBSCANなど、より洗練されたクラスタリング手法をここに実装)
                
                # 設計者注釈：疑似クエリを発行するため、DBSCANで塊を選択するより高精度。

                # MVPでは、まず近傍をそのままクラスタ候補とする
                neighbor_grids = self._get_neighbor_grids(seed_cube.int3_grid)
                cluster_candidate = [c for c in active_cubes if c.int3_grid in neighbor_grids and not c.is_archived]
                
                if len(cluster_candidate) < self.cluster_threshold:
                    continue
                
                # 6. 重心計算 & 8. 疑似クエリ生成
                centroid = self._calculate_cluster_centroid(cluster_candidate)
                pseudo_query = self._find_pseudo_query_cube(centroid, cluster_candidate)
                
                if not pseudo_query: continue

                # 11-12. 疑似クエリでクラスタを精製
                final_cluster = self._refine_cluster_with_pseudo_query(pseudo_query, cluster_candidate)
                
                if len(final_cluster) < self.cluster_threshold:
                    continue
                
                if final_cluster:
                    # 13. Origin生成
                    new_origin_summary, new_origin_cube = self._create_origin_from_cluster(final_cluster)
                    
                    if new_origin_cube and self._check_origin_quality(new_origin_summary, final_cluster):
                        # 14. 品質チェックOK！
                        self.hub.cube_add(new_origin_cube)
                        for old_cube in final_cluster:
                            old_cube.is_archived = True
                        
                        origin_created_in_this_loop = True
                        print(f"  ✨ New Origin created. Restarting defrag cycle...")
                        break
                    else:
                        # ✨【FIX】品質NG or 生成失敗の場合
                        print(f"  - Origin rejected for cluster around seed {seed_cube.cube_id[:8]}. Marking as processed for this turn.")
                        # このターンの処理対象から外すために、defrag_turnを更新
                        for cube_in_failed_cluster in final_cluster:
                            cube_in_failed_cluster.defrag_turn = self.hub.defrag_turn

            if not origin_created_in_this_loop:
                break
        
        # 最終的なGCを呼び出す
        gc_eng = GCEngine(self.hub)
        gc_eng.run_gc_cycle()

