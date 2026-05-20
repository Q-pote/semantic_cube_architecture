"""
kvhub.py
Stateful: 世界の記憶を保持。
※ v2.1 修正版: 検索時に確実に `calculate_hybrid_trust` (MiniCube語義検索) を使用し、代謝を発動する。
"""

from common_types import SemanticCube
from trust_evaluator import map_diff_to_7bucket, calculate_exact_hybrid_score, update_alpha, cosine_similarity
import numpy as np
from typing import List, Tuple

class KVHub:
    def __init__(self):
        self.cubes: List[SemanticCube] = []
        self.thread_mode: str = "smart"  # "smart" (代謝あり) or "protected" (代謝なし)
        # ✨ 追加: 検索時のMiniCube語義一致の厳しさ設定（デフォルト: 標準）
        self.mc_strictness = "standard" 

    def _get_mc_threshold(self) -> float:
        """設定値からMiniCubeの一致閾値を返す"""
        thresholds = {
            "fuzzy": 0.5,    # 曖昧（広く語義を拾う）
            "standard": 0.7, # 標準
            "strict": 0.85   # 厳密（完全に一致した語義しか拾わない）
        }
        return thresholds.get(self.mc_strictness, 0.7)

    def put(self, cube: SemanticCube):
        """新しいキューブを追加。初期α値を計算しておく。"""
        update_alpha(cube)
        self.cubes.append(cube)

    def get_all(self) -> List[SemanticCube]:
        """GC対象でないアクティブなキューブを取得"""
        return [c for c in self.cubes if not c.trust.gc_flag]


    def search_by_trust(self, q_cube, current_turn: int, k: int = 5):
        bucketed_candidates = []
        
        # 🚀【第一段階：粗探索（相対評価によるバケット判定）】
        rough_scores = []
        for c in self.get_all():
            if c.cube_id == q_cube.cube_id: continue
            
            # 生ベクトルによるコサイン類似度
            rough = cosine_similarity(q_cube.vector, c.vector)
            rough_scores.append((c, rough))
            
        if not rough_scores:
            return []
            
        # ✨ Copilotの神ロジック：候補群の中での最大スコアを基準点(0)とする
        max_score = max(r for _, r in rough_scores)
        
        for c, rough in rough_scores:
            # 最大スコアからの「相対的なズレ（差分）」を計算
            diff = rough - max_score
            bucket_id, gravity = map_diff_to_7bucket(diff)
            
            # 論文の設計方針: Bucket 0, 1, 5, 6 は引力=0 として除外
            if gravity == 0.0:
                continue 
                
            # 優先バケット(2, 3, 4)のみ次へ進む
            bucketed_candidates.append((c, bucket_id, gravity))

        mc_thresh = self._get_mc_threshold() # ✨ 追加: 設定値を取得

        cands = []
        for c, bucket_id, gravity in bucketed_candidates:
            # ✨ 修正: 取得した閾値を渡す
            exact_score = calculate_exact_hybrid_score(q_cube, c, mc_threshold=mc_thresh)
            
            final_score = exact_score * gravity
            cands.append((c, final_score, bucket_id))

        cands.sort(key=lambda x: (x[1], abs(3 - x[2])), reverse=True)
        top_k = [(c, s) for c, s, b in cands[:k]]
        
        # 代謝メカニズム（参照ボーナス）
        if self.thread_mode == "smart":
            for c, final_score, bucket_id in cands[:k]:
                # final_score は 0.0〜1.0 の純粋な引力なので、そのままボーナスに使える
                ref_bonus = final_score 
                c.trust.ref_count += (1.0 + ref_bonus)
                c.trust.last_used_at_turn = current_turn
                c.trust.avg_similarity = (c.trust.avg_similarity * 0.8) + (final_score * 0.2)
                update_alpha(c)
                
        return top_k


    def garbage_collection(self):
        """α値が低く gc_flag が立ったキューブを空間から消去（忘却）する"""
        if self.thread_mode == "protected":
            print("  🔒 [KVHub] Protected mode: Garbage Collection is disabled.")
            return
            
        before_count = len(self.cubes)
        self.cubes = [c for c in self.cubes if not c.trust.gc_flag]
        after_count = len(self.cubes)
        
        if before_count - after_count > 0:
            print(f"  🧹 [KVHub] Garbage Collection 発動: {before_count - after_count} 個のキューブ(α<0.5)を忘却しました。")
