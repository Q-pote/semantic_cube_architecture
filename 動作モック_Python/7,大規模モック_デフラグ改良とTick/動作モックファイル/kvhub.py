"""
kvhub.py
Stateful: 世界の記憶を保持。
※ v3.5: global_turn と previous_5 の管理、イベント駆動強化の実装。
"""

from common_types import SemanticCube
from trust_evaluator import map_diff_to_7bucket, calculate_exact_hybrid_score, cosine_similarity
import numpy as np
from typing import List, Tuple

class KVHub:
    def __init__(self):
        self.cubes: List[SemanticCube] = []
        self.thread_mode: str = "smart"
        self.mc_strictness = "standard" 
        
        # ✨ v3.5 追加: 宇宙の時間と短期記憶のバッファ
        self.global_turn: int = 0
        self.previous_5: List[str] = []

    def advance_turn(self):
        """Tick加算（宇宙の時間を進める）"""
        self.global_turn += 1

    def add_to_previous(self, cube_id: str):
        """直近5ターンのIDリストを更新 (FIFO)"""
        self.previous_5.append(cube_id)
        if len(self.previous_5) > 5:
            self.previous_5.pop(0)

    def trigger_event(self, cube: SemanticCube, event_type: str):
        """イベント駆動でアルファ値（記憶の強度）を更新する"""
        if cube.trust.is_privileged: return
        
        current_alpha = cube.get_alpha(self.global_turn)
        if event_type == "similar":
            new_alpha = current_alpha + 0.1
        elif event_type == "srag_hit":
            new_alpha = current_alpha + 0.2
        elif event_type == "defrag_absorbed":
            new_alpha = current_alpha * 0.3

        cube.trust.alpha_base = max(0.0, min(1.0, new_alpha))

        # イベント発生により、時間を現在にリセット
        cube.trust.last_updated_turn = self.global_turn

    def put(self, cube: SemanticCube):
        """新しいキューブを追加し、短期記憶バッファに入れる"""
        cube.trust.last_updated_turn = self.global_turn

        # 初期 alpha_base をクリップして格納
        cube.trust.alpha_base = max(0.0, min(1.0, cube.trust.alpha_base))

        self.cubes.append(cube)
        self.add_to_previous(cube.cube_id)

    def get_all(self) -> List[SemanticCube]:
        """実効アルファ値が 0.01 以上の生きているキューブを取得（O(N)計算）"""
        return [c for c in self.cubes if c.get_alpha(self.global_turn) >= 0.01 and not c.trust.is_archived]

    def search_by_trust(self, q_cube, k: int = 5):
        bucketed_candidates = []
        rough_scores = []
        
        for c in self.get_all():
            if c.cube_id == q_cube.cube_id: continue
            rough = cosine_similarity(q_cube.vector, c.vector)
            rough_scores.append((c, rough))
            
        if not rough_scores: return []
            
        max_score = max(r for _, r in rough_scores)
        
        for c, rough in rough_scores:
            diff = rough - max_score
            bucket_id, gravity = map_diff_to_7bucket(diff)
            if gravity == 0.0: continue 
            bucketed_candidates.append((c, bucket_id, gravity))

        mc_thresh = 0.7 
        cands = []
        for c, bucket_id, gravity in bucketed_candidates:
            exact_score = calculate_exact_hybrid_score(q_cube, c, mc_threshold=mc_thresh)
            cands.append((c, exact_score * gravity, bucket_id))

        cands.sort(key=lambda x: (x[1], abs(3 - x[2])), reverse=True)
        top_k = [(c, s) for c, s, b in cands[:k]]
        
        # ✨ 代謝メカニズム（イベント発火）
        if self.thread_mode == "smart":
            top_k_ids = [c.cube_id for c, _ in top_k]
            for c, final_score, bucket_id in cands:
                # 候補に残ったものは "similar(類似)"、採用されたものは "srag_hit(参照)"
                if c.cube_id in top_k_ids:
                    self.trigger_event(c, "srag_hit")
                    c.trust.ref_count += 1.0 # 互換性のため維持
                else:
                    self.trigger_event(c, "similar")
                    
        return top_k
