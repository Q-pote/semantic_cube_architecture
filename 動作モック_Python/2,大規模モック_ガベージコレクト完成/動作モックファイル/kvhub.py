"""
kvhub.py
Stateful: 世界の記憶を保持。
※ v2.1 修正版: 検索時に確実に `calculate_hybrid_trust` (MiniCube語義検索) を使用し、代謝を発動する。
"""

from common_types import SemanticCube
from trust_evaluator import calculate_hybrid_trust, update_alpha
from typing import List, Tuple

class KVHub:
    def __init__(self):
        self.cubes: List[SemanticCube] = []
        self.thread_mode: str = "smart"  # "smart" (代謝あり) or "protected" (代謝なし)

    def put(self, cube: SemanticCube):
        """新しいキューブを追加。初期α値を計算しておく。"""
        update_alpha(cube)
        self.cubes.append(cube)

    def get_all(self) -> List[SemanticCube]:
        """GC対象でないアクティブなキューブを取得"""
        return [c for c in self.cubes if not c.trust.gc_flag]

    def search_by_trust(self, q_cube: SemanticCube, current_turn: int, k: int = 5) -> List[Tuple[SemanticCube, float]]:
        """
        与えられたクエリキューブ(q_cube)を基準に、Hybrid Trustスコアが高い順にTop-Kのキューブを検索・抽出する。
        検索と同時に『記憶の代謝（参照ボーナス付与）』を発動する。
        """
        cands = []
        for c in self.get_all():
            # ✨ 修正の核心: ここで確実に calculate_hybrid_trust を呼ぶ！ ✨
            score = calculate_hybrid_trust(c, q_cube, current_turn)
            cands.append((c, score))
            
        # スコアの降順でソート
        cands.sort(key=lambda x: x[1], reverse=True)
        top_k = cands[:k]
        
        # --- 参照メカニズムのアップデート（代謝・α更新） ---
        if self.thread_mode == "smart":
            for c, score in top_k:
                # 検索にヒット（参照）されたキューブに、意味強度(score)をボーナスとして加算
                ref_bonus = score
                c.trust.ref_count += (1.0 + ref_bonus)
                c.trust.last_used_at_turn = current_turn
                # avg_similarityを簡易的に更新（指数移動平均）
                c.trust.avg_similarity = (c.trust.avg_similarity * 0.8) + (score * 0.2)
                # α値の再計算
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
