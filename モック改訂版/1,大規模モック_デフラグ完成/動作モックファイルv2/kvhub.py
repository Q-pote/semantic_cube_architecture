"""
kvhub.py
Stateful: 唯一状態を持つクラス。生成された SemanticCube を保持し、検索結果を返す。
※ v2.0: 検索時に MiniCube の語義スコアを合成した `calculate_hybrid_trust` を使用。
"""

from common_types import SemanticCube
from trust_evaluator import calculate_hybrid_trust
from typing import List, Tuple

class KVHub:
    def __init__(self):
        # 世界の記憶（キューブの集合）を保持するリスト
        self.cubes: List[SemanticCube] = []

    def put(self, cube: SemanticCube):
        """新しいキューブを記憶層に追加する"""
        self.cubes.append(cube)

    def get_all(self) -> List[SemanticCube]:
        """アーカイブされていないすべてのアクティブなキューブを取得する"""
        return [c for c in self.cubes if c.role != "archived"]

    def search_by_trust(self, q_cube: SemanticCube, current_turn: int, k: int = 5) -> List[Tuple[SemanticCube, float]]:
        """
        与えられたクエリキューブ(q_cube)を基準に、Hybrid Trustスコアが高い順にTop-Kのキューブを検索・抽出する。
        """
        # 全キューブに対する Hybrid Trustスコア を計算
        cands = [(c, calculate_hybrid_trust(c, q_cube, current_turn)) for c in self.get_all()]
        # スコアの降順でソート
        cands.sort(key=lambda x: x[1], reverse=True)
        # 上位 K 件を返す
        return cands[:k]
