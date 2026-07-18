# data_hub.py

from typing import Dict, List
from common_types import SemanticCube

class DataHub:
    """
    システムの全ての状態（記憶）を一元管理する中央ハブ。
    智也の設計案v5.1を実装。
    """
    def __init__(self):
        # 1. 全セマンティックキューブ（長期記憶）
        self.cube_pool: Dict[str, SemanticCube] = {}
        
        # 2. 直近対話ログ（短期記憶 / UI用）
        self.active_history: List[SemanticCube] = []
        
        # 3. SRAG検索結果（ワーク領域）
        self.srag_results: List[SemanticCube] = []

        # 4. GC後にアーカイブされたキューブを移すプール
        self.archived_cube_pool: Dict[str, SemanticCube] = {}
        
        self.global_turn: int = 0 # 現在のグローバルターン数

        self.defrag_turn: int = 0 # デフラグの進行状況を示すターン数

    def tick(self):
        """グローバルターンを進める"""
        self.global_turn += 1

    def cube_add(self, cube: SemanticCube):
        """キューブを長期記憶に追加"""
        self.cube_pool[cube.cube_id] = cube

    def cube_get_all(self) -> List[SemanticCube]:
        """生きている全キューブを取得"""
        return [c for c in self.cube_pool.values() if not c.is_archived]

    def history_push(self, cube: SemanticCube):
        """直近対話ログにキューブを追加（10個制限）"""
        self.active_history.append(cube)
        if len(self.active_history) > 10:
            self.active_history.pop(0)

    def history_get_all(self) -> List[SemanticCube]:
        """直近対話ログを全件取得"""
        return self.active_history

    def search_change_set(self, cubes: List[SemanticCube]):
        """SRAGの一時保存エリアを上書き"""
        self.srag_results = cubes

    def search_get_all(self) -> List[SemanticCube]:
        """SRAGの一時保存エリアを取得"""
        return self.srag_results

    def archived_get_all(self) -> List[SemanticCube]:
        """アーカイブされたキューブを全件取得"""
        return list(self.archived_cube_pool.values())
    