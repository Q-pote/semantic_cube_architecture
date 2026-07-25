# gc_engine.py (v1.0)

from data_hub import DataHub
from common_types import SemanticCube

class GCEngine:
    """
    Garbage Collection Engine.
    アーカイブ済みのキューブを、最終的にどう処理するかを決定する。
    """
    def __init__(self, hub: DataHub):
        self.hub = hub
        print("✅ GC_Engine initialized.")

    def run_gc_cycle(self):
        """
        GCの1サイクルを実行する。
        Defragの後に呼ばれることを想定。
        """
        print("\n" + "="*20 + " GC Engine Cycle Start " + "="*20)
        
        # Hubの全キューブをスキャン
        all_cubes = list(self.hub.cube_pool.values()) # コピーに対してループ
        
        gc_count = 0
        for cube in all_cubes:
            if cube.is_archived:
                # 【Phase 1: 論理削除】
                # メインのプールから削除し、アーカイブ用のプールに移動する
                
                # 1. メインプールから削除
                del self.hub.cube_pool[cube.cube_id]
                
                # 2. アーカイブプールに追加 (DataHubに archived_cube_pool を追加する必要あり)
                self.hub.archived_cube_pool[cube.cube_id] = cube
                
                gc_count += 1
                print(f"  - Moved to Archive: Cube '{cube.cube_id}'")

        print(f"✅ GC cycle complete. Moved {gc_count} cubes to archive.")
