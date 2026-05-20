"""
logger.py
Pure function: KVHubの状態などをJSON形式で出力する。
"""

import json
from common_types import SemanticCube

def export_cubes_to_json(cubes: list[SemanticCube], query_cube: SemanticCube = None) -> str:
    """キューブ群の空間座標、MiniCube、代謝パラメータをJSON出力する。"""
    export_data = {"cubes": []}
    
    for c in cubes:
        cube_data = {
            "cube_id": c.cube_id,
            "role": c.role,
            "grid_index": c.trust.grid_index,
            "metabolism": {
                "ref_count": round(c.trust.ref_count, 3),
                "alpha": round(c.trust.alpha, 3),
                "gc_flag": c.trust.gc_flag
            },
            "summary": c.summary[:30] + "...",
            "mini_cubes": [{"phrase": mc.phrase, "confidence": mc.confidence} for mc in c.mini_cubes]
        }
        export_data["cubes"].append(cube_data)
    
    if query_cube:
        export_data["query_cube"] = {
            "cube_id": query_cube.cube_id,
            "role": "query_anchor",
            "grid_index": query_cube.trust.grid_index,
            "summary": query_cube.summary[:30] + "...",
            "mini_cubes": [{"phrase": mc.phrase, "confidence": mc.confidence} for mc in query_cube.mini_cubes]
        }
        
    return json.dumps(export_data, ensure_ascii=False, indent=2)

def print_log(phase: str, message: str):
    print(f"\n[{phase}] {message}")
