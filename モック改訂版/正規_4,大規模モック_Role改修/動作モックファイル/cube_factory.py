"""
cube_factory.py
Pure functions: テキストからベクトル・MiniCubeを生成し、SemanticCube をインスタンス化する。
"""

import os
import uuid
import time
import math
import json
import numpy as np
import google.generativeai as genai
from typing import List, Dict
from common_types import SemanticCube, MiniCube, Orientation, OrientationBins, TrustStruct

def embed_text(text: str) -> np.ndarray:
    time.sleep(0.5) # API制限回避用
    emb_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    try:
        res = genai.embed_content(model=emb_model, content=text, task_type="retrieval_document")
        return np.array(res["embedding"])
    except Exception as e:
        print(f"⚠️ [API Error] Embedding failed: {e}. Using dummy vector.")
        return np.random.rand(768)

def map_to_grid(vec: np.ndarray, bins: int = 100) -> list[int]:
    vals = np.array([vec[0], vec[1], vec[2]], dtype=float)
    norm = np.clip((vals + 0.1) / 0.2, 0.0, 1.0)
    return [int(max(0, min(bins - 1, n * (bins - 1)))) for n in norm]

def compute_orientation(vec: np.ndarray) -> Orientation:
    norm = np.linalg.norm([vec[0], vec[1], vec[2]]) + 1e-9
    azimuth = math.atan2(vec[1]/norm, vec[0]/norm) % (2*math.pi)
    elevation = math.asin(vec[2]/norm)
    return Orientation(azimuth, elevation, float(norm))

def extract_keyphrases(text: str, model_name: str) -> List[Dict]:
    prompt = f"""あなたはテキスト解析の専門家としてふるまってください。
以下のテキストから主要なキーフレーズを3〜6個抽出してください。
回答は必ず以下のJSON形式のフォーマットで配列のみとし、
テキストに含まれていない事実は追加しないでください。
[
  {{"phrase": "キーフレーズ1", "confidence": 0.9}}
]
【要約】
{text}
"""
    try:
        time.sleep(0.5) # API制限回避用
        model = genai.GenerativeModel(model_name)
        response_text = model.generate_content(prompt).text.strip()
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        return json.loads(response_text.strip())
    except Exception as e:
        print(f"⚠️ [API Error] Keyphrase extraction failed: {e}. Fallback to dummy.")
        return [{"phrase": text[:20], "confidence": 0.5}]


# ✨ 修正: 引数に `parent_id` と `origin_id` を追加！（デフォルトはNone）
def make_cube(text: str, role: str, turn: int, ref_count: float, llm_model_name: str, parent_id: str = None, origin_id: str = None) -> SemanticCube:
    vec = embed_text(text)
    grid = map_to_grid(vec)
    ori = compute_orientation(vec)
    
    az_bin = int(max(0, min(359, (ori.azimuth / (2 * math.pi)) * 360)))
    el_bin = int(max(0, min(359, ((ori.elevation + (math.pi / 2)) / math.pi) * 360)))
    str_bin = int(max(0, min(359, math.log1p(ori.strength) * 100)))
    ori_bins = OrientationBins(az_bin, el_bin, str_bin)

    trust = TrustStruct(grid_index=grid, role=role, orientation=ori, 
                        orientation_bins=ori_bins, created_at_turn=turn, 
                        last_used_at_turn=turn, ref_count=float(ref_count))
    
    mini_cubes = []
    phrase_dicts = extract_keyphrases(text, llm_model_name)
    for pd in phrase_dicts:
        phrase = pd.get("phrase", "")
        conf = float(pd.get("confidence", 0.5))
        if phrase:
            emb = embed_text(phrase)
            mini_cubes.append(MiniCube(phrase=phrase, embedding=emb, confidence=conf))

    # ✨ 修正: parent_id と origin_id を SemanticCube に渡す
    cube = SemanticCube(
        cube_id=str(uuid.uuid4())[:8], 
        role=role, 
        summary=text, 
        vector=vec, 
        trust=trust, 
        mini_cubes=mini_cubes,
        parent_id=parent_id,
        origin_id=origin_id
    )
    
    # ✨ 自分が core の場合、大元の先祖（origin_id）を自分自身のIDに設定する
    if role == "core" and origin_id is None:
        cube.origin_id = cube.cube_id

    print(f"🧊 [Cube Factory] ID:{cube.cube_id} / Role:{cube.role:7s} / Alpha:{cube.trust.alpha:.2f} / Parent:{parent_id} / Origin:{cube.origin_id}")
    return cube
