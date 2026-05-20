"""
cube_factory.py
Pure functions: テキストからベクトルを生成し、正規化して SemanticCube を生成する。
※ v2.0: LLMによるキーフレーズ抽出と、MiniCubeの埋め込み処理を追加。
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
    """テキストをベクトルに変換する"""
    time.sleep(1) # API制限回避用
    emb_model = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    try:
        res = genai.embed_content(model=emb_model, content=text, task_type="retrieval_document")
        return np.array(res["embedding"])
    except Exception as e:
        print(f"⚠️ [API Error] Embedding failed: {e}. Using dummy vector.")
        return np.random.rand(768)

def map_to_grid(vec: np.ndarray, bins: int = 100) -> list[int]:
    """呪い解除済: 絶対的な範囲で正規化し、空間の内部に立体的に分布させる"""
    vals = np.array([vec[0], vec[1], vec[2]], dtype=float)
    norm = np.clip((vals + 0.1) / 0.2, 0.0, 1.0)
    return [int(max(0, min(bins - 1, n * (bins - 1)))) for n in norm]

def compute_orientation(vec: np.ndarray) -> Orientation:
    """ベクトルの向き（方位）を計算する"""
    norm = np.linalg.norm([vec[0], vec[1], vec[2]]) + 1e-9
    azimuth = math.atan2(vec[1]/norm, vec[0]/norm) % (2*math.pi)
    elevation = math.asin(vec[2]/norm)
    return Orientation(azimuth, elevation, float(norm))

def extract_keyphrases(text: str, model_name: str) -> List[Dict]:
    """LLMを用いてテキストからキーフレーズとConfidenceを抽出する（JSON出力）"""
    prompt = f"""あなたはテキスト解析の専門家です。以下の要約から主要なキーフレーズを3〜6個抽出してください。
出力は必ず以下のJSON形式の配列のみとし、マークダウン記法(```json)やその他のテキストは一切含めないでください。
新事実を追加しないでください。

[
  {{"phrase": "キーフレーズ1", "confidence": 0.9}},
  {{"phrase": "キーフレーズ2", "confidence": 0.8}}
]

【要約】
{text}
"""
    try:
        time.sleep(1)
        model = genai.GenerativeModel(model_name)
        response_text = model.generate_content(prompt).text.strip()
        
        # 不要なマークダウンブロックの除去
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
             response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        parsed = json.loads(response_text.strip())
        return parsed
    except Exception as e:
        print(f"⚠️ [API Error] Keyphrase extraction failed: {e}. Fallback to dummy.")
        # フォールバック（抽出失敗時はテキスト全体を1つのフレーズとする）
        return [{"phrase": text[:20], "confidence": 0.5}]

def make_cube(text: str, role: str, turn: int, ref_count: int, llm_model_name: str) -> SemanticCube:
    """すべての情報を結合して SemanticCube を生成し、MiniCubeを内包する"""
    vec = embed_text(text)
    grid = map_to_grid(vec)
    ori = compute_orientation(vec)
    
    # 簡易ビン化
    az_bin = int(max(0, min(359, (ori.azimuth / (2 * math.pi)) * 360)))
    el_bin = int(max(0, min(359, ((ori.elevation + (math.pi / 2)) / math.pi) * 360)))
    str_bin = int(max(0, min(359, math.log1p(ori.strength) * 100)))
    ori_bins = OrientationBins(az_bin, el_bin, str_bin)

    trust = TrustStruct(grid_index=grid, role=role, orientation=ori, 
                        orientation_bins=ori_bins, created_at_turn=turn, 
                        last_used_at_turn=turn, ref_count=ref_count)
    
    # ✨ v2.0: MiniCubeの生成と格納 ✨
    mini_cubes = []
    phrase_dicts = extract_keyphrases(text, llm_model_name)
    for pd in phrase_dicts:
        phrase = pd.get("phrase", "")
        conf = float(pd.get("confidence", 0.5))
        if phrase:
            emb = embed_text(phrase)
            mini_cubes.append(MiniCube(phrase=phrase, embedding=emb, confidence=conf))

    cube = SemanticCube(str(uuid.uuid4())[:8], role, text, vec, trust, mini_cubes=mini_cubes)
    print(f"🧊 [Cube Factory] ID:{cube.cube_id} / Role:{cube.role:7s} / Grid:{cube.trust.grid_index} / MiniCubes:{len(cube.mini_cubes)}個")
    return cube
