"""
cube_factory.py
Pure functions: テキストからベクトル・MiniCubeを生成し、SemanticCube をインスタンス化する。
※ v3.1: 派生系(parent_id, origin_id)の記録、および実時刻(created_at_time)の自動保存に対応。
"""

import os
import uuid
import time
import math
import json
import numpy as np
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Tuple
from common_types import SemanticCube, MiniCube, Orientation, OrientationBins, TrustStruct
from sklearn.decomposition import PCA


# PCAモデルの初期化
print("🔧 Initializing placeholder PCA model...")
dummy_data_for_pca = np.random.rand(100, 3072) # 3072次元のダミーデータ
pca_model = PCA(n_components=3)
pca_model.fit(dummy_data_for_pca)
print("✅ PCA model is ready.")


def embed_text(text: str) -> np.ndarray:
    emb_model = os.environ.get("GEMINI_EMBEDDING_MODEL")
    try:
        time.sleep(0.2) # API制限回避用
        
        res = genai.embed_content(model=emb_model, content=text, task_type="retrieval_document")
        return np.array(res["embedding"])
    except Exception as e:
        print(f"⚠️ [API Error] Embedding failed: {e}. Using dummy vector.")
        return np.random.rand(3072)

def map_to_grid(vec: np.ndarray, bins: int = 8) -> list[int]:
    vals = np.array([vec[0], vec[1], vec[2]], dtype=float)
    norm = np.clip((vals + 0.1) / 0.2, 0.0, 1.0)
    # 🔍 [Debug] グリッド化の様子を出力
    grid = [int(max(0, min(bins - 1, n * (bins - 1)))) for n in norm]
    print(f"    🔍 [Debug-Factory] Vec[0:3]: [{vals[0]:.2f}, {vals[1]:.2f}, {vals[2]:.2f}] -> Grid: {grid}")
    
    return [int(max(0, min(bins - 1, n * (bins - 1)))) for n in norm]

def extract_keyphrases(text: str, model_name: str) -> List[Dict]:
    prompt = f"""あなたはテキスト解析の専門家としてふるまってください。
以下のテキストから名詞や形容詞などの単語キーワードを1~10個程度抽出してください。
回答は必ず以下のJSON形式のフォーマットで配列のみとし、
テキストに含まれていない事実は追加しないでください。
[
  {{"phrase": "キーフレーズ1", "confidence": 0.9}}
]
【要約対象のテキスト】
{text}
"""
    try:
        time.sleep(0.2) # API制限回避用
        model = genai.GenerativeModel(model_name)
        response_text = model.generate_content(prompt).text.strip()
        if response_text.startswith("```json"): response_text = response_text[7:]
        if response_text.startswith("```"): response_text = response_text[3:]
        if response_text.endswith("```"): response_text = response_text[:-3]
        return json.loads(response_text.strip())
    except Exception as e:
        print(f"⚠️ [API Error] Keyphrase extraction failed: {e}. Fallback to dummy.")
        return [{"phrase": text[:20], "confidence": 0.1}]

def quantize_embedding(embedding_vector: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    ✨ [v2.5] 新しい量子化関数
    高次元ベクトルから vec_int3, int3_grid, grid_mesh を生成する。
    """
    # --- [vec_int3 の生成] --- (N次元, 8階調)
    clipped_vec = np.clip(embedding_vector, -3.0, 3.0)
    norm_vec = (clipped_vec + 3.0) / 6.0
    vec_int3 = np.floor(norm_vec * 8).astype(int)
    vec_int3 = np.clip(vec_int3, 0, 7)

    # --- [int3_grid, grid_mesh の生成] --- (3次元)
    # PCAで3次元に圧縮
    vec_3d = pca_model.transform([embedding_vector])[0]

    # 3次元空間を正規化
    space_min, space_max = -5.0, 5.0
    norm_3d = np.clip((vec_3d - space_min) / (space_max - space_min), 0.0, 1.0)

    # int3_grid: 8階調 (0~7) にマッピング
    int3_grid = tuple(np.floor(norm_3d * 8).astype(int))

    # grid_mesh: 1000階調 (0~999) にマッピング
    grid_mesh = tuple(np.floor(norm_3d * 1000).astype(int))

    return vec_int3, int3_grid, grid_mesh

def compute_orientation(vec: np.ndarray) -> Orientation:
    norm = np.linalg.norm([vec[0], vec[1], vec[2]]) + 1e-9
    azimuth = math.atan2(vec[1]/norm, vec[0]/norm) % (2*math.pi)
    elevation = math.asin(vec[2]/norm)
    return Orientation(azimuth, elevation, float(norm))

def make_cube(text: str, role: str, turn: int, parent_id: str = None, origin_id: str = None, response: str = None) -> SemanticCube:
    """
    改造されたmake_cube関数。v2.5の量子化ロジックを注入。
    """

    gemini_model = os.environ.get("GEMINI_LLM_MODEL")
    # 1. 基本的なエンベディング (v3.5互換)
    vec = embed_text(text)
    
    # ✨ 2. [v2.5] 新しい量子化処理
    vec_int3, int3_grid, grid_mesh = quantize_embedding(vec)

    # 3. v3.5互換の向きと信頼度構造の生成
    ori = compute_orientation(vec)
    az_bin = int(max(0, min(359, (ori.azimuth / (2 * math.pi)) * 360)))
    el_bin = int(max(0, min(359, ((ori.elevation + (math.pi / 2)) / math.pi) * 360)))
    str_bin = int(max(0, min(359, math.log1p(ori.strength) * 100)))
    ori_bins = OrientationBins(az_bin, el_bin, str_bin)
    trust = TrustStruct(
        grid_index=list(int3_grid), # v3.5互換のためリストに変換
        role=role, orientation=ori, orientation_bins=ori_bins,
        created_at_turn=turn, last_used_at_turn=turn,
        created_at_time=datetime.now().timestamp(), last_updated_turn=turn, alpha_base=1.0
    )

    # 4. MiniCubeの生成 (ここもv2.5の量子化を適用)
    mini_cubes = []
    phrase_dicts = extract_keyphrases(text, model_name=gemini_model)
    for pd in phrase_dicts:
        phrase = pd.get("phrase", "")
        conf = float(pd.get("confidence", 0.5))
        if phrase:
            emb = embed_text(phrase)
            # MiniCube用の量子化
            mc_vec_int3, mc_int3_grid, _ = quantize_embedding(emb)
            mini_cubes.append(MiniCube(
                phrase=phrase, embedding=emb, confidence=conf,
                vec_int3=mc_vec_int3, int3_grid=mc_int3_grid
            ))

    # 5. v2.5の属性を注入してSemanticCubeを最終的に生成
    cube = SemanticCube(
        cube_id=str(uuid.uuid4())[:8],
        role=role, summary=text, vector=vec, trust=trust,
        mini_cubes=mini_cubes, parent_id=parent_id, origin_id=origin_id,
        response_and_answer=response,
        # ✨ [v2.5] 新しい属性をインスタンスにセット
        vector_int3=vec_int3,
        int3_grid=int3_grid,
        grid_mesh=grid_mesh
    )

    if role == "core" and origin_id is None:
        cube.origin_id = cube.cube_id

    print(f"🧊 [Cube Factory] New Cube '{cube.summary[:20]}...' created.")
    print(f"    - Cube ID: {cube.cube_id}")
    print(f"    - Grid Mesh (1000): {cube.grid_mesh}")
    if cube.mini_cubes:
        for mc in cube.mini_cubes:
            print(f"    - MiniCube '{mc.phrase}' Grid: {mc.int3_grid}")

    return cube


# LLMResponseを追加するための関数
def add_Response(cube: SemanticCube, response_test: str) -> None:
    cube.response_and_answer = response_test

