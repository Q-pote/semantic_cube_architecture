"""
common_types.py
Semantic Cube Runtime v2.1 共通データ構造
※ v2.1: αチャンネル（代謝モデル）と float 型 ref_count を追加。
"""

import os
import numpy as np
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 🔑 API & 環境変数の設定 (グローバル設定)
# ==========================================
os.environ["GEMINI_API_KEY"] = "API key"  # ここに実際のAPIキーを設定してください
os.environ["GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
os.environ["GEMINI_LLM_MODEL"] = "gemini-3.1-flash-lite"

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ==========================================
# 🧱 データ構造 (Data Structures)
# ==========================================
@dataclass
class Orientation:
    azimuth: float = 0.0
    elevation: float = 0.0
    strength: float = 1.0

@dataclass
class OrientationBins:
    az_bin: int = 0; el_bin: int = 0; str_bin: int = 0

@dataclass
class MiniCube:
    phrase: str
    embedding: np.ndarray
    confidence: float

@dataclass
class TrustStruct:
    grid_index: list = field(default_factory=lambda: [0, 0, 0])
    role: str = "history"
    orientation: Orientation = field(default_factory=Orientation)
    orientation_bins: OrientationBins = field(default_factory=OrientationBins)
    gravity: float = 1.0
    
    # --- 履歴パラメータ ---
    created_at_turn: int = 0
    last_used_at_turn: int = 0
    # ✨ v2.1: ref_count を float 化（ボーナス加算対応）
    ref_count: float = 0.0 
    
    # --- ✨ v2.1 代謝（αチャンネル）パラメータ ---
    avg_similarity: float = 1.0           # 参照された時の類似度の平均
    replacement_closeness: float = 0.0    # 新Originによる代替可能性 (0.0〜1.0)
    alpha: float = 1.0                    # 記憶の透明度 (濃いほど忘れない)
    gc_flag: bool = False                 # Garbage Collection 候補フラグ

@dataclass
class SemanticCube:
    cube_id: str
    role: str
    summary: str
    vector: np.ndarray
    trust: TrustStruct = field(default_factory=TrustStruct)
    mini_cubes: List[MiniCube] = field(default_factory=list)
