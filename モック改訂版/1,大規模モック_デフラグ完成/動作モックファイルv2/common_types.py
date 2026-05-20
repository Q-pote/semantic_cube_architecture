"""
common_types.py
Semantic Cube Runtime の共通データ構造（Data Classes）を定義する。
※ v2.0: MiniCube（語義ベクトル）のデータ構造を追加。
"""

import os
import numpy as np
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 🔑 API & 環境変数の設定 (グローバル設定)
# ==========================================
os.environ["GEMINI_API_KEY"] = "API key"
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
    az_bin: int = 0
    el_bin: int = 0
    str_bin: int = 0

@dataclass
class TrustStruct:
    grid_index: list = field(default_factory=lambda: [0, 0, 0])
    role: str = "history"
    orientation: Orientation = field(default_factory=Orientation)
    orientation_bins: OrientationBins = field(default_factory=OrientationBins)
    gravity: float = 1.0
    created_at_turn: int = 0
    last_used_at_turn: int = 0
    ref_count: int = 0

# ✨ v2.0 新規追加: 最小構成の MiniCube ✨
@dataclass
class MiniCube:
    phrase: str
    embedding: np.ndarray
    confidence: float

@dataclass
class SemanticCube:
    cube_id: str
    role: str
    summary: str
    vector: np.ndarray
    trust: TrustStruct = field(default_factory=TrustStruct)
    # ✨ v2.0 新規追加: MiniCubeのリスト ✨
    mini_cubes: List[MiniCube] = field(default_factory=list)
