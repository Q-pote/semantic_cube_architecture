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
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"
os.environ["GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
# os.environ["GEMINI_LLM_MODEL"] = "gemini-3.1-flash-lite"
os.environ["GEMINI_LLM_MODEL"] = "gemini-3.5-flash"

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
    # ✨ 修正: デフォルトのroleを "topic"（話題）に変更
    role: str = "topic" # "core", "topic", "identity", "ethics" の4種
    # "core"はデフラグ時にのみ生成されえる.
    orientation: Orientation = field(default_factory=Orientation)
    orientation_bins: OrientationBins = field(default_factory=OrientationBins)
    gravity: float = 1.0
    
    # --- 履歴パラメータ ---
    created_at_turn: int = 0
    last_used_at_turn: int = 0
    ref_count: float = 0.0 
    
    # --- 代謝（αチャンネル）パラメータ ---
    avg_similarity: float = 1.0
    replacement_closeness: float = 0.0
    alpha: float = 1.0
    gc_flag: bool = False
    
    # ✨ 追加: 特権キューブ（代謝・GCの対象外）かどうかを判定するプロパティ
    @property
    def is_privileged(self) -> bool:
        return self.role in ["identity", "ethics"]

@dataclass
class SemanticCube:
    cube_id: str
    role: str
    summary: str
    vector: np.ndarray
    trust: TrustStruct = field(default_factory=TrustStruct)
    mini_cubes: List[MiniCube] = field(default_factory=list)
    # ✨ 追加したプロパティは、引数ズレを防ぐために一番下に置く！
    vector_int3: np.ndarray = field(default=None) 
    parent_id: str = None  # 派生元の直近キューブ(Active)のID
    origin_id: str = None  # 大元となる中心重力(Core)のID
    