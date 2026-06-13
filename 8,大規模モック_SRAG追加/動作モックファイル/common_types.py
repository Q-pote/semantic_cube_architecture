"""
common_types.py
※ v3.5: イベント駆動アルファ代謝 (O(1)アーキテクチャ) の実装
"""

import os
import uuid
import numpy as np
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List, Tuple

os.environ["GEMINI_API_KEY"] = "your_api_key_here"
os.environ["GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
os.environ["GEMINI_LLM_MODEL"] = "gemini-3.1-flash-lite"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

@dataclass
class Orientation:
    """向き（v3.5互換）"""
    azimuth: float = 0.0
    elevation: float = 0.0
    strength: float = 1.0

@dataclass
class OrientationBins:
    """量子化された向き（v3.5互換）"""
    az_bin: int = 0
    el_bin: int = 0
    str_bin: int = 0

@dataclass
class MiniCube:
    """
    語義DNA (v2.5の属性を注入)
    """
    phrase: str
    embedding: np.ndarray
    confidence: float
    # ✨ [v2.5] SRAGのための新しい属性
    vec_int3: np.ndarray = field(default=None)
    int3_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))

@dataclass
class TrustStruct:
    """信頼度・代謝構造（v3.5互換）"""
    grid_index: list = field(default_factory=lambda: [0, 0, 0])
    role: str = "topic"
    orientation: Orientation = field(default_factory=Orientation)
    orientation_bins: OrientationBins = field(default_factory=OrientationBins)
    gravity: float = 1.0
    created_at_turn: int = 0
    last_used_at_turn: int = 0
    ref_count: float = 0.0
    created_at_time: float = 0.0
    alpha_base: float = 1.0
    last_updated_turn: int = 0
    is_archived: bool = False

    @property
    def is_privileged(self) -> bool:
        return self.role in ["identity", "ethics"]

@dataclass
class SemanticCube:
    """
    意味の基本単位 (v2.5の属性を注入)
    """
    cube_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: str = ""
    summary: str = ""
    vector: np.ndarray = None
    trust: TrustStruct = field(default_factory=TrustStruct)
    mini_cubes: List[MiniCube] = field(default_factory=list)
    parent_id: str = None
    origin_id: str = None
    response_and_answer: str = None
    # ✨ [v2.5] SRAGのための新しい属性
    vector_int3: np.ndarray = field(default=None)
    int3_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))
    grid_mesh: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))

    # ✨ [v3.5] 互換のため仮置き
    trust_score: float = 1.0

    def get_alpha(self, global_turn: int) -> float:
        if self.trust.is_privileged:
            return 1.0
        turns_passed = global_turn - self.trust.last_updated_turn
        grace_period = 5
        decay_rate = 0.05
        effective_turns_for_decay = max(0, turns_passed - grace_period)
        decay = decay_rate * effective_turns_for_decay
        return max(0.0, self.trust.alpha_base - decay)