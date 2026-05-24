"""
common_types.py
※ v3.5: イベント駆動アルファ代謝 (O(1)アーキテクチャ) の実装
"""

import os
import numpy as np
import google.generativeai as genai
from dataclasses import dataclass, field
from typing import List

os.environ["GEMINI_API_KEY"] = "GEMINI_API_KEYをここにセット"
os.environ["GEMINI_EMBEDDING_MODEL"] = "gemini-embedding-2"
os.environ["GEMINI_LLM_MODEL"] = "gemini-3.5-flash"
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

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
    role: str = "topic" 
    orientation: Orientation = field(default_factory=Orientation)
    orientation_bins: OrientationBins = field(default_factory=OrientationBins)
    gravity: float = 1.0
    
    # --- 履歴パラメータ ---
    created_at_turn: int = 0
    last_used_at_turn: int = 0
    ref_count: float = 0.0 
    created_at_time: float = 0.0  
    
    # --- 代謝（αチャンネル）パラメータ (✨v3.5 O(1) イベント駆動) ---
    alpha_base: float = 1.0       # 計算の基準となるアルファ値
    last_updated_turn: int = 0    # 最後にイベントが起きたターン
    is_archived: bool = False     # GC対象だが物理削除しない監査用フラグ
    
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
    vector_int3: np.ndarray = field(default=None) 
    parent_id: str = None  
    origin_id: str = None  
    response_and_answer: str = None  

    # ✨ v3.5 O(1) アルファ代謝のオンデマンド計算メソッド
    def get_alpha(self, global_turn: int) -> float:
        if self.trust.is_privileged:
            return 1.0
        
        turns_passed = global_turn - self.trust.last_updated_turn
        
        # 司令官の「間接閾値排除」の思想:
        # 5ターンまでは decayが0またはマイナスになり、max関数で0.0に打ち消されるか、
        # あるいは単純に5ターンの猶予期間(grace_period)を変数化する。
        grace_period = 5
        decay_rate = 0.05
        
        # 経過ターンから保護期間を引いた分だけ減衰させる（マイナスなら減衰ゼロ）
        effective_turns_for_decay = max(0, turns_passed - grace_period)
        decay = decay_rate * effective_turns_for_decay
        
        return max(0.0, self.trust.alpha_base - decay)
