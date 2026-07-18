# common_types.py

import uuid
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import List, Tuple, Literal

# --- enum設計 ---
CubeRole = Literal["history", "origin", "knowledge", "ethics", "identity"]
CubeSource = Literal["USER", "LLM"]
PhraseRole = Literal["subject", "predicate", "key_phrase"]

SearchMode = Literal["standard", "exploratory"]

# ---------------------------------
@dataclass
class TrustStruct:
    """信頼度・代謝構造"""
    score: float = 1.0
    alpha: float = 1.0
    ref_count: int = 0

# ---------------------------------

@dataclass
class MatchInfo:
    """どのキーワードがマッチしたかの詳細情報"""
    query_phrase: str
    target_phrase: str
    score: float
    parent_cube_id: str = ""

# ---------------------------------

@dataclass
class SearchResult:
    """SRAGの検索結果"""
    cube: SemanticCube
    final_score: float
    reason: str
    matched_minicubes: List[MatchInfo] = field(default_factory=list)

# ---------------------------------

@dataclass
class TensionResult:
    has_tension: bool = False
    tension_score: float = 0.0
    # LLMに通知するための「異なっている意見」のサマリーを保持する
    divergent_summaries: List[str] = field(default_factory=list)

# ---------------------------------
@dataclass
class MiniCube:
    """キーフレーズ"""
    minicube_id: str = field(default_factory=lambda: "mc_" + str(uuid.uuid4())[:8])
    phrase: str = ""
    role: PhraseRole = "key_phrase"
    confidence: float = 1.0
    embedding: np.ndarray = None
    vec_int3: np.ndarray = None
    int3_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # インデックス用 (8階調)
    int8_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # 描画・分布観測用 (256階調)
    grid_mesh: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # クラスタ・葛藤演算用 (10000階調)
    parent_cube_id: str = ""    # 親キューブのIDを保持するデバッグ実装

# ---------------------------------
@dataclass
class SemanticCube:
    """意味の基本単位（思考の原子）"""
    cube_id: str = field(default_factory=lambda: "cb_" + str(uuid.uuid4())[:8])
    
    role: CubeRole = "history"
    source: CubeSource = "USER"
    summary: str = "" # 原文を格納
    
    embedding_vector: np.ndarray = None
    vec_int3: np.ndarray = None
    int3_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # インデックス用 (8階調)
    int8_grid: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # 描画・分布観測用 (256階調)
    grid_mesh: Tuple[int, int, int] = field(default_factory=lambda: (0, 0, 0))       # クラスタ・葛藤演算用 (10000階調)
    
    trust: TrustStruct = field(default_factory=TrustStruct)
    mini_cubes: List[MiniCube] = field(default_factory=list)
    
    reactionLink_ID: str = None
    response_and_answer: str = None # result_summary
    
    defrag_target: bool = True
    is_archived: bool = False
    create_datetime: datetime = field(default_factory=lambda: datetime.now().strftime("%y%m%d_%H%M%S"))
    create_turn: int = 0
    defrag_turn: int = 0
    last_used_turn: int = 0 # getAlphaで使うため追加

    def get_alpha(self, global_turn: int) -> float:
        """現在の実効アルファ値（生命力）を計算する"""
        if self.role in ["knowledge", "ethics", "identity"]:
            return 1.0
        
        turns_passed = global_turn - self.last_used_turn
        grace_period = 5    # 5ターンの猶予期間を設ける
        decay_rate = 0.1    # 1ターンあたりの減衰率を設定
        
        effective_turns_for_decay = max(0, turns_passed - grace_period)
        decay = decay_rate * effective_turns_for_decay
        
        # alpha_base は TrustStruct から削除し、1.0固定で計算
        current_alpha = max(0.0, 1.0 - decay)
        self.trust.alpha = current_alpha
        return current_alpha
