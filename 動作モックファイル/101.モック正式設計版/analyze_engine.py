# analyze_engine.py (v9.2 ハイブリッド・トピックロック版)

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from common_types import SearchResult, SemanticCube, TensionResult

class AnalyzeEngine:
    """
    SRAGで集められたキューブ群から、意味の「張力（Tension）」を検知する。
    「ハイブリッド・トピックロック」思想を完全実装。
    """
    def __init__(self, topic_threshold: float = 0.8, tension_threshold: float = 0.65):
        # MiniCube同士が「同じトピック」だと見なす類似度の閾値
        self.topic_threshold = topic_threshold
        # SemanticCube同士が「葛藤している」と見なす類似度の閾値
        self.tension_threshold = tension_threshold
        print(f"✅ AnalyzeEngine (v9.2) initialized. topic_threshold={topic_threshold}, tension_threshold={tension_threshold}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1 is None or v2 is None: return 0.0
        # np.linalg.normが0になるのを防ぐ
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _find_shared_topic(self, cube1: SemanticCube, cube2: SemanticCube) -> Optional[Tuple[str, str]]:
        """
        【トピックロック】
        2つのキューブが、意味的に非常に近いMiniCubeを持っているか（＝共通の話題か）を判定する。
        """
        for mc1 in cube1.mini_cubes:
            for mc2 in cube2.mini_cubes:
                # MiniCubeのベクトル同士の類似度で判定！
                if self._cosine_similarity(mc1.embedding, mc2.embedding) > self.topic_threshold:
                    # 閾値を超えたら、その時点で「共通トピックあり」と見なす
                    return (mc1.phrase, mc2.phrase) # マッチしたキーワードのペアを返す
        return None

    def detect_tension(self, search_results: List[SearchResult]) -> TensionResult:
        """
        SRAGの結果を分析し、意味の張力（Tension）を検知する。
        """
        if len(search_results) < 2:
            return TensionResult(has_tension=False)

        cubes = [res.cube for res in search_results]
        
        min_similarity_found = 1.0
        most_divergent_pair = (None, None)
        shared_topic_phrases = ("", "")

        # 1. キューブの全ペアを総当たり
        for i in range(len(cubes)):
            for j in range(i + 1, len(cubes)):
                cube_a = cubes[i]
                cube_b = cubes[j]
                
                # 2. 【トピックロック】まず、2つが同じ話題について語っているか？
                topic_match = self._find_shared_topic(cube_a, cube_b)
                
                if topic_match:
                    # 3. 【張力測定】同じ話題なら、本体ベクトルの類似度を測る
                    main_similarity = self._cosine_similarity(cube_a.embedding_vector, cube_b.embedding_vector)
                    
                    # 4. 最も「似ていない（そっぽを向いている）」ペアを探す
                    if main_similarity < min_similarity_found:
                        min_similarity_found = main_similarity
                        most_divergent_pair = (cube_a, cube_b)
                        shared_topic_phrases = topic_match

        # 5. 最小類似度が閾値を下回っていれば、「葛藤（Tension）」と判定
        if min_similarity_found < self.tension_threshold:
            tension_score = 1.0 - min_similarity_found
            print(f"    ⚡ Tension Detected! Topic: '{shared_topic_phrases[0]}' vs '{shared_topic_phrases[1]}'")
            print(f"      Min similarity: {min_similarity_found:.3f} (Tension Score: {tension_score:.3f})")
            
            return TensionResult(
                has_tension=True,
                tension_score=tension_score,
                # topic=f"{shared_topic_phrases[0]} <-> {shared_topic_phrases[1]}",
                divergent_summaries=[most_divergent_pair[0].summary, most_divergent_pair[1].summary]
            )
        
        return TensionResult(has_tension=False)

