# cube_engine.py (v7.1 完全整合版)

import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple

# 外部サービスとデータ型をインポート
from llm_service import LLMService
from common_types import SemanticCube, MiniCube, CubeRole, CubeSource, PhraseRole

# -----------------------------------------------------
# 1. 動的プロジェクター：DynamicProjector
# -----------------------------------------------------
class DynamicProjector:
    """
    PCAと動的スケーラーを組み合わせ、3段階の解像度で座標を生成する。
    """
    def __init__(self, embedding_dim=3072, n_components=3):
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.is_fitted = False
        print("🔧 DynamicProjector initialized. Waiting for fitting...")

    def fit(self, data: np.ndarray):
        """
        与えられたデータ群全体から、PCAとスケーラーを「学習」する。
        """
        if data.shape[0] < self.n_components:
            print("⚠️ Warning: Not enough data to fit PCA. Skipping.")
            return
            
        # print(f"⚙️ Fitting DynamicProjector with {data.shape[0]} samples...")
        self.pca_model = PCA(n_components=self.n_components)
        projected_data = self.pca_model.fit_transform(data)
        
        # ✨ [0.0, 1.0]の範囲で正規化するスケーラーを学習
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(projected_data)
        
        self.is_fitted = True
        print("✅ DynamicProjector is fitted and ready.")

    def project(self, embedding_vector: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        """
        1つのベクトルを受け取り、3段階の解像度で座標を返す。
        """
        if not self.is_fitted:
            # 学習前はダミー値を返す
            vec_int8 = np.zeros(self.embedding_dim, dtype=int)
            return vec_int8, (0,0,0), (0,0,0), (0,0,0)

        # --- [vec_int8 の生成] --- (255階調)
        clipped_vec = np.clip(embedding_vector, -3.0, 3.0)
        norm_vec = (clipped_vec + 3.0) / 6.0
        vec_int8 = np.floor(norm_vec * 255).astype(int)
        vec_int8 = np.clip(vec_int8, 0, 255)

        # --- [3つのグリッドを同時生成] ---
        coords_3d = self.pca_model.transform([embedding_vector])[0]
        norm_coords = self.scaler.transform([coords_3d])[0]
        norm_coords = np.clip(norm_coords, 0.0, 0.9999999)

        # 1. int3_grid: 8階調 [0-7]
        int3_grid = tuple((norm_coords * 8).astype(int))
        # 2. int8_grid: 256階調 [0-255]
        int8_grid = tuple((norm_coords * 256).astype(int))
        # 3. grid_mesh: 10000階調 [0-9999]
        grid_mesh = tuple((norm_coords * 10000).astype(int))

        return vec_int8, int3_grid, int8_grid, grid_mesh
    
# -----------------------------------------------------
# 2. CubeEngine 本体 (v7.1)
# -----------------------------------------------------
class CubeEngine:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.projector = DynamicProjector()
        print("✅ CubeEngine initialized.")
        
    def fit_projector(self, initial_vectors: np.ndarray):
        """プロジェクターをデータで学習させる"""
        self.projector.fit(initial_vectors)

    def _embed_text(self, text: str) -> np.ndarray:
        """LLMServiceにベクトル化を依頼する"""
        # print(f"    - Embedding text: '{text[:30]}...'")
        result = self.llm_service.call_api_embed(text)
        return result.data

    def _extract_keyphrases(self, text: str) -> List[Dict]:
        """LLMServiceにキーワード抽出を依頼する"""
        # print(f"    - Extracting keyphrases from: '{text[:20]}...'")

        # ここで作成するプロンプトは、SRAGの要になるヒット率に関係する。
        # role : genre を定義して、これは何の話題かを付与することで、SRAGが周辺記憶を検知できるようにする。
        # 相槌や挨拶など、特定ジャンルを持たないものはgenre1ロールは無しとする。
        # 従来通りの「主語・述語・名詞句」の抽出は通常通り行う。
        # これはSRAGヒット率の下限を担保する目的。

        prompt = f"""【あなたへのオーダー】
以下のテキストを解析して、代表的な「主語」と「述語」、それから重要な名詞や概念を表すフレーズを抽出してください。
抽出した各フレーズに対して、0.0〜1.0の範囲で信頼度を付与してください。

また、その会話が何のジャンルについての会話なのかを推測して、role : genre を付与してください。
ジャンルが特定できない場合は、role : genre は無しとしてください。

- 例題１：　『今日は友達と久しぶりにボールを蹴りに行った。PK合戦をやって負けた。』この文章なら
-  {{"phrase": "サッカー", "role": "genre", "confidence": 0.5}}の様になります。
- 例題２：　[偏向報道, 責任転嫁, メディア, 政治家, 影響力] ➔ "Social"などです。
- 例題３：　[アーキテクチャ, デザインパターン, ソフトウェア, システムテスト, 技術検証] ➔ "architecture"などです。
- 例題４：　[うんうん, こんにちは, そうですね, なるほど] ➔ role : genre は無しになります。

【回答方法のオーダー】
抽出結果は以下の「回答フォーマット」を参考にJSON形式のみで回答してください。

【回答フォーマット】
[
  {{"phrase": "推測した単語", "role": "genre", "confidence": 0.65}}
  {{"phrase": "抽出した単語", "role": "subject", "confidence": 0.95}},
  {{"phrase": "抽出した単語", "role": "predicate", "confidence": 0.90}},
  {{"phrase": "抽出した単語", "role": "key_phrase", "confidence": 0.85}}
]

【roleの意味】
- "genre": 会話のジャンル（例：スポーツ、政治、科学など）
- "subject": 主語（何が、誰が）
- "predicate": 述語（どうする、どんなだ、何だ）
- "key_phrase": その他の重要な名詞や概念フレーズ
    
【解析対象のテキスト】
{text}
"""
        result = self.llm_service.call_api_generate(prompt)
        
        if result.http_code == 200:
            try:
                response_text = result.data
                if response_text.startswith("```json"): response_text = response_text[7:]
                if response_text.startswith("```"): response_text = response_text[3:]
                if response_text.endswith("```"): response_text = response_text[:-3]
                return json.loads(response_text.strip())
            except Exception as e:
                print(f"⚠️ [JSON Parse Error] Keyphrase parsing failed: {e}")
                return [{"phrase": text[:20], "role": "key_phrase", "confidence": 0.1}]
        else:
            return [{"phrase": text[:20], "role": "key_phrase", "confidence": 0.1}]


    def make_cube(self, query: str, role: CubeRole, source: CubeSource, current_turn: int) -> SemanticCube:
        """
        ベクトル化から3段階の座標生成までを、一気通貫で行う。
        """
        print(f"\n🧊 CubeEngine: Starting 'make_cube' for '{query[:50]}...'")
        
        # 1. キーワード抽出
        keywords = self._extract_keyphrases(query)

        # 2. メインベクトル生成
        main_vector = self._embed_text(query)

        # 3. 【最重要】プロジェクターで4つの座標データを一括生成！
        vec_int3, int3_grid, int8_grid, grid_mesh = self.projector.project(main_vector)

        # 4. MiniCube群の生成
        mini_cubes = []
        for kw_data in keywords:
            phrase = kw_data.get("phrase", "")
            phrase_role = kw_data.get("role", "key_phrase")
            confidence = float(kw_data.get("confidence", 0.5))
            print(f"    - keyphrase: '{phrase}'  role: '{phrase_role}'  with confidence {confidence}")

            if not phrase: continue
            # キーワード単体でEmbedding APIを叩く！
            kw_vector = self._embed_text(phrase)
            mc_vec_int3, mc_int3_grid, mc_int8_grid, mc_grid_mesh = self.projector.project(kw_vector)
            
            mini_cubes.append(MiniCube(
                phrase=phrase,
                role=phrase_role,
                embedding=kw_vector,
                vec_int3=mc_vec_int3,
                int3_grid=mc_int3_grid,
                int8_grid=mc_int8_grid,
                grid_mesh=mc_grid_mesh,
                confidence=confidence
            ))

        # 5. 最終的なキューブを組み立てる
        new_cube = SemanticCube(
            role=role,
            source=source,
            summary=query,
            embedding_vector=main_vector,
            vec_int3=vec_int3,
            int3_grid=int3_grid,
            int8_grid=int8_grid,
            grid_mesh=grid_mesh,
            mini_cubes=mini_cubes,
            create_turn=current_turn,
            last_used_turn=current_turn
        )
        
        print(f"✅ CubeEngine: Cube '{new_cube.cube_id}' created successfully.")
        print(f"    - int3 Grid : {new_cube.int3_grid}")
        print(f"    - Fine Grid : {new_cube.int8_grid}")
        print(f"    - Grid Mesh :  {new_cube.grid_mesh}")

        return new_cube
