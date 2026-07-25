# llm_service.py (v2 Embedding統合版)

import os
import time
import numpy as np
import google.generativeai as genai
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union # Unionを追加

# -----------------------------------------------------
# 1. APIからの応答を格納する標準データクラス
# -----------------------------------------------------
@dataclass
class LLMResult:
    """LLMからの応答を標準化するデータクラス"""
    http_code: int
    reason: str
    # generateはstr、embedはnp.ndarrayを返すため、Unionで両対応
    data: Union[str, np.ndarray] 

# -----------------------------------------------------
# 2. LLMProviderインターフェース (embedメソッド追加)
# -----------------------------------------------------
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResult:
        """プロンプトからテキストを生成する"""
        pass

    @abstractmethod
    def embed(self, text: str) -> LLMResult:
        """テキストから意味ベクトルを生成する"""
        pass

# -----------------------------------------------------
# 3. GeminiProvider (embedメソッド実装)
# -----------------------------------------------------
class GeminiProvider(LLMProvider):
    def __init__(self, llm_model_name: str = None, embed_model_name: str = None):

        self.llm_model_name = llm_model_name or os.environ.get("GEMINI_LLM_MODEL", "gemini-3.5-flash")
        self.embed_model_name = embed_model_name or os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
        
        try:
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            self.llm_model = genai.GenerativeModel(self.llm_model_name)
            # Embeddingモデルは都度指定するため、ここでは初期化しない
            # print(f"✅ GeminiProvider initialized. LLM: {self.llm_model_name}, Embedding: {self.embed_model_name}")
        except Exception as e:
            print(f"🚨 FATAL ERROR: Geminiの初期化に失敗しました: {e}")
            self.llm_model = None

    def generate(self, prompt: str) -> LLMResult:
        """テキスト生成APIを叩く"""
        if not self.llm_model:
            return LLMResult(500, "Initialization Failed", "LLMモデルが初期化されていません。")
        try:
            time.sleep(float(os.environ["API_CALL_WAIT_TIME"]))  # レート制限対策
            response = self.llm_model.generate_content(prompt)
            return LLMResult(200, "OK", response.text.strip())
        except Exception as e:
            return LLMResult(500, str(e), "LLM呼び出しに失敗しました。")


    def embed(self, text: str) -> LLMResult:
        """【新機能】Embedding APIを叩く"""
        try:
            time.sleep(float(os.environ["API_CALL_WAIT_TIME"])) # API制限対策
            res = genai.embed_content(model=self.embed_model_name, content=text, task_type="retrieval_document")
            return LLMResult(200, "OK", np.array(res["embedding"]))
        except Exception as e:
            print(f"⚠️ [API Error] Embedding failed: {e}. Using dummy vector.")
            # エラー時はダミーのベクトルを返す
            dummy_vector = np.random.randn(3072)
            return LLMResult(500, str(e), dummy_vector)

# -----------------------------------------------------
# 4. LLMService (embed_textメソッド追加)
# -----------------------------------------------------
class LLMService:
    def __init__(self, provider: LLMProvider):
        # ... (実装は同じ) ...
        self.provider = provider
        # print(f"✅ LLMService activated with provider: {type(provider).__name__}")

    def call_api_generate(self, prompt: str) -> LLMResult:
        """テキスト生成を呼び出す"""
        # print(f"📡 LLMService: Sending 'generate' request to {type(self.provider).__name__}...")
        return self.provider.generate(prompt)

    def call_api_embed(self, text: str) -> LLMResult:
        """【新機能】ベクトル埋め込みを呼び出す"""
        # print(f"📡 LLMService: Sending 'embed' request to {type(self.provider).__name__}...")
        return self.provider.embed(text)
