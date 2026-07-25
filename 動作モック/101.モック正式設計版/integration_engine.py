# integration_engine.py (v2.0 SRAG連動版)

from llm_service import LLMService
from common_types import SemanticCube, SearchResult, TensionResult
from typing import List

class IntegrationEngine:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        print("✅ IntegrationEngine (v2.0) initialized.")

    def single(self, query_cube: SemanticCube, history: List[SemanticCube]) -> str:
        """
        SRAGが何も見つけられなかった時に呼ばれる。
        """
        # print("    -> Executing 'single' route (No SRAG context).")
        
        # 直近の対話履歴をプロンプト用に整形
        history_text = "\n".join([f"- {h.source}: {h.create_datetime} : {h.summary}" for h in history])
        
        prompt = f"""ユーザーとの自然な対話をしてください。
以下の【直近の対話履歴】の流れを踏まえながらユーザーの入力プロンプトに回答してください。
また【直近の対話履歴】に何も含まれていない場合は新規スレッドでの会話となります。

【直近の対話履歴】
{history_text}

【ユーザーの入力プロンプト】
{query_cube.summary}

【メタデータ】
 - {query_cube.create_datetime}
"""
        # デバッグ表示
        print( "--------------------------------" )
        print(f" -> Prompt for 'single':\n{prompt}\n")
        print( "--------------------------------" )

        result = self.llm_service.call_api_generate(prompt)
        return result.data if result.http_code == 200 else "（エラー）"

    def integration(self, query_cube: SemanticCube, search_results: List[SearchResult], history: List[SemanticCube] , tension_meta: TensionResult = None, conflict_detect_switch: bool = True) -> str:
        """
        【記憶参照モード】
        SRAGが1件以上の根拠を見つけてきた時に呼ばれる。
        """
        print(f"    -> Executing 'integration' route with {len(search_results)} SRAG results.")
        
        # SRAGの検索結果をプロンプト用に整形
        srag_context = ""
        for i, res in enumerate(search_results):
            matched_phrases = ", ".join([m.target_phrase for m in res.matched_minicubes])
            srag_context += f"【根拠データ{i+1} (Score: {res.final_score:.3f})】\n"
            srag_context += f" - 【過去の対話内容】:【 {res.cube.summary}】\n"
            srag_context += f" - 【関連キーワード】: [{matched_phrases}]\n\n"

        history_text = "\n".join([f"- {h.source}: {h.create_datetime} : {h.summary}" for h in history])
        
        prompt = ""

        # ここにifを挟んで、葛藤検知の有無をチェック
        if conflict_detect_switch and tension_meta and tension_meta.has_tension:

            # --------------------------------------------------
            # 葛藤検知っぽいプロンプトテンプレート使用 (Tension Mode)
            # --------------------------------------------------
            print(f"    ⚡ Tension routing activated! (Score: {tension_meta.tension_score:.3f})")

            tension_info = f"⚡【意見の対立情報】\n - 張力スコア: {tension_meta.tension_score:.3f}\n"
            divergent_summaries = "\n".join([f"- {s}" for s in tension_meta.divergent_summaries])
            tension_info += f"【異なる意見のサマリー】\n{divergent_summaries}\n"
            
            # 葛藤検知時のプロンプトテンプレート
            prompt = f"""ユーザーと自然な対話をしてください。
以下の【関連する過去の記憶】をヒントとして参照しながら、
【直近の対話履歴】の流れを踏まえてユーザーの入力プロンプトに解答してください。
また、以下の【意見の対立情報】のように幾つかの意見が葛藤を示している可能性があるため、
意見をまとめず、異なる視点を尊重しつつ回答を作成してください。

【関連する過去の記憶（検索結果）】
{srag_context}

{tension_info}

【直近の対話履歴】
{history_text}

【ユーザーの入力プロンプト】
{query_cube.summary}

【メタデータ】
- {query_cube.create_datetime}
"""
        # 葛藤検知無しのプロンプトテンプレート
        else:
            prompt = f"""ユーザーと自然な対話をしてください。
以下の【関連する過去の記憶】をヒントとして参照しながら、
【直近の対話履歴】の流れを踏まえてユーザーの入力プロンプトに解答してください。

【関連する過去の記憶（検索結果）】
{srag_context}

【直近の対話履歴】
{history_text}

【ユーザーの入力プロンプト】
{query_cube.summary}

【メタデータ】
- {query_cube.create_datetime}
"""
        
        # デバッグ表示
        print( "--------------------------------" )
        print(f" -> Prompt for 'integration':\n{prompt}\n")
        print( "--------------------------------" )

        result = self.llm_service.call_api_generate(prompt)
        return result.data if result.http_code == 200 else "（エラー）"

