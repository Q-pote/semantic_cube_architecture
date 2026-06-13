# runtime.py 完全版（v3.5 対話イベントループテスト用）

import os
from common_types import SemanticCube, MiniCube
from srag import srag_search
from cube_factory import make_cube, add_Response, embed_text, quantize_embedding
from kvhub import KVHub
from partial_engine import make_partial, call_llm
from integration_engine import integrate_responses
from logger import print_log, export_cubes_to_json
from defrag_engine import run_defrag_cycle


def simulate_user_turn(kv: KVHub, user_text: str, model: str):
    """
    【v3.6 フロントエンド・ライフサイクル】
    SRAGによる軽量なプロンプトパッキングと、1回のLLM呼び出し。
    """
    print("\n" + "="*60)
    print(f"👤 [Turn {kv.global_turn}] User: {user_text}")
    print("-" * 60)
    
    # 1. ユーザー入力からセマンティックキューブ生成 (MiniCube埋め込み含む)
    q_cube = make_cube(user_text, "topic", kv.global_turn, 1.0, model)
    
    # 2. SRAG検索 ＆ イベント代謝
    res = kv.search_by_trust(q_cube, k=3)
    
    # 3. 直近5ターン (previous_5) のコンテキストを取得
    recent_contexts = []
    for cid in kv.previous_5:
        # KVHub内の全キューブ（アーカイブ含む）からIDで検索
        for c in kv.cubes: 
            if c.cube_id == cid:
                recent_contexts.append(f"Q: {c.summary}\nA: {c.response_and_answer}")
                break
    
    recent_text = "\n".join(recent_contexts) if recent_contexts else "なし"
    
    # 4. 検索でヒットした候補キューブの情報をテキスト化（既存の回答を再利用！）
    candidates_text = ""
    if res:
        blocks = []
        for i, (c, s) in enumerate(res):
            print(f"  🔍 参照キューブ: [{c.cube_id[:8]}] (α={c.get_alpha(kv.global_turn):.2f}) {c.summary[:20]}...")
            kp = ", ".join([mc.phrase for mc in c.mini_cubes]) if c.mini_cubes else "None"
            block = f"[{i+1}] (ID: {c.cube_id})\n"
            block += f"  過去の質問: {c.summary[:20]}\n"
            block += f"  過去の回答: {c.response_and_answer[:20]}\n"
            block += f"  関連スコア: {s:.3f}, キーフレーズ: [{kp}]"
            blocks.append(block)
        candidates_text = "\n\n".join(blocks)
    else:
        candidates_text = "関連する過去の記憶は見つかりませんでした。"

    # 5. LLMへの1回のプロンプト送信（SRAGフォーマット）
    prompt = f"""あなたは賢明なAIアシスタントです。以下の「直近の会話の流れ」と「関連する過去の記憶」を踏まえて、ユーザーの最新の質問に答えてください。

【直近の会話の流れ (Recent Context)】
{recent_text[:50]}

【関連する過去の記憶 (Semantic Memories)】
{candidates_text[:50]}

【ユーザーの最新の質問】
{user_text}
"""
    
    final_ans = call_llm(prompt, model)
        
    print(f"\n🤖 [LLM Response]:\n{final_ans[:200]}")
    
    # 6. LLMからの回答をキューブに埋め込み
    add_Response(q_cube, final_ans)
    
    # 7. KVHubへ登録 (previous_5への入れ替えも内部で実行)
    kv.put(q_cube)
    
    # 8. global_tick のインクリメント (宇宙の時間を1つ進める)
    kv.advance_turn()
    
    print("-" * 60)
    print(f"📊 [KVHub Status] 次のターン: {kv.global_turn}")
    print(f"📊 [Active Cubes]: {len(kv.get_all())} 個 (実効α>=0.01)")
    print(f"📊 [Previous 5]: {[cid[:8] for cid in kv.previous_5]}")
    print("="*60)

def run_semantic_cube_runtime():
    model = os.environ.get("GEMINI_LLM_MODEL", "")
    kv = KVHub()

    print("🌍 Semantic Cube Runtime v3.5 (Event-Driven Metabolism) 起動...")
    
    # 対話ループのシミュレーション
    dialogues = [
        # --- 序盤：リモートワークと日常（既存の6ターン） ---
        "リモートワークは通勤時間がなくて最高だね。朝の満員電車に押し込まれるストレスがなくなるだけで、1日の始まりが全然違う気がするよ。",
        "でも、オフィスで雑談からアイデアが生まれることも多いよ。ちょっとした立ち話がそのままプロジェクトの方向性を変えることもあるから、完全リモートにはまだ迷いがあるんだ。",
        "今日のランチは近所の定食屋で唐揚げ定食を食べた。揚げたてでサクサクしてて、レモンをかけたらさらに美味しくて、午後の仕事のやる気が少し戻った気がするよ。",
        "オンライン会議ツールがもっと進化すれば、完全リモートでもいけそう。音声の遅延とか、ホワイトボード機能の使い勝手が改善されたら、対面との差はほとんどなくなると思うんだ。",
        "最近読んだSF小説の結末がどうしても納得いかない。途中までは世界観も設定も完璧だったのに、最後だけ急に駆け足になった感じがして、読み終わったあとモヤモヤが残ってるんだよね。",
        "カレーの隠し味にはやっぱりチョコレートが一番だよね。ほんの少し入れるだけでコクが深くなって、家庭の味が一段階プロっぽくなるから、つい毎回入れちゃうんだ。",

        # --- 中盤：新しい話題（天気・靴）と、カレーの「葛藤（自己矛盾）」の種まき ---
        "明日の天気は雨らしいね。せっかくの休日なのに出かける気が失せるなぁ。",
        "そういえば、革靴って本当に雨に弱いよね。お気に入りの靴が雨で濡れて、内側のボンディングが剥がれちゃって最悪だったよ。",
        "靴のメンテナンスって大事だよね。防水スプレーを忘れた自分を呪いたい。",
        "実はさ、俺の中で「カレーは飲み物だ」っていう認識が強くなってきたんだ。あのスパイシーな液体をごくごく喉に流し込む感覚、たまらないよね。",
        "でもやっぱり、ゴロゴロした野菜や肉をしっかり噛み締めて食べるのがカレーの醍醐味だから、食べ物派に戻ろうかな……。飲み物って言うのはちょっと極端すぎたかも。",
        
        # --- 後盤：コーラの宗教戦争（対立意見）とノイズ ---
        "コーラといえば、やっぱり元祖のコカ・コーラ（赤）一択だよね。あのガツンとくる甘さと炭酸の強さは、他じゃ絶対に味わえない。",
        "いやいや、コカ・コーラ ゼロ（黒）の方が優秀でしょ。カロリーを気にせずあの味を楽しめるんだから、現代人の最適解だよ。",
        "二人とも分かってないな。ペプシコーラ（青）の爽やかな後味と、少しだけ柑橘系の香りがする絶妙なバランスこそが最強なんだって。",
        "あーあ、急に雨が降ってきた。さっき天気予報見たのに傘持ってこなかったよ。",
        "ペプシもいいけど、やっぱり映画館で飲む赤コーラには敵わないなぁ。",

        # --- 終盤：過去の記憶の引き出し（SRAGテスト）と単発ノイズ ---
        "そういえば、前にリモートワークの話したけど、やっぱりコミュニケーション不足は課題だよね。",
        "今日の夕飯、また唐揚げにしようかな。レモン買い忘れないようにしないと。",
        "あはは、確かに。", # ← 超単発ノイズ（GC対象になるかテスト）
        "SF小説って、やっぱり結末のオチが綺麗に決まらないと名作とは呼べないよね。モヤモヤする作品は二度と読まないよ。"
    ]

    
    for text in dialogues:
        simulate_user_turn(kv, text, model)
        
    # ※ この後、次のターンで「デフラグアルゴリズムの修正(v5.0)」を呼び出す
    defrag_result = run_defrag_cycle(kv, model)

    kv = defrag_result  # デフラグ後のKVHubを取得して上書き

    print("\n" + "="*60)
    print("🧹 デフラグサイクル完了  ---  テスト終了")
    # print(f"デフラグ済みキューブの状態：{ export_cubes_to_json(kv.get_all()) }")

if __name__ == "__main__":

    model = os.environ.get("GEMINI_LLM_MODEL", "")
    kv = KVHub()

    print("\n" + "="*60)
    print("🚀 Operation Start: Testing End-to-End Flow (v1.1)...")
    print("="*60 + "\n")
    
    # --- ✨【修正点】ダミーDBをmake_cube_prototypeで生成 ---
    print("📚 Creating spatially consistent in-memory database...")
    db_cubes = []
    dummy_texts = [
        "リモートワークは生産性を高める。", "オフィスの雑談からイノベーションが生まれる。",
        "在宅勤務は集中できる環境が大事。", "唐揚げ定食はレモンをかけると美味しい。",
        "SF小説は結末が肝心だ。", "カレーの隠し味はチョコレートが良い。",
        # クエリと意図的に近いデータ
        "自宅での作業効率を上げる方法", "新しい働き方としてのテレワーク"
    ]
    for text in dummy_texts:
        # 司令官の指示通り、DBの各要素もmake_cubeで生成する
        db_cubes.append(make_cube(text, "topic", kv.global_turn))
    print(f"✅ {len(db_cubes)} dummy cubes created with consistent quantization.")

    # --- クエリを生成 (こちらも同じ関数で) ---
    print("\n" + "--- Query Generation ---")
    query_text = "在宅での働き方について教えて"
    query_cube = make_cube(query_text, "topic", kv.global_turn)
    print(f"✅ Query cube created for: '{query_text}'")
    
    # --- 標準モードで検索実行 ---
    standard_results = srag_search(query_cube, db_cubes, mode="standard")
    
    # --- ✨【追加】make_partialで最終回答を生成 ---
    if standard_results:
        print("\n--- End-to-End Flow: Standard Search & Partial Response ---")
        top_result = standard_results[0]
        print(f"  - SRAG Top Hit: [{top_result.final_score:.3f}] {top_result.cube.summary}")
        
        # 最終回答の生成
        final_answer = make_partial(query_text, top_result.cube, top_result.final_score, model)
        
        print(f"\n✨ Final Answer Simulation:\n   '{final_answer}'")
    else:
        print("\n--- Standard Search returned no results. ---")


    print("\n✅ Operation Phase 3 Complete. The system is naturally well-formed.")
