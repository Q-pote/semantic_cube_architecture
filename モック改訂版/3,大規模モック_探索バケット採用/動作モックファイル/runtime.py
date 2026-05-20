"""
runtime.py
Semantic Cube Runtime v2.1 (生命的代謝モデル統合版) メインコントローラ
Test1〜Test5 のシナリオを、1つの KVHub (世界) の中で連続して実行する。
※ v2.1: 参照ボーナス、新デフラグアルゴリズム、GC(忘却)、スレッドモード切り替えを統合。
"""

import os
from cube_factory import make_cube
from kvhub import KVHub
from partial_engine import make_partial
from integration_engine import integrate_responses
from conflict_engine import determine_route_by_physics, analyze_conflict
from defrag_engine import defrag_cluster
from logger import export_cubes_to_json, print_log

def run_semantic_cube_runtime():
    model = os.environ.get("GEMINI_LLM_MODEL", "")
    kv = KVHub()
    turn = 100

    print("==================================================")
    print("🌍 Semantic Cube Runtime v2.1 世界線の立ち上げ...")
    print("==================================================\n")

    # ------------------------------------------------
    # 🚀 Test 1: 代表選抜（初期キャリブレーション）
    # ------------------------------------------------
    print_log("Test 1", "代表選抜（初期キャリブレーション）開始")
    kv.thread_mode = "smart" # Smartモード（代謝あり）で開始
    
    o1 = make_cube("リモートワークは通勤時間を削減し、生産性を高める働き方である。", "origin", 1, 1.0, model)
    kv.put(o1)
    
    h1_texts = [
        "在宅勤務は、自分に合った環境を整えられるため集中しやすい。",
        "通勤ストレスがないことで、仕事以外の時間も充実しやすい。",
        "オンライン会議ツールの発達により、リモートでも十分に協働できる。",
        "静かな環境で作業できるため、集中力が途切れにくい。",
        "移動時間がゼロになることで、1日の可処分時間が増える。",
        # --- 追加: 完全に無関係なノイズ（テスト用） ---
        "今日のランチは近所の定食屋で唐揚げ定食を食べた。",
        "明日の天気は午後から雨が降るらしいから傘が必要だ。",
        "最近読んだSF小説の結末がどうしても納得いかない。",
        "休日は一日中ゲームをして過ごすのが最高の贅沢だ。",
        "新しいスニーカーを買ったけど、靴擦れして痛い。",
        # --- 追加: 短い相槌や意味のない文脈（テスト用） ---
        "なるほど、確かにそうですね。",
        "へえ、それは面白い。",
        "ふむふむ。",
        "ちょっと待って。",
        "それはどうかなぁ。"
    ]
    for t in h1_texts:
        kv.put(make_cube(t, "history", turn-5, 1, model))
    
    q1 = make_cube("リモートワークのメリットを、経営者の視点から整理してほしい。", "history", turn, 1, model)

     # Test1時点での空間分布とMiniCube情報をJSON出力
    print("\n📊 [JSON Export Test1]")
    print(export_cubes_to_json(kv.get_all(), q1))   
 

    # 検索実行（ここで参照ボーナスが付与され、αが更新される）
    res1 = kv.search_by_trust(q1, turn, k=1)

    print(f"  👉 Top1 選抜: [{res1[0][0].role}] Trust:{res1[0][1]:.3f} | {res1[0][0].summary}...")
    
    # ------------------------------------------------
    # 🚀 Test 2: 順次回答 → 統合／葛藤分岐 (高トラスト)
    # ------------------------------------------------
    print("\n" + "-"*50)
    print_log("Test 2", "順次回答 → 統合／葛藤分岐 開始")
    
    o2 = make_cube("オフィスでの対面コミュニケーションは、新しいアイデアを生み出す土壌となる。", "origin", 1, 1.0, model)
    kv.put(o2)
    for t in ["対面での議論は意思決定のスピードが上がる。", "部署をまたいだ出会いがコラボレーションを生む。"]:
        kv.put(make_cube(t, "history", turn-5, 1, model))

    kv.mc_strictness = "standard"
    print(f"  💡 検索設定: [MiniCube語義一致度: {kv.mc_strictness.upper()}]")

    q2 = make_cube("生産性とイノベーションの両面から方針を提案して。", "history", turn, 1, model)
    res2 = kv.search_by_trust(q2, turn, k=4)
    
    print(f"  👉 抽出 Top4:")
    partials2 = []
    for i, (c, s) in enumerate(res2):
        print(f"     {i+1}. [{c.role}] Hybrid Trust:{s:.3f} (α:{c.trust.alpha:.2f}) | {c.summary[:50]}...")
        p_ans = make_partial(q2.summary, c, s, model)
        partials2.append((c.cube_id, p_ans, s))
    
    is_conf2 = determine_route_by_physics([c for c, s in res2])
    print(f"  👉 物理的葛藤検知: {is_conf2}")
    
    final_ans2 = analyze_conflict(partials2, model) if is_conf2 else integrate_responses(partials2, model)
    print(f"  👉 最終回答 (Test2):\n{final_ans2}...\n")

    # ------------------------------------------------
    # 🚀 Test 3: 高トラスト葛藤（カレー戦争）
    # ------------------------------------------------
    turn += 10
    print("\n" + "-"*50)
    print_log("Test 3", "高トラスト葛藤（カレー戦争）開始")
    
    # ※カレー戦争のような「重要なテーマ」と仮定して Protected モードに切り替え
    kv.thread_mode = "protected"
    print("  🔒 スレッドモードを [Protected] に変更。代謝(GC)を一時停止します。")
    
    o3a = make_cube("カレーのおいしさは、ルーの一体感と口当たりの滑らかさに宿る。", "origin", turn, 1.0, model)
    o3b = make_cube("カレーのおいしさは、具材の食感と素材の旨味に宿る。", "origin", turn, 1.0, model)
    kv.put(o3a); kv.put(o3b)
    
    for t in ["カレーの魅力は、滑らかなルーが生む一体感にある。忙しい時でも、飲むように味わえる手軽さが“おいしさ”の一部を形作る。", 
              "カレーの魅力は、具材を噛むことで広がる多層的な風味にある。素材の旨味がしっかり残ることが“おいしさ”の核になる。"]:
        kv.put(make_cube(t, "history", turn, 1, model))
    
    kv.mc_strictness = "fuzzy"
    print(f"  💡 検索設定: [MiniCube語義一致度: {kv.mc_strictness.upper()}]")
    
    q3 = make_cube("カレーの“本質的なおいしさ”って何だと思う？具材？ルー？それとも別の要素？", "history", turn, 1, model)
    res3 = kv.search_by_trust(q3, turn, k=4)
    
    print(f"  👉 抽出 Top4:")
    partials3 = []
    for i, (c, s) in enumerate(res3):
        print(f"     {i+1}. [{c.role}] Hybrid Trust:{s:.3f} (α:{c.trust.alpha:.2f}) | {c.summary[:50]}...")
        partials3.append((c.cube_id, make_partial(q3.summary, c, s, model), s))
        
    is_conf3 = determine_route_by_physics([c for c, s in res3])
    print(f"  👉 物理的葛藤検知: {is_conf3}")
    
    final_ans3 = analyze_conflict(partials3, model) if is_conf3 else integrate_responses(partials3, model)
    print(f"  👉 最終メタ判断 (Test3):\n{final_ans3}...\n")

    # ------------------------------------------------
    # 🚀 Test 4: 低トラスト葛藤（採用保留・却下）
    # ------------------------------------------------
    turn += 10
    print("\n" + "-"*50)
    print_log("Test 4", "低トラスト葛藤（ノイズ判定）開始")
    
    # 日常会話に戻ったとして Smart モードに戻す
    kv.thread_mode = "smart"
    print("  💡 スレッドモードを [Smart] に戻しました。代謝(GC)が有効になります。")
    
    kv.put(make_cube("カレーの隠し味にはチョコレートが良い。", "history", turn, 1, model))
    kv.put(make_cube("カレーを食べる時はスプーンよりフォークだ。", "history", turn, 1, model))
    
    q4 = make_cube("働き方の未来における、カレーのスパイスの役割とは？", "history", turn, 1, model)
    res4 = kv.search_by_trust(q4, turn, k=3) # ここでノイズ群が参照される
    
    print(f"  👉 抽出 Top3 (ノイズ想定):")
    partials4 = []
    for i, (c, s) in enumerate(res4):
        print(f"     {i+1}. [{c.role}] Hybrid Trust:{s:.3f} (α:{c.trust.alpha:.2f}) | {c.summary[:50]}...")
        partials4.append((c.cube_id, make_partial(q4.summary, c, s, model), s))
        
    is_conf4 = determine_route_by_physics([c for c, s in res4])
    print(f"  👉 物理的葛藤検知: {is_conf4}")
    
    final_ans4 = analyze_conflict(partials4, model) if is_conf4 else integrate_responses(partials4, model)
    print(f"  👉 最終メタ判断 (Test4):\n{final_ans4}...\n")

    # ------------------------------------------------
    # 🚀 Test 5: 新8ステップ・デフラグ ＆ ガベージコレクション 開始
    # ------------------------------------------------
    turn += 50
    print("\n" + "-"*50)
    print_log("Test 5", "新8ステップ・連続デフラグ ＆ ガベージコレクション 開始")
    
    # 1. 忘却処理(GC)の実行（まずはゴミを消す）
    kv.garbage_collection()

    # ✨ 【追加】複数クラスタの連続デフラグ処理
    max_defrag_loops = 5 # 無限ループ防止のための上限
    loop_count = 0
    
    while loop_count < max_defrag_loops:
        # デフラグの対象になり得る（まだ代替されていない）Historyを探す
        eligible_cubes = [
            c for c in kv.get_all() 
            if c.role == "history" and c.trust.replacement_closeness < 0.5
        ]
        
        # 対象が2個未満（もうまとめるべきクラスタがない）なら終了
        if len(eligible_cubes) < 2:
            print(f"\n  👉 デフラグ可能なクラスタがなくなったため、連続処理を終了します。")
            break
            
        loop_count += 1
        print(f"\n🌀 [DefragEngine] --- 連続デフラグ 第 {loop_count} サイクル ---")
        
        # ⚠️ 注意: defrag_cluster 内部の seed 抽出も、eligible_cubes から選ぶように
        # 引数として eligible_cubes だけを渡すか、内部ロジックを少し調整する必要がある。
        # ここではモックとして、kv.get_all() を渡し、内部で replacement_closeness をチェックするようにする。
        new_origin = defrag_cluster(kv.get_all(), turn, model, strictness="standard") # aggressive, standard, strict から選ぶ
        
        if new_origin:
            kv.put(new_origin)
            print(f"  👉 新Originを空間に配置しました。ID: {new_origin.cube_id}")
        else:
            # Seedはあったがクラスタが形成できなかった場合は抜ける
            print(f"  👉 有効なクラスタが形成されませんでした。処理を抜けます。")
            break

    # 2. デフラグ後の追加ガベージコレクション
    print("\n  👉 デフラグ後の追加ガベージコレクション...")
    kv.garbage_collection()

    print("\n==================================================")
    print("🏁 全テスト完了。v2.1 Runtime(代謝モデル)で完走成功！")
    print("==================================================")

if __name__ == "__main__":
    run_semantic_cube_runtime()
