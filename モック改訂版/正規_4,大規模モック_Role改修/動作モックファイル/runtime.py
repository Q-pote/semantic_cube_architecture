# runtime.py 完全版（カオス空間テスト用）

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
    print("🌍 Semantic Cube Runtime v3.0 世界線の立ち上げ...")
    print("==================================================\n")

    # ------------------------------------------------
    # 🌌 【Phase 0】カオス空間の初期化（全キューブを一括投入）
    # ------------------------------------------------
    print_log("Phase 0", "全記憶キューブの空間配置（カオス状態の作成）")
    kv.thread_mode = "smart"
    
    # --- 話題A: リモートワークとオフィス ---
    o1 = make_cube("リモートワークは通勤時間を削減し、生産性を高める働き方である。", "topic", turn-10, 1.0, model)
    o2 = make_cube("オフィスでの対面コミュニケーションは、新しいアイデアを生み出す土壌となる。", "topic", turn-10, 1.0, model)
    kv.put(o1); kv.put(o2)
    
    topic_a_texts = [
        "在宅勤務は、自分に合った環境を整えられるため集中しやすい。",
        "通勤ストレスがないことで、仕事以外の時間も充実しやすい。",
        "オンライン会議ツールの発達により、リモートでも十分に協働できる。",
        "静かな環境で作業できるため、集中力が途切れにくい。",
        "移動時間がゼロになることで、1日の可処分時間が増える。",
        "対面での議論は意思決定のスピードが上がる。",
        "部署をまたいだ出会いがコラボレーションを生む。"
    ]
    for t in topic_a_texts:
        # parent_id として便宜上 o1 のIDをセット（血脈のモック）
        kv.put(make_cube(t, "topic", turn-5, 1.0, model, parent_id=o1.cube_id))

    # --- 話題B: カレー戦争 ---
    o3a = make_cube("カレーのおいしさは、ルーの一体感と口当たりの滑らかさに宿る。", "topic", turn-8, 1.0, model)
    o3b = make_cube("カレーのおいしさは、具材の食感と素材の旨味に宿る。", "topic", turn-8, 1.0, model)
    kv.put(o3a); kv.put(o3b)
    
    topic_b_texts = [
        "カレーの魅力は、滑らかなルーが生む一体感にある。忙しい時でも、飲むように味わえる手軽さが“おいしさ”の一部を形作る。", 
        "カレーの魅力は、具材を噛むことで広がる多層的な風味にある。素材の旨味がしっかり残ることが“おいしさ”の核になる。",
        "カレーの隠し味にはチョコレートが良い。",
        "カレーを食べる時はスプーンよりフォークだ。"
    ]
    for t in topic_b_texts:
        kv.put(make_cube(t, "topic", turn-3, 1.0, model, parent_id=o3a.cube_id))

    # --- 話題C: 完全なノイズ・相槌 ---
    noise_texts = [
        "今日のランチは近所の定食屋で唐揚げ定食を食べた。",
        "明日の天気は午後から雨が降るらしいから傘が必要だ。",
        "最近読んだSF小説の結末がどうしても納得いかない。",
        "休日は一日中ゲームをして過ごすのが最高の贅沢だ。",
        "新しいスニーカーを買ったけど、靴擦れして痛い。",
        "へえ、それは面白い。",
        "ふむふむ。",
        "ちょっと待って。",
        "なるほどね。"
    ]
    for t in noise_texts:
        kv.put(make_cube(t, "topic", turn-1, 1.0, model))

    print(f"  👉 全 {len(kv.get_all())} 個のキューブを空間に配置完了。")



    # ------------------------------------------------
    # 🚀 Test 1: 検索の純度テスト（リモートワーク）
    # ------------------------------------------------
    print("\n" + "-"*50)
    print_log("Test 1", "検索の純度テスト（カオス空間からの抽出）")
    
    q1 = make_cube("リモートワークのメリットを、経営者の視点から整理してほしい。", "topic", turn, 1.0, model)
    
    res1 = kv.search_by_trust(q1, turn, k=4) # 試しに Top-4 まで拾ってみる
    print(f"  👉 抽出 Top4:")
    for i, (c, s) in enumerate(res1):
        print(f"     {i+1}. [{c.role}] Trust:{s:.3f} | {c.summary[:50]}...")
    
    # ------------------------------------------------
    # 🚀 Test 2: 葛藤分岐テスト（リモート vs オフィス）
    # ------------------------------------------------
    turn += 10
    print("\n" + "-"*50)
    print_log("Test 2", "順次回答 → 統合／葛藤分岐 開始")
    
    kv.mc_strictness = "standard"
    print(f"  💡 検索設定: [MiniCube語義一致度: {kv.mc_strictness.upper()}]")

    q2 = make_cube("生産性とイノベーションの両面から方針を提案して。", "topic", turn, 1.0, model)
    res2 = kv.search_by_trust(q2, turn, k=4)
    
    print(f"  👉 抽出 Top4:")
    partials2 = []
    for i, (c, s) in enumerate(res2):
        print(f"     {i+1}. [{c.role}] Trust:{s:.3f} (α:{c.trust.alpha:.2f}) | {c.summary[:50]}...")
        p_ans = make_partial(q2.summary, c, s, model)
        partials2.append((c.cube_id, p_ans, s))
    
    is_conf2 = determine_route_by_physics([c for c, s in res2])
    print(f"  👉 物理的葛藤検知: {is_conf2}")
    
    final_ans2 = analyze_conflict(partials2, model) if is_conf2 else integrate_responses(partials2, model)
    print(f"  👉 最終回答 (Test2):\n{final_ans2}...\n")

    # ------------------------------------------------
    # 🚀 Test 3: カレー戦争（Fuzzy設定での拾い上げ）
    # ------------------------------------------------
    turn += 10
    print("\n" + "-"*50)
    print_log("Test 3", "高トラスト葛藤（カレー戦争）開始")
    
    kv.mc_strictness = "fuzzy"
    print(f"  💡 検索設定: [MiniCube語義一致度: {kv.mc_strictness.upper()}]")
    
    q3 = make_cube("カレーの“本質的なおいしさ”って何だと思う？具材？ルー？それとも別の要素？", "topic", turn, 1.0, model)
    res3 = kv.search_by_trust(q3, turn, k=4)
    
    print(f"  👉 抽出 Top4:")
    partials3 = []
    for i, (c, s) in enumerate(res3):
        print(f"     {i+1}. [{c.role}] Trust:{s:.3f} (α:{c.trust.alpha:.2f}) | {c.summary[:50]}...")
        partials3.append((c.cube_id, make_partial(q3.summary, c, s, model), s))
        
    is_conf3 = determine_route_by_physics([c for c, s in res3])
    print(f"  👉 物理的葛藤検知: {is_conf3}")
    
    final_ans3 = analyze_conflict(partials3, model) if is_conf3 else integrate_responses(partials3, model)
    print(f"  👉 最終メタ判断 (Test3):\n{final_ans3}...\n")

    # ------------------------------------------------
    # 🚀 Test 4: 連続デフラグ（Multi-Defrag）
    # ------------------------------------------------
    turn += 50
    print("\n" + "-"*50)
    print_log("Test 4", "新8ステップ・連続デフラグ ＆ ガベージコレクション 開始")
    
    # 1. 忘却処理(GC)の実行
    kv.garbage_collection()

    max_defrag_loops = 5
    loop_count = 0
    
    while loop_count < max_defrag_loops:
        # デフラグの対象になり得る（まだ代替されていない）Topicを探す
        eligible_cubes = [
            c for c in kv.get_all() 
            if c.role == "topic" and c.trust.replacement_closeness < 0.5
        ]
        
        if len(eligible_cubes) < 2:
            print(f"\n  👉 デフラグ可能なクラスタがなくなったため、連続処理を終了します。")
            break
            
        loop_count += 1
        print(f"\n🌀 [DefragEngine] --- 連続デフラグ 第 {loop_count} サイクル ---")
        
        # ⚠️ strictness = "standard" (相対diff=-0.4) を指定
        new_core = defrag_cluster(kv.get_all(), turn, model, strictness="standard")
        
        if new_core:
            kv.put(new_core)
            print(f"  👉 新Coreを空間に配置しました。ID: {new_core.cube_id}")
        else:
            print(f"  👉 有効なクラスタが形成されませんでした。処理を抜けます。")
            break

    # 2. デフラグ後の追加GC
    print("\n  👉 デフラグ後の追加ガベージコレクション...")
    kv.garbage_collection()

    print("\n==================================================")
    print("🏁 全テスト完了。Semantic Cube v3.0 での完走成功！")
    print("==================================================")

if __name__ == "__main__":
    run_semantic_cube_runtime()
