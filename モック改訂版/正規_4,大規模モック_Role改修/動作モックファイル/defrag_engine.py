# defrag_engine.py 修正版
import math
from typing import List, Optional
from common_types import SemanticCube
from trust_evaluator import calculate_exact_hybrid_score, update_alpha, cosine_similarity
from cube_factory import make_cube
from partial_engine import call_llm

def select_defrag_seed(all_cubes: List[SemanticCube]) -> Optional[SemanticCube]:
    """ref_countが大きい、かつ【未処理の】topicキューブをシードに選ぶ"""
    # ✨ 修正: "history" ではなく "topic" を対象にする
    history_cubes = [
        c for c in all_cubes 
        if c.role == "topic" and c.trust.replacement_closeness < 0.5
    ]
    if not history_cubes:
        return None
    history_cubes.sort(key=lambda c: c.trust.ref_count, reverse=True)
    return history_cubes[0]

def defrag_cluster(all_cubes: List[SemanticCube], current_turn: int, model_name: str, strictness: str = "standard") -> Optional[SemanticCube]:
    
    # "standard" ならトップから -0.45 までのズレを許容する
    threshold_diff_map = {
        "aggressive": -0.6, 
        "standard": -0.4,  # ✨ 修正: 許容差分
        "strict": -0.2
    }
    allowed_diff = threshold_diff_map.get(strictness, -0.4)

    seed_cube = select_defrag_seed(all_cubes)
    if not seed_cube:
        return None
        
    print(f"\n🌀 [DefragEngine] 起動: Seed Cube [{seed_cube.cube_id[:4]}] (ref_count: {seed_cube.trust.ref_count:.2f})")
    

    # ✨ デフラグ強度に合わせて、MiniCubeのDNA照合の厳しさも連動させる！
    mc_thresh = {"aggressive": 0.5, "standard": 0.65, "strict": 0.8}.get(strictness, 0.65)

    # --- Step 2: 局所クラスタ抽出（純粋なコサイン類似度＋相対評価） ---
    scored = []
    for c in all_cubes:
        if c.trust.gc_flag: continue
        
        # ✨ 復活！！ マクロ(ベクトル) ＋ ミクロ(MiniCube語義) の真のハイブリッド評価！
        # デフラグ用に少しだけ寛容にした mc_thresh を渡す
        sim = calculate_exact_hybrid_score(seed_cube, c, mc_threshold=mc_thresh)
        scored.append((c, sim))
    
    if not scored: return None
    
    # クラスタ内の最大類似度を基準(0)とする
    max_sim = max(s for _, s in scored)
    
    # ✨ 修正: 相対的なズレが許容範囲内、かつ「絶対的な類似度」が 0.3 以上のものを抽出
    local_cluster = [c for c, s in scored if (s - max_sim) >= allowed_diff and s >= 0.3]
    
    print(f"  ├ Step 2: 局所クラスタ抽出 ({len(local_cluster)} 個)")

    if len(local_cluster) < 2:
         print("  └ 同一クラスタとみなせるキューブが不足しているため、デフラグをスキップします。")
         return None

    # --- Step 3: クラスタの重心位置を計算 ---
    weights = []
    vectors = []
    for c in local_cluster:
        w = math.log1p(c.trust.ref_count) # 参照頻度が高いものほど重心に影響
        weights.append(w)
        vectors.append(c.vector)
        
    weights_sum = sum(weights) + 1e-9
    center_vec = sum(w * v for w, v in zip(weights, vectors)) / weights_sum
    print(f"  ├ Step 3: 重心ベクトル算出完了")

    # --- Step 4 & 5: 重心から近い順＆ref_countの大きい順に再選抜 ---
    
    re_scored = []
    for c in local_cluster:
        sim_to_center = cosine_similarity(center_vec, c.vector)
        final_score = sim_to_center * (1.0 + math.log1p(c.trust.ref_count))
        re_scored.append((c, final_score))
        
    re_scored.sort(key=lambda x: x[1], reverse=True)
    final_cluster = [c for c, score in re_scored[:10]] # 最大10個に絞る
    
    print(f"  ├ Step 4-5: 重心からの再選抜完了 ({len(final_cluster)} 個)")
    print(f"\n  👀 [監査ログ] デフラグ入力（再選抜されたキューブ群）:")
    for i, c in enumerate(final_cluster):
        print(f"      [{i+1}] (ref_count: {c.trust.ref_count:.2f}) {c.summary[:40]}...")

    # --- Step 6: 意訳（LLMによる抽象化） ---
    texts = "\n".join([f"- {c.summary}" for c in final_cluster])
    
    prompt = f"""あなたは情報の整理と要約を行うアシスタントとして回答してください。
以下の【意見クラスタ】は、AIの記憶から抽出された複数の断片です。
これらを統合し、事実と本来の文脈に基づいた「１つの要約文章」を作成してください。

【厳格なルール】
1. 事実に基づき統合すること。入力された情報にない「新たな価値観」「教訓」「哲学的な意味づけ」を絶対に捏造しないこと（ポエムの禁止）。
2. 全く無関係なノイズ（日常の些末な出来事など）が混ざっている場合は、それらを無視し、中核となる話題のみを抽出すること。
3. 抽出するに足る具体的な話題が存在しない場合や、相槌のみの場合は、「情報が分散しており統合不可」とだけ出力すること。
4. 可能な限り、元の文章のニュアンスや重要なキーワードを保持しつつ、全体を包括するような洞察や結論を導き出すことを目指してください。
5. 根拠となる文章以外には、余計な情報やノイズを一切含めないでください。

【意見クラスタ】
{texts}"""
    
    print(f"\n  ├ Step 6: LLM による意訳（抽象化）処理中...")
    new_origin_text = call_llm(prompt, model_name)
    print(f"\n  👀 [監査ログ] デフラグ出力（LLMが生成した新Origin）:")
    print(f"      {new_origin_text}\n")
    
    # --- Step 7: 新Core生成 ---
    # ✨ 修正: 新しい中心は "origin" ではなく "core" として誕生する
    new_origin = make_cube(new_origin_text, "core", turn=current_turn, ref_count=1.0, llm_model_name=model_name)
    
    # 古いtopicキューブ群の代謝パラメータを更新
    for c in final_cluster:
        c.trust.replacement_closeness = 0.8
        update_alpha(c)

    return new_origin
