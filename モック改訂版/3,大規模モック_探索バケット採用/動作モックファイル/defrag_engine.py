# defrag_engine.py 修正版
import math
from typing import List, Optional
from common_types import SemanticCube
from trust_evaluator import calculate_exact_hybrid_score, update_alpha
from cube_factory import make_cube
from partial_engine import call_llm

def select_defrag_seed(all_cubes: List[SemanticCube]) -> Optional[SemanticCube]:
    """ref_countが大きい、かつ【未処理の】historyキューブをシードに選ぶ"""
    history_cubes = [
        c for c in all_cubes 
        if c.role == "history" and c.trust.replacement_closeness < 0.5  # ✨ ココを追加
    ]
    if not history_cubes:
        return None
    history_cubes.sort(key=lambda c: c.trust.ref_count, reverse=True)
    return history_cubes[0]

def defrag_cluster(all_cubes: List[SemanticCube], current_turn: int, model_name: str, strictness: str = "standard") -> Optional[SemanticCube]:
    """
    【相対評価版・デフラグエンジン】
    シードキューブに対する「最大の類似度」を基準点とし、
    そこからの相対的な差分（diff）が許容範囲内のキューブをクラスタとして抽出する。
    """
    # ✨ 相対評価における「許容するズレ（diff）」の閾値
    # "standard" ならトップから -0.3 までのズレを許容して同じクラスタとみなす
    threshold_diff_map = {
        "aggressive": -0.66, # 寛容：広く集める
        "standard": -0.45,   # 標準
        "strict": -0.2     # 厳格：非常に似ているものだけ
    }
    allowed_diff = threshold_diff_map.get(strictness, -0.3)

    seed_cube = select_defrag_seed(all_cubes)
    if not seed_cube:
        print("🌀 [DefragEngine] デフラグ可能なHistoryキューブが存在しません。")
        return None
        
    print(f"\n🌀 [DefragEngine] 起動: Seed Cube [{seed_cube.cube_id[:4]}] (ref_count: {seed_cube.trust.ref_count:.2f})")
    print(f"  ├ デフラグ強度: [{strictness.upper()}] (Allowed Diff: {allowed_diff})")
    
    # --- Step 2: 局所クラスタ抽出（相対評価） ---
    # シードに対する全てのキューブの類似度を計算
    scored = []
    for c in all_cubes:
        if c.trust.gc_flag: continue # 消えかけのゴミは無視
        # デフラグ時は純粋な意味の近さを見たいので、exact_scoreを使う
        sim = calculate_exact_hybrid_score(seed_cube, c)
        scored.append((c, sim))
    
    if not scored: return None
    
    # クラスタ内の最大類似度を基準(0)とする（Copilotの検索ロジックと同じ発想）
    max_sim = max(s for _, s in scored)
    
    # 基準からのズレ(diff)が許容範囲内(allowed_diff以上)のものをクラスタに含める
    local_cluster = [c for c, s in scored if (s - max_sim) >= allowed_diff]
    
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
    from trust_evaluator import cosine_similarity
    
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
    prompt = f"""以下は同じ方向性を持つ意見のクラスタです。
これらの共通点を抽象化し、新しい「Origin」としてふさわしい文章に要約して生成してください。
１つの文章で他の根拠となった文の置き換えになれるようにうまくまとめてください。
事実や根拠を単純に羅列するのではなく、全体を包括するような洞察や結論を導き出すことを目指してください。

根拠となる文章以外には、余計な情報やノイズを一切含めないでください。

【意見クラスタ】
{texts}"""
    
    print(f"\n  ├ Step 6: LLM による意訳（抽象化）処理中...")
    new_origin_text = call_llm(prompt, model_name)
    print(f"\n  👀 [監査ログ] デフラグ出力（LLMが生成した新Origin）:")
    print(f"      {new_origin_text}\n")
    
    # --- Step 7: 新Origin生成 ---
    # 新しいOriginは平等に ref_count=1 で誕生する
    new_origin = make_cube(new_origin_text, "origin", turn=current_turn, ref_count=1.0, llm_model_name=model_name)
    
    # 古いキューブ群の代謝パラメータ（代替可能性）を更新
    for c in final_cluster:
        c.trust.replacement_closeness = 0.8 # 即死(0.9)を避け、少しだけ余韻を残す(0.8)
        update_alpha(c)

    return new_origin
