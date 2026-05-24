"""
defrag_engine.py (v5.1: イベント駆動 × 地形ベースデフラグ仕様)
"""

import random
import numpy as np
from typing import List, Dict, Set, Tuple

# 司令官のエコシステムから必要な「関数」と「型」をインポート
from common_types import SemanticCube
from partial_engine import call_llm
from cube_factory import make_cube
from trust_evaluator import calculate_exact_hybrid_score, angular_distance
from logger import print_log
from kvhub import KVHub

# ==========================================
# ヘルパー関数群 (v5.1 地形ベース仕様)
# ==========================================

def _build_density_grid(cubes: List[SemanticCube]) -> Dict[Tuple[int, int, int], List[SemanticCube]]:
    """ステップ①: 全キューブをスキャンし、int3グリッド（密度マップ）を構築する"""
    grid_map: Dict[Tuple[int, int, int], List[SemanticCube]] = {}
    for cube in cubes:
        coord_list = cube.trust.grid_index
        if len(coord_list) == 3:
            coord = tuple(coord_list)
            if coord not in grid_map:
                grid_map[coord] = []
            grid_map[coord].append(cube)
    return grid_map

def _get_neighboring_cubes(seed_coord: Tuple[int, int, int], grid_map: Dict[Tuple[int, int, int], List[SemanticCube]]) -> List[SemanticCube]:
    """シードのグリッドとその隣接（27空間）から近傍キューブを収集する"""
    neighbors = []
    x, y, z = seed_coord
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                target_coord = (x + dx, y + dy, z + dz)
                if target_coord in grid_map:
                    neighbors.extend(grid_map[target_coord])
    return neighbors

def _form_hybrid_cluster(seed: SemanticCube, pool: List[SemanticCube], processed: Set[str], global_turn: int) -> List[SemanticCube]:
    """ステップ④: 近傍プールに対し、MiniCube(話題)とOrientation(方向)でクラスタを形成する"""
    cluster = [seed]
    
    seed_phrases = set([mc.phrase for mc in seed.mini_cubes])
    
    for candidate in pool:
        if candidate.cube_id == seed.cube_id or candidate.cube_id in processed:
            continue
            
        candidate_phrases = set([mc.phrase for mc in candidate.mini_cubes])
        
        # 1. MiniCube検索（話題一致）
        common_phrases = seed_phrases.intersection(candidate_phrases)
        if len(common_phrases) == 0:
            continue
            
        # 2. 近傍探索（方向一致）
        direction_sim = angular_distance(seed.trust.orientation, candidate.trust.orientation)
        
        if direction_sim < 0.4: # 方向が近い
            cluster.append(candidate)
            
    return cluster

def _run_logical_garbage_collection(all_cubes: List[SemanticCube], kv_hub) -> int:
    """
    ステップ⑧: 論理GC。物理削除はせず、アーカイブフラグを立てて非表示にする。
    監査性と安全性の両立（第3.5世代仕様）。
    """
    gc_count = 0
    for cube in all_cubes:
        if cube.trust.is_privileged or cube.trust.is_archived:
            continue
            
        current_alpha = cube.get_alpha(kv_hub.global_turn)
        
        # アルファ値が限界まで低下し、かつ直近5ターン保護に該当しない場合
        turns_passed = kv_hub.global_turn - cube.trust.last_updated_turn
        
        if current_alpha < 0.01 and turns_passed > 5:
            # 物理削除(self.cubes.remove等)は行わず、フラグだけ立てる
            cube.trust.is_archived = True
            print_log("Defrag-v5.1", f"🗑️ 論理GC実行: [{cube.cube_id[:8]}] をアーカイブ（非表示化）しました。")
            gc_count += 1
            
    return gc_count

# ==========================================
# メイン実行関数 (runtime.pyから呼ばれる)
# ==========================================

def run_defrag_cycle(kv_hub, model_name: str) -> KVHub:
    """
    [v5.1] 地形ベース・デフラグ × イベント駆動アルファ代謝 統合版
    """
    global_turn = kv_hub.global_turn
    print_log("Defrag-v5.1", f"⚡ デフラグサイクル起動 (Global Turn: {global_turn})")
    
    # アーカイブされていない（生きている）キューブだけを取得
    all_active_cubes = kv_hub.get_all()
    
    if len(all_active_cubes) < 5:
        print_log("Defrag-v5.1", "💤 アクティブキューブ少数につきスキップ。")
        return kv_hub

    new_origins = []
    processed_cube_ids: Set[str] = set()

    # --- ステップ①＆②: 空間グリッド構築と濃い領域の特定 ---
    grid_map = _build_density_grid(all_active_cubes)
    dense_grids = [g for g, c_list in grid_map.items() if len(c_list) >= 2]
    
    print_log("Defrag-v5.1", f"🌌 密度マップ構築完了。ホットスポットを {len(dense_grids)} 箇所特定。")

    # --- ステップ③: 濃い領域の中から「alpha_current が低い」キューブをシードにする ---
    seed_candidates = []
    for g in dense_grids:
        for cube in grid_map[g]:
            # ★ 指示通り、turns_passed > 5 のIF文を完全削除！
            if not cube.trust.is_privileged:
                seed_candidates.append(cube)
                
    # 実効アルファが「低い順」にソート
    # ※若い記憶はアルファが減衰していない(1.0付近)ため自然と後回しになり、
    # ※古いノイズ(0.01等)が優先的にシード（お掃除対象）として選ばれる。
    seed_candidates.sort(key=lambda x: x.get_alpha(global_turn))
                
    # 実効アルファが「低い順」にソート（弱いものから掃除する）
    seed_candidates.sort(key=lambda x: x.get_alpha(global_turn))

    # --- ステップ⑥: 濃い領域がなくなるまでループ ---
    for seed_cube in seed_candidates:

        print_log("Defrag-v5.1", f"🔍 シード候補キューブ: [{seed_cube.cube_id[:8]}] | α={seed_cube.get_alpha(global_turn):.2f} | グリッド={tuple(seed_cube.trust.grid_index)} | ミニキューブ数={len(seed_cube.mini_cubes)}")

        if seed_cube.cube_id in processed_cube_ids:
            continue
            
        seed_coord = tuple(seed_cube.trust.grid_index)
        neighborhood_pool = _get_neighboring_cubes(seed_coord, grid_map)
        
        # --- ステップ④: シードを中心にクラスタ形成 ---
        final_cluster = _form_hybrid_cluster(seed_cube, neighborhood_pool, processed_cube_ids, global_turn)

        # --- ステップ⑤: クラスタ判定 ---
        if len(final_cluster) >= 2:
            print_log("Defrag-v5.1", f"🧹 弱記憶 [{seed_cube.cube_id[:8]}] (α={seed_cube.get_alpha(global_turn):.2f}) を起点に {len(final_cluster)}個のクラスタ形成。")
            
            # 🔍 [Debug] クラスタの中身を曝け出すぜ！
            print("\n" + "="*50)
            print(f"🔥 [DEBUG] 形成されたクラスタの中身 (Origin生成前)")
            for i, c in enumerate(final_cluster):
                kp = ", ".join([mc.phrase for mc in c.mini_cubes]) if c.mini_cubes else "None"
                print(f"  [{i+1}] ID:{c.cube_id[:8]} | α:{c.get_alpha(global_turn):.2f} | Key:{kp[:20]} | Sum:{c.summary[:30]}...")
            print("="*50 + "\n")

            # --- ステップ⑥: Origin生成 (LLM) ---
            cluster_details = []
            for i, c in enumerate(final_cluster):
                kp = ", ".join([mc.phrase for mc in c.mini_cubes]) if c.mini_cubes else "None"
                detail = f"- [{i+1}] 根拠ID (ID: {c.cube_id})\n  質問: 【{c.summary}】\n  抽出キーフレーズ: 【{kp}】\n  回答: 【{c.response_and_answer or 'None'}】"
                cluster_details.append(detail)
            
            texts = "\n\n".join(cluster_details)
            prompt = f"""情報の整理と要約をおこなってください。
以下の意見のクラスタは、プロンプトの記憶から抽出された複数の情報です。
これらを総合的に分析・統合し、キーフレーズ（Keyphrases）や、事実に基づいた文章にまとめてください。

【ルール】
1. 入力された情報にない「新たな価値観」「教訓」「哲学的な意味づけ」を付加しないでください。（ポエム化は禁止）。
2. 無関係なノイズ（日常の些末な出来事など）が混ざっている場合は、それらを完全に無視し、中核となる話題のみを抽出してください。
3. 元の文章のニュアンスをうまく保持しつつ、全体を包括するような結論を導き出してください。

==【統合対象の意見クラスタ】==
{texts}"""
            
            print_log("Defrag-v5.1", "🧠 LLMによるOrigin（抽象化）生成中...")
            # print("-" * 40)
            # print_log("Defrag-v5.1", f"📋 プロンプト内容:\n{prompt}")
            # print("-" * 40)
            
            new_origin_text = call_llm(prompt, model_name)

            new_origin = make_cube(new_origin_text, "core", global_turn, 1.0, model_name)
            new_origin.vector = seed_cube.vector.copy()
            new_origins.append(new_origin)
            kv_hub.put(new_origin)

            # 🔍 [Debug] LLMが吐き出したOriginの確認！
            print("\n" + "="*50)
            print(f"✨ [DEBUG] 爆誕した新Origin (ID:{new_origin.cube_id[:8]})")
            print(f"【要約テキスト】:\n{new_origin.summary[:300]}...")
            print("="*50 + "\n")

            # --- ステップ⑦: デフラグに使ったキューブの alpha_base を弱化 ---
            for old_cube in final_cluster:
                # KVHubのイベントトリガーを通して弱化（アーキテクチャの統一）
                kv_hub.trigger_event(old_cube, "defrag_absorbed")
                old_cube.parent_id = new_origin.cube_id
                processed_cube_ids.add(old_cube.cube_id)
        else:
            # 似ているキューブが少ない → ただのノイズとして扱う
            # （ここでは何もしない。いずれGC対象になる）
            pass

    print_log("Defrag-v5.1", f"🧹 デフラグサイクル完了。新規Origin: {len(new_origins)}個。次はGCフェーズへ...")
    print("-" * 60)
    print_log("Defrag-v5.1", f"🗑️ GCフェーズ開始。GC前のキューブ数：{len(all_active_cubes)}")
    print("-" * 60)
    # --- ステップ⑧: GC (論理削除・アーカイブ化) ---
    gc_count = _run_logical_garbage_collection(all_active_cubes, kv_hub)
    
    print_log("Defrag-v5.1", f"✨ 代謝完了。新規Origin: {len(new_origins)}個 / アーカイブ(GC): {gc_count}個")
    print("-" * 60)
    return kv_hub
