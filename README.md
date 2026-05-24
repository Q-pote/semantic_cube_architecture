# Semantic Cube Architecture
（意味キューブアーキテクチャ）

---
### A Data & Memory Architecture for Next-Generation AI

論文はかなり高難易度なので、AIに読んでもらってわかる言語に翻訳してもらってください。
---

**Semantic Cube Architecture は “データ＆メモリアーキテクチャ” です。  

これは LLM の外側で動作する “思考するメモリ” です。  

既存の LLM 本体には一切手を加える必要はありません。  

Apache-2.0 ライセンスで公開します。**

---

## 🚀 Overview

Semantic Cube Architecture は、  
**「意味単位で記憶し、方向で推論し、代謝で整理する」**  
次世代 AI のためのメモリ・推論アーキテクチャです。

従来の LLM が抱える構造的限界：

- 文脈は一次元で線形  
- 記憶は直列で保持できない  
- 意味空間は暗黙的で操作できない  
- 長期記憶の統合ができない  
- 推論の方向性を追跡できない  

これらを克服するために、Semantic Cube Architecture は  
**“LLM の外側にある意味空間”** を構築し、  
そこに **記憶・推論・代謝・方向性検知** を委譲します。

---

## 🧠 Core Concepts

### 1. Semantic Cube（意味キューブ）
意味単位で記憶する最小ブロック。  
構造体には以下を含む：

- `summary`（短い要約）  
- `vector_float32` / `vector_int3`  
- `MiniCube`（語義DNA）  
- `trust`（gravity, freshness, provenance）  
- `generation_number`  
- `origin` / `active` の依存関係  

---

### 2. Attraction Map（引力マップ）
候補キューブ間の  
**距離 × 方向 × 重み**  
を組み合わせて視点クラスタを検出する仕組み。

- スカラー類似度  
- 方向性減衰  
- 重み（trust × provenance）  
- 葛藤検知（Conflict Detection）  

---

### 3. α Metabolism（代謝）
`global_turn` を基準に  
**実効 α（生命力）** をオンデマンド計算し、  
古い記憶を自然に弱化 → デフラグ → GC。

- 直近保護  
- 減衰モデル  
- defrag_absorbed イベント  
- 段階的 GC（保持 → アーカイブ → 削除）  

---

### 4. Atomic Topic Shift（話題の原子性）
進行ベクトルの角度変化で  
**論点の転換** を検知する新しい理論。

- 進行ベクトル  
- sV（方向類似度）  
- sG（重心類似度）  
- rM（MiniCube Jaccard）  

---

## 🧪 Implementation Mock

`動作モック_Python` には以下の PoC が含まれます：

- int3 量子化  
- 7階調バケット  
- Attraction Map  
- α代謝  
- デフラグ  
- GC  
- SRAG（Semantic Retrieval-Augmented Generation）  

---

## 📚 Documentation

論文（第1〜7章）は `ドキュメント清書` に格納されています。



- **第1章**：意味キューブとは  
- **第2章**：データ構造  
- **第3章**：軽量化と SRAG  
- **第4章**：引力マップと葛藤検知  
- **第5章**：代謝・デフラグ・GC  
- **第6章**：話題の原子性とエコーインサイト  
- **第7章**：デュアルエンジン構想  

---

## 🤝 Contributing

Semantic Cube Architecture はオープンな研究プロジェクトです。  
Issue / PR / Discussion を歓迎します。

！　主に私のポートフォリオかもしれません。！

---

## 📄 License

本プロジェクトは **Apache License 2.0** の下で公開されています。

