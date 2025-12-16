# 開発者向けコード理解ガイド / Developer's Code Understanding Guide

このガイドは、T5Gemma-TTSのコードベースを理解したい開発者向けです。

This guide is for developers who want to understand the T5Gemma-TTS codebase.

## 🎯 目的 / Purpose

ユーザー向けではなく、**開発者がコードアーキテクチャを理解するため**のシンプル化版です。

Not for users, but for **developers to understand the code architecture**.

## 📁 コアファイル / Core Files

### 1. アーキテクチャ / Architecture

```
models/
├── t5gemma.py          # コアモデル: PM-RoPE実装
│   ├── PMCrossAttention         # カスタムクロスアテンション
│   ├── PMDecoderLayer          # PM-RoPE対応デコーダ層
│   └── T5GemmaVoiceModel       # メインモデルクラス
└── utils.py            # ヘルパー関数
```

### 2. データ処理 / Data Processing

```
data/
├── tokenizer.py        # AudioTokenizer (XCodec2ラッパー)
└── combined_dataset.py # 学習用データローダー
```

### 3. 推論 / Inference

```
inference_tts_utils.py     # 推論ロジック (inference_one_sample)
inference_commandline_hf.py # CLIエントリーポイント
```

## 🔍 アーキテクチャの理解 / Understanding Architecture

### ステップ1: 基本的なフロー / Basic Flow

```
テキスト → Encoder → Hidden States → Decoder → 音声トークン → XCodec2 → 音声波形
Text    → Encoder → Hidden States → Decoder → Audio Tokens → XCodec2 → Waveform
```

### ステップ2: PM-RoPE (Progress-Monitoring RoPE)

通常のTransformer cross-attentionとの違い:

**通常のcross-attention:**
```python
Q = decoder_hidden_states @ W_q  # Query from decoder
K = encoder_hidden_states @ W_k  # Key from encoder
V = encoder_hidden_states @ W_v  # Value from encoder
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**PM-RoPE cross-attention:**
```python
Q = decoder_hidden_states @ W_q
Q = apply_rope(Q, decoder_position_ids)  # ← デコーダ位置情報を追加

K = encoder_hidden_states @ W_k
K = apply_rope(K, encoder_position_ids)  # ← エンコーダ位置情報を追加

V = encoder_hidden_states @ W_v
attention = softmax(Q @ K.T / sqrt(d)) @ V
```

**なぜ？/ Why?**
- デコーダが生成の進行状況を把握できる
- エンコーダのどの部分に注目すべきかがより明確になる
- Decoder can track generation progress
- Clearer focus on which encoder parts to attend to

### ステップ3: コード探索の順序 / Code Exploration Order

開発者がコードを理解する推奨順序:

**1. まず `minimal_example.py` を読む**
- 全体の流れを理解
- 各ステップの役割を把握

**2. 次に `models/t5gemma.py` を読む**
- `PMCrossAttention` クラス (L22-100付近)
  - `_apply_rotary_with_progress` メソッドでRoPE適用
  - `forward` メソッドでクロスアテンション実行
- `PMDecoderLayer` クラス (L150-250付近)
  - デコーダ層の実装
  - PM-RoPEの統合
- `T5GemmaVoiceModel` クラス (L300-600付近)
  - メインのforward実装
  - 位置IDの計算

**3. `inference_tts_utils.py` を読む**
- `inference_one_sample` 関数 (L107-)
  - 実際の推論フロー
  - テキスト準備
  - 音声トークン生成
  - デコード処理

**4. `data/tokenizer.py` を読む**
- `AudioTokenizer` クラス
  - XCodec2のラッパー
  - エンコード/デコード処理

## 🔧 主要な概念 / Key Concepts

### 1. XCodec2

**何？/ What?**
ニューラル音声コーデック。音声を離散トークンに変換（およびその逆）。

Neural audio codec. Converts audio to discrete tokens (and vice versa).

**コード:**
```python
# エンコード: 音声 → トークン
tokens = audio_tokenizer.encode(waveform)  # [T] -> [1, 1, T]

# デコード: トークン → 音声  
waveform = audio_tokenizer.decode(tokens)  # [1, 1, T] -> [T]
```

### 2. 自己回帰生成 / Autoregressive Generation

デコーダは1トークンずつ生成:

```python
# t=0: BOS -> token_1
# t=1: [BOS, token_1] -> token_2
# t=2: [BOS, token_1, token_2] -> token_3
# ...
# t=N: [...] -> EOS
```

**コード:** `model.generate()` が内部でこれを実行

### 3. 特殊トークン / Special Tokens

- `empty_token` (BOS): 音声シーケンスの開始
- `y_sep_token`: 参照音声と生成音声の区切り
- `x_sep_token`: テキストと音声の区切り（廃止）
- `eos_token`: シーケンスの終了

## 📊 データフロー図 / Data Flow Diagram

```
Input: "Hello world"
  ↓
[Text Tokenizer (SentencePiece)]
  ↓
Token IDs: [1234, 5678, 2]
  ↓
[T5Gemma Encoder]
  ↓
Encoder Hidden States: [B, Seq, Dim]
  ↓
[T5Gemma Decoder with PM-RoPE] ← Autoregressive
  ↓                                  ↓
Audio Token IDs: [0, 123, 456, ...]  ← Self-attention
  ↓                                  ← Cross-attention (PM-RoPE)
[XCodec2 Decoder]
  ↓
Waveform: [-0.5, 0.3, -0.2, ...]
  ↓
Output: speech.wav
```

## 🛠️ デバッグ Tips / Debugging Tips

### 形状を追跡する / Track Shapes

```python
# テキスト
input_ids: [B, Seq_text]

# エンコーダ出力
encoder_hidden_states: [B, Seq_text, Dim]

# デコーダ入力 (音声トークン)
decoder_input_ids: [B, Seq_audio]

# デコーダ出力
decoder_hidden_states: [B, Seq_audio, Dim]
logits: [B, Seq_audio, Vocab_audio]

# 生成されたトークン
generated_ids: [B, Seq_audio]

# 音声波形
waveform: [B, T_samples] where T_samples = Seq_audio * hop_length
```

### 位置IDを理解する / Understanding Position IDs

PM-RoPEは2種類の位置IDを使用:

```python
# デコーダ位置: 生成の進行状況
decoder_position_ids: [0, 1, 2, 3, ...]  # 生成したトークン数

# エンコーダ位置: テキストのどこを見るか
encoder_position_ids: [0, 0, 1, 1, 2, 2, ...]  # 対応するテキスト位置
```

## 📝 コードリーディングの例 / Code Reading Example

`models/t5gemma.py` の `PMCrossAttention.forward()` を読む:

```python
def forward(self, hidden_states, ..., pm_decoder_position_ids, pm_encoder_position_ids):
    # 1. Query projection
    query_states = self.q_proj(hidden_states)
    
    # 2. PM-RoPE on queries (デコーダ側)
    if pm_decoder_position_ids is not None:
        query_states = self._apply_rotary_with_progress(
            query_states, hidden_states, 
            pm_decoder_position_ids,  # ← 生成進行状況
            self.decoder_rotary_emb
        )
    
    # 3. Key projection
    key_states = self.k_proj(encoder_hidden_states)
    
    # 4. PM-RoPE on keys (エンコーダ側)
    if pm_encoder_position_ids is not None:
        key_states = self._apply_rotary_with_progress(
            key_states, encoder_hidden_states,
            pm_encoder_position_ids,  # ← テキスト位置
            self.encoder_rotary_emb
        )
    
    # 5. Attention計算
    attn_output = scaled_dot_product_attention(query, key, value)
    
    return attn_output
```

## 🎓 学習リソース / Learning Resources

### 関連論文 / Related Papers

1. **T5** - Text-to-Text Transfer Transformer
2. **RoPE** - Rotary Position Embedding
3. **XCodec2** - Neural Audio Codec
4. **VoiceStar** - 元のアーキテクチャのインスピレーション

### 重要な実装箇所 / Key Implementation Points

- `models/t5gemma.py:32-48` - RoPE適用関数
- `models/t5gemma.py:50-120` - PM-RoPEクロスアテンション
- `inference_tts_utils.py:107-300` - 推論パイプライン
- `data/tokenizer.py:50-150` - XCodec2ラッパー

## 🚀 実行方法 / How to Run

### 最小限の例 / Minimal Example

```bash
# 概念的な理解のため
python minimal_example.py
```

### 完全な推論 / Full Inference

```bash
# 実際に動作する推論
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "Hello, world!"
```

## 💡 次のステップ / Next Steps

1. `minimal_example.py` を実行して全体像を把握
2. `models/t5gemma.py` でPM-RoPEの実装を理解
3. `inference_tts_utils.py` で推論フローを追跡
4. 自分で小さな変更を加えて実験

## ❓ よくある質問 / FAQ

**Q: なぜEncoder-Decoderアーキテクチャ？**
A: テキスト（エンコーダ）と音声（デコーダ）のモダリティが異なるため。クロスアテンションで両者を関連付ける。

**Q: PM-RoPEの利点は？**
A: 通常のアテンションより生成の進行状況を明示的に追跡できる。

**Q: XCodec2の役割は？**
A: 連続的な音声波形を離散トークンに変換。Transformerが処理しやすくなる。

**Q: 学習はどうやる？**
A: `main.py` と `steps/trainer.py` を参照。分散学習対応。

---

**このガイドで理解できない部分があれば、該当するコードファイルを読んでください。**

**If you don't understand something, read the actual code files mentioned above.**
