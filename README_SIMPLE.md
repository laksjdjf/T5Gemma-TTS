# T5Gemma-TTS - 超シンプル版 / Super Simple Version

[![Model](https://img.shields.io/badge/Model-HuggingFace-yellow)](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)

これは T5Gemma-TTS の超シンプルな推論専用バージョンです。最小限の依存関係とコードで、簡単にテキスト音声合成を試せます。

This is a super simple inference-only version of T5Gemma-TTS. With minimal dependencies and code, you can easily try text-to-speech.

## 🚀 クイックスタート / Quick Start

### 1. インストール / Installation

```bash
git clone https://github.com/laksjdjf/T5Gemma-TTS.git
cd T5Gemma-TTS
pip install -r requirements_simple.txt
```

**Note:** GPU利用の場合は先にPyTorchをインストール / For GPU, install PyTorch first:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 使い方 / Usage

#### 基本的な使い方 / Basic Usage

```bash
# 英語 / English
python inference_simple.py "Hello, this is a test of text to speech."

# 日本語 / Japanese
python inference_simple.py "こんにちは、これは音声合成のテストです。"

# 中国語 / Chinese
python inference_simple.py "你好，这是语音合成测试。"
```

#### 出力ファイル名を指定 / Specify Output File

```bash
python inference_simple.py "Hello!" --output my_speech.wav
```

#### カスタムモデルを使用 / Use Custom Model

```bash
python inference_simple.py "Hello!" --model path/to/your/model
```

## 📦 必要なもの / Requirements

- Python 3.8+
- PyTorch 2.5+
- transformers
- torchaudio
- その他（requirements_simple.txt参照）

## 💡 機能 / Features

- ✅ 簡単な使い方 / Easy to use
- ✅ 最小限の依存関係 / Minimal dependencies
- ✅ 多言語対応（英語、日本語、中国語） / Multilingual support (EN, JA, ZH)
- ✅ 自動言語検出 / Automatic language detection
- ✅ 自動音声長推定 / Automatic duration estimation

## 🚫 含まれないもの / What's NOT Included

このシンプル版には以下の機能は含まれません：

This simple version does NOT include:

- ❌ Voice cloning（音声クローニング）
- ❌ Duration control（音声長の手動指定）
- ❌ Training code（学習コード）
- ❌ Advanced parameters（高度なパラメータ）
- ❌ Gradio UI（Web UI）

これらの機能が必要な場合は、[完全版のREADME](README.md)を参照してください。

For these features, see the [full README](README.md).

## 📝 オプション / Options

```bash
python inference_simple.py --help
```

| オプション | デフォルト | 説明 |
|-----------|---------|------|
| `text` | (必須) | 音声合成するテキスト / Text to synthesize |
| `--model` | `Aratako/T5Gemma-TTS-2b-2b` | モデルのパス / Model path |
| `--output` | `output.wav` | 出力ファイル名 / Output filename |

## 🎯 推奨環境 / Recommended Environment

- GPU: NVIDIA GPU with 8GB+ VRAM
- CPU: 動作しますが遅いです / Works but slow
- OS: Linux, macOS, Windows (WSL2 recommended)

## ⚠️ 制限事項 / Limitations

- リアルタイム生成には不向き / Not suitable for real-time
- 生成に数秒～数十秒かかります / Takes several seconds to generate
- GPU推奨（CPUでも動作しますが非常に遅い） / GPU recommended

## 🔗 リンク / Links

- [完全版 README / Full README](README.md)
- [モデルカード / Model Card](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)
- [デモ / Demo](https://huggingface.co/spaces/Aratako/T5Gemma-TTS-Demo)

## 📄 ライセンス / License

MIT License - [LICENSE](LICENSE)

---

**より高度な機能が必要ですか？** [完全版のドキュメント](README.md)をご覧ください。

**Need more advanced features?** Check the [full documentation](README.md).
