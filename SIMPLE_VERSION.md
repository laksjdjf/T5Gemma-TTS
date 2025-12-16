# 超シンプル版について / About the Super Simple Version

## 🎯 目的 / Purpose

このプロジェクトには2つのバージョンがあります：

This project has two versions:

### 完全版 / Full Version
- 学習コード、高度な機能、カスタマイズ可能なパラメータ
- Training code, advanced features, customizable parameters
- ファイル: `inference_commandline_hf.py`, `inference_gradio.py`
- ドキュメント: [README.md](README.md), [README_ja.md](README_ja.md)

### 超シンプル版 / Super Simple Version ⭐
- **推論のみ**、最小限の依存関係、簡単な使い方
- **Inference only**, minimal dependencies, easy to use
- ファイル: `inference_simple.py`
- ドキュメント: [README_SIMPLE.md](README_SIMPLE.md)

## 📋 比較表 / Comparison

| 機能 / Feature | 完全版 / Full | シンプル版 / Simple |
|---------------|--------------|-------------------|
| 基本的なTTS / Basic TTS | ✅ | ✅ |
| 多言語対応 / Multilingual | ✅ | ✅ |
| Voice Cloning | ✅ | ❌ |
| Duration Control | ✅ | ❌ |
| 学習コード / Training | ✅ | ❌ |
| Gradio UI | ✅ | ❌ |
| 依存関係 / Dependencies | 多い / Many | 少ない / Minimal |
| セットアップの簡単さ / Setup | 普通 / Normal | 簡単 / Easy |

## 🚀 クイックスタート / Quick Start

### シンプル版を使う場合 / Using Simple Version

```bash
# インストール / Install
pip install -r requirements_simple.txt

# 実行 / Run
python inference_simple.py "Hello, world!"
```

### 完全版を使う場合 / Using Full Version

```bash
# インストール / Install
pip install -r requirements.txt

# 実行 / Run
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "Hello, world!"
```

## 🔄 どちらを選ぶべきか？ / Which Should You Choose?

### シンプル版を選ぶべき場合 / Choose Simple Version If:
- ✅ 音声合成を素早く試したい
- ✅ Quick TTS testing
- ✅ 最小限のセットアップで済ませたい
- ✅ Minimal setup preferred
- ✅ 基本機能だけで十分
- ✅ Basic features are enough

### 完全版を選ぶべき場合 / Choose Full Version If:
- ✅ Voice cloningが必要
- ✅ Need voice cloning
- ✅ 音声の長さを細かく制御したい
- ✅ Need duration control
- ✅ モデルの学習が必要
- ✅ Need model training
- ✅ Gradio UIを使いたい
- ✅ Want Gradio UI

## 📁 ファイル一覧 / File List

### シンプル版関連ファイル / Simple Version Files
```
inference_simple.py          # メインスクリプト / Main script
README_SIMPLE.md             # ドキュメント / Documentation
requirements_simple.txt      # 依存関係 / Dependencies
examples_simple.py           # 使用例 / Examples
SIMPLE_VERSION.md            # このファイル / This file
```

### 共通ファイル / Common Files
```
data/                        # データ処理モジュール
models/                      # モデル定義
duration_estimator.py        # 音声長推定
inference_tts_utils.py       # 推論ユーティリティ
```

## 💡 ヒント / Tips

### シンプル版の使い方 / Using Simple Version

```bash
# 基本 / Basic
python inference_simple.py "テキスト"

# 出力ファイル指定 / Specify output
python inference_simple.py "テキスト" --output my_audio.wav

# カスタムモデル / Custom model
python inference_simple.py "テキスト" --model path/to/model

# ヘルプ / Help
python inference_simple.py --help
```

### よくある質問 / FAQ

**Q: シンプル版で音声クローニングはできますか？**
A: いいえ、シンプル版は基本的なTTS機能のみです。音声クローニングには完全版を使用してください。

**Q: Can I do voice cloning with the simple version?**
A: No, the simple version only has basic TTS. Use the full version for voice cloning.

**Q: シンプル版の方が速いですか？**
A: いいえ、生成速度は同じです。違いは機能と依存関係の数です。

**Q: Is the simple version faster?**
A: No, generation speed is the same. The difference is in features and dependencies.

**Q: シンプル版から完全版への移行は簡単ですか？**
A: はい、両方とも同じモデルを使用しているので、完全版に切り替えるだけで高度な機能を使えます。

**Q: Is it easy to migrate from simple to full version?**
A: Yes, both use the same model, so you can just switch to the full version for advanced features.

## 📚 詳細情報 / More Information

- シンプル版の使い方: [README_SIMPLE.md](README_SIMPLE.md)
- Simple version usage: [README_SIMPLE.md](README_SIMPLE.md)
- 完全版の使い方: [README.md](README.md)
- Full version usage: [README.md](README.md)
- モデル情報: [HuggingFace Model Card](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)
- Model info: [HuggingFace Model Card](https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b)

---

**質問や問題がありますか？** GitHubのIssuesで報告してください。

**Questions or issues?** Report them in GitHub Issues.
