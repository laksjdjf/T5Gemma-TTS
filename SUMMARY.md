# 超シンプル版の実装完了 / Super Simple Version Implementation Complete

## 🎉 実装完了 / Implementation Complete

T5Gemma-TTSの**超シンプル版**の実装が完了しました！

The **super simple version** of T5Gemma-TTS has been implemented!

## 📁 新規追加ファイル / New Files Added

1. **inference_simple.py** - シンプルな推論スクリプト / Simple inference script
2. **README_SIMPLE.md** - シンプル版のREADME（日英対応） / Simple version README (JP/EN)
3. **requirements_simple.txt** - 最小限の依存関係 / Minimal dependencies
4. **examples_simple.py** - 使用例スクリプト / Example usage script
5. **SIMPLE_VERSION.md** - 完全版との比較文書 / Comparison with full version
6. **IMPLEMENTATION_NOTES.md** - 実装の詳細（日本語） / Implementation details (Japanese)

## 📝 変更ファイル / Modified Files

1. **README.md** - シンプル版へのリンク追加 / Added link to simple version
2. **README_ja.md** - シンプル版へのリンク追加 / Added link to simple version
3. **.gitignore** - *.wavを追加 / Added *.wav

## 💡 主な特徴 / Key Features

### シンプル版 / Simple Version

✅ **最小限の依存関係** / Minimal dependencies
- 学習不要なパッケージを除外
- 推論に必要なものだけ

✅ **簡単な使い方** / Easy to use
```bash
python inference_simple.py "こんにちは"
```

✅ **多言語対応** / Multilingual
- 英語、日本語、中国語に対応
- 自動言語検出

✅ **自動音声長推定** / Auto duration estimation
- ユーザーが指定する必要なし

❌ **含まれない機能** / Not included
- Voice cloning（音声クローニング）
- Duration control（音声長の手動制御）
- Training（学習機能）
- Gradio UI

## 📊 比較 / Comparison

| 項目 / Item | 完全版 / Full | シンプル版 / Simple |
|------------|--------------|-------------------|
| コマンド行数 / CLI lines | ~200 | ~130 |
| 必須パラメータ / Required params | 1 | 1 |
| 依存関係 / Dependencies | ~30 packages | ~15 packages |
| 学習機能 / Training | ✅ | ❌ |
| Voice Cloning | ✅ | ❌ |
| 基本TTS / Basic TTS | ✅ | ✅ |

## 🚀 使い方 / How to Use

### インストール / Installation

```bash
# シンプル版 / Simple version
pip install -r requirements_simple.txt

# 完全版 / Full version
pip install -r requirements.txt
```

### 実行 / Run

```bash
# シンプル版 / Simple version
python inference_simple.py "Hello, world!"

# 完全版 / Full version
python inference_commandline_hf.py \
    --model_dir Aratako/T5Gemma-TTS-2b-2b \
    --target_text "Hello, world!"
```

## 📖 ドキュメント / Documentation

- **シンプル版**: [README_SIMPLE.md](README_SIMPLE.md)
- **完全版**: [README.md](README.md), [README_ja.md](README_ja.md)
- **比較**: [SIMPLE_VERSION.md](SIMPLE_VERSION.md)
- **実装詳細**: [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)

## 🎯 目的 / Purpose

### 誰のため？ / Who is it for?

- 🎓 **学生** - TTS技術を学びたい / Students learning TTS
- 🔬 **研究者** - 素早く試したい / Researchers wanting quick tests
- 👨‍💻 **開発者** - シンプルな統合が必要 / Developers needing simple integration
- 🚀 **初心者** - 複雑な設定を避けたい / Beginners avoiding complex setup

### なぜ作った？ / Why was it created?

1. **エントリーバリアを下げる** / Lower entry barrier
2. **学習曲線を緩やかに** / Gentler learning curve
3. **迅速なプロトタイピング** / Rapid prototyping
4. **デプロイの簡素化** / Simplified deployment

## ✨ 次のステップ / Next Steps

### ユーザー向け / For Users

1. [README_SIMPLE.md](README_SIMPLE.md)を読む / Read README_SIMPLE.md
2. `requirements_simple.txt`をインストール / Install requirements_simple.txt
3. `inference_simple.py`を実行 / Run inference_simple.py
4. より高度な機能が必要なら完全版へ / Move to full version if needed

### 開発者向け / For Developers

- より多くの使用例を追加 / Add more examples
- パフォーマンスの最適化 / Performance optimization
- より詳細なエラーメッセージ / Better error messages
- チュートリアルの作成 / Create tutorials

## 🙏 謝辞 / Acknowledgments

この超シンプル版は、元のT5Gemma-TTSプロジェクトの素晴らしい仕事に基づいています。

This super simple version is based on the excellent work of the original T5Gemma-TTS project.

---

**質問やフィードバックは大歓迎です！** / **Questions and feedback are welcome!**

GitHub Issues: https://github.com/laksjdjf/T5Gemma-TTS/issues
