# 解決策：T5Gemma-TTS 超シンプル版 / Solution: T5Gemma-TTS Super Simple Version

## 問題 / Problem

**要求**: このリポジトリのスーパーシンプルバージョンをつくって

**Request**: Create a super simple version of this repository

## 解決策 / Solution

T5Gemma-TTSの**超シンプル版**を実装しました。これは完全版と並行して使用でき、初心者や素早くTTSを試したいユーザーに最適です。

Implemented a **super simple version** of T5Gemma-TTS that runs alongside the full version, perfect for beginners and users who want to quickly try TTS.

## 実装内容 / What Was Implemented

### 1. コアファイル / Core Files

#### `inference_simple.py`
最小限のパラメータで動作するシンプルな推論スクリプト。

Simple inference script with minimal parameters.

**特徴 / Features:**
- 3つのパラメータのみ（text, --model, --output）
- 自動言語検出
- 自動音声長推定
- エラーハンドリング
- デバイス互換性（CPU/GPU、bfloat16/float16自動選択）

**使用例 / Usage:**
```bash
python inference_simple.py "Hello, world!"
python inference_simple.py "こんにちは" --output my_audio.wav
```

### 2. ドキュメント / Documentation

#### `README_SIMPLE.md`
シンプル版専用のREADME（日英両対応）

Dedicated README for simple version (bilingual JP/EN).

**内容 / Contents:**
- インストール手順
- 基本的な使い方
- 含まれない機能の明示
- よくある質問

#### `SIMPLE_VERSION.md`
完全版とシンプル版の詳細な比較文書

Detailed comparison between full and simple versions.

**内容 / Contents:**
- 機能比較表
- 使い分けのガイド
- FAQ

#### `IMPLEMENTATION_NOTES.md`
実装の技術的詳細（日本語）

Technical implementation details (Japanese).

#### `SUMMARY.md`
実装完了のサマリー（日英両対応）

Implementation summary (bilingual JP/EN).

### 3. 依存関係 / Dependencies

#### `requirements_simple.txt`
推論に必要な最小限のパッケージのみ

Only minimal packages needed for inference.

**削減されたパッケージ / Removed packages:**
- 学習関連（wandb, tensorboard）
- Gradio UI
- その他推論に不要なもの

**結果 / Result:** ~30パッケージ → ~15パッケージ

### 4. 使用例 / Examples

#### `examples_simple.py`
複数の使用例を実行するデモスクリプト

Demo script showing multiple usage examples.

**例 / Examples:**
- 英語TTS
- 日本語TTS
- 中国語TTS
- カスタム出力ファイル名

### 5. 既存ファイルの更新 / Updates to Existing Files

#### `README.md` & `README_ja.md`
メインREADMEにシンプル版へのリンクを追加

Added prominent links to simple version in main READMEs.

#### `.gitignore`
生成された音声ファイル（*.wav）を除外

Added *.wav to exclude generated audio files.

## 設計原則 / Design Principles

### 1. シンプルさ優先 / Simplicity First
- 必要最小限の機能のみ
- 複雑な設定を排除
- 分かりやすいAPI

### 2. 既存コードの再利用 / Code Reuse
- `inference_tts_utils`を活用
- `duration_estimator`を活用
- 新規コードは最小限

### 3. 後方互換性 / Backward Compatibility
- 完全版は一切変更なし
- 新規ファイルとして追加
- 既存ユーザーへの影響ゼロ

### 4. 初心者フレンドリー / Beginner Friendly
- 明確なエラーメッセージ
- デバイス自動検出
- 適切なデフォルト値

## 技術的改善 / Technical Improvements

### エラーハンドリング / Error Handling
```python
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    print("Error: transformers is not installed.")
    print("Please install it with: pip install transformers")
    sys.exit(1)
```

### デバイス互換性 / Device Compatibility
```python
if device == "cuda" and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16
```

### ディレクトリ処理 / Directory Handling
```python
output_dir = os.path.dirname(output_path)
if output_dir:  # Only create if path includes directory
    os.makedirs(output_dir, exist_ok=True)
```

## 統計 / Statistics

- **新規ファイル / New files**: 7
- **変更ファイル / Modified files**: 3
- **追加行数 / Lines added**: 744
- **削減された依存関係 / Dependencies reduced**: ~50%

## 使用方法 / How to Use

### インストール / Installation

```bash
# リポジトリをクローン / Clone repository
git clone https://github.com/laksjdjf/T5Gemma-TTS.git
cd T5Gemma-TTS

# PyTorchをインストール（GPU使用の場合） / Install PyTorch (for GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 依存関係をインストール / Install dependencies
pip install -r requirements_simple.txt
```

### 実行 / Execution

```bash
# 基本的な使い方 / Basic usage
python inference_simple.py "テキストをここに入力"

# 出力ファイルを指定 / Specify output file
python inference_simple.py "Hello!" --output speech.wav

# カスタムモデルを使用 / Use custom model
python inference_simple.py "Hello!" --model path/to/model
```

## 利点 / Benefits

### ユーザー向け / For Users
1. **迅速な試用** - 複雑な設定なしで即座に使用可能
2. **学習しやすい** - シンプルなインターフェース
3. **軽量** - 少ない依存関係
4. **明確** - 何ができて何ができないかが明確

### 開発者向け / For Developers
1. **保守性** - 既存コードへの影響なし
2. **拡張性** - 将来的な機能追加が容易
3. **テスト性** - シンプルなコードでテストが容易
4. **文書化** - 充実したドキュメント

### プロジェクト向け / For the Project
1. **アクセシビリティ** - より多くのユーザーが利用可能
2. **教育価値** - 学習教材として最適
3. **コミュニティ** - 初心者の参加を促進
4. **採用率** - エントリーバリアを下げる

## 今後の展開 / Future Work

### 短期 / Short Term
- [ ] より詳細な使用例の追加
- [ ] チュートリアル動画の作成
- [ ] ユーザーフィードバックの収集

### 中期 / Medium Term
- [ ] バッチ処理のサポート
- [ ] より詳細なエラーメッセージ
- [ ] パフォーマンスの最適化

### 長期 / Long Term
- [ ] 他の言語への対応拡大
- [ ] より高度な機能の段階的追加
- [ ] コミュニティからの貢献の統合

## 結論 / Conclusion

この超シンプル版の実装により、T5Gemma-TTSはより広いユーザー層にアクセス可能になり、同時に既存の高度なユーザーも引き続き完全版の全機能を利用できます。

This super simple version makes T5Gemma-TTS accessible to a wider audience while maintaining full functionality for advanced users.

---

**ドキュメント / Documentation:**
- [README_SIMPLE.md](README_SIMPLE.md) - シンプル版の使い方
- [SIMPLE_VERSION.md](SIMPLE_VERSION.md) - 詳細な比較
- [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) - 実装詳細
- [SUMMARY.md](SUMMARY.md) - 実装サマリー

**質問や提案は歓迎です！ / Questions and suggestions welcome!**
