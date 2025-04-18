# Z-works 系統用蓄電池 営業アシスタントAI

Z-works社の系統用蓄電池製品に関する営業活動をサポートするAIアシスタントです。製品情報の提供、技術的な質問への回答、見積もり支援などを行います。

## 概要

このプロジェクトは、Z-works社の系統用蓄電池製品の営業をサポートするためのAIチャットボットを提供します。OpenAI APIを使用して、製品知識、競合分析、導入事例などに基づいた回答を生成します。

## 特徴

- **製品情報の提供**: Z-works社の蓄電池製品の詳細情報を提供
- **技術サポート**: 技術的な質問に対する正確な回答
- **競合分析**: 競合他社製品との比較情報
- **見積支援**: お客様の要件に基づいた適切な製品提案
- **導入事例の紹介**: 類似顧客の成功事例の共有
- **モダンなWebインターフェース**: 使いやすいチャットUIを提供

## プロジェクト構成

```
z-works-battery-assistant/
├── api_server.py         # FastAPIバックエンド
├── implementation_patched.py  # AIアシスタント実装
├── patch_langchain.py    # Windows互換性パッチ
├── windows_compat.py     # Windows用互換モジュール
├── requirements_web.txt  # 必要なパッケージ
├── .env                  # 環境設定
├── .env.example          # 環境設定例
├── static/               # WebフロントエンドUI
└── docs/                 # 製品資料・技術情報
    ├── products.md       # 製品カタログ
    ├── technical_specs.md # 技術仕様書
    ├── use_cases.md      # 導入事例
    ├── faq.md            # よくある質問と回答
    └── competitive_analysis.md # 競合比較分析
```

## セットアップ手順

1. **必要なパッケージのインストール**:

```bash
pip install -r requirements_web.txt
```

2. **OpenAI APIキーの設定**:

`.env`ファイルを作成または編集し、OpenAI APIの接続情報を設定します。

```
# OpenAI API設定
OPENAI_API_KEY=your_api_key_here

# アプリケーション設定
MODEL_NAME=gpt-4  # 使用するモデル
TEMPERATURE=0.7   # 温度パラメータ

# ドキュメント設定
DOCS_DIR=./docs   # 製品資料を置くディレクトリ
DB_DIR=./db       # ベクトルDBの保存ディレクトリ
```

3. **製品資料の準備**:

`docs`ディレクトリに製品情報、技術仕様、導入事例などのマークダウンファイルを配置します。

## 実行方法

アプリケーションを起動するには以下のコマンドを実行します:

```bash
python api_server.py
```

ブラウザで `http://localhost:8000` にアクセスすると、チャットインターフェースが表示されます。

## 機能拡張予定

- ユーザー認証機能の追加
- 会話履歴の保存と分析
- 見積書の自動生成
- CRMとの連携
- モバイルアプリ対応

## 開発者向け情報

OpenAI APIとLangChainを使用したRAGアプローチを実装しています。`docs`ディレクトリに追加された資料は自動的にベクトル化され、ユーザーの質問に関連する情報が検索されてAIの回答に活用されます。

---

© 2023 Z-works Corporation. All rights reserved. 