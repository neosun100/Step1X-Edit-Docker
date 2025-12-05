<div align="center">
  <img src="assets/logo.png" height=100>
  <h1>Step1X-Edit Docker デプロイ版</h1>
  <p>🎨 インテリジェントGPU管理を備えたAI画像編集システム</p>
  
  [English](README_NEW.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](Dockerfile)
  [![GPU](https://img.shields.io/badge/GPU-CUDA%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
</div>

## 📖 プロジェクト概要

インテリジェントGPUメモリ管理を備えたStep1X-EditのプロダクショングレードのDockerデプロイメント。遅延ロード、即時オフロードをサポートし、単一コンテナで3つのアクセスモード（UI + API + MCP）を提供します。

### ✨ 主な機能

- 🚀 **ワンクリックデプロイ** - 最適なGPUを自動選択して起動
- 🧠 **スマートGPU管理** - 遅延ロード + 即時オフロード（アイドル時<1GB）
- 🎨 **モダンWeb UI** - ドラッグ&ドロップアップロード、リアルタイムプレビュー
- 🔌 **REST API** - 完全なAPIインターフェース、Swaggerドキュメント
- 🤖 **MCPサポート** - Model Context Protocol、AIアシスタント連携
- 🌍 **多言語対応** - 英語、簡体字中国語、繁体字中国語、日本語
- 🐳 **Docker最適化** - 単一コンテナ、外部アクセス対応
- 📊 **GPUモニタリング** - リアルタイムステータス表示と手動制御

## 🚀 クイックスタート

### 前提条件

- NVIDIA GPU（24GB以上のVRAM）
- NVIDIAドライバ 525以上
- Docker 20.10以上
- nvidia-docker2

### 3ステップで起動

```bash
# 1. 環境設定
cp .env.example .env
# .envのMODEL_PATHを編集

# 2. サービス起動（最適なGPUを自動選択）
bash start.sh

# 3. アクセス
# UI:  http://0.0.0.0:8000
# API: http://0.0.0.0:8000/docs
```

## 📦 インストール

### 方法1：Dockerデプロイ（推奨）

#### nvidia-dockerのインストール

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 環境変数の設定

```bash
# テンプレートをコピー
cp .env.example .env

# 設定を編集
nano .env
```

必須設定：
```bash
MODEL_PATH=/path/to/Step1X-Edit-model  # モデルパス
PORT=8000                               # サービスポート
GPU_IDLE_TIMEOUT=60                     # GPUアイドルタイムアウト（秒）
```

#### サービス起動

```bash
# ワンクリック起動（GPUを自動選択）
bash start.sh

# または手動起動
docker-compose up -d
```

#### デプロイメント検証

```bash
# テストスイートを実行
bash test_deployment.sh

# ヘルスチェック
curl http://0.0.0.0:8000/health

# ログ確認
docker-compose logs -f
```

### 方法2：直接実行

#### 依存関係のインストール

```bash
# Pythonパッケージをインストール
pip install -r requirements.txt

# flash-attentionをインストール
python scripts/get_flash_attn.py
# 出力に従って対応するwheelファイルをダウンロードしてインストール
```

#### サーバー起動

```bash
# 環境変数を設定
export MODEL_PATH=/path/to/model
export PORT=8000
export GPU_IDLE_TIMEOUT=60

# 統合サーバーを起動
python unified_server.py
```

## ⚙️ 設定

### 環境変数

| 変数名 | デフォルト値 | 説明 |
|--------|-------------|------|
| `PORT` | 8000 | サービスポート |
| `HOST` | 0.0.0.0 | バインドアドレス（全インターフェース） |
| `NVIDIA_VISIBLE_DEVICES` | 0 | GPU ID（start.shが自動選択） |
| `GPU_IDLE_TIMEOUT` | 60 | 自動オフロードタイムアウト（秒） |
| `MODEL_PATH` | - | Step1X-Editモデルパス（必須） |
| `ENABLE_UI` | true | Web UIを有効化 |
| `ENABLE_API` | true | REST APIを有効化 |
| `ENABLE_MCP` | true | MCPサーバーを有効化 |
| `DEFAULT_NUM_STEPS` | 28 | デフォルト推論ステップ数 |
| `DEFAULT_GUIDANCE_SCALE` | 6.0 | デフォルトCFGスケール |
| `DEFAULT_SIZE_LEVEL` | 1024 | デフォルト解像度 |

## 💻 使用方法

### Web UI

1. ブラウザを開く：`http://0.0.0.0:8000`
2. 画像をドラッグ&ドロップまたはクリックしてアップロード
3. 編集指示を入力（例：「人物に赤い帽子を追加」）
4. パラメータを調整：
   - **ステップ数** (10-50)：高いほど品質が向上
   - **ガイダンススケール** (1-15)：高いほどプロンプトの影響が強い
   - **解像度** (512/768/1024)：出力サイズ
   - **シード値**：再現可能な結果のため
5. 「画像を編集」をクリック
6. 比較結果を確認

### REST API

#### 画像編集

```bash
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@input.jpg" \
  -F "prompt=人物に赤い帽子を追加" \
  -F "num_steps=28" \
  -F "guidance_scale=6.0" \
  -F "size_level=1024" \
  --output result.png
```

#### GPUステータス確認

```bash
curl http://0.0.0.0:8000/api/gpu/status
```

#### 手動GPU制御

```bash
# CPUにオフロード（メモリに保持）
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# 完全解放（全キャッシュをクリア）
curl -X POST http://0.0.0.0:8000/api/gpu/release
```

#### APIドキュメント

インタラクティブSwagger UI：`http://0.0.0.0:8000/docs`

### MCP（Model Context Protocol）

#### Pythonクライアント

```python
from mcp import ClientSession

async with ClientSession() as session:
    result = await session.call_tool(
        "edit_image",
        {
            "image_path": "input.jpg",
            "prompt": "赤い帽子を追加",
            "num_steps": 28,
            "guidance_scale": 6.0
        }
    )
    print(f"保存先: {result['output_path']}")
```

詳細は [MCP_GUIDE.md](MCP_GUIDE.md) を参照

## 🧠 GPUメモリ管理

### インテリジェントリソース管理

```
未ロード ──初回(20-30s)──> GPU ──完了(2s)──> CPU ──次回(2-5s)──> GPU
   ↑                                          ↓
   └──────────────タイムアウト/解放(1s)────────┘
```

### メモリ状態

| 状態 | GPUメモリ | 説明 |
|------|-----------|------|
| 未ロード | <1GB | モデル未ロード |
| CPUキャッシュ | <1GB | メモリ内、高速リロード（2-5秒） |
| GPUアクティブ | ~40GB | GPU上、処理中 |

詳細は [GPU_MANAGEMENT.md](GPU_MANAGEMENT.md) を参照

## 📊 パフォーマンス

### ベンチマーク（H800 GPU）

| 操作 | 時間 | GPUメモリ |
|------|------|-----------|
| 初回ロード（ディスク→GPU） | 20-30秒 | ~40GB |
| 編集（1024px, 28ステップ） | 15-20秒 | ~40GB |
| リロード（CPU→GPU） | 2-5秒 | ~40GB |
| オフロード（GPU→CPU） | ~2秒 | <1GB |
| 解放（全クリア） | ~1秒 | <1GB |

## 🛠️ 技術スタック

- **フレームワーク**：FastAPI, Gradio
- **AI/ML**：PyTorch, Transformers, Diffusers
- **GPU**：CUDA 12.1, Flash Attention
- **コンテナ**：Docker, nvidia-docker2
- **プロトコル**：MCP（Model Context Protocol）
- **API**：REST, WebSocket, Swagger/OpenAPI

## 🧪 テスト

```bash
# 完全なテストスイートを実行
bash test_deployment.sh
```

## 🤝 コントリビューション

コントリビューションを歓迎します！以下の手順に従ってください：

1. リポジトリをフォーク
2. フィーチャーブランチを作成（`git checkout -b feature/amazing`）
3. 変更をコミット（`git commit -m '素晴らしい機能を追加'`）
4. ブランチにプッシュ（`git push origin feature/amazing`）
5. プルリクエストを作成

## 📝 変更履歴

### v1.2.0 (2025-12-06)
- ✨ 統合サーバー追加（UI + API + MCP）
- 🧠 インテリジェントGPUメモリ管理実装
- 🐳 Dockerデプロイメント、自動GPU選択
- 📚 完全なドキュメント
- 🧪 テストスイート

## 📄 ライセンス

このプロジェクトはApache License 2.0の下でライセンスされています - 詳細は [LICENSE](LICENSE) ファイルを参照してください

## 📞 お問い合わせとサポート

- **GitHub Issues**：[問題報告や機能リクエスト](https://github.com/neosun100/Step1X-Edit/issues)
- **Discord**：[コミュニティに参加](https://discord.gg/j3qzuAyn)
- **ドキュメント**：[完全なドキュメント](DEPLOYMENT.md)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/Step1X-Edit&type=Date)](https://star-history.com/#neosun100/Step1X-Edit)

## 📱 公式アカウントをフォロー

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<div align="center">
  Step1X-Editコミュニティが ❤️ を込めて制作
  <br>
  <sub>このプロジェクトが役に立ったら、⭐️ をお願いします</sub>
</div>
