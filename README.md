# HyperStrike 8K Optimizer for Apex Legends

Apex Legendsのプレイ映像とコントローラー入力を**完全ローカルのAI**で解析し、
HyperStrike 8K（HyperStrike Hub）の各設定項目の最適値を提案するWindows用ツールです。

- 画面キャプチャとスティック/トリガー入力（約120Hz）を同期記録
- YOLOベースの敵検出（GPU優先: DirectML / CUDA、CPUフォールバック）で
  クロスヘア偏差からオーバーシュート/アンダーシュートを計測
- 射撃ボタン入力から**射撃区間を実測分離**し、リコイルとエイム癖を混同しない解析
- 武器（HUDのOCR）とアタッチメント（スロット色）の自動認識によるリコイル補正
- APEXゲーム内感度（視点/ADS/詳細スコープ）を組み込んだ提案
- HyperStrike Hubのバックアップ(.json)の**読込**と、解析結果を反映した**プロファイル書出**
- シーズン対応カスタムモデルの学習パイプライン同梱（[docs/TRAINING.md](docs/TRAINING.md)）

> **本ツールはオフライン解析専用です。** ゲームのメモリへのアクセス、入力の注入・自動化は
> 一切行いません。記録済みのプレイを事後解析して設定値を提案するだけのツールです。

## 動作環境

- Windows 10/11（64bit）
- HyperStrike 8K（HyperStrike Hub / FW 2.5x以降）
- インターネット接続（初回セットアップのみ）
- Pythonのインストールは**不要**（ポータブル環境を自動構築）

## クイックスタート

1. [Releases](../../releases) から最新のZIPをダウンロードして展開
2. `setup_portable.bat` をダブルクリック（初回のみ。約5〜10分）
   - ポータブルPython・依存パッケージ・DirectML推論ランタイム・検出モデルを
     `runtime` / `models` フォルダに自動構築します。PCには何もインストールしません
3. `start_portable.bat` をダブルクリック → ブラウザが自動で開きます
4. 画面の指示に従って: HyperStrike Hubのバックアップを読込 → モニターと射撃ボタンを選択 →
   計測（訓練場か実戦を2〜5分）→ 解析 → 提案されたプロファイルを書出してHubで復元


## 主なファイル

| ファイル | 役割 |
|---|---|
| `app.py` | ローカルサーバー本体（キャプチャ・入力記録・API） |
| `analyzer.py` | 解析エンジン（メトリクス・提案・GPU推論） |
| `static/index.html` | ブラウザUI |
| `prep_dataset.py`  | カスタムモデル学習パイプライン |
| `setup_portable.bat` / `start_portable.bat` | セットアップ / 起動 |

## プライバシー

すべての処理はローカルで完結します。スクリーンショット・入力ログ・解析結果が
外部へ送信されることはありません（初回セットアップのパッケージ取得を除き通信しません）。

## ライセンス

本リポジトリは **AGPL-3.0** で公開されています
（[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)（AGPL-3.0）に由来する
検出モデルを利用するため）。詳細は `LICENSE` を参照してください。

- Apex Legends は Electronic Arts Inc. / Respawn Entertainment の商標です。
  本ツールは非公式であり、EA/Respawnとは無関係です。
- 学習用データセット（ゲームのスクリーンショット）は各自のローカルでのみ利用し、
  再配布しないでください。
