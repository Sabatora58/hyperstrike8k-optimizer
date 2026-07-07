# -*- coding: utf-8 -*-
"""
prep_dataset.py — S29データセット整形（擬似ラベリング）
dataset/raw の生スクリーンショットに対し、現在のモデル(apex.onnx等)で
Enemy/Teammate を自動検出してYOLO形式ラベルを生成し、train/valid に分割する。

使い方:  python prep_dataset.py [--conf 0.30] [--keep-empty 0.1]
出力:    dataset/apex_s29/{train,valid}/{images,labels}/ と apex_s29.yaml

注意: 擬似ラベルは完璧ではありません。学習品質を上げたい場合は
labelImg等で labels/*.txt を目視修正してから train_s29.bat を実行してください。
クラス: 0=Teammate, 1=Enemy（訓練所ダミーはEnemyとして扱われます）
"""
import argparse
import glob
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import analyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30,
                    help="擬似ラベルの信頼度しきい値")
    ap.add_argument("--keep-empty", type=float, default=0.10,
                    help="検出なし画像を負例として残す割合(0-1)")
    ap.add_argument("--valid", type=float, default=0.10, help="検証データ割合")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    raw = sorted(glob.glob(os.path.join(base, "dataset", "raw", "*.jpg")))
    if not raw:
        print("dataset/raw に画像がありません。アプリの「学習用データセット収集」を"
              "ONにして計測（訓練所＋実戦）を行ってください。")
        return 1

    engine = analyzer.VisionEngine()
    if engine.session is None:
        print("モデルが読み込めません（models/apex.onnx か yolov8n.onnx が必要）")
        return 1
    print(f"モデル: {engine.model_name} / プロバイダ: {engine.active_provider}")

    out = os.path.join(base, "dataset", "apex_s29")
    for split in ("train", "valid"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, split, sub), exist_ok=True)

    random.seed(29)
    labeled = empty_kept = skipped = 0
    for i, path in enumerate(raw):
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        targets = engine.detect_persons(img, conf_th=args.conf)
        lines = []
        for t in targets:
            b = t.get("box")
            if not b:
                continue
            # 2クラスモデルはclsをそのまま、80クラス(person)はEnemy(1)として扱う
            cls = t.get("cls", 1) if engine.num_classes == 2 else 1
            cx = (b[0] + b[2] / 2) / w
            cy = (b[1] + b[3] / 2) / h
            bw = b[2] / w
            bh = b[3] / h
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        if not lines and random.random() > args.keep_empty:
            skipped += 1
            continue
        split = "valid" if random.random() < args.valid else "train"
        name = os.path.basename(path)
        shutil.copy2(path, os.path.join(out, split, "images", name))
        with open(os.path.join(out, split, "labels",
                               name.rsplit(".", 1)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        if lines:
            labeled += 1
        else:
            empty_kept += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(raw)} 処理済み…")

    yaml_path = os.path.join(base, "dataset", "apex_s29.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"""path: {out}
train: train/images
val: valid/images
names:
  0: Teammate
  1: Enemy
""")
    print(f"完了: ラベル付き {labeled}枚 / 負例 {empty_kept}枚 / 除外 {skipped}枚")
    print(f"データセット定義: {yaml_path}")
    print("次のステップ: (任意)labelImg等でラベルを目視修正 → train_s29.bat を実行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
