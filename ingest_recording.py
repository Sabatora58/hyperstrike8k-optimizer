# -*- coding: utf-8 -*-
"""外部レコーダー(AimSync Recorder等)のマッチ録画を取り込んで解析する。

アプリ内蔵の自前キャプチャ(最大600フレーム)より遥かに高品質な、
フルマッチの120Hzコントローラーログ＋60fps動画から
HyperStrike 8K の最適設定を算出する。

使い方:
    python ingest_recording.py "<録画フォルダ>"
録画フォルダに input_log.jsonl（必須）と apex_record.mp4（任意）が必要。
動画があり ffmpeg が使える場合はADS/射撃区間のフレームでVision解析も行う。
結果は logs/analysis_*.json に保存され、アプリのUI履歴・RC暴走ガードから参照される。
"""
import sys, os, json, glob, shutil, subprocess, tempfile, math

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
import analyzer


def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # よくある同梱場所も探索
    for pat in (r"C:\Users\*\Documents\**\recorder\ffmpeg.exe",
                os.path.join(_APP_DIR, "ffmpeg.exe")):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def load_samples(folder):
    jl = os.path.join(folder, "input_log.jsonl")
    if not os.path.exists(jl):
        raise FileNotFoundError(f"input_log.jsonl が見つかりません: {folder}")
    samples = []
    with open(jl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # 射撃/ADSはRB/LB等のボタン。アナライザーはrt>0.5/ad>0.5で判定するため写像
            fire = d.get("fire")
            ads = d.get("ads")
            fv = d.get("fire_value", 1.0 if fire else 0.0)
            av = d.get("ads_value", 1.0 if ads else 0.0)
            samples.append({
                "t": d.get("time_ms", d.get("main_receive_time_ms", 0)),
                "lx": d.get("left_x", 0.0), "ly": d.get("left_y", 0.0),
                "rx": d.get("right_x", 0.0), "ry": d.get("right_y", 0.0),
                "rt": 1.0 if (fire or fv > 0.5) else 0.0,
                "ad": 1.0 if (ads or av > 0.5) else 0.0,
            })
    return samples


def vision_from_video(folder, samples, engine):
    """動画のADS/射撃区間フレームをIFFモデルで解析してvisionMetricsを返す"""
    video = None
    for name in ("apex_record.mp4", "record.mp4"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            video = p
            break
    if not video:
        print("[ingest] 動画が見つかりません → 入力ログのみで解析します")
        return None, 0, 0
    ff = _find_ffmpeg()
    if not ff:
        print("[ingest] ffmpeg が見つかりません → 入力ログのみで解析します")
        return None, 0, 0
    try:
        import cv2
    except ImportError:
        print("[ingest] OpenCV未導入 → 入力ログのみで解析します")
        return None, 0, 0

    ads_iv = analyzer.firing_intervals(samples, key="ad", tail_ms=100.0)
    fire_iv = analyzer.firing_intervals(samples, key="rt", tail_ms=150.0)
    # ADSが少ない近距離ヒップファイア主体でも標本を確保するため射撃区間も対象に含める
    engage_iv = analyzer.firing_intervals(
        [{"t": s["t"], "e": 1.0 if (s["ad"] > 0.5 or s["rt"] > 0.5) else 0.0}
         for s in samples], key="e", tail_ms=120.0)

    tmp = tempfile.mkdtemp(prefix="hs_ingest_")
    fps = 4
    try:
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-i", video,
                        "-vf", f"fps={fps},scale=1536:960", "-q:v", "4",
                        os.path.join(tmp, "f_%05d.jpg")],
                       check=True)
        frames = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
        frame_results = []
        enemy_frames = 0
        for i, p in enumerate(frames):
            t = i * (1000.0 / fps)
            if not analyzer._in_intervals(t, engage_iv):
                continue
            kept, rej = engine.detect_persons(cv2.imread(p), return_rejected=True)
            if kept:
                enemy_frames += 1
            frame_results.append({"t": t, "targets": kept, "rejected": rej})
        bot_hits = sum(1 for r in frame_results
                       for t in r["targets"] if t.get("bot"))
        all_hits = sum(len(r["targets"]) for r in frame_results)
        trusted_model = (engine.num_classes == 2) or (all_hits > 0
                                                      and bot_hits / all_hits >= 0.7)
        vm = analyzer.summarize_vision(frame_results, firing_iv=fire_iv,
                                       samples=samples, iff_model=trusted_model)
        return vm, len(frame_results), enemy_frames
    except subprocess.CalledProcessError as e:
        print(f"[ingest] ffmpegフレーム抽出に失敗: {e}")
        return None, 0, 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print("使い方: python ingest_recording.py \"<録画フォルダ>\"")
        sys.exit(2)
    folder = sys.argv[1].strip('"')
    print(f"[ingest] 録画フォルダ: {folder}")
    samples = load_samples(folder)
    dur = (samples[-1]["t"] - samples[0]["t"]) / 1000.0 if samples else 0
    print(f"[ingest] サンプル数={len(samples)} 時間={dur:.0f}秒")

    engine = analyzer.VisionEngine()
    print(f"[ingest] モデル={engine.model_name} プロバイダ={engine.active_provider}")

    stick_m = analyzer.analyze_stick_log(samples)
    vision_m, used, enemy = vision_from_video(folder, samples, engine)
    if vision_m:
        print(f"[ingest] Vision: 解析{used}フレーム 敵検出{enemy} "
              f"信頼度={vision_m.get('confidence')}")

    # 現在の設定はアプリのsettings.jsonを基準に（なければデフォルト）
    current = json.loads(json.dumps(analyzer.DEFAULT_CURRENT))
    sp = os.path.join(_APP_DIR, "settings.json")
    if os.path.exists(sp):
        try:
            with open(sp, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("current"), dict):
                current = d["current"]
            print("[ingest] 現在の設定を settings.json から読込")
        except Exception:
            pass
    # APEX profile.cfg があれば感度・応答曲線・スコープ倍率を自動同期
    cfgd = analyzer.read_apex_profile_cfg()
    if cfgd:
        current = json.loads(json.dumps(current))
        current.setdefault("apex", {}).update(cfgd)
        print(f"[ingest] APEX profile.cfg同期: {cfgd}")

    # 直近の解析ログを履歴に（RC暴走ガード用）
    history = None
    logs = sorted(glob.glob(os.path.join(_APP_DIR, "logs", "analysis_*.json")))
    if logs:
        try:
            with open(logs[-1], encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    rec, reasons, audit = analyzer.build_recommendation(
        current, stick_m, vision_m, history=history)
    effects = analyzer.predict_effects(current, rec)

    import time
    log_dir = os.path.join(_APP_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out = os.path.join(log_dir, time.strftime("analysis_%Y%m%d_%H%M%S.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "external_recording",
            "sourceFolder": folder,
            "provider": engine.active_provider,
            "aiUsed": engine.session is not None,
            "sampleCount": len(samples),
            "current": current,
            "stickMetrics": stick_m,
            "visionMetrics": vision_m,
            "audit": audit,
            "recommendation": rec,
            "effects": effects,
            "reasons": reasons,
        }, f, ensure_ascii=False, indent=2)

    print("\n===== 推奨設定の要点 =====")
    for r in reasons:
        print(" -", r)
    print("\n----- 予測されるゲーム内効果 -----")
    for e in effects:
        print(" *", e)
    print(f"\n[ingest] 解析ログ保存: {out}")
    print("[ingest] アプリを起動すると、この結果が結果画面/履歴に反映されます。")


if __name__ == "__main__":
    main()
