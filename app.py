# -*- coding: utf-8 -*-
"""
HyperStrike 8K Local Optimizer v1.4 — 完全ローカル動作（Claudeアカウント不要）
起動:  python app.py  →  ブラウザが自動で開きます
- モニター選択: Apexを表示しているモニターを選んでキャプチャ
- 自動記録: アーム状態にすると、スティック操作＋画面の動きを検知して
  訓練所/マッチ開始時に自動で記録開始、非アクティブが続くと自動停止→解析
"""
import os
import sys

# 埋め込み版Python(runtime)対策: ._pth環境ではスクリプトのフォルダが
# importパスに入らないため、app.pyのあるフォルダを明示的に追加する
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ---- サイレント起動(pythonw / --noconsole EXE)対応 ----
# stdoutが無い場合は全出力を hyperstrike.log へ。以降のimport失敗も記録される
if sys.stdout is None or sys.stderr is None:
    _log = open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                             "hyperstrike.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log
    import atexit, traceback
    def _excepthook(t, v, tb):
        traceback.print_exception(t, v, tb, file=sys.stderr)
    sys.excepthook = _excepthook
    print("=" * 60)
    import time as _t
    print(f"[HyperStrike Local] 起動 {_t.strftime('%Y-%m-%d %H:%M:%S')} (silent mode)")

import threading
import time
import json
import math
import webbrowser
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

import analyzer

UI_VERSION = "v4.8"

FRAME_CAP = 600          # 記録フレーム上限（JPEG圧縮保持なので600枚でも約200MB）
CAP_FAST_SEC = 0.15      # ADS/射撃中のキャプチャ間隔（Vision時系列ペアを増やし信頼度向上）
CAP_SLOW_SEC = 0.6       # 通常時のキャプチャ間隔
CAP_SLOW_LATE = 1.5      # バッファ6割消費後の通常間隔（エイム時用に枠を温存）


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


try:
    import mss
except ImportError:
    mss = None
try:
    import pygame
except ImportError:
    pygame = None

app = FastAPI()


@app.middleware("http")
async def _local_origin_guard(request, call_next):
    """公開配布向けCSRF対策: 外部サイト起点のリクエストを拒否する。
    - Hostが127.0.0.1/localhost以外 → 拒否
    - Originヘッダが付いていて127.0.0.1/localhost以外 → 拒否
    （通常のUI操作はsame-originなので影響なし）"""
    from fastapi.responses import JSONResponse as _JR
    host = (request.headers.get("host") or "").split(":")[0]
    if host not in ("127.0.0.1", "localhost"):
        return _JR({"error": "forbidden host"}, status_code=403)
    origin = request.headers.get("origin")
    if origin:
        o = origin.split("//")[-1].split(":")[0]
        if o not in ("127.0.0.1", "localhost"):
            return _JR({"error": "forbidden origin"}, status_code=403)
    return await call_next(request)
engine = analyzer.VisionEngine()

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                             "settings.json")


def save_settings():
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump({"current": state["current"],
                       "bindings": state["bindings"],
                       "monitor": state.get("monitorIndex", state.get("monitor", 1))},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[警告] 設定保存に失敗: {e}")


def load_settings():
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d.get("current"), dict):
                state["current"] = d["current"]
            if isinstance(d.get("bindings"), dict):
                state["bindings"].update(d["bindings"])
            if "monitor" in d:
                if "monitorIndex" in state:
                    state["monitorIndex"] = int(d["monitor"])
                state["monitor"] = int(d["monitor"])
            print(f"[HyperStrike Local] 保存済み設定を読込: {SETTINGS_PATH}")
    except Exception as e:
        print(f"[警告] 設定読込に失敗: {e}")

state = {
    "recording": False,
    "armed": False,           # 自動記録待機
    "autoStarted": False,     # 現在の記録が自動開始によるものか
    "monitorIndex": 1,        # mssのモニター番号（1=プライマリ）
    "samples": [],
    "frames": [],
    "result": None,
    "resultSeq": 0,           # 解析結果の更新カウンタ（UIが新着を検知する用）
    "progress": {"phase": "", "done": 0, "total": 0},
    "bindings": {"fire": {"type": "axis", "index": 5},   # 既定: R2トリガー
                 "ads":  {"type": "axis", "index": 4}},  # 既定: L2トリガー
    "lastPoll": 0.0,           # UIからの最終ポーリング時刻
    "goodbyeAt": 0.0,          # タブが閉じられた通知の時刻
    "current": json.loads(json.dumps(analyzer.DEFAULT_CURRENT)),
    "status": "待機中",
    "screenMotion": 0.0,
    "stickActive": 0.0,       # 直近2秒のスティック活動率 0-1
}
lock = threading.Lock()
_activity_win = []            # (t, active:bool)
_last_active_t = time.perf_counter()

AUTO_START_ACTIVITY = 0.6     # 直近2秒の60%以上スティック操作で開始候補
AUTO_START_MOTION = 6.0       # 画面差分しきい値（0-255平均絶対差）
AUTO_STOP_IDLE_SEC = 30       # 無操作がこの秒数続いたら自動停止
MIN_AUTO_SAMPLES = 600        # 自動停止時、これ未満なら解析せず破棄


def _begin_recording(auto: bool):
    with lock:
        state["samples"].clear()
        state["frames"].clear()
        state["recording"] = True
        state["autoStarted"] = auto
        state["status"] = ("プレイ検知 → 自動記録中" if auto else "記録中（Apexをプレイしてください）")


def _load_prev_analysis():
    """直近の解析ログを読み込む（RC暴走防止：前回推奨との比較に使用）"""
    try:
        import glob
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        logs = sorted(glob.glob(os.path.join(base, "logs", "analysis_*.json")))
        if logs:
            with open(logs[-1], encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[警告] 前回解析ログの読込に失敗: {e}")
    return None


def _frame_img(fr):
    """キャプチャフレーム（JPEG圧縮 or 生配列）をBGR画像に復元する"""
    if "img" in fr:
        return fr["img"]
    import cv2
    return cv2.imdecode(np.frombuffer(fr["jpg"], dtype=np.uint8), cv2.IMREAD_COLOR)


def _run_analysis():
    with lock:
        state["recording"] = False
        state["status"] = "解析中…"
        samples = list(state["samples"])
        frames = list(state["frames"])
        current = state["current"]
    # ---- APEX profile.cfg 自動同期（感度・応答曲線・スコープ倍率のground truth） ----
    cfg_note = None
    try:
        cfgd = analyzer.read_apex_profile_cfg()
        if cfgd:
            current = json.loads(json.dumps(current))
            current.setdefault("apex", {}).update(cfgd)
            cfg_note = ("APEX設定自動同期: profile.cfgから "
                        + ", ".join(f"{k}={v}" for k, v in cfgd.items()
                                    if k != "optics")
                        + ("、スコープ倍率" if "optics" in cfgd else "")
                        + " を反映（ゲーム内実設定を優先）")
            print(f"[HyperStrike Local] {cfg_note}")
    except Exception as e:
        print(f"[警告] profile.cfg同期に失敗: {e}")

    # ---- 武器の自動認識（右下HUDの武器名をWindows標準OCRで読取り） ----
    weapon_note = None
    slot_note = None
    ocr_texts = []
    weapon_stats = None    # マルチ武器: 射撃時間内訳と武器別リコイル指標
    _auto_weapon = current.get("apex", {}).get("weapon") == "自動認識"
    _auto_barrel = current.get("apex", {}).get("barrel") == "自動認識"
    if _auto_weapon or _auto_barrel:
        with lock:
            state["progress"] = {"phase": "武器自動認識 (OCR)", "done": 0, "total": 1}
        detected, votes = None, {}
        weapon_timeline = []          # [(t_ms, 武器)] マッチ中の武器切替を追跡
        try:
            if not _auto_weapon:
                raise ImportError("weapon manual")
            import winocr
            import cv2
            texts = []
            # 時間軸に沿って最大24フレームをOCRし「いつどの武器か」を記録
            picks = frames[:: max(1, len(frames) // 24)][:24]
            for fr in picks:
                img = _frame_img(fr)
                h, w = img.shape[:2]
                # 右下HUD: 武器名/弾薬表示の領域（余裕を持って切出し。
                # アスペクト比16:10等でもHUDは右下アンカーなので比率指定で概ね追従）
                crop = img[int(h * 0.85):int(h * 0.985), int(w * 0.70):int(w * 0.995)]
                up = cv2.resize(crop, None, fx=3.0, fy=3.0,
                                interpolation=cv2.INTER_CUBIC)
                # 前処理バリエーション: 原色 + 白文字抽出の二値化（ゲームフォント対策）
                gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                _, binimg = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
                variants = [up, cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)]
                frame_weapon = None
                for v in variants:
                    for lang in ("en", "ja"):
                        try:
                            r = winocr.recognize_cv2_sync(v, lang)
                            if r and r.get("text"):
                                texts.append(r["text"])
                                if frame_weapon is None:
                                    frame_weapon = analyzer.match_weapon(r["text"])
                        except Exception:
                            continue
                    if frame_weapon:      # このフレームは判定済み → 残り前処理を省略
                        break
                if frame_weapon:
                    weapon_timeline.append((fr["t"], frame_weapon))
            ocr_texts = sorted(set(t.strip() for t in texts if t.strip()))[:40]
            detected, votes = analyzer.detect_weapon_from_text(texts)
        except ImportError as _ie:
            if str(_ie) == "weapon manual":
                weapon_note = None
            else:
                weapon_note = ("武器自動認識: winocr未導入のためスキップ"
                           "（setup_portable.bat再実行で導入されます）→「なし/その他」で解析")
        except Exception as e:
            weapon_note = f"武器自動認識に失敗: {e} →「なし/その他」で解析"
        # ---- アタッチメントスロットのレアリティ色検出（バレル推定） ----
        if _auto_barrel:
            try:
                import cv2
                rarities = []
                picks2 = frames[:: max(1, len(frames) // 6)][:6]
                for fr in picks2:
                    img = _frame_img(fr)
                    h, w = img.shape[:2]
                    # 武器名の上段にあるアタッチメントスロット列（右下HUD）
                    band = img[int(h * 0.825):int(h * 0.862), int(w * 0.80):int(w * 0.985)]
                    bw = band.shape[1] // 4
                    frame_r = []
                    for si in range(4):
                        cell = band[:, si * bw:(si + 1) * bw]
                        frame_r.append(analyzer.classify_slot_color(
                            cell.reshape(-1, 3).mean(axis=0)))
                    rarities.append(frame_r)
                # スロットごとに多数決
                slot_major = []
                for si in range(4):
                    col = [r[si] for r in rarities]
                    slot_major.append(max(set(col), key=col.count))
                # 最高レアリティをバレル推定に採用（スロット順は武器で異なるため保守的に）
                order = ["gold", "purple", "blue", "white"]
                best = next((c for c in order if c in slot_major), "empty")
                barrel = analyzer.RARITY_TO_BARREL.get(best, "なし")
                current = json.loads(json.dumps(current))
                current["apex"]["barrel"] = barrel
                slot_note = (f"アタッチメント自動認識: スロット色={slot_major} → "
                             f"バレル推定「{barrel}」でリコイル低減を適用"
                             f"（スロット順は武器により異なるため最高レアリティで推定。"
                             f"実際と違う場合は手動選択してください）")
            except Exception as e:
                current = json.loads(json.dumps(current))
                current["apex"]["barrel"] = "なし"
                slot_note = f"アタッチメント自動認識に失敗: {e} → バレル「なし」で解析"
            if slot_note:
                print(f"[HyperStrike Local] {slot_note}")

        # ---- マルチ武器対応: 射撃バーストごとに使用武器を割当て（v4.6） ----
        if _auto_weapon and weapon_timeline:
            tl_votes = {}
            for _, w in weapon_timeline:
                tl_votes[w] = tl_votes.get(w, 0) + 1
            weapon_stats = analyzer.weapon_firing_stats(
                samples, analyzer.firing_intervals(samples), weapon_timeline)
            dominant = weapon_stats.get("dominant")
            # 支配的武器もOCR2フレーム以上の裏付けを要求（誤読1回での確定を防止）
            if dominant and tl_votes.get(dominant, 0) < 2:
                dominant = None
            if dominant:
                current = json.loads(json.dumps(current))
                current["apex"]["weapon"] = dominant
                parts = []
                for w, st in sorted(weapon_stats["perWeapon"].items(),
                                    key=lambda kv: -kv[1]["fireMs"]):
                    seg = f"{w} {st['fireRatio']*100:.0f}%"
                    if "holdJitter" in st:
                        seg += f"(リコイル制御σ{st['holdJitter']:.2f})"
                    parts.append(seg)
                weapon_note = ("武器自動認識(マルチ): 射撃時間の内訳 "
                               + " / ".join(parts)
                               + f" → リコイル評価の代表武器「{dominant}」"
                               f"（OCR{len(weapon_timeline)}フレームで追跡）")
                detected = dominant
            else:
                # 射撃区間なし/射撃時の裏付け不足 → 全体多数決(2フレーム以上)へ
                mj = max(tl_votes.items(), key=lambda kv: kv[1])
                if mj[1] >= 2:
                    detected = mj[0]
                    current = json.loads(json.dumps(current))
                    current["apex"]["weapon"] = detected
                    weapon_note = (f"武器自動認識: 「{detected}」"
                                   f"({mj[1]}/{len(weapon_timeline)}フレーム、"
                                   f"射撃バーストとの対応付けは不成立)")
                else:
                    detected = None
        if _auto_weapon and detected is None and weapon_note is None and votes:
            best = max(votes.items(), key=lambda kv: kv[1])
            weapon_note = (f"武器自動認識: 「{best[0]}」はOCR裏付け不足のため不採用 → "
                           f"「なし/その他」で解析（誤読防止。手動選択も可能です）")
            current = json.loads(json.dumps(current))
            current["apex"]["weapon"] = "なし/その他"
        elif _auto_weapon and detected is None and weapon_note is None:
            current = json.loads(json.dumps(current))
            current["apex"]["weapon"] = "なし/その他"
            sample = " / ".join(ocr_texts[:3]) if ocr_texts else "（読取テキストなし）"
            weapon_note = (f"武器自動認識: HUDから武器名を判定できず →「なし/その他」で解析"
                           f"（OCR読取例: {sample}）")
        print(f"[HyperStrike Local] {weapon_note}")

    with lock:
        state["progress"] = {"phase": "入力ログ解析", "done": 0, "total": max(1, len(frames))}
    stick_m = analyzer.analyze_stick_log(samples)
    frame_results = []
    for i, fr in enumerate(frames):
        targets, rejected = engine.detect_persons(_frame_img(fr),
                                                  return_rejected=True)
        frame_results.append({"t": fr["t"], "targets": targets,
                              "rejected": rejected})
        with lock:
            state["progress"] = {"phase": f"AI画像解析 ({engine.active_provider})",
                                 "done": i + 1, "total": len(frames)}
    # 検出プレビュー保存: 検出があったフレームに枠を描いて logs/detections/ へ（最大12枚）
    try:
        import cv2
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        det_dir = os.path.join(base, "logs", "detections",
                               time.strftime("%Y%m%d_%H%M%S"))
        saved = 0
        cls_names = {0: "Teammate", 1: "Enemy"}
        for fr, res in zip(frames, frame_results):
            if (not res["targets"] and not res.get("rejected")) or saved >= 12:
                continue
            img = _frame_img(fr).copy()
            # 除外された検出はグレー枠+理由（フィルタの妥当性を目視確認できる）
            for tg in res.get("rejected", []):
                b = tg.get("box")
                if not b:
                    continue
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                              (128, 128, 128), 1)
                cv2.putText(img, f"except: {tg.get('reject','?')} {tg['conf']:.2f}",
                            (b[0], max(12, b[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
            for tg in res["targets"]:
                b = tg.get("box")
                if not b:
                    continue
                is_enemy = (engine.num_classes == 2 and tg.get("cls") == 1) \
                           or engine.num_classes != 2
                color = (60, 60, 230) if is_enemy else (230, 180, 60)  # BGR
                if tg.get("bot"):
                    color = (60, 200, 60)   # 訓練場Botモデルの検出は緑
                cv2.rectangle(img, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 2)
                label = "Bot" if tg.get("bot") else \
                        (cls_names.get(tg.get("cls"), f"cls{tg.get('cls')}")
                         if engine.num_classes == 2 else "person")
                cv2.putText(img, f"{label} {tg['conf']:.2f}", (b[0], max(12, b[1]-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            os.makedirs(det_dir, exist_ok=True)
            cv2.imwrite(os.path.join(det_dir, f"t{int(fr['t'])}ms.jpg"), img)
            saved += 1
        if saved:
            print(f"[HyperStrike Local] 検出プレビュー {saved}枚保存: {det_dir}")
    except Exception as e:
        print(f"[警告] 検出プレビュー保存に失敗: {e}")

    with lock:
        state["progress"] = {"phase": "提案生成", "done": len(frames), "total": max(1, len(frames))}
    firing_iv = analyzer.firing_intervals(samples)
    # 信頼できる検出 = IFFモデル、または訓練場Bot特化モデルが検出の主体(70%以上)
    bot_hits = sum(1 for r in frame_results for t in r["targets"] if t.get("bot"))
    all_hits = sum(len(r["targets"]) for r in frame_results)
    trusted_model = (engine.num_classes == 2) or (all_hits > 0
                                                  and bot_hits / all_hits >= 0.7)
    vision_m = analyzer.summarize_vision(frame_results, firing_iv, samples=samples,
                                         iff_model=trusted_model)
    history = _load_prev_analysis()
    rec, reasons, audit = analyzer.build_recommendation(current, stick_m, vision_m,
                                                        history=history)
    effects = analyzer.predict_effects(current, rec)
    if weapon_note:
        reasons.insert(0, weapon_note)
        audit.insert(0, {"rule": "_武器自動認識", "value": None, "threshold": "-",
                         "fired": True, "action": weapon_note})
    if slot_note:
        reasons.insert(1 if weapon_note else 0, slot_note)
        audit.insert(1 if weapon_note else 0,
                     {"rule": "_アタッチメント自動認識", "value": None, "threshold": "-",
                      "fired": True, "action": slot_note})
    if cfg_note:
        reasons.insert(0, cfg_note)
        audit.insert(0, {"rule": "_APEX設定同期", "value": None, "threshold": "-",
                         "fired": True, "action": cfg_note})
    # 解析ログをファイル保存（アプリフォルダ/logs/）
    log_path = None
    try:
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        log_dir = os.path.join(base, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir, time.strftime("analysis_%Y%m%d_%H%M%S.json"))
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "provider": engine.active_provider,
                "model": getattr(engine, "model_name", "なし"),
                "aiUsed": engine.session is not None,
                "sampleCount": len(samples),
                "frameCount": len(frames),
                "enemyFrames": sum(1 for r in frame_results if r["targets"]),
                "rejectedDetections": sum(len(r.get("rejected", []))
                                          for r in frame_results),
                "current": current,
                "stickMetrics": stick_m,
                "visionMetrics": vision_m,
                "weaponStats": weapon_stats,
                "ocrTexts": ocr_texts,
                "audit": audit,
                "recommendation": rec,
                "effects": effects,
                "reasons": reasons,
            }, f, ensure_ascii=False, indent=2)
        print(f"[HyperStrike Local] 解析ログ保存: {log_path}")
    except Exception as e:
        print(f"[警告] 解析ログ保存に失敗: {e}")
    with lock:
        state["result"] = {
            "rec": rec, "reasons": reasons, "audit": audit, "effects": effects,
            "stickMetrics": stick_m, "visionMetrics": vision_m,
            "aiUsed": engine.session is not None,
            "provider": engine.active_provider,
            "logPath": log_path,
        }
        state["resultSeq"] += 1
        state["status"] = "解析完了" + ("（自動）" if state["autoStarted"] else "")
        state["progress"] = {"phase": "完了", "done": 0, "total": 0}
    return state["result"]


def input_loop():
    """常時パッドをサンプリング。記録中はログへ、非記録中も活動検知に使う"""
    global _last_active_t
    if pygame is None:
        return
    pygame.init()
    pygame.joystick.init()
    js = None
    t0 = None
    while True:
        pygame.event.pump()
        if js is None or pygame.joystick.get_count() == 0:
            if pygame.joystick.get_count() > 0:
                js = pygame.joystick.Joystick(0)
                js.init()
            else:
                js = None
                time.sleep(0.3)
                continue
        now = time.perf_counter()
        lx, ly = js.get_axis(0), js.get_axis(1)
        rx = js.get_axis(2) if js.get_numaxes() > 3 else 0.0
        ry = js.get_axis(3) if js.get_numaxes() > 3 else 0.0
        # 射撃/ADSは任意ボタンに変更可能なため、学習済みバインディングで読む
        def _read(b):
            try:
                if b["type"] == "axis":
                    if js.get_numaxes() > b["index"]:
                        return (js.get_axis(b["index"]) + 1.0) / 2.0
                else:
                    if js.get_numbuttons() > b["index"]:
                        return 1.0 if js.get_button(b["index"]) else 0.0
            except Exception:
                pass
            return 0.0
        rt = _read(state["bindings"]["fire"])
        ad = _read(state["bindings"]["ads"])
        active = math.hypot(rx, ry) > 0.15 or math.hypot(lx, ly) > 0.15
        if active:
            _last_active_t = now
        _activity_win.append((now, active))
        while _activity_win and now - _activity_win[0][0] > 2.0:
            _activity_win.pop(0)
        if _activity_win:
            state["stickActive"] = sum(1 for _, a in _activity_win if a) / len(_activity_win)

        if state["recording"]:
            if t0 is None or len(state["samples"]) == 0:
                t0 = now
            with lock:
                state["samples"].append({
                    "t": (now - t0) * 1000,
                    "lx": lx, "ly": ly, "rx": rx, "ry": ry, "rt": rt, "ad": ad,
                })
        else:
            t0 = None
        time.sleep(1 / 120)


def capture_loop():
    """選択モニターをキャプチャ。非記録中も低頻度で画面差分を計測（自動開始判定用）"""
    if mss is None:
        return
    prev_small = None
    t0 = None
    while True:                       # 外側: mssコンテキストの再生成ループ
      try:                            # BitBlt失敗(画面モード変更/スリープ/ロック)で
        with getattr(mss, "MSS", mss.mss)() as sct:   # スレッドごと死なないよう保護
          while True:
            idx = state["monitorIndex"]
            if idx >= len(sct.monitors):
                idx = 1
            mon = sct.monitors[idx]
            if state["recording"]:
                if t0 is None or len(state["frames"]) == 0:
                    t0 = time.perf_counter()
                if len(state["frames"]) < FRAME_CAP:
                    shot = sct.grab(mon)
                    img = np.array(shot)[:, :, :3]
                    # JPEG圧縮で保持（600枚でもメモリ圧を抑え、記録180秒超を全カバー）
                    entry = {"t": (time.perf_counter() - t0) * 1000}
                    try:
                        import cv2
                        ok, buf = cv2.imencode(".jpg", img,
                                               [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if ok:
                            entry["jpg"] = buf.tobytes()
                        else:
                            entry["img"] = img
                    except ImportError:
                        entry["img"] = img
                    # 生フレーム保持(cv2不在)時は120枚まで（メモリ保護）
                    if "img" in entry and len(state["frames"]) >= 120:
                        entry = None
                    if entry is not None:
                        with lock:
                            state["frames"].append(entry)
                # ADS/射撃中は高頻度キャプチャ（エイム評価の時系列標本を確保）。
                # 非エイム時はバッファ残量に応じて間引き、エイム用の枠を温存する
                last = state["samples"][-1] if state["samples"] else None
                engaged = bool(last and (last.get("rt", 0) > 0.5
                                         or last.get("ad", 0) > 0.5))
                slow = (CAP_SLOW_LATE if len(state["frames"]) > FRAME_CAP * 0.6
                        else CAP_SLOW_SEC)
                time.sleep(CAP_FAST_SEC if engaged else slow)
            else:
                t0 = None
                # 画面の動き量（縮小グレースケール差分）
                shot = sct.grab(mon)
                img = np.array(shot)[:, :, :3]
                small = img[::16, ::16].mean(axis=2).astype(np.float32)
                if prev_small is not None and prev_small.shape == small.shape:
                    state["screenMotion"] = float(np.abs(small - prev_small).mean())
                prev_small = small
                time.sleep(1.0)
      except Exception as e:
        print(f"[警告] 画面キャプチャに失敗（{e}）。2秒後に再試行します")
        time.sleep(2.0)


def auto_loop():
    """アーム中の自動開始・自動停止の判定"""
    while True:
        time.sleep(0.5)
        if not state["armed"]:
            continue
        if not state["recording"]:
            screen_ok = (mss is None) or (state["screenMotion"] >= AUTO_START_MOTION)
            if state["stickActive"] >= AUTO_START_ACTIVITY and screen_ok:
                _begin_recording(auto=True)
        else:
            if state["autoStarted"]:
                idle = time.perf_counter() - _last_active_t
                if idle >= AUTO_STOP_IDLE_SEC:
                    if len(state["samples"]) >= MIN_AUTO_SAMPLES:
                        threading.Thread(target=_run_analysis, daemon=True).start()
                    else:
                        with lock:
                            state["recording"] = False
                            state["status"] = "記録が短すぎたため破棄（自動待機中）"


@app.get("/", response_class=HTMLResponse)
def index():
    with open(resource_path("static/index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read(),
                            headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/monitors")
def monitors():
    if mss is None:
        return {"monitors": []}
    with getattr(mss, "MSS", mss.mss)() as sct:
        mons = []
        for i, m in enumerate(sct.monitors):
            if i == 0:
                continue  # 0は全画面合成
            mons.append({"index": i, "width": m["width"], "height": m["height"],
                         "left": m["left"], "top": m["top"]})
        return {"monitors": mons, "selected": state["monitorIndex"]}


@app.post("/api/monitor")
async def set_monitor(payload: dict):
    with lock:
        state["monitorIndex"] = int(payload.get("index", 1))
        save_settings()
    return {"ok": True, "selected": state["monitorIndex"]}


@app.get("/api/settings")
def get_settings():
    return {"current": state["current"], "bindings": state["bindings"],
            "monitor": state.get("monitorIndex", state.get("monitor", 1))}


@app.post("/api/bindings")
async def set_bindings(payload: dict):
    with lock:
        for k in ("fire", "ads"):
            b = payload.get(k)
            if isinstance(b, dict) and b.get("type") in ("axis", "button"):
                state["bindings"][k] = {"type": b["type"], "index": int(b["index"])}
        save_settings()
    return {"ok": True, "bindings": state["bindings"]}


@app.post("/api/reset_settings")
def reset_settings():
    with lock:
        state["current"] = json.loads(json.dumps(analyzer.DEFAULT_CURRENT))
        state["bindings"] = {"fire": {"type": "axis", "index": 5},
                             "ads": {"type": "axis", "index": 4}}
        try:
            if os.path.exists(SETTINGS_PATH):
                os.remove(SETTINGS_PATH)
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/arm")
async def set_arm(payload: dict):
    with lock:
        state["armed"] = bool(payload.get("enabled", False))
        if state["armed"] and not state["recording"]:
            state["status"] = "自動記録 待機中（プレイ開始を検知します）"
        elif not state["armed"] and not state["recording"]:
            state["status"] = "待機中"
    return {"ok": True, "armed": state["armed"]}


@app.get("/api/status")
def status():
    state["lastPoll"] = time.perf_counter()
    state["goodbyeAt"] = 0.0   # ポーリング再開=タブ復帰
    with lock:
        return {
            "recording": state["recording"],
            "armed": state["armed"],
            "autoStarted": state["autoStarted"],
            "samples": len(state["samples"]),
            "frames": len(state["frames"]),
            "provider": engine.active_provider,
            "model": getattr(engine, "model_name", "なし"),
            "gpu": engine.gpu_active,
            "status": state["status"],
            "monitorIndex": state["monitorIndex"],
            "screenMotion": round(state["screenMotion"], 1),
            "stickActive": round(state["stickActive"], 2),
            "resultSeq": state["resultSeq"],
            "progress": state["progress"],
            "bindings": state["bindings"],
            "recent": state["samples"][-200:],
        }


@app.post("/api/current")
async def set_current(payload: dict):
    with lock:
        state["current"] = payload
        save_settings()
    return {"ok": True}


@app.post("/api/start")
def start():
    _begin_recording(auto=False)
    return {"ok": True}


@app.post("/api/stop")
def stop():
    threading.Thread(target=_run_analysis, daemon=True).start()
    return {"ok": True, "async": True}


@app.post("/api/goodbye")
def goodbye():
    state["goodbyeAt"] = time.perf_counter()
    return {"ok": True}


@app.post("/api/shutdown")
def shutdown():
    def _die():
        time.sleep(0.3)
        os._exit(0)
    threading.Thread(target=_die, daemon=True).start()
    return {"ok": True}


@app.get("/api/result")
def result():
    return JSONResponse(state["result"] or {})


def lifecycle_loop():
    """ブラウザタブが閉じられたら自動終了し、ログファイル等のロックを解放する。
    記録中/自動待機中は終了しない。"""
    while True:
        time.sleep(10)
        if state["recording"] or state["armed"]:
            continue
        now = time.perf_counter()
        lp = state["lastPoll"]
        if state["goodbyeAt"] and now - state["goodbyeAt"] > 60 and now - lp > 60:
            print("[HyperStrike Local] ブラウザが閉じられたため自動終了します")
            os._exit(0)
        if lp and now - lp > 900:   # 15分間UIからの応答なし
            print("[HyperStrike Local] 15分間UI無応答のため自動終了します")
            os._exit(0)


def _reclaim_stale_instances(ports=range(8720, 8725)):
    """旧バージョンの取り残しプロセスに終了要求を送り、ポートとログのロックを回収"""
    import urllib.request
    for p in ports:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/status",
                                        timeout=0.6) as r:
                body = r.read(2000).decode("utf-8", "ignore")
            if '"provider"' in body:   # 本アプリの旧インスタンスと判定
                print(f"[HyperStrike Local] ポート{p}の旧インスタンスを終了します")
                req = urllib.request.Request(f"http://127.0.0.1:{p}/api/shutdown",
                                             method="POST", data=b"{}")
                req.add_header("Content-Type", "application/json")
                try:
                    urllib.request.urlopen(req, timeout=0.6)
                except Exception:
                    pass
                time.sleep(1.0)
        except Exception:
            continue


def _find_port(preferred=8720, tries=10):
    import socket
    for i in range(tries):
        port = preferred + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                if i > 0:
                    print(f"[警告] ポート{preferred}は既に使用中です（旧バージョンが起動したままの可能性）。")
                    print("[警告] タスクバーの黒いウィンドウや古いEXEを終了することを推奨します。")
                    print(f"[HyperStrike Local] 代わりにポート {port} で起動します。")
                return port
            except OSError:
                continue
    raise RuntimeError("空きポートが見つかりません (8720-8729)")


if __name__ == "__main__":
    load_settings()
    threading.Thread(target=input_loop, daemon=True).start()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=auto_loop, daemon=True).start()
    _reclaim_stale_instances()
    threading.Thread(target=lifecycle_loop, daemon=True).start()
    port = _find_port()
    try:
        _html = open(resource_path("static/index.html"), encoding="utf-8").read()
        import re as _re
        _m = _re.search(r"UI (v[0-9.]+)", _html)
        _ver = _m.group(1) if _m else "不明"
        print(f"[HyperStrike Local] UI version: {_ver} (期待: {UI_VERSION})")
        if _ver != UI_VERSION:
            print("[警告] UIとサーバのバージョンが一致しません。最新ZIPのフォルダから起動してください。")
    except Exception as e:
        print(f"[警告] index.html を確認できません: {e}")
    print(f"[HyperStrike Local] 推論プロバイダ: {engine.active_provider}")
    print(f"[HyperStrike Local] 利用可能プロバイダ: {getattr(engine, 'available_providers', [])}")
    if (os.name == "nt" and engine.session is not None
            and engine.active_provider == "CPUExecutionProvider"):
        print("[警告] GPUプロバイダが見つかりません。CPU版onnxruntimeがDirectML版を")
        print("[警告] 上書きしている可能性があります。fix_gpu.bat を実行してください。")
    print(f"[HyperStrike Local] http://127.0.0.1:{port} を起動します（ブラウザが自動で開きます）")
    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    except Exception:
        import traceback
        traceback.print_exc()
        raise
