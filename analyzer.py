# -*- coding: utf-8 -*-
"""
HyperStrike 8K Local Analyzer
- 敵検出: YOLOv8n (ONNX) をローカル推論。GPUを優先使用。
- 実行プロバイダ優先順: TensorRT > CUDA > DirectML > CPU
- 入力ログ解析 + 画面解析 → HyperStrike Hub 実機スキーマに沿った推奨値を生成
"""
import math
import os
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

def _model_path():
    """Apex特化モデル(apex.onnx)を最優先、なければ汎用yolov8n.onnx。
    EXE実行時はEXEと同じフォルダの models/ を優先"""
    import sys
    bases = []
    if getattr(sys, "frozen", False):
        bases.append(os.path.dirname(sys.executable))
        if getattr(sys, "_MEIPASS", ""):
            bases.append(sys._MEIPASS)
    bases.append(os.path.dirname(os.path.abspath(__file__)))
    candidates = []
    for name in ("apex.onnx", "yolov8n.onnx"):   # apex優先
        for b in bases:
            candidates.append(os.path.join(b, "models", name))
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[-1]


MODEL_PATH = _model_path()

# ------------------------------------------------------------
# 実機 HyperStrike Hub スキーマ（hs2.evua.cc 解析結果 / FW 2.6x）
# ------------------------------------------------------------
SCHEMA = {
    "pollingRate":   {"label": "ポーリングレート", "options": [250, 500, 1000, 2000, 4000, 8000], "unit": "Hz"},
    "reportRate":    {"label": "レポート頻度", "options": [250, 500, 1000, 2000, 4000, 8000, 32000], "unit": "Hz"},
    "stickSampling": {"label": "スティックサンプリング", "options": ["Extreme", "Excellent", "Good", "Robust"]},
    "quantization":  {"label": "スティック量子化(X360)", "options": ["無制限", "11bit", "10bit", "9bit", "8bit"]},
    "advSampling":   {"label": "高度サンプリングレベル", "options": ["オフ", "14bit", "15bit", "16bit"]},
    # 左右スティック共通レンジ
    "centerDZ": {"label": "中心デッドゾーン", "min": 0, "max": 30, "step": 1, "unit": "%"},
    "antiDZ":   {"label": "アンチデッドゾーン", "min": 0, "max": 30, "step": 1, "unit": "%"},
    "outerDZ":  {"label": "外周デッドゾーン", "min": 0, "max": 30, "step": 1, "unit": "%"},
    "curvePreset": {"label": "カーブプリセット",
                    "options": ["デフォルト", "クイック", "精密", "安定", "デジタル", "ダイナミック", "カスタム"]},
    "curveAdjust": {"label": "カーブ調整", "min": -5, "max": 5, "step": 1},
    # カスタムカーブ: P1〜P8 各 {in, out}（0〜1000・単調増加、P0=0/0, P9=1000/1000 固定）
    "curvePoints": {"label": "カスタムカーブP1-P8", "min": 0, "max": 1000, "len": 8},
    "rcEnabled":  {"label": "RC2.0 有効"},
    "rcMode":     {"label": "RCモード", "options": ["ベーシック", "アドバンスド"]},
    "rcStrength": {"label": "全域RC強度", "min": -500, "max": 500, "step": 1},
    # RC2.0 アドバンスド: 速度段 P1〜P5 各 {speed(0〜255・単調増加・最終255), rc(-500〜500)}
    # RC特性: 負値=入力変化へのブースト(初動応答UP/近距離追い向き)、正値=平滑化(安定/遅延増)
    "rcAdvanced": {"label": "速度RCカーブP1-P5", "speedMin": 0, "speedMax": 255,
                   "rcMin": -500, "rcMax": 500, "len": 5},
}

DEFAULT_CURRENT = {
    "pollingRate": 8000, "reportRate": 8000,
    "stickSampling": "Excellent", "quantization": "無制限", "advSampling": "オフ",
    # RCフィルター(RC2.0)使用可否。ApexでRCフィルターが処分対象化したため既定はOFF。
    # OFF時は解析でRCを無効化(Hub書出もenabled:false)し、補正はカーブ/DZ/感度で行う。
    "useRcFilter": False,
    "rs": {"centerDZ": 0, "antiDZ": 0, "outerDZ": 0,
           "curvePreset": "デフォルト", "curveAdjust": 0,
           "curvePoints": [{"in": 125, "out": 125}, {"in": 251, "out": 251},
                           {"in": 376, "out": 376}, {"in": 502, "out": 502},
                           {"in": 635, "out": 635}, {"in": 702, "out": 702},
                           {"in": 769, "out": 769}, {"in": 884, "out": 884}],
           "rcEnabled": True, "rcMode": "アドバンスド", "rcStrength": -70,
           "rcAdvanced": [{"speed": 32, "rc": -70}, {"speed": 80, "rc": -70},
                          {"speed": 128, "rc": -70}, {"speed": 192, "rc": -70},
                          {"speed": 255, "rc": -70}]},
    "ls": {"centerDZ": 0, "antiDZ": 0, "outerDZ": 0,
           "curvePreset": "デフォルト", "curveAdjust": 0,
           "curvePoints": [{"in": 125, "out": 125}, {"in": 251, "out": 251},
                           {"in": 376, "out": 376}, {"in": 502, "out": 502},
                           {"in": 635, "out": 635}, {"in": 702, "out": 702},
                           {"in": 769, "out": 769}, {"in": 884, "out": 884}],
           "rcEnabled": True, "rcMode": "アドバンスド", "rcStrength": -70,
           "rcAdvanced": [{"speed": 32, "rc": -70}, {"speed": 80, "rc": -70},
                          {"speed": 128, "rc": -70}, {"speed": 192, "rc": -70},
                          {"speed": 255, "rc": -70}]},
}



# ------------------------------------------------------------
# APEXゲーム内感度モデル（VPKファイル実データ）
# 出典: r/CompetitiveApex wvr1w8 "Values for standard sensitivities from vpk files"
# 各感度プリセット: (yaw°/s, pitch°/s, 加速yaw上限, 加速pitch上限, 加速遅延s, 加速到達s)
# 加速(Accel Max Speed)は全倒し維持時にランプアップして基本速度へ加算される
# ------------------------------------------------------------
APEX_LOOK_SENS = {                      # 腰撃ち (Looksensitivity)
    1: (50, 50, 60, 0, 0.05, 0.5),
    2: (80, 50, 150, 120, 0.0, 0.3),
    3: (160, 120, 220, 0, 0.0, 0.33),
    4: (240, 200, 220, 0, 0.0, 0.3),
    5: (380, 240, 0, 0, 0.0, 0.0),
    6: (450, 300, 0, 0, 0.0, 0.0),
    7: (500, 500, 0, 0, 0.0, 0.0),
    8: (500, 500, 0, 0, 0.0, 0.0),
}
APEX_ADS_SENS = {                       # ADS (Looksensitivity_zoomed)
    1: (35, 35, 20, 0, 0.05, 0.5),
    2: (60, 50, 35, 35, 0.0, 0.5),
    3: (110, 75, 30, 30, 0.25, 1.0),
    4: (150, 80, 0, 0, 0.0, 0.0),
    5: (200, 90, 0, 0, 0.0, 0.0),
    6: (450, 300, 0, 0, 0.0, 0.0),
    7: (500, 500, 0, 0, 0.0, 0.0),
    8: (500, 500, 0, 0, 0.0, 0.0),
}
# 応答曲線（VPK aimcurveのTRANSFORM定義: クラシック=2乗, 安定=3乗, リニア=1乗）
# cfgのgamepad_look_curve値はメニュー順 0=クラシック,1=安定,2=微調整,3=高速,4=リニア
RESPONSE_CURVES = {
    0: ("クラシック", 2.0),
    1: ("安定", 3.0),
    2: ("微調整", 3.0),     # Fine Aim: 区分線形+cubed（指数は近似）
    3: ("高速", 2.0),       # High Velocity: 区分線形+squared（指数は近似）
    4: ("リニア", 1.0),
}
APEX_DEFAULT_OPTICS = {"1x": 1.0, "2x": 1.0, "3x": 1.0, "4x": 1.0,
                       "6x": 1.0, "8x": 1.0, "10x": 1.0}


def read_apex_profile_cfg(path=None):
    """APEXのprofile.cfgから感度関連設定を読み取る（あればグラウンドトゥルース）。
    戻り値: apex辞書へマージ可能なdict（読めなければNone）"""
    import re
    if path is None:
        path = os.path.join(os.path.expanduser("~"), "Saved Games", "Respawn",
                            "Apex", "profile", "profile.cfg")
    try:
        if not os.path.exists(path):
            return None
        kv = {}
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                mm = re.match(r'(\S+)\s+"([^"]*)"', line.strip())
                if mm:
                    kv[mm.group(1)] = mm.group(2)
        out = {}
        if "gamepad_aim_speed" in kv:      # cfgは0始まり → 表示感度は+1
            out["lookSens"] = max(1, min(8, int(float(kv["gamepad_aim_speed"])) + 1))
        if "gamepad_look_curve" in kv:
            out["lookCurve"] = max(0, min(4, int(float(kv["gamepad_look_curve"]))))
        if "gamepad_use_per_scope_sensitivity_scalars" in kv:
            out["perOptic"] = kv["gamepad_use_per_scope_sensitivity_scalars"] == "1"
        optic_keys = ["1x", "2x", "3x", "4x", "6x", "8x", "10x"]
        optics = {}
        for i, ok in enumerate(optic_keys):
            k = f"gamepad_ads_advanced_sensitivity_scalar_{i}"
            if k in kv:
                optics[ok] = round(float(kv[k]), 3)
        if optics:
            out["optics"] = optics
        return out or None
    except Exception:
        return None

# 武器リコイルプロファイル（vert/horiz: 0-1 強度, drift: 水平ドリフト方向）
WEAPON_RECOIL = {
    "なし/その他":  {"vert": 0.0, "horiz": 0.0, "drift": 0},
    "R-301":       {"vert": 0.5, "horiz": 0.30, "drift": 0},
    "フラットライン": {"vert": 0.7, "horiz": 0.60, "drift": 0},
    "ネメシス":     {"vert": 0.5, "horiz": 0.30, "drift": 0},
    "ヘムロック":   {"vert": 0.6, "horiz": 0.35, "drift": 0},
    "R-99":        {"vert": 0.6, "horiz": 0.40, "drift": 0},
    "CAR":         {"vert": 0.6, "horiz": 0.45, "drift": 0},
    "ボルト":       {"vert": 0.4, "horiz": 0.20, "drift": 0},
    "オルタネーター": {"vert": 0.45, "horiz": 0.25, "drift": 0},
    "プラウラー":   {"vert": 0.55, "horiz": 0.40, "drift": 0},
    "スピットファイア": {"vert": 0.6, "horiz": 0.50, "drift": -1},
    "ディヴォーション": {"vert": 0.75, "horiz": 0.45, "drift": 0},
    "ハボック":     {"vert": 0.8, "horiz": 0.30, "drift": 0},
    "L-STAR":      {"vert": 0.65, "horiz": 0.45, "drift": 0},
    "RE-45":       {"vert": 0.35, "horiz": 0.25, "drift": 0},
    "P2020":       {"vert": 0.20, "horiz": 0.15, "drift": 0},
    "ウィングマン":  {"vert": 0.45, "horiz": 0.20, "drift": 0},
    "ランページ":   {"vert": 0.50, "horiz": 0.35, "drift": 0},
    "30-30":       {"vert": 0.45, "horiz": 0.20, "drift": 0},
}

APEX_DEFAULT = {"lookSens": 4, "adsSens": 3, "perOptic": False, "lookCurve": 0,
                "optics": dict(APEX_DEFAULT_OPTICS), "weapon": "なし/その他",
                "barrel": "なし"}

# バレルスタビライザーLv → リコイル低減率（コミュニティ実測の近似値）
BARREL_REDUCTION = {"なし": 0.0, "Lv1": 0.10, "Lv2": 0.15, "Lv3/4": 0.25}


def classify_slot_color(bgr_mean):
    """アタッチメントスロットの平均色(BGR) → レアリティ分類"""
    import colorsys
    b, g, r = [x / 255.0 for x in bgr_mean]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18:
        return "empty"
    if s < 0.20:
        return "white"          # Lv1
    deg = h * 360
    if 190 <= deg <= 250:
        return "blue"           # Lv2
    if 250 < deg <= 310:
        return "purple"         # Lv3
    if 25 <= deg <= 60:
        return "gold"           # Lv4
    return "unknown"


RARITY_TO_BARREL = {"white": "Lv1", "blue": "Lv2", "purple": "Lv3/4",
                    "gold": "Lv3/4", "empty": "なし", "unknown": "なし"}

# OCRテキスト → 武器キーのエイリアス（大文字化・空白除去後に部分一致）
WEAPON_ALIASES = {
    "R-301": ["R301", "R-301", "R30I", "CARBINE"],
    "フラットライン": ["FLATLINE", "フラットライン", "VK-47", "VK47"],
    "ネメシス": ["NEMESIS", "ネメシス"],
    "ヘムロック": ["HEMLOK", "HEMLOCK", "ヘムロック"],
    "R-99": ["R99", "R-99", "R9S"],
    "CAR": ["CAR"],
    "ボルト": ["VOLT", "ボルト"],
    "オルタネーター": ["ALTERNATOR", "オルタネーター"],
    "プラウラー": ["PROWLER", "プラウラー"],
    "スピットファイア": ["SPITFIRE", "スピットファイア"],
    "ディヴォーション": ["DEVOTION", "ディヴォーション", "ディボーション"],
    "ハボック": ["HAVOC", "ハボック"],
    "L-STAR": ["LSTAR", "L-STAR", "L STAR"],
    "RE-45": ["RE45", "RE-45"],
    "P2020": ["P2020"],
    "ウィングマン": ["WINGMAN", "ウィングマン", "ウイングマン"],
    "ランページ": ["RAMPAGE", "ランページ"],
    "30-30": ["3030", "30-30", "リピーター", "REPEATER"],
}


def _norm_weapon_text(s):
    """OCRテキストの正規化: 空白・ハイフン・長音・ピリオド除去+大文字化
    （HUDの「C.A.R.」「R E ー 45」等のOCR表記ゆれを吸収）"""
    return (s.upper().replace(" ", "").replace("　", "")
            .replace("-", "").replace("ー", "").replace(".", ""))


def match_weapon(raw):
    """1テキストから武器を判定（エイリアス部分一致）。なければNone"""
    if not raw:
        return None
    s = _norm_weapon_text(raw)
    for weapon, aliases in WEAPON_ALIASES.items():
        for al in aliases:
            if _norm_weapon_text(al) in s:
                return weapon
    return None


def weapon_firing_stats(samples, fire_iv, timeline, tol_ms=8000.0):
    """OCR武器タイムライン[(t,武器)]から各射撃バーストの使用武器を割り当て、
    武器別の射撃時間比率とリコイル制御指標を返す（マッチ中の武器切替対応）。
    戻り値: {"perWeapon": {武器: {"fireMs","fireRatio","holdMean","holdJitter"}},
             "dominant": 射撃時間が最長の武器 or None}"""
    if not fire_iv:
        return {"perWeapon": {}, "dominant": None}

    def weapon_at(t):
        best, bd = None, float("inf")
        for wt, w in timeline:
            d = abs(wt - t)
            if d < bd:
                bd, best = d, w
        return best if bd <= tol_ms else None   # 近傍のOCR標本のみ信頼

    per = {}
    for s0, e0 in fire_iv:
        w = weapon_at((s0 + e0) / 2.0) or "不明"
        seg = per.setdefault(w, {"ms": 0.0, "ry": []})
        seg["ms"] += (e0 - s0)
        for s in samples:
            if s0 <= s["t"] <= e0:
                seg["ry"].append(s.get("ry", 0.0))
            elif s["t"] > e0:
                break
    total = sum(v["ms"] for v in per.values()) or 1.0
    out = {}
    for w, v in per.items():
        e = {"fireMs": round(v["ms"]), "fireRatio": round(v["ms"] / total, 3)}
        if len(v["ry"]) >= 30:
            mean = sum(v["ry"]) / len(v["ry"])
            var = sum((y - mean) ** 2 for y in v["ry"]) / len(v["ry"])
            e["holdMean"] = round(mean, 4)       # 正=下方向(リコイル制御)
            e["holdJitter"] = round(var ** 0.5, 4)
        out[w] = e
    known = [(w, v["fireMs"]) for w, v in out.items() if w != "不明"]
    dominant = max(known, key=lambda kv: kv[1])[0] if known else None
    return {"perWeapon": out, "dominant": dominant}


def detect_weapon_from_text(texts):
    """OCRで得た文字列群から武器を多数決で判定。
    戻り値: (武器名 or None, {武器: 票数})"""
    votes = {}
    for raw in texts:
        if not raw:
            continue
        s = _norm_weapon_text(raw)
        hit_in_this_frame = set()
        for weapon, aliases in WEAPON_ALIASES.items():
            for al in aliases:
                if _norm_weapon_text(al) in s:
                    hit_in_this_frame.add(weapon)
                    break
        for w in hit_in_this_frame:
            votes[w] = votes.get(w, 0) + 1
    if not votes:
        return None, votes
    best = max(votes.items(), key=lambda kv: kv[1])
    return best[0], votes


def apply_game_context(stick_m, vision_m, apex):
    """ゲーム内感度と武器リコイルでメトリクスを補正する。
    - 水平リコイルが強い武器: 微小域の反転はリコイル制御なので割引く
    - 垂直リコイルが強い武器: 下方向の偏差バイアスは正常挙動として緩和
    - 感度から推定最大旋回速度を算出（提案の文脈に使用）
    """
    m = dict(stick_m) if stick_m else None
    vm = dict(vision_m) if vision_m else None
    apex = apex or APEX_DEFAULT
    rec = dict(WEAPON_RECOIL.get(apex.get("weapon", "なし/その他"),
                                 WEAPON_RECOIL["なし/その他"]))
    notes = []
    barrel = apex.get("barrel", "なし")
    red = BARREL_REDUCTION.get(barrel, 0.0)
    if red > 0 and (rec["vert"] > 0 or rec["horiz"] > 0):
        rec["vert"] *= (1 - red)
        rec["horiz"] *= (1 - red)
        notes.append(f"バレルスタビライザー{barrel}によりリコイル評価を{red*100:.0f}%低減して補正")
    if m:
        if m.get("firingSegmented"):
            notes.append(f"射撃区間（射撃ボタン実測 {m.get('firingRatio', 0)*100:.0f}%）を"
                         f"エイム指標から除外済み。統計的リコイル割引は不使用（精度向上）")
        elif rec["horiz"] > 0:
            factor = 1.0 - 0.5 * rec["horiz"]
            m["reversalRatio"] *= factor
            m["bandReversal"] = [b * factor for b in m.get("bandReversal", [0]*5)]
            notes.append(f"武器「{apex.get('weapon')}」の水平リコイル(強度{rec['horiz']:.1f})分、"
                         f"反転補正の評価を{(1-factor)*100:.0f}%割引"
                         f"（射撃ボタン未記録時のフォールバック）")
    if vm:
        if m and m.get("firingSegmented"):
            pass   # Vision側も射撃フレーム除外済みのため縦バイアス補正は不要
        elif rec["vert"] > 0 and vm.get("verticalBias") is not None:
            # 垂直リコイル制御中は下向き入力が正常 → 縦バイアスを緩和
            vm["verticalBias"] = vm["verticalBias"] * (1.0 - 0.6 * rec["vert"])
        if rec["drift"] != 0 and vm.get("horizontalBias") is not None:
            vm["horizontalBias"] -= 0.08 * rec["drift"]
            notes.append(f"武器の水平ドリフト特性を偏差バイアスから控除")
    # VPK実データから腰撃ち/ADSのyaw・pitch・加速を取得
    lk = APEX_LOOK_SENS.get(int(apex.get("lookSens", 4)), APEX_LOOK_SENS[4])
    ad = APEX_ADS_SENS.get(int(apex.get("adsSens", 3)), APEX_ADS_SENS[3])
    ctx = {"yawSpeed": lk[0], "pitchSpeed": lk[1],
           "turnExtraYaw": lk[2], "turnRampDelay": lk[4], "turnRampTime": lk[5],
           "adsYawSpeed": ad[0], "adsPitchSpeed": ad[1], "adsExtraYaw": ad[2],
           "recoil": rec, "notes": notes}
    return m, vm, ctx


def sens_recommendations(over, under, apex, ctx):
    """コントローラ側では吸収しきれない場合のゲーム内感度提案"""
    out = []
    look = int(apex.get("lookSens", 4))
    ads = int(apex.get("adsSens", 3))
    if over > 0.45 and look >= 5:
        out.append(f"オーバーシュートが強く視点感度{look}(推定yaw {ctx['yawSpeed']:.0f}°/s)は高め → "
                   f"ゲーム内の視点感度を {look}→{look-1} に下げる検討を推奨"
                   f"（コントローラ側の調整だけでは吸収しきれない水準）")
    elif under > 0.50 and look <= 3:
        out.append(f"アンダーシュートが強く視点感度{look}(推定yaw {ctx['yawSpeed']:.0f}°/s)は低め → "
                   f"視点感度 {look}→{look+1} への引き上げ、またはALCでyaw微調整を推奨")
    if under > 0.45 and ads <= 2:
        out.append(f"ADS感度{ads}(推定yaw {ctx['adsYawSpeed']:.0f}°/s)が低く追い切れていない可能性 → "
                   f"ADS感度 {ads}→{ads+1} を検討")
    if over > 0.45 and ads >= 5:
        out.append(f"ADS感度{ads}が高くADS中の行き過ぎの一因の可能性 → ADS感度 {ads}→{ads-1} を検討")
    if apex.get("perOptic"):
        out.append("詳細スコープ感度が有効です。中遠距離でのみ問題が出る場合は該当倍率の値のみを"
                   "±0.05刻みで調整してください（全倍率一括変更は筋肉記憶を崩します）")
    return out

# ------------------------------------------------------------
# GPU優先の推論セッション
# ------------------------------------------------------------
class VisionEngine:
    PROVIDER_PRIORITY = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "DmlExecutionProvider",      # DirectML (Windows / AMD・Intel・NVIDIA)
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]

    def __init__(self, model_path: str = MODEL_PATH):
        self.session = None
        self.model_name = "なし"
        self.num_classes = None
        self.available_providers = []
        self.active_provider = "なし（モデル未ロード）"
        if ort is None:
            self.active_provider = "onnxruntime 未インストール"
            return
        if not os.path.exists(model_path):
            self.active_provider = "モデルファイルなし（ヒューリスティックのみ）"
            return
        try:
            ort.set_default_logger_severity(4)  # CUDAランタイム欠落等のエラースパムを抑止
        except Exception:
            pass
        available = ort.get_available_providers()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 4
        # 優先順に1つずつ試し、DLL欠落(CUDA未導入等)は静かに次へフォールバック
        self.session = None
        for p in [x for x in self.PROVIDER_PRIORITY if x in available]:
            try:
                plist = [p] if p == "CPUExecutionProvider" else [p, "CPUExecutionProvider"]
                self.session = ort.InferenceSession(model_path, sess_options=so, providers=plist)
                break
            except Exception:
                continue
        if self.session is None:
            self.session = ort.InferenceSession(model_path, sess_options=so,
                                                providers=["CPUExecutionProvider"])
        self.active_provider = self.session.get_providers()[0]
        self.available_providers = list(available)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.imgsz = int(shp[2]) if isinstance(shp[2], int) else 640
        # fp16モデル対応（apex_7w_8n.onnx等は入力がfloat16）
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        self.model_name = os.path.basename(model_path)
        self.num_classes = None   # 初回推論時に出力形状から自動判定
        # 訓練場Bot特化モデル(models/apex_bot.onnx)があれば併用ロード。
        # 実戦モデルはダミー人形をほぼ検出できないため、検出結果をマージして
        # 訓練場計測でも高密度なエイム標本を得る（v4.3）
        self.bot = None
        try:
            bot_path = os.path.join(os.path.dirname(model_path), "apex_bot.onnx")
            if os.path.exists(bot_path):
                bsess = None
                for p in [x for x in self.PROVIDER_PRIORITY if x in available]:
                    try:
                        plist = ([p] if p == "CPUExecutionProvider"
                                 else [p, "CPUExecutionProvider"])
                        bsess = ort.InferenceSession(bot_path, sess_options=so,
                                                     providers=plist)
                        break
                    except Exception:
                        continue
                if bsess is None:
                    bsess = ort.InferenceSession(bot_path, sess_options=so,
                                                 providers=["CPUExecutionProvider"])
                binp = bsess.get_inputs()[0]
                self.bot = {
                    "session": bsess, "input": binp.name,
                    "dtype": np.float16 if "float16" in binp.type else np.float32,
                    "imgsz": (int(binp.shape[2])
                              if isinstance(binp.shape[2], int) else 640),
                }
                self.model_name += "+bot"
        except Exception:
            self.bot = None

    @property
    def gpu_active(self) -> bool:
        return self.active_provider not in ("CPUExecutionProvider",) and self.session is not None

    # 誤検出フィルタ（v3.8）: エイム解析はレティクル付近の敵だけが対象。
    # 壁バナー/キルフィード/HUD/自キャラの腕を位置・サイズで除外する
    ROI_X = 0.60          # 画面中心から横方向この割合を超える検出は対象外
    ROI_Y = 0.55          # 同・縦方向
    MAX_AREA_RATIO = 0.35  # 画面の35%超を占めるボックスは誤検出（自キャラ等）
    VIEWMODEL_TOP = 0.58   # ボックス上端が画面下部(58%より下)開始→自分の腕/武器

    def _raw_pred(self, session, input_name, dtype, imgsz, frame_bgr):
        """1モデル分の推論 → YOLOv8生予測 (N, 4+nc)"""
        img = self._letterbox(frame_bgr, imgsz)
        blob = (img[:, :, ::-1].transpose(2, 0, 1)[None]
                .astype(dtype) / dtype(255.0))
        out = session.run(None, {input_name: blob})[0].astype(np.float32)
        return out[0].T if out.shape[1] < out.shape[2] else out[0]

    def detect_persons(self, frame_bgr: np.ndarray, conf_th: float = 0.35,
                       return_rejected: bool = False):
        """フレーム内の敵候補を検出し、画面中心からの正規化偏差リストを返す。
        メインモデル（IFF/汎用）と訓練場Botモデル(apex_bot.onnx)の検出をマージ。
        return_rejected=True でフィルタ除外分も (kept, rejected) で返す"""
        if self.session is None:
            return ([], []) if return_rejected else []
        h, w = frame_bgr.shape[:2]
        pred = self._raw_pred(self.session, self.input_name, self.input_dtype,
                              self.imgsz, frame_bgr)
        if self.num_classes is None:
            self.num_classes = pred.shape[1] - 4
        # ターゲットクラスの決定:
        #  - 80クラス(COCO): class 0 = person のみ
        #  - 2クラス(Apex IFFモデル): class 1 = Enemy のみ（味方を除外）
        #  - 1クラス/その他: 全クラスを対象
        if self.num_classes >= 60:
            target_cls = {0}
        elif self.num_classes == 2:
            target_cls = {1}
        else:
            target_cls = None
        sources = [(pred, target_cls, self.imgsz, False)]
        if self.bot is not None:
            bpred = self._raw_pred(self.bot["session"], self.bot["input"],
                                   self.bot["dtype"], self.bot["imgsz"], frame_bgr)
            sources.append((bpred, None, self.bot["imgsz"], True))
        results = []
        rejected = []
        for pred_, tcls, imgsz_, is_bot in sources:
          for row in pred_:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < conf_th or (tcls is not None and cls_id not in tcls):
                continue
            cx, cy = float(row[0]), float(row[1])
            # letterbox 逆変換
            scale = imgsz_ / max(h, w)
            pad_x = (imgsz_ - w * scale) / 2
            pad_y = (imgsz_ - h * scale) / 2
            px = (cx - pad_x) / scale
            py = (cy - pad_y) / scale
            bw = float(row[2]) / scale
            bh = float(row[3]) / scale
            dev_x = (px - w / 2) / (w / 2)
            dev_y = (py - h / 2) / (h / 2)
            det = {
                "devX": dev_x,                    # -1〜1（右が正）
                "devY": dev_y,                    # -1〜1（下が正）
                "conf": conf,
                "cls": 1 if is_bot else cls_id,   # Botはエイム対象（敵相当）
                "box": [int(px - bw / 2), int(py - bh / 2), int(bw), int(bh)],
            }
            if is_bot:
                det["bot"] = True
            # レティクル周辺ROI外/巨大/画面下部開始（自キャラ腕）は解析対象から除外
            # 理由コードはASCII（cv2.putTextは日本語不可のためプレビュー描画兼用）
            if abs(dev_x) > self.ROI_X or abs(dev_y) > self.ROI_Y:
                det["reject"] = "out-of-ROI"        # HUD/バナー/画面端
                rejected.append(det)
            elif bw * bh > self.MAX_AREA_RATIO * w * h:
                det["reject"] = "too-large"          # 巨大ボックス=誤検出
                rejected.append(det)
            elif (py - bh / 2) > self.VIEWMODEL_TOP * h:
                det["reject"] = "viewmodel"          # 自キャラの腕/武器
                rejected.append(det)
            else:
                results.append(det)
        kept = self._nms(results)   # 両モデルが同一対象を検出した場合の重複も除去
        if return_rejected:
            return kept, self._nms(rejected)
        return kept

    @staticmethod
    def _nms(dets, iou_th=0.45):
        """YOLOv8生出力は同一対象に複数ボックスを返すため重複を除去する"""
        if len(dets) <= 1:
            return dets
        dets = sorted(dets, key=lambda d: -d["conf"])
        keep = []
        for d in dets:
            x1, y1, w1, h1 = d["box"]
            dup = False
            for k in keep:
                x2, y2, w2, h2 = k["box"]
                ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                inter = ix * iy
                union = w1 * h1 + w2 * h2 - inter
                if union > 0 and inter / union > iou_th:
                    dup = True
                    break
            if not dup:
                keep.append(d)
        return keep

    @staticmethod
    def _letterbox(img, size):
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        try:
            import cv2
            resized = cv2.resize(img, (nw, nh))
        except ImportError:
            resized = img[:: max(1, h // nh), :: max(1, w // nw)][:nh, :nw]
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas


# ------------------------------------------------------------
# 入力ログ解析（ヒューリスティック）
# ------------------------------------------------------------
def firing_intervals(samples, th=0.5, tail_ms=150.0, key="rt"):
    """射撃(トリガー/ボタン)区間 [(start_ms, end_ms), ...] を抽出。tailはリコイル整定分の余韻。
    key="ad" でADS区間の抽出にも使える"""
    iv = []
    start = None
    for s in samples:
        firing = s.get(key, 0.0) > th
        if firing and start is None:
            start = s["t"]
        elif not firing and start is not None:
            iv.append((start, s["t"] + tail_ms))
            start = None
    if start is not None and samples:
        iv.append((start, samples[-1]["t"] + tail_ms))
    # 重複区間を結合
    merged = []
    for s0, e0 in iv:
        if merged and s0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e0))
        else:
            merged.append((s0, e0))
    return merged


def _in_intervals(t, intervals):
    for s0, e0 in intervals:
        if s0 <= t <= e0:
            return True
        if s0 > t:
            break
    return False


REF_DT_MS = 1000.0 / 120.0   # 速度正規化の基準サンプル間隔（120Hz）


def _axis_metrics(samples, kx, ky, exclude=None):
    """指定軸ペア(右=rx/ry, 左=lx/ly)のメトリクスを算出。
    速度はサンプルレート非依存に正規化（基準120Hz）。実測レートが違っても
    RC速度段の分位点が安定するようにする（外部62Hz録画対応・v3.9）。"""
    n = len(samples)
    snapback = jitter = center_n = reversals = micro_n = sat = fast = 0
    overshoot_ev = fast_move_ev = 0     # 行き過ぎ痕跡（高速接近→反転）と高速移動の母数
    band_edges = [32, 80, 128, 192, 256]
    band_rev = [0] * 5
    band_n = [0] * 5
    speeds = []
    mags = []
    prev_vmag_n = 0.0                    # 前ステップの正規化速度(0-1相当)
    # 切り返し(左右反転)の横断時間: 片側|x|>0.35 → 反対側|x|>0.35 までのms。
    # RC参照なしでカーブの低〜中域ブースト量を実測から決めるための指標(v4.8)
    rev_transits = []
    _rev_side = 0
    _rev_last_t = None
    for i in range(2, n):
        skip_aim = exclude[i] if exclude is not None else False
        p2, p1, p0 = samples[i - 2], samples[i - 1], samples[i]
        x2, x1, x0 = p2[kx], p1[kx], p0[kx]
        mag = math.hypot(p0[kx], p0[ky])
        v1 = x1 - x2
        v0 = x0 - x1
        vmag_raw = math.hypot(p0[kx] - p1[kx], p0[ky] - p1[ky])
        # dtで正規化 → 「基準8.33ms当たりの移動量」に換算（レート非依存）
        dt = p0.get("t", 0) - p1.get("t", 0)
        scale = REF_DT_MS / dt if dt and dt > 0 else 1.0
        scale = min(4.0, max(0.25, scale))   # 欠測フレームでの暴発を抑制
        vmag_n = vmag_raw * scale
        v255 = min(255, int(vmag_n * 765))
        if v255 > 2:
            speeds.append(v255)
        if not skip_aim and mag > 0.03:
            mags.append(mag)
        if not skip_aim:
            for b in range(5):
                if v255 < band_edges[b]:
                    band_n[b] += 1
                    if 0.05 < mag < 0.6 and v1 * v0 < 0:
                        band_rev[b] += 1
                    break
        if (not skip_aim and abs(x1) < 0.12 and x2 * x0 < 0
                and abs(x2) > 0.35 and abs(v1) > 0.15):
            snapback += 1
        if not skip_aim and mag < 0.08:
            center_n += 1
            # 中心付近での方向反転のみをジッターと数える（通過中の等速移動は除外）
            if v1 * v0 < 0 and abs(v0) > 0.01 and abs(v1) > 0.01:
                jitter += 1
        if not skip_aim and 0.05 < mag < 0.35 and v1 * v0 < 0:
            reversals += 1
        if not skip_aim and 0.05 < mag < 0.30:
            micro_n += 1
        # 真のオーバーシュート痕跡: 高速接近(前ステップ高速) の直後に反転し、
        # かつ反転が中〜高倒し量(標的方向)で起きる。通常のトラッキング微修正
        # (低速・小反転)は数えない。フォールバックのover指標に使う。
        if not skip_aim and prev_vmag_n > 0.12:
            fast_move_ev += 1
            if v1 * v0 < 0 and vmag_n > 0.05 and mag > 0.06:
                overshoot_ev += 1
        prev_vmag_n = vmag_n
        if mag > 0.97:
            sat += 1
        if vmag_n > 0.25:
            fast += 1
        # 切り返し横断時間の計測（振幅0.35以上の左右反転のみ＝意図的な切り返し）
        if abs(x0) > 0.35:
            s_now = 1 if x0 > 0 else -1
            if _rev_side == -s_now and _rev_last_t is not None:
                tr = p0.get("t", 0) - _rev_last_t
                if 0 < tr <= 600:
                    rev_transits.append(tr)
            _rev_side = s_now
            _rev_last_t = p0.get("t", 0)
    # 実測スティック速度の分位点（RC速度段の最適化に使用）
    quantiles = None
    if len(speeds) >= 100:
        speeds.sort()
        pick = lambda q: speeds[min(len(speeds) - 1, int(q * len(speeds)))]
        quantiles = [pick(0.35), pick(0.60), pick(0.80), pick(0.93)]
    # 倒し量の分位点8点 → カスタムカーブ入力座標の最適配置（よく使う倒し量に点を密集）
    mag_quantiles = None
    if len(mags) >= 200:
        mags.sort()
        mpick = lambda q: mags[min(len(mags) - 1, int(q * len(mags)))]
        pts = [int(min(0.98, max(0.02, mpick((i + 1) / 9))) * 1000) for i in range(8)]
        for i in range(1, 8):                      # 昇順・最小間隔15を保証
            if pts[i] < pts[i - 1] + 15:
                pts[i] = pts[i - 1] + 15
        mag_quantiles = [min(980, p) for p in pts]
        for i in range(6, -1, -1):
            if mag_quantiles[i] >= mag_quantiles[i + 1]:
                mag_quantiles[i] = mag_quantiles[i + 1] - 15
    # 分布ヒストグラム（v5.0: 最適ノット配置/RC段k-meansの入力。分位点8点より高情報）
    mag_hist = None
    if len(mags) >= 200:
        mag_hist = [0] * 40
        for mg in mags:
            mag_hist[min(39, int(mg * 40))] += 1
    speed_hist = None
    if len(speeds) >= 100:
        speed_hist = [0] * 32
        for v in speeds:
            speed_hist[min(31, v * 32 // 256)] += 1
    # 切り返し統計（中央値と頻度/分）
    rev_transit_ms = None
    rev_per_min = 0.0
    dur_min = ((samples[-1].get("t", 0) - samples[0].get("t", 0)) / 60000.0
               if n >= 2 else 0.0)
    if rev_transits:
        rev_transits.sort()
        rev_transit_ms = rev_transits[len(rev_transits) // 2]
        if dur_min > 0:
            rev_per_min = len(rev_transits) / dur_min
    return {
        "sampleCount": n,
        "snapbackRate": snapback / (n / 100),
        "jitterRatio": jitter / center_n if center_n else 0.0,
        "reversalRatio": reversals / micro_n if micro_n else 0.0,
        "revTransitMs": rev_transit_ms,     # 切り返し横断時間の中央値[ms]
        "revPerMin": round(rev_per_min, 1),  # 切り返し頻度[回/分]
        # 高速接近直後の反転＝行き過ぎ痕跡。Vision不在時のover代替（reversalより厳格）
        "overshootProxy": overshoot_ev / fast_move_ev if fast_move_ev else 0.0,
        "saturationRatio": sat / n,
        "fastMoveRatio": fast / n,
        "bandReversal": [band_rev[b] / band_n[b] if band_n[b] else 0.0 for b in range(5)],
        "speedQuantiles": quantiles,
        "magQuantiles": mag_quantiles,
        "magHist": mag_hist,       # 倒し量分布40bin（エイム時のみ）
        "speedHist": speed_hist,   # 正規化速度分布32bin(0-255を8刻み)
    }


def analyze_stick_log(samples):
    """samples: [{t, lx, ly, rx, ry, rt?, ad?}] （-1〜1、rt/adは0〜1）
    右スティックのエイム指標は射撃区間(rt>0.5+150ms)を除外して算出。
    戻り値: 右スティックメトリクス（従来互換キー）＋ ls / 射撃・ADS指標"""
    n = len(samples)
    if n < 60:
        return None
    has_rt = any(s.get("rt", 0.0) > 0.5 for s in samples)
    exclude = None
    if has_rt:
        iv = firing_intervals(samples)
        exclude = [_in_intervals(s["t"], iv) for s in samples]
    rs = _axis_metrics(samples, "rx", "ry", exclude=exclude)
    ls = _axis_metrics(samples, "lx", "ly")
    rs["ls"] = ls
    rs["firingSegmented"] = bool(has_rt)
    if has_rt:
        fire_ry = [s["ry"] for s, ex in zip(samples, exclude) if ex]
        rs["firingRatio"] = sum(exclude) / n
        if len(fire_ry) >= 30:
            mean = sum(fire_ry) / len(fire_ry)
            var = sum((y - mean) ** 2 for y in fire_ry) / len(fire_ry)
            rs["recoilHoldMean"] = mean          # 正=下方向(リコイル制御)
            rs["recoilHoldJitter"] = var ** 0.5  # 制御の震え
    if any("ad" in s for s in samples):
        rs["adsRatio"] = sum(1 for s in samples if s.get("ad", 0.0) > 0.5) / n
    return rs


def summarize_vision(frame_results, firing_iv=None, samples=None, iff_model=True):
    """frame_results: [{t, targets:[{devX,devY,conf}]}]
    改良版(v3.7):
    - ADS区間のフレームを優先使用（意図的に狙っている時間だけをエイム評価に使う）
    - 時系列ペア(<900ms)で「収束(近づいている)」「行き過ぎ(左右符号反転)」を判定
      （通常キャプチャ間隔0.6〜1.5sのフレームがペア成立できるよう900ms。
       600msでは非射撃フレームが構造的に一度もペアにならず信頼度が
       常に半減する不整合があった: v4.9修正）
    - サンプル数から confidence(0-1) を算出。少数フレームの偶然で
      設定変更が駆動されるのを防ぐ（呼び出し側でゲート）"""
    ads_iv = []
    if samples and any(s.get("ad", 0.0) > 0.5 for s in samples):
        ads_iv = firing_intervals(samples, key="ad", tail_ms=100.0)
    hits = []
    for fr in frame_results:
        if firing_iv and _in_intervals(fr["t"], firing_iv):
            continue   # 射撃中の偏差はリコイル影響下のため除外
        if fr["targets"]:
            best = min(fr["targets"], key=lambda t: math.hypot(t["devX"], t["devY"]))
            engaged = _in_intervals(fr["t"], ads_iv) if ads_iv else False
            hits.append({"t": fr["t"], "engaged": engaged, **best})
    if len(hits) < 3:
        return None

    def _summ(use, mode):
        """部分集合(use)のメトリクスと信頼度を計算"""
        pair_n = conv_n = over_ev = 0
        for a, b in zip(use, use[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0 or dt > 900:
                continue
            pair_n += 1
            d0 = math.hypot(a["devX"], a["devY"])
            d1 = math.hypot(b["devX"], b["devY"])
            if d1 < d0 - 0.01:
                conv_n += 1
            if (a["devX"] * b["devX"] < 0
                    and abs(a["devX"]) > 0.06 and abs(b["devX"]) > 0.06):
                over_ev += 1
        under_n = sum(1 for h in use
                      if math.hypot(h["devX"], h["devY"]) > 0.20)
        over = over_ev / pair_n if pair_n else 0.0
        under = under_n / len(use)
        conv = conv_n / pair_n if pair_n else None
        # 収束率が高い＝遠方から追い付いている最中 → 偏差残存をアンダーシュートと
        # 誤認しないよう減免
        if pair_n >= 5 and conv is not None:
            under *= (1.0 - 0.5 * conv)
        confidence = min(1.0, len(use) / 30.0) * (1.0 if mode == "ads" else 0.6)
        # 汎用モデル(yolov8n等・敵味方識別なし)は味方や無関係な人型も数えるため
        # 検証済みの実害あり（味方追跡でunder誇張）→ 信頼度を大幅減
        if not iff_model:
            confidence *= 0.45
        # 時系列ペアが無い＝収束減免が働かない静的距離のみのunder → 信頼度減
        if pair_n < 5:
            confidence *= 0.6
        return {"use": use, "mode": mode, "pair_n": pair_n, "conv": conv,
                "over": over, "under": under, "confidence": confidence}

    # ADS限定と全フレームの両候補を計算し、最終信頼度が高い方を採用する。
    # 「ADS優先」固定だと、訓練場のように射撃≒ADSのプレイでは
    # 「ADS中かつ非射撃」の少数標本(13フレーム等)が大量の良標本(171フレーム)より
    # 優先されてしまう実害があった（v4.4で修正）
    eng = [h for h in hits if h["engaged"]]
    cands = [_summ(hits, "all")]
    if len(eng) >= 12:
        cands.append(_summ(eng, "ads"))
    best_s = max(cands, key=lambda c: (c["confidence"],
                                       1 if c["mode"] == "ads" else 0))
    use = best_s["use"]
    mode = best_s["mode"]
    pair_n = best_s["pair_n"]
    conv = best_s["conv"]
    over = best_s["over"]
    under = best_s["under"]
    confidence = best_s["confidence"]
    hx = sum(h["devX"] for h in use) / len(use)
    vy = sum(h["devY"] for h in use) / len(use)
    return {
        "frames": len(hits),
        "usedFrames": len(use),
        "mode": mode,                      # "ads"=ADS中のみ / "all"=全フレーム
        "iffModel": bool(iff_model),
        "pairCount": pair_n,
        "convergenceRatio": round(conv, 3) if conv is not None else None,
        "overshootRatio": over,
        "undershootRatio": under,
        "horizontalBias": hx,
        "verticalBias": vy,
        "confidence": round(confidence, 2),
    }


# ------------------------------------------------------------
# 提案エンジン（実機スキーマにクランプ）
# ------------------------------------------------------------
def _clamp(key, v):
    s = SCHEMA[key]
    return max(s["min"], min(s["max"], int(round(v))))


def _clamp_rc(v):
    return max(-500, min(500, int(round(v))))


def _rc_equivalent_boost(cur_side, prev_side=None):
    """旧RC設定から「カーブへ移植すべき初動ブースト量」（平均|負RC|）を推定する。
    現在の設定に負RCが残っていればそこから、既にゼロ化済みの場合は
    前回解析ログのcurrent（履歴）から取得。RC由来の情報が無ければ0。"""
    for side in (cur_side, prev_side):
        if not side:
            continue
        adv = side.get("rcAdvanced") or []
        negs = [abs(t["rc"]) for t in adv if t.get("rc", 0) < 0]
        # rcStrength(ベーシック用)へのフォールバックはアドバンスド段が無い場合のみ
        # （段が全0=情報なしのときに未使用の残骸rcStrengthを拾わないため）
        if not negs and not adv and side.get("rcStrength", 0) < 0:
            negs = [abs(side["rcStrength"])]
        # 注: rcEnabledは条件にしない。RC不使用移行では「値は残っているが
        # 無効化済み」が正に移植対象（有効フラグを条件にすると、UIでRC値を
        # 参照入力しただけの移行手順が発動しない実害があった: v4.7修正）
        if negs:
            return sum(negs) / len(negs)
    return 0.0


def _disable_rc(side):
    """RCフィルターを無効化する（BAN対策モード）。Hub書出でenabled:falseになり、
    全ティア/全域のRC強度を0にする。カーブ/DZ/感度側で補正する前提。"""
    side["rcEnabled"] = False
    side["rcStrength"] = 0
    if side.get("rcAdvanced"):
        side["rcAdvanced"] = [{"speed": t["speed"], "rc": 0}
                              for t in side["rcAdvanced"]]


def _adjust_curve_regions(points, low_delta, mid_delta, high_delta):
    """カスタムカーブの出力値を領域別に調整（単調増加・0〜1000を維持）
    low(P1-P3)=微操作域, mid(P4-P6)=追いエイム域, high(P7-P8)=フリック域"""
    weights = []
    n = len(points)
    for i in range(n):
        if i < 3:
            w = (low_delta, math.sin(math.pi * (i + 1) / 6))     # P1-P3
        elif i < 6:
            w = (mid_delta, math.sin(math.pi * (i - 2) / 4))     # P4-P6
        else:
            w = (high_delta, math.sin(math.pi * (i - 5) / 3))    # P7-P8
        weights.append(w[0] * w[1])
    new = [max(0, min(1000, int(round(p["out"] + weights[i]))))
           for i, p in enumerate(points)]
    for i in range(1, n):
        if new[i] <= new[i - 1]:
            new[i] = new[i - 1] + 1
    if new[-1] > 1000:
        overflow = new[-1] - 1000
        new = [max(0, v - overflow) for v in new]
    return [{"in": p["in"], "out": new[i]} for i, p in enumerate(points)]


def _adjust_curve_outputs(points, mid_delta):
    return _adjust_curve_regions(points, 0, mid_delta, 0)


def _optimize_rc_speeds(current_tiers, quantiles, speed_hist=None):
    """RC速度段の境界を実測から最適化。
    v5.0: speedHistがあれば重み付き1次元k-means(Lloyd法)で「各ティア内の
    速度ばらつきが最小」になる境界を算出（各段は一定RC値を持つため、
    段内分散最小＝RC割当ての量子化誤差最小）。ヒスト不足時は分位点法。
    最終段は255固定、昇順・最小間隔2を保証。変化が小さければNone"""
    if len(current_tiers) != 5:
        return None
    prop = None
    if speed_hist and sum(speed_hist) >= 100:
        vals = [b * 8 + 4 for b in range(32)]         # ビン代表値(0-252)
        init = list(quantiles[:4]) if quantiles else [32, 80, 128, 192]
        centers = sorted(set(max(2, min(250, int(c))) for c in init))
        centers.append(min(252, centers[-1] + 30))
        while len(centers) < 5:
            centers.append(min(252, centers[-1] + 20))
        centers = centers[:5]
        for _ in range(30):
            bnd = [(centers[i] + centers[i + 1]) / 2.0 for i in range(4)]
            new_c = []
            for ci in range(5):
                lo = bnd[ci - 1] if ci > 0 else -1.0
                hi = bnd[ci] if ci < 4 else 1e9
                s = wv = 0.0
                for v, wt in zip(vals, speed_hist):
                    if lo < v <= hi:
                        s += v * wt
                        wv += wt
                new_c.append(s / wv if wv else centers[ci])
            if all(abs(a - b) < 0.5 for a, b in zip(new_c, centers)):
                centers = new_c
                break
            centers = sorted(new_c)
        bnd = [(centers[i] + centers[i + 1]) / 2.0 for i in range(4)]
        prop = []
        prev = 0
        for b in bnd:
            s = max(prev + 2, min(250, int(round(b))))
            prop.append(s)
            prev = s
        prop.append(255)
    elif quantiles:
        prop = []
        prev = 0
        for q in quantiles[:4]:
            s = max(prev + 2, min(250, int(q)))
            prop.append(s)
            prev = s
        prop.append(255)
    if prop is None:
        return None
    cur = [t["speed"] for t in current_tiers]
    if max(abs(a - b) for a, b in zip(prop, cur)) < 8:
        return None   # 現状と大差なし
    return prop


W_LOW = [1.0, 0.8, 0.5, 0.2, 0.0]    # 低速帯の重み(P1→P5)
W_HIGH = [0.0, 0.2, 0.5, 0.8, 1.0]   # 高速帯の重み


# 帯域反転の正常トラッキング水準。これを超えた分だけを「過剰な振動」として扱う。
# 実マッチ検証で通常のトラッキングは低速帯で0.08〜0.10の反転を含むため、
# 閾値なしで弱体化すると正常なエイムでRCブーストが削られてしまう（v3.9で修正）。
BAND_REV_FLOOR = 0.15


def optimize_rc_tiers(tiers, band_rev, over, under, jitter):
    """5ティア全てのRC値を実測から最適化。
    - 各帯の反転が正常水準(0.15)を超えた分のみ: 負ブーストを0方向へ
    - 微振動(jitter): 低速帯の負ブーストを緩和
    - アンダーシュート: 低速寄り重みで負方向へ強化
    - オーバーシュート: 高速寄り重みで0方向へ
    戻り値: (新tiers, 変更ログ, ティア別delta)"""
    new = []
    changed = []
    deltas = []
    for i, t in enumerate(tiers):
        rc = t["rc"]
        br_raw = band_rev[i] if i < len(band_rev) else 0.0
        br = max(0.0, br_raw - BAND_REV_FLOOR)   # 正常水準を超えた過剰分のみ
        delta = 0.0
        if rc < 0:
            delta += min(60.0, br * abs(rc))
            if i < 2:
                delta += min(40.0, max(0.0, jitter - 0.10) * abs(rc) * 0.6)
        if under > 0.30:
            delta -= (under - 0.30) * 160.0 * W_LOW[i]
        if over > 0.35:
            delta += (over - 0.35) * 200.0 * W_HIGH[i]
        nrc = _clamp_rc(round(rc + delta))
        deltas.append(round(delta))
        if abs(nrc - rc) >= 3:
            changed.append(f"P{i+1}(速度{t['speed']}): {rc}→{nrc}")
        else:
            nrc = rc
        new.append({"speed": t["speed"], "rc": nrc})
    return new, changed, deltas


def _curve_value(points, x):
    """(0,0)-P1..P8-(1000,1000)の折れ線でxの出力を補間"""
    xs = [0] + [p["in"] for p in points] + [1000]
    ys = [0] + [p["out"] for p in points] + [1000]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            if xs[i] == xs[i - 1]:
                return ys[i]
            r = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + r * (ys[i] - ys[i - 1])
    return 1000


def optimize_curve_points(points, mag_q, low_delta, mid_delta, high_delta):
    """入力座標を実測倒し量の分位点へ再配置し、出力は旧カーブ形状を補間した上で
    領域別デルタを適用（単調増加・0〜1000保証）"""
    ins = list(mag_q) if mag_q and len(mag_q) == 8 else [p["in"] for p in points]
    n = 8
    # 実測分位点が高倒し量に偏っている場合（左スティックの全倒し移動等）、
    # 低・中域の制御点が消えてカーブの粒度を失うため、最大間隔を制限して再分配
    if mag_q and len(mag_q) == 8:
        ins[0] = min(ins[0], 180)
        for i in range(1, n):
            ins[i] = min(ins[i], ins[i - 1] + 280)
        for i in range(1, n):                       # 単調増加を回復
            if ins[i] <= ins[i - 1]:
                ins[i] = ins[i - 1] + 15
        if ins[-1] > 980:
            ins[-1] = 980
            for i in range(n - 2, -1, -1):
                if ins[i] >= ins[i + 1]:
                    ins[i] = ins[i + 1] - 15
    outs = [int(round(_curve_value(points, x))) for x in ins]
    for i in range(n):
        if i < 3:
            w = low_delta * math.sin(math.pi * (i + 1) / 6)
        elif i < 6:
            w = mid_delta * math.sin(math.pi * (i - 2) / 4)
        else:
            w = high_delta * math.sin(math.pi * (i - 5) / 3)
        outs[i] = max(0, min(1000, int(round(outs[i] + w))))
    for i in range(1, n):
        if ins[i] <= ins[i - 1]:
            ins[i] = ins[i - 1] + 1
        if outs[i] <= outs[i - 1]:
            outs[i] = outs[i - 1] + 1
    if outs[-1] > 1000:
        over = outs[-1] - 1000
        outs = [max(0, o - over) for o in outs]
    return [{"in": min(999, ins[i]), "out": outs[i]} for i in range(n)]


# ------------------------------------------------------------
# 設定→ゲーム内挙動の物理モデル（v4.5: VPK実データ準拠）
# HyperStrike内変換: 倒し量 → DZ → カーブ → アンチDZ → XInput出力
# Apex側: 出力^(応答曲線指数) × 感度別yaw速度 (+全倒し維持で加速加算)
# ------------------------------------------------------------


def _hs_output(side, mag):
    """HyperStrikeの静的変換: スティック倒し量mag(0-1) → XInput出力(0-1)
    ※RC2.0は速度依存の動的補正のため静的応答には含めない"""
    m = mag * 1000.0
    dz = side.get("centerDZ", 0) * 10.0
    odz = 1000.0 - side.get("outerDZ", 0) * 10.0
    if m <= dz:
        return 0.0
    if m >= odz:
        return 1.0
    t = (m - dz) / (odz - dz) * 1000.0
    if side.get("curvePreset") == "カスタム" and side.get("curvePoints"):
        out = _curve_value(side["curvePoints"], t)
    else:
        out = t
    adz = side.get("antiDZ", 0) * 10.0
    out = adz + out * (1000.0 - adz) / 1000.0
    return max(0.0, min(1.0, out / 1000.0))


def _apex_curve_exp(apex):
    """ゲーム内応答曲線の指数（VPK: クラシック=2乗/安定=3乗/リニア=1乗）"""
    return RESPONSE_CURVES.get(int(apex.get("lookCurve", 0)),
                               RESPONSE_CURVES[0])[1]


def turn_rate(side, apex, mag, ads=False, sustained=False):
    """倒し量mag(0-1)でのゲーム内視点旋回速度[°/s]（VPK実データ準拠）。
    sustained=True で全倒し維持時の加速(Accel Max Speed)を加算"""
    sens_key = "adsSens" if ads else "lookSens"
    tbl = APEX_ADS_SENS if ads else APEX_LOOK_SENS
    row = tbl.get(int(apex.get(sens_key, 4)), tbl[4])
    out = _hs_output(side, mag)
    rate = row[0] * (out ** _apex_curve_exp(apex))
    if sustained and out >= 0.995:
        rate += row[2]     # 全倒しランプアップ後の追加yaw
    return rate


def predict_effects(current, rec):
    """現行設定→推奨設定でゲームプレイがどう変わるかを予測して日本語で返す。
    「設定値がAPEXにどう反映されるか」を可視化するための変換モデル出力"""
    apex = current.get("apex", APEX_DEFAULT)
    out = []
    for label, mag in (("微操作域(倒し15%)", 0.15),
                       ("追いエイム域(倒し40%)", 0.40),
                       ("フリック域(倒し80%)", 0.80)):
        c = turn_rate(current["rs"], apex, mag)
        n = turn_rate(rec["rs"], apex, mag)
        if abs(n - c) >= 1.0:
            pct = f" ({(n - c) / c * 100:+.0f}%)" if c > 0.5 else ""
            out.append(f"右 {label}: 腰撃ち視点旋回 約{c:.0f}→{n:.0f}°/s{pct}")
    ca = turn_rate(current["rs"], apex, 0.15, ads=True)
    na = turn_rate(rec["rs"], apex, 0.15, ads=True)
    if abs(na - ca) >= 0.5:
        out.append(f"右 ADS微操作(倒し15%): 約{ca:.1f}→{na:.1f}°/s")

    rc_off = not rec["rs"].get("rcEnabled", True)
    if rc_off:
        out.append("RCフィルター無効（BAN対策）: RC2.0は使用せず、補正はカーブ/DZ/"
                   "ゲーム内感度で行います")
    else:
        def _p1(s):
            adv = s.get("rcAdvanced") or []
            return adv[0]["rc"] if adv else s.get("rcStrength", 0)
        c1, n1 = _p1(current["rs"]), _p1(rec["rs"])
        if c1 != n1:
            out.append(f"右 初動レスポンス(RC P1): ブースト強度 {abs(c1)/5:.0f}%→{abs(n1)/5:.0f}%"
                       f"（負RC=入力変化の増幅。強すぎると微振動・行き過ぎの原因）")
    cw = _hs_output(current["ls"], 0.4)
    nw = _hs_output(rec["ls"], 0.4)
    if abs(nw - cw) >= 0.02:
        out.append(f"左 歩き速度域(倒し40%): 移動出力 {cw*100:.0f}%→{nw*100:.0f}%")

    def _full_mag(s):
        for mm in range(50, 101):
            if _hs_output(s, mm / 100.0) >= 0.999:
                return mm
        return 100
    cf, nf = _full_mag(current["ls"]), _full_mag(rec["ls"])
    if cf != nf:
        out.append(f"左 最高移動速度に必要な倒し量: {cf}%→{nf}%")
    if not out:
        out.append("静的応答（カーブ/DZ）に大きな変化なし（RC等の動的補正のみ変更）")
    # 参考情報: 使用中の感度のVPK実値と応答曲線
    lk = APEX_LOOK_SENS.get(int(apex.get("lookSens", 4)), APEX_LOOK_SENS[4])
    ad = APEX_ADS_SENS.get(int(apex.get("adsSens", 3)), APEX_ADS_SENS[3])
    cname = RESPONSE_CURVES.get(int(apex.get("lookCurve", 0)),
                                RESPONSE_CURVES[0])[0]
    out.append(f"参考(VPK実値): 腰撃ち感度{apex.get('lookSens')}="
               f"yaw{lk[0]}/pitch{lk[1]}°/s"
               + (f"(+全倒し加速{lk[2]})" if lk[2] else "")
               + f"、ADS感度{apex.get('adsSens')}=yaw{ad[0]}/pitch{ad[1]}°/s"
               + (f"(+加速{ad[2]})" if ad[2] else "")
               + f"、応答曲線={cname}")
    out.append("※°/s値はVPKファイル実データ（感度テーブル・応答曲線TRANSFORM）に"
               "基づく推定値です")
    return out


# ------------------------------------------------------------
# RC変更の安全ガード（v3.7: 解析を繰り返すたびに強化され続ける暴走を防止）
# ------------------------------------------------------------
RC_STEP_MAX = 45   # 1回の解析で許すRC変化量の上限


def _apply_rc_guards(old_tiers, new_tiers, prev_rec_tiers, metric_improved):
    """1) 1回あたりの変化量を±RC_STEP_MAXに制限
    2) 前回の推奨が適用済みなのに指標が改善していない場合、同方向の強化を半減
    3) |RC|>=350のティアは改善の証拠なしにさらに強化しない
    戻り値: (ガード後tiers, 注記リスト)"""
    notes = []
    applied = bool(prev_rec_tiers and len(prev_rec_tiers) == len(old_tiers)
                   and all(abs(p["rc"] - o["rc"]) <= 5
                           for p, o in zip(prev_rec_tiers, old_tiers)))
    out = []
    for i, (o, nt) in enumerate(zip(old_tiers, new_tiers)):
        rc0 = o["rc"]
        d = nt["rc"] - rc0
        strengthening = abs(rc0 + d) > abs(rc0)
        if strengthening and applied and not metric_improved:
            if abs(d) >= 6:
                notes.append(f"P{i+1}: 前回強化で改善が確認できないため増分を半減")
            d = int(d / 2)
        if strengthening and abs(rc0) >= 350 and not metric_improved:
            if d:
                notes.append(f"P{i+1}: |RC|{abs(rc0)}は既に大きく、改善の証拠が"
                             f"ないため追加強化を保留")
            d = 0
        d = max(-RC_STEP_MAX, min(RC_STEP_MAX, d))
        out.append({"speed": nt["speed"], "rc": _clamp_rc(rc0 + d)})
    return out, notes


# ------------------------------------------------------------
# 最適ノット配置エンジン（v5.0）
# 区分線形近似の理論: ノットは「使用密度 × 目標カーブの曲率」の重みに
# 比例して配置すると近似誤差が最小化される。計測ヒストグラムを直接使用。
# ------------------------------------------------------------
def _target_curve_grid(points, low_d, mid_d, high_d, step=5):
    """旧カーブ+領域デルタ（連続エンベロープ）の目標カーブをグリッドで返す"""
    xs = list(range(0, 1001, step))
    ys = []
    for x in xs:
        base = _curve_value(points, x)
        d = 0.0
        if x < 333:
            d += low_d * math.sin(math.pi * x / 333.0)
        if 250 <= x <= 750:
            d += mid_d * math.sin(math.pi * (x - 250) / 500.0)
        if x > 667:
            d += high_d * math.sin(math.pi * (x - 667) / 333.0)
        ys.append(max(0.0, min(1000.0, base + d)))
    for i in range(1, len(ys)):                 # 単調化
        if ys[i] < ys[i - 1]:
            ys[i] = ys[i - 1]
    return xs, ys


def optimal_curve_inputs(xs, ys, mag_hist, fallback_ins, min_gap=25):
    """使用密度(magHist)×目標曲率で8ノットを最適配置。ヒスト不足時はfallback"""
    if not mag_hist or sum(mag_hist) < 50:
        return list(fallback_ins)
    n = len(xs)
    total = float(sum(mag_hist))
    dens = []
    for x in xs:
        b = min(39, x * 40 // 1000)
        dens.append(mag_hist[b] / total)
    mean_d = sum(dens) / n
    # キャップ+床: 極端に尖った使用分布でも全域カバーを失わないようにする
    dens = [min(d, 3.0 * mean_d) + 0.30 * mean_d for d in dens]
    curv = [0.0] * n
    for i in range(2, n - 2):
        curv[i] = abs(ys[i + 2] - 2 * ys[i] + ys[i - 2])
    mean_c = (sum(curv) / n) or 1.0
    w = [dens[i] * (curv[i] + 0.35 * mean_c) for i in range(n)]
    cum = [0.0]
    for v in w:
        cum.append(cum[-1] + v)
    tot = cum[-1] or 1.0
    ins = []
    j = 0
    for k in range(1, 9):
        tgt = tot * k / 9.0
        while j < n and cum[j + 1] < tgt:
            j += 1
        ins.append(xs[min(n - 1, j)])
    ins[0] = max(20, min(ins[0], 300))           # P1は必ず低域に
    # 全域カバー保証: 端点(1000)側を含む各区間の最大幅を280に制限
    if 1000 - ins[-1] > 280:
        ins[-1] = 1000 - 280
    for i in range(6, -1, -1):
        if ins[i + 1] - ins[i] > 280:
            ins[i] = ins[i + 1] - 280
    for i in range(1, 8):
        if ins[i] < ins[i - 1] + min_gap:
            ins[i] = ins[i - 1] + min_gap
    if ins[-1] > 980:
        ins[-1] = 980
        for i in range(6, -1, -1):
            if ins[i] >= ins[i + 1]:
                ins[i] = ins[i + 1] - min_gap
    return ins


LEFT_UNIFORM_INS = [111, 222, 333, 444, 556, 667, 778, 889]   # P1-P8均等基準


def optimize_left_curve_points(points, mag_q, low_delta, high_delta):
    """左スティック専用: P1〜P8を全域にバランス配置する（v4.9）。
    各点は均等基準±70の帯内で実測分位点へ寄せる。移動スティックは
    歩き〜走り〜全力の全域に制御点が必要で、分位点直採用だと全倒し付近に
    点が密集して低速域の粒度を失うため（実測: 8点中5点が920-980に集中）。
    出力は旧カーブ形状の補間＋領域デルタ（低域=ジッター平坦化/高域=飽和対策）"""
    ins = []
    for i in range(8):
        base = LEFT_UNIFORM_INS[i]
        tgt = mag_q[i] if (mag_q and len(mag_q) == 8) else base
        ins.append(int(max(base - 70, min(base + 70, tgt))))
    for i in range(1, 8):                    # 単調増加・最小間隔40
        if ins[i] < ins[i - 1] + 40:
            ins[i] = ins[i - 1] + 40
    if ins[-1] > 980:
        ins[-1] = 980
        for i in range(6, -1, -1):
            if ins[i] >= ins[i + 1]:
                ins[i] = ins[i + 1] - 40
    outs = [int(round(_curve_value(points, x))) for x in ins]
    for i in range(8):
        if i < 3:
            w = low_delta * math.sin(math.pi * (i + 1) / 6)      # P1-P3
        elif i < 6:
            w = 0.0                                              # P4-P6は形状維持
        else:
            w = high_delta * math.sin(math.pi * (i - 5) / 3)     # P7-P8
        outs[i] = max(0, min(1000, int(round(outs[i] + w))))
    for i in range(1, 8):
        if outs[i] <= outs[i - 1]:
            outs[i] = outs[i - 1] + 1
    if outs[-1] > 1000:
        over = outs[-1] - 1000
        outs = [max(0, o - over) for o in outs]
    return [{"in": min(999, ins[i]), "out": outs[i]} for i in range(8)]


def build_recommendation(current, stick_m, vision_m, history=None):
    import copy
    rec = copy.deepcopy(current)
    reasons = []
    audit = []

    def check(rule, value, op, th, action=""):
        fired = (value > th) if op == ">" else (value < th)
        audit.append({"rule": rule,
                      "value": round(float(value), 4) if value is not None else None,
                      "threshold": f"{op}{th}", "fired": bool(fired),
                      "action": action if fired else "変更なし"})
        return fired

    raw_stick = dict(stick_m) if stick_m else None
    apex = current.get("apex", APEX_DEFAULT)
    # RCフィルター使用可否（既定OFF=BAN対策）。OFFならRCは一切最適化せず無効化する。
    use_rc = bool(current.get("useRcFilter", False))
    if use_rc:
        reasons.append("⚠ 警告: RCフィルター(RC2.0)は現在Apexで検出・処分の対象です。"
                       "リスクを避けるにはUIの「RCフィルターを使用」をOFFにしてください"
                       "（OFFにするとRCを使わない設定を算出します）。")
        audit.append({"rule": "_RCフィルター", "value": None, "threshold": "-", "fired": True,
                      "action": "RC使用モード（BANリスクあり）"})
    else:
        audit.append({"rule": "_RCフィルター", "value": None, "threshold": "-", "fired": True,
                      "action": "RC不使用モード（BAN対策）: RCを無効化しカーブ/DZ/感度で補正"})
    stick_m, vision_m, ctx = apply_game_context(stick_m, vision_m, apex)
    m = stick_m or {}
    vm = vision_m or {}
    if raw_stick and raw_stick.get("firingSegmented"):
        audit.append({"rule": "_射撃区間分離", "value": None, "threshold": "-", "fired": True,
                      "action": f"射撃区間{raw_stick.get('firingRatio',0)*100:.0f}%を除外"
                                f"（ADS率: {raw_stick.get('adsRatio',0)*100:.0f}%）"})
    audit.append({"rule": "_入力データ", "value": None, "threshold": "-", "fired": True,
                  "action": f"サンプル数={m.get('sampleCount', 0)}, "
                            f"Vision敵検出フレーム={vm.get('frames', 0) if vm else 0}, "
                            f"武器={apex.get('weapon')}"})
    if not vm:
        audit.append({"rule": "_Vision解析", "value": None, "threshold": "-", "fired": False,
                      "action": "敵検出フレーム不足のためVision系判定は不発。入力ログのみで判定"})
    reasons.extend(ctx["notes"])

    band_rev = m.get("bandReversal", [0.0] * 5)
    # Vision指標は信頼度(confidence)でゲート＆スケールする（v3.7）:
    # 少数の敵検出フレームの偶然でRC/カーブの大変更が駆動されるのを防ぐ
    # over/underの決定（v3.9で改修）:
    # 実マッチ検証でreversalRatio(通常トラッキング微修正)は真のover(≈0)の数百倍に
    # 誇張されRCを誤って弱体化させていた。Vision不在時はreversalではなく
    # overshootProxy(高速接近→反転の痕跡)を使い、under信号は持たない。
    vconf = vm.get("confidence", 1.0) if vm else 0.0
    osp = m.get("overshootProxy", 0.0)
    if vm and vm.get("iffModel") is False:
        reasons.append("注意: 敵味方識別モデル(apex.onnx)が未導入のため汎用人物検出で"
                       "解析しています（味方も検出されVision精度が低下）。"
                       "update_model_apex.bat の実行を推奨します。")
    if vm and vconf >= 0.25:
        over = vm.get("overshootRatio", 0.0) * vconf
        under = vm.get("undershootRatio", 0.0) * vconf
        if over == 0:
            over = osp * (1.0 - vconf)   # Vision補完（弱信頼分だけstick痕跡で補う）
        audit.append({"rule": "_Vision信頼度", "value": round(vconf, 2),
                      "threshold": ">=0.25", "fired": True,
                      "action": f"mode={vm.get('mode','?')} 標本={vm.get('usedFrames',0)}"
                                f"フレーム / over・underを信頼度{vconf:.2f}でスケール"})
    else:
        over = osp
        under = 0.0
        if vm:
            audit.append({"rule": "_Vision信頼度", "value": round(vconf, 2),
                          "threshold": ">=0.25", "fired": False,
                          "action": f"敵検出フレーム不足 → Vision指標は不使用。"
                                    f"stick痕跡overshootProxy={osp:.3f}で保守的判定"})
            tips = []
            if vm.get("iffModel") is False:
                tips.append("update_model_apex.bat で敵味方識別モデルを導入")
            tips.append("射撃訓練場でBot撃ちを2〜3分計測（Bot特化モデルで高信頼データが"
                        "取れます。ADS追いエイムを多めに）")
            tips.append("計測中はApex画面に他ウィンドウを重ねない")
            reasons.append("Vision信頼度が実用水準(0.25)未満です。改善手順: "
                           + " ／ ".join(f"{i+1}) {t}" for i, t in enumerate(tips)))
    audit.append({"rule": "_over/under決定", "value": None, "threshold": "-", "fired": True,
                  "action": f"over={over:.3f}（overshootProxy={osp:.3f}, "
                            f"reversalRatio={m.get('reversalRatio',0):.3f}は不使用） under={under:.3f}"})
    j = m.get("jitterRatio", 0)
    lm = m.get("ls") or {}
    audit.append({"rule": "_複合指標", "value": None, "threshold": "-", "fired": True,
                  "action": f"over={over:.3f} under={under:.3f} jitter={j:.3f} "
                            f"帯域反転={[round(b,2) for b in band_rev]}"})

    # ============================================================
    # デッドゾーン最小化ポリシー: DZは極力0にし、カーブ/RCで補正する
    # ============================================================
    dz_notes = []
    for p, jp, jj in (("rs", "右", j), ("ls", "左", lm.get("jitterRatio", 0))):
        cur_s = current[p]
        for key, label in (("centerDZ", "中心"), ("antiDZ", "アンチ"), ("outerDZ", "外周")):
            if cur_s.get(key, 0) > 0:
                rec[p][key] = 0
                dz_notes.append(f"{jp}:{label}DZ {cur_s[key]}%→0%")
        # ハード由来ドリフトの最終手段のみ最小DZを許容（ジッター30%超）
        if jj > 0.30:
            rec[p]["centerDZ"] = min(2, _clamp("centerDZ", round(jj * 5)))
            dz_notes.append(f"{jp}:重度ジッター({jj*100:.0f}%)のため中心DZ最小値"
                            f"{rec[p]['centerDZ']}%のみ許容")
    audit.append({"rule": "DZ最小化ポリシー", "value": None, "threshold": "-",
                  "fired": bool(dz_notes),
                  "action": " / ".join(dz_notes) if dz_notes else "全DZ既に0"})
    if dz_notes:
        reasons.append("デッドゾーン最小化: " + " / ".join(dz_notes)
                       + f"（補正はカーブ低域{'とRC' if use_rc else ''}で行います）")

    # ============================================================
    # 右スティック: RC全ティア + 速度段 + カーブを実測から最適化
    # ============================================================
    rs = rec["rs"]
    cur_rs = current["rs"]
    check("オーバーシュート", over, ">", 0.35, "高速帯RCを0方向へ/カーブ高域抑制")
    check("アンダーシュート", under, ">", 0.30, "低速帯RC強化/カーブ中域増")
    check("中心ジッター率(右)", j, ">", 0.15, "低速帯RC緩和/カーブ低域平坦化")

    # 前回解析（履歴）: 前回推奨が適用済みで指標が改善したかを判定（暴走防止）
    prev_vm = (history or {}).get("visionMetrics") or {}
    prev_under = prev_vm.get("undershootRatio")
    prev_conf = prev_vm.get("confidence", 1.0)
    metric_improved = True   # 履歴なし/前回Vision不発なら「証拠なし」として通常動作
    if prev_under is not None and prev_conf >= 0.25 and vconf >= 0.25:
        metric_improved = under < prev_under - 0.05

    if not use_rc:
        _disable_rc(rs)
        audit.append({"rule": "RC無効化(右)", "value": None, "threshold": "-", "fired": True,
                      "action": "RCフィルターOFF → 右スティックRCを無効化(enabled:false/全RC=0)"})
        reasons.append("右: RCフィルターを無効化しました（BAN対策）。初動応答・行き過ぎの"
                       "補正はカスタムカーブとゲーム内感度で行います。")

    use_adv = (cur_rs.get("rcMode") == "アドバンスド" and cur_rs.get("rcEnabled")
               and use_rc)
    if use_adv:
        adv, changed, deltas = optimize_rc_tiers(cur_rs["rcAdvanced"], band_rev,
                                                 over, under, j)
        prev_rc = ((history or {}).get("recommendation") or {}) \
            .get("rs", {}).get("rcAdvanced")
        adv, guard_notes = _apply_rc_guards(cur_rs["rcAdvanced"], adv,
                                            prev_rc, metric_improved)
        # ガード後の実変化量で監査ログと変更リストを再構成
        deltas = [a["rc"] - o["rc"] for a, o in zip(adv, cur_rs["rcAdvanced"])]
        changed = []
        for i, (o, a) in enumerate(zip(cur_rs["rcAdvanced"], adv)):
            if abs(deltas[i]) < 3:
                adv[i] = {"speed": a["speed"], "rc": o["rc"]}
                deltas[i] = 0
            elif a["rc"] != o["rc"]:
                changed.append(f"P{i+1}(速度{o['speed']}): {o['rc']}→{a['rc']}")
        if guard_notes:
            audit.append({"rule": "RC暴走ガード(右)", "value": None, "threshold": "-",
                          "fired": True, "action": " / ".join(guard_notes)})
            reasons.append("右: RC安全ガード発動 — " + " / ".join(guard_notes))
        for i in range(5):
            audit.append({"rule": f"RC最適化(右) P{i+1}", "value": deltas[i],
                          "threshold": "|Δ|>=3", "fired": abs(deltas[i]) >= 3,
                          "action": changed and next((c for c in changed
                                     if c.startswith(f"P{i+1}(")), "変更なし") or "変更なし"})
        sb = m.get("snapbackRate", 0)
        if check("スナップバック率(右)", sb, ">", 0.8, "P5のRC緩和"):
            old5 = adv[4]["rc"]
            adv[4]["rc"] = _clamp_rc(old5 + 30)
            changed.append(f"P5: {old5}→{adv[4]['rc']}（スナップバック対策）")
        q = m.get("speedQuantiles")
        new_speeds = _optimize_rc_speeds(adv, q, m.get("speedHist"))
        audit.append({"rule": "RC速度段最適化(右)", "value": None,
                      "threshold": "実測分位点との乖離>=8", "fired": bool(new_speeds),
                      "action": (f"{[t['speed'] for t in adv]}→{new_speeds}"
                                 if new_speeds else "変更なし（実測分布と概ね一致）")})
        if new_speeds:
            for i, s in enumerate(new_speeds):
                adv[i]["speed"] = s
            reasons.append(f"右: 実測速度分布（分位点 {q}）からRC速度段を {new_speeds} に再配置")
        rs["rcAdvanced"] = adv
        if changed:
            reasons.append("右: RC値を帯域別実測から最適化 → " + " / ".join(changed))
    elif use_rc:
        cur_rc = cur_rs["rcStrength"]
        if over > 0.35:
            rs["rcStrength"] = _clamp_rc(cur_rc + int(20 + 60 * over))
            reasons.append(f"右: 全域RC強度 {cur_rc}→{rs['rcStrength']}（行き過ぎ抑制）")
        elif under > 0.30:
            rs["rcStrength"] = _clamp_rc(cur_rc - int(20 + 50 * under))
            reasons.append(f"右: 全域RC強度 {cur_rc}→{rs['rcStrength']}（初動応答強化）")

    # カーブ: 入力座標=実測倒し量の分位点、出力=領域別デルタ
    if cur_rs.get("curvePreset") == "カスタム":
        # 低域平坦化は正常トラッキング水準(帯域反転0.15/ジッター0.15)を超えた場合のみ。
        # 超過分だけを使い、通常のエイム微修正でカーブが削られないようにする(v3.9)
        low_d = (-min(60, int(max(0.0, band_rev[0] - BAND_REV_FLOOR) * 200
                             + max(0.0, j - 0.15) * 100))
                 if (band_rev[0] > BAND_REV_FLOOR or j > 0.15) else 0)
        mid_d = int(max(0, under - 0.25) * 120) - int(max(0, over - 0.30) * 120)
        high_d = -int(max(0, over - 0.35) * 80)
        # RC無効時: 旧RC(負=初動ブースト)が担っていた応答をカーブへ静的近似で移植。
        # under検出時のみの補償だと、リニアカーブ+RC構成からの移行時に
        # 「RC無し・カーブもリニア＝無補正」のプロファイルが出力される実害があった
        # ため、旧RC強度から常にカーブブーストを合成する（v4.2）
        if not use_rc:
            prev_rs = ((history or {}).get("current") or {}).get("rs")
            rc_mag = _rc_equivalent_boost(cur_rs, prev_rs)
            cur_shape = max(abs(p["out"] - p["in"]) for p in cur_rs["curvePoints"])
            if rc_mag > 0 and cur_shape < 30:   # 既にカーブへ移植済みなら二重適用しない
                lift = int(min(120, rc_mag * 0.45))
                low_d += int(lift * 0.35)
                mid_d += lift
                reasons.append(f"右: 旧RCブースト(平均-{rc_mag:.0f})の応答感をカーブで近似再現"
                               f"（低域+{int(lift*0.35)}/中域+{lift}）— RC不使用の静的代替です")
                audit.append({"rule": "RC→カーブ移植(右)", "value": round(rc_mag),
                              "threshold": "旧RC<0かつカーブ未成形", "fired": True,
                              "action": f"低域+{int(lift*0.35)}/中域+{lift}"})
            elif rc_mag == 0 and cur_shape < 30:
                # RC参照値なし: 実測の切り返し特性からカーブブーストを直接算出。
                # 横断時間(片側0.35→反対側0.35)が機械的下限(~90ms)を大きく超える
                # ＝リニア低域の応答不足で切り返しが重い → 低〜中域を持ち上げる
                rt_ms = m.get("revTransitMs")
                rpm = m.get("revPerMin", 0)
                if rt_ms is not None and rpm >= 4:
                    lift = int(min(100, max(0.0, rt_ms - 90) * 0.9))
                    fired_rev = lift >= 15
                    if fired_rev:
                        low_d += int(lift * 0.35)
                        mid_d += lift
                        reasons.append(
                            f"右: 実測の切り返し特性（横断{rt_ms:.0f}ms・{rpm:.0f}回/分）から"
                            f"低〜中域をカーブで補強（低域+{int(lift*0.35)}/中域+{lift}）"
                            f"— RC参照値なしの実測最適化")
                    audit.append({"rule": "切り返し実測最適化(右)",
                                  "value": round(rt_ms),
                                  "threshold": "横断>90ms かつ 4回/分以上",
                                  "fired": fired_rev,
                                  "action": (f"低域+{int(lift*0.35)}/中域+{lift}"
                                             if fired_rev else
                                             f"切り返しは十分機敏（{rt_ms:.0f}ms）→ 変更なし")})
                elif rpm < 4:
                    audit.append({"rule": "切り返し実測最適化(右)", "value": rpm,
                                  "threshold": "4回/分以上", "fired": False,
                                  "action": "切り返し標本不足（計測中に左右の振り向きを"
                                            "数回入れると実測最適化が働きます）"})
            if under > 0.30:
                boost = int(min(90, (under - 0.30) * 160))
                low_d += int(boost * 0.5)
                mid_d += boost
                reasons.append(f"右: RC無効のため初動不足(under={under:.2f})をカーブ低〜中域で補償"
                               f"（低域+{int(boost*0.5)}/中域+{boost}）")
        mq = m.get("magQuantiles")
        mh = m.get("magHist")
        placement = "分位点"
        if mh and sum(mh) >= 50:
            # v5.0: 目標カーブ(旧形状+デルタ)の曲率×使用密度で最適ノット配置。
            # 出力は目標カーブそのものをノット位置で標本化（デルタ込みで整合）
            gx, gy = _target_curve_grid(cur_rs["curvePoints"], low_d, mid_d, high_d)
            fb = [p["in"] for p in optimize_curve_points(
                cur_rs["curvePoints"], mq, 0, 0, 0)]
            ins = optimal_curve_inputs(gx, gy, mh, fb, min_gap=25)
            outs = [int(round(gy[min(len(gy) - 1, x // 5)])) for x in ins]
            for i in range(1, 8):
                if outs[i] <= outs[i - 1]:
                    outs[i] = outs[i - 1] + 1
            if outs[-1] > 1000:
                ov = outs[-1] - 1000
                outs = [max(0, o - ov) for o in outs]
            newpts = [{"in": min(999, ins[i]), "out": outs[i]} for i in range(8)]
            placement = "最適配置(密度×曲率)"
        else:
            newpts = optimize_curve_points(cur_rs["curvePoints"], mq,
                                           low_d, mid_d, high_d)
        # in=outの点はどこに置いてもリニア（感度不変）。リニア→リニアの座標移動は
        # 無意味な「変更風」表示になるだけなので抑止する（ユーザー指摘・v4.2）
        new_flat = max(abs(p["out"] - p["in"]) for p in newpts) < 12
        cur_flat = max(abs(p["out"] - p["in"]) for p in cur_rs["curvePoints"]) < 12
        if new_flat and cur_flat:
            newpts = cur_rs["curvePoints"]
            reasons.append("右: カーブは実質リニアのまま維持（in=outの点は位置に関わらず"
                           "感度同一のため座標も変更しません。補正不要の判定であり不具合"
                           "ではありません）")
        if newpts != cur_rs["curvePoints"]:
            rs["curvePoints"] = newpts
            reasons.append(f"右: カスタムカーブを実測から再構成 — 入力座標を"
                           f"{placement}{[p['in'] for p in newpts]}へ、"
                           f"出力を領域別調整(低域{low_d:+d}/中域{mid_d:+d}/高域{high_d:+d})")
        audit.append({"rule": "カーブ最適化(右)", "value": None, "threshold": "-",
                      "fired": newpts != cur_rs["curvePoints"],
                      "action": (f"in={placement} low={low_d:+d} "
                                 f"mid={mid_d:+d} high={high_d:+d}"
                                 + ("（リニア→リニアのため座標移動も抑止）"
                                    if (new_flat and cur_flat) else ""))})
    elif over > 0.35 or under > 0.35:
        reasons.append("右: カーブプリセットを「カスタム」にすると、実測に基づく"
                       "点単位の最適化が可能になります")

    # 水平偏差 → 再キャリブレーション提案
    hb = vm.get("horizontalBias", 0)
    if check("水平偏差バイアス", abs(hb), ">", 0.15, "再キャリブレーション提案"):
        side = "右" if hb > 0 else "左"
        reasons.append(f"クロスヘア偏差が{side}に偏り → キャリブレーションウィザードの"
                       f"「中心キャリブレーション（4方向サンプリング）」再実行を推奨")

    # サンプリング/高度サンプリング
    sb = m.get("snapbackRate", 0)
    if sb > 0.8:
        order = ["Extreme", "Excellent", "Good", "Robust"]
        cur = current.get("stickSampling", "Excellent")
        if cur in order and order.index(cur) < 3:
            rec["stickSampling"] = order[order.index(cur) + 1]
            reasons.append(f"スナップバック{sb:.1f}回/100 → サンプリングを"
                           f"「{rec['stickSampling']}」へ（安定寄り）")
    if check("アンダーシュート(高度サンプリング)", under, ">", 0.30,
             "高度サンプリング14bit") and current.get("advSampling") == "オフ":
        # 高度サンプリング=ADCオーバーサンプリング。分解能が上がる代わりに
        # 平均化で応答が僅かに鈍る（ローパス+微小レイテンシ）ため、
        # 切り返しの多い操作スタイルには提案しない（ユーザー実感と物理の整合）
        rpm = m.get("revPerMin", 0) or 0
        if rpm >= 15:
            reasons.append(f"微調整の分解能不足の兆候はありますが、切り返しが多い"
                           f"スタイル({rpm:.0f}回/分)のため高度サンプリングは"
                           f"「オフ」維持を推奨（オーバーサンプリングの平均化は"
                           f"切り返し応答に不利）")
        else:
            rec["advSampling"] = "14bit"
            reasons.append("微調整の分解能不足の兆候 → 高度サンプリング「14bit」を提案"
                           "（注: 平均化で応答が僅かに滑らか/遅くなるトレードオフあり。"
                           "切り返し重視ならオフ維持も妥当です）")

    # リコイル制御診断
    if m.get("firingSegmented") and m.get("recoilHoldJitter") is not None:
        rj = m["recoilHoldJitter"]
        if check("リコイル制御ジッター(射撃中)", rj, ">", 0.12, "低速帯RC/カーブ低域の再確認"):
            reasons.append(f"射撃中のリコイル制御に震え（σ={rj:.2f}）→ 今回の低速帯RC・"
                           f"カーブ低域の提案値で追い撃ちの安定を検証してください")

    # ============================================================
    # 左スティック: 右と同じ実測最適化（移動特性の重みで）
    # ============================================================
    ls = rec["ls"]
    cur_ls = current["ls"]
    lj = lm.get("jitterRatio", 0)
    lsb = lm.get("snapbackRate", 0)
    lsat = lm.get("saturationRatio", 0)
    l_band = lm.get("bandReversal", [0.0] * 5)
    check("中心ジッター率(左)", lj, ">", 0.15, "左:低速帯RC緩和/カーブ低域平坦化")
    check("スナップバック率(左)", lsb, ">", 0.8, "左:P5のRC緩和")
    check("外周飽和率(左)", lsat, ">", 0.50, "左:カーブ高域を持ち上げ最高速到達を短縮")

    if not use_rc:
        _disable_rc(ls)
        audit.append({"rule": "RC無効化(左)", "value": None, "threshold": "-", "fired": True,
                      "action": "RCフィルターOFF → 左スティックRCを無効化(enabled:false/全RC=0)"})

    if cur_ls.get("rcMode") == "アドバンスド" and cur_ls.get("rcEnabled") and use_rc:
        # 移動スティック: over/underの代わりにスナップバックを高速帯の抑制信号に使う
        l_over = min(1.0, lsb / 2.0) if lsb > 0.8 else 0.0
        ladv, l_changed, l_deltas = optimize_rc_tiers(cur_ls["rcAdvanced"], l_band,
                                                      l_over, 0.0, lj)
        prev_lrc = ((history or {}).get("recommendation") or {}) \
            .get("ls", {}).get("rcAdvanced")
        ladv, _ = _apply_rc_guards(cur_ls["rcAdvanced"], ladv, prev_lrc, True)
        l_deltas = [a["rc"] - o["rc"] for a, o in zip(ladv, cur_ls["rcAdvanced"])]
        l_changed = []
        for i, (o, a) in enumerate(zip(cur_ls["rcAdvanced"], ladv)):
            if abs(l_deltas[i]) < 3:
                ladv[i] = {"speed": a["speed"], "rc": o["rc"]}
                l_deltas[i] = 0
            elif a["rc"] != o["rc"]:
                l_changed.append(f"P{i+1}(速度{o['speed']}): {o['rc']}→{a['rc']}")
        for i in range(5):
            audit.append({"rule": f"RC最適化(左) P{i+1}", "value": l_deltas[i],
                          "threshold": "|Δ|>=3", "fired": abs(l_deltas[i]) >= 3,
                          "action": next((c for c in l_changed
                                          if c.startswith(f"P{i+1}(")), "変更なし")})
        if lsb > 0.8:
            old5 = ladv[4]["rc"]
            ladv[4]["rc"] = _clamp_rc(old5 + 30)
            l_changed.append(f"P5: {old5}→{ladv[4]['rc']}（スナップバック対策）")
        lq = lm.get("speedQuantiles")
        l_speeds = _optimize_rc_speeds(ladv, lq, lm.get("speedHist"))
        audit.append({"rule": "RC速度段最適化(左)", "value": None,
                      "threshold": "実測分位点との乖離>=8", "fired": bool(l_speeds),
                      "action": (f"{[t['speed'] for t in ladv]}→{l_speeds}"
                                 if l_speeds else "変更なし")})
        if l_speeds:
            for i, s in enumerate(l_speeds):
                ladv[i]["speed"] = s
            reasons.append(f"左: 実測速度分布（分位点 {lq}）からRC速度段を {l_speeds} に再配置")
        ls["rcAdvanced"] = ladv
        if l_changed:
            reasons.append("左: RC値を実測から最適化 → " + " / ".join(l_changed))

    if cur_ls.get("curvePreset") == "カスタム":
        l_low = (-min(60, int(max(0.0, l_band[0] - BAND_REV_FLOOR) * 200
                             + max(0.0, lj - 0.15) * 100))
                 if (l_band[0] > BAND_REV_FLOOR or lj > 0.15) else 0)
        l_high = int(max(0.0, lsat - 0.40) * 120)   # 飽和が多い→高域を持ち上げ最高速へ早く
        lmq = lm.get("magQuantiles")
        # P1〜P8を全域バランス配置（均等基準±70で実測へ寄せる）: v4.9
        l_pts = optimize_left_curve_points(cur_ls["curvePoints"], lmq, l_low, l_high)
        # リニア→リニアの座標移動は感度不変で無意味なため抑止（右と同様）
        l_new_flat = max(abs(p["out"] - p["in"]) for p in l_pts) < 12
        l_cur_flat = max(abs(p["out"] - p["in"]) for p in cur_ls["curvePoints"]) < 12
        if l_new_flat and l_cur_flat:
            l_pts = cur_ls["curvePoints"]
        if l_pts != cur_ls["curvePoints"]:
            ls["curvePoints"] = l_pts
            reasons.append(f"左: カーブをP1〜P8バランス配置で再構成"
                           f"（全域を均等カバーしつつ実測分布へ寄せ、"
                           f"低域{l_low:+d}/高域{l_high:+d} — 高域増は最高移動速度への到達短縮）")
        audit.append({"rule": "カーブ最適化(左)", "value": None, "threshold": "-",
                      "fired": l_pts != cur_ls["curvePoints"],
                      "action": f"バランス配置 in={[p['in'] for p in l_pts]} "
                                f"low={l_low:+d} high={l_high:+d}"
                                + ("（リニア維持）" if (l_new_flat and l_cur_flat) else "")})

    reasons.extend(sens_recommendations(over, under, apex, ctx))
    if not reasons:
        reasons.append("大きな問題は検出されませんでした。現行設定の維持を推奨します。")
    return rec, reasons, audit
