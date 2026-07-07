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
# APEXゲーム内感度モデル（VPK由来のコミュニティ換算値）
# プリセット感度 n (1-8) ≒ ALC yaw 62.5*n [deg/s]、pitch ≒ yaw*0.75
# 詳細スコープ感度(Per Optic)は各倍率への乗数
# ------------------------------------------------------------
APEX_SENS_YAW = {n: 62.5 * n for n in range(1, 9)}
APEX_DEFAULT_OPTICS = {"1x": 1.0, "2x": 1.0, "3x": 1.0, "4x": 1.0,
                       "6x": 1.0, "8x": 1.0, "10x": 1.0}

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
}

APEX_DEFAULT = {"lookSens": 4, "adsSens": 3, "perOptic": False,
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
}


def detect_weapon_from_text(texts):
    """OCRで得た文字列群から武器を多数決で判定。
    戻り値: (武器名 or None, {武器: 票数})"""
    votes = {}
    for raw in texts:
        if not raw:
            continue
        s = raw.upper().replace(" ", "").replace("\u3000", "")
        hit_in_this_frame = set()
        for weapon, aliases in WEAPON_ALIASES.items():
            for al in aliases:
                if al.replace(" ", "").replace("-", "") in s.replace("-", ""):
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
    yaw = APEX_SENS_YAW.get(int(apex.get("lookSens", 4)), 250.0)
    ads_yaw = APEX_SENS_YAW.get(int(apex.get("adsSens", 3)), 187.5)
    ctx = {"yawSpeed": yaw, "pitchSpeed": yaw * 0.75,
           "adsYawSpeed": ads_yaw, "adsPitchSpeed": ads_yaw * 0.75,
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
        self.input_name = self.session.get_inputs()[0].name
        shp = self.session.get_inputs()[0].shape
        self.imgsz = int(shp[2]) if isinstance(shp[2], int) else 640
        self.model_name = os.path.basename(model_path)
        self.num_classes = None   # 初回推論時に出力形状から自動判定

    @property
    def gpu_active(self) -> bool:
        return self.active_provider not in ("CPUExecutionProvider",) and self.session is not None

    def detect_persons(self, frame_bgr: np.ndarray, conf_th: float = 0.35):
        """フレーム内の人型(敵候補)を検出し、画面中心からの正規化偏差リストを返す"""
        if self.session is None:
            return []
        h, w = frame_bgr.shape[:2]
        img = self._letterbox(frame_bgr, self.imgsz)
        blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob})[0]
        # YOLOv8 出力: (1, 4+nc, 8400) → 転置して boxes
        pred = out[0].T if out.shape[1] < out.shape[2] else out[0]
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
        results = []
        for row in pred:
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < conf_th or (target_cls is not None and cls_id not in target_cls):
                continue
            cx, cy = float(row[0]), float(row[1])
            # letterbox 逆変換
            scale = self.imgsz / max(h, w)
            pad_x = (self.imgsz - w * scale) / 2
            pad_y = (self.imgsz - h * scale) / 2
            px = (cx - pad_x) / scale
            py = (cy - pad_y) / scale
            bw = float(row[2]) / scale
            bh = float(row[3]) / scale
            results.append({
                "devX": (px - w / 2) / (w / 2),   # -1〜1（右が正）
                "devY": (py - h / 2) / (h / 2),   # -1〜1（下が正）
                "conf": conf,
                "cls": cls_id,
                "box": [int(px - bw / 2), int(py - bh / 2), int(bw), int(bh)],
            })
        return results

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
def firing_intervals(samples, th=0.5, tail_ms=150.0):
    """射撃(トリガー/ボタン)区間 [(start_ms, end_ms), ...] を抽出。tailはリコイル整定分の余韻"""
    iv = []
    start = None
    for s in samples:
        firing = s.get("rt", 0.0) > th
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


def _axis_metrics(samples, kx, ky, exclude=None):
    """指定軸ペア(右=rx/ry, 左=lx/ly)のメトリクスを算出"""
    n = len(samples)
    snapback = jitter = center_n = reversals = micro_n = sat = fast = 0
    band_edges = [32, 80, 128, 192, 256]
    band_rev = [0] * 5
    band_n = [0] * 5
    speeds = []
    for i in range(2, n):
        skip_aim = exclude[i] if exclude is not None else False
        p2, p1, p0 = samples[i - 2], samples[i - 1], samples[i]
        x2, x1, x0 = p2[kx], p1[kx], p0[kx]
        mag = math.hypot(p0[kx], p0[ky])
        v1 = x1 - x2
        v0 = x0 - x1
        vmag = math.hypot(p0[kx] - p1[kx], p0[ky] - p1[ky])
        v255 = min(255, int(vmag * 765))
        if v255 > 2:
            speeds.append(v255)
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
            if abs(v0) > 0.015:
                jitter += 1
        if not skip_aim and 0.05 < mag < 0.35 and v1 * v0 < 0:
            reversals += 1
        if not skip_aim and 0.05 < mag < 0.30:
            micro_n += 1
        if mag > 0.97:
            sat += 1
        if abs(v0) > 0.25:
            fast += 1
    # 実測スティック速度の分位点（RC速度段の最適化に使用）
    quantiles = None
    if len(speeds) >= 100:
        speeds.sort()
        pick = lambda q: speeds[min(len(speeds) - 1, int(q * len(speeds)))]
        quantiles = [pick(0.35), pick(0.60), pick(0.80), pick(0.93)]
    return {
        "sampleCount": n,
        "snapbackRate": snapback / (n / 100),
        "jitterRatio": jitter / center_n if center_n else 0.0,
        "reversalRatio": reversals / micro_n if micro_n else 0.0,
        "saturationRatio": sat / n,
        "fastMoveRatio": fast / n,
        "bandReversal": [band_rev[b] / band_n[b] if band_n[b] else 0.0 for b in range(5)],
        "speedQuantiles": quantiles,
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


def summarize_vision(frame_results, firing_iv=None):
    """frame_results: [{t, targets:[{devX,devY,conf}]}]"""
    hits = []
    for fr in frame_results:
        if firing_iv and _in_intervals(fr["t"], firing_iv):
            continue   # 射撃中の偏差はリコイル影響下のため除外
        if fr["targets"]:
            best = min(fr["targets"], key=lambda t: math.hypot(t["devX"], t["devY"]))
            hits.append({"t": fr["t"], **best})
    if len(hits) < 3:
        return None
    over = under = 0
    hx = 0.0
    for i, h in enumerate(hits):
        d = math.hypot(h["devX"], h["devY"])
        if d > 0.22:
            under += 1
        hx += h["devX"]
        if i > 0 and h["devX"] * hits[i - 1]["devX"] < 0 \
                and abs(h["devX"]) > 0.08 and abs(hits[i - 1]["devX"]) > 0.08:
            over += 1
    vy = sum(h["devY"] for h in hits) / len(hits)
    return {
        "frames": len(hits),
        "overshootRatio": over / len(hits),
        "undershootRatio": under / len(hits),
        "horizontalBias": hx / len(hits),
        "verticalBias": vy,
    }


# ------------------------------------------------------------
# 提案エンジン（実機スキーマにクランプ）
# ------------------------------------------------------------
def _clamp(key, v):
    s = SCHEMA[key]
    return max(s["min"], min(s["max"], int(round(v))))


def _clamp_rc(v):
    return max(-500, min(500, int(round(v))))


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


def _optimize_rc_speeds(current_tiers, quantiles):
    """実測スティック速度の分位点からRC速度段の境界を提案。
    最終段は255固定、昇順・最小間隔2を保証。変化が小さければNone"""
    if not quantiles or len(current_tiers) != 5:
        return None
    prop = []
    prev = 0
    for q in quantiles[:4]:
        s = max(prev + 2, min(250, int(q)))
        prop.append(s)
        prev = s
    prop.append(255)
    cur = [t["speed"] for t in current_tiers]
    if max(abs(a - b) for a, b in zip(prop, cur)) < 8:
        return None   # 現状と大差なし
    return prop


def build_recommendation(current, stick_m, vision_m):
    import copy
    rec = copy.deepcopy(current)
    reasons = []
    audit = []

    def check(rule, value, op, th, action=""):
        """判定を監査ログに記録して結果を返す"""
        fired = (value > th) if op == ">" else (value < th)
        audit.append({
            "rule": rule,
            "value": round(float(value), 4) if value is not None else None,
            "threshold": f"{op}{th}",
            "fired": bool(fired),
            "action": action if fired else "変更なし",
        })
        return fired

    raw_stick = dict(stick_m) if stick_m else None
    apex = current.get("apex", APEX_DEFAULT)
    stick_m, vision_m, ctx = apply_game_context(stick_m, vision_m, apex)
    m = stick_m or {}
    vm = vision_m or {}
    if raw_stick and raw_stick.get("firingSegmented"):
        audit.append({"rule": "_射撃区間分離", "value": None, "threshold": "-", "fired": True,
                      "action": f"射撃ボタン実測により射撃区間{raw_stick.get('firingRatio',0)*100:.0f}%を"
                                f"エイム指標から除外（ADS率: "
                                f"{raw_stick.get('adsRatio',0)*100:.0f}%）"})
    audit.append({"rule": "_入力データ", "value": None, "threshold": "-", "fired": True,
                  "action": f"サンプル数={m.get('sampleCount', 0)}, "
                            f"Vision敵検出フレーム={vm.get('frames', 0) if vm else 0}, "
                            f"武器={apex.get('weapon')}, リコイル割引後の反転率="
                            f"{m.get('reversalRatio', 0):.3f}"})
    if not vm:
        audit.append({"rule": "_Vision解析", "value": None, "threshold": "-", "fired": False,
                      "action": "敵検出フレームが3未満のためVision系判定(オーバー/アンダーシュート/偏差)は全て不発。"
                                "入力ログのみで判定"})
    reasons.extend(ctx["notes"])
    rs = rec["rs"]
    cur_rs = current["rs"]

    # --- 中心ジッター → 中心デッドゾーン ---
    j = m.get("jitterRatio", 0)
    if check("中心ジッター率(拡大)", j, ">", 0.15, "中心DZ拡大"):
        rs["centerDZ"] = _clamp("centerDZ", cur_rs["centerDZ"] + max(1, round(j * 10)))
        reasons.append(f"ニュートラル時ジッター率 {j*100:.0f}% → 右スティック中心デッドゾーンを "
                       f"{cur_rs['centerDZ']}% → {rs['centerDZ']}% に拡大（範囲0〜30%）")
    elif check("中心ジッター率(縮小)", j, "<", 0.03,
               "中心DZ縮小") and cur_rs["centerDZ"] > 2:
        rs["centerDZ"] = _clamp("centerDZ", cur_rs["centerDZ"] - 1)
        reasons.append(f"ドリフト兆候なし → 中心デッドゾーンを {rs['centerDZ']}% に縮小し初動を軽く")

    # ============================================================
    # RC2.0（動的フィルター特性: 負値=変化ブースト/初動応答UP、正値=平滑化/遅延増）
    # アドバンスド有効時は速度帯域(P1〜P5)ごとに個別調整する
    # ============================================================
    band_rev = m.get("bandReversal", [0] * 5)
    over = vm.get("overshootRatio", 0) or m.get("reversalRatio", 0)
    under = vm.get("undershootRatio", 0)
    audit.append({"rule": "_複合指標", "value": None, "threshold": "-", "fired": True,
                  "action": f"over={over:.3f} (Vision符号反転率 or 入力反転率), "
                            f"under={under:.3f} (Vision偏差>0.22率)"})
    check("オーバーシュート(RC/カーブ変更)", over, ">", 0.35, "高速帯RCを0方向へ/カーブ緩和")
    check("アンダーシュート(RC変更)", under, ">", 0.40, "低速帯RC強化/アンチDZ増")
    if current["rs"].get("rcMode") == "アドバンスド":
        _adv = current["rs"].get("rcAdvanced", [])
        for _i in range(min(2, len(_adv))):
            _br = band_rev[_i] if _i < len(band_rev) else 0
            _rc = _adv[_i].get("rc", 0)
            check(f"低速帯ブースト過多 P{_i+1}(反転率×|RC|)",
                  _br * abs(_rc), ">", 25,
                  f"P{_i+1}のRC負値を0方向へ緩和")
    use_adv = cur_rs.get("rcMode") == "アドバンスド" and cur_rs.get("rcEnabled")

    if use_adv:
        adv = [dict(p) for p in cur_rs["rcAdvanced"]]
        changed_tiers = []
        for i in range(5):
            tier = adv[i]
            br = band_rev[i] if i < len(band_rev) else 0
            # 高速帯(P4/P5)でオーバーシュート → 負値ブーストを弱める(0方向へ)
            if i >= 3 and (over > 0.35 or br > 0.30):
                delta = int(20 + 60 * max(over, br))
                tier["rc"] = _clamp_rc(tier["rc"] + delta if tier["rc"] < 0
                                       else tier["rc"] + delta // 2)
                changed_tiers.append(f"P{i+1}(速度{tier['speed']}): "
                                     f"{cur_rs['rcAdvanced'][i]['rc']}→{tier['rc']}")
            # 低速帯(P1/P2)でアンダーシュート/初動遅れ → 負値ブーストを強める
            # （オーバーシュート検出時は矛盾するため発動しない）
            elif i <= 1 and over <= 0.35 and (under > 0.40 or m.get("fastMoveRatio", 1) < 0.02):
                delta = int(20 + 50 * under)
                tier["rc"] = _clamp_rc(tier["rc"] - delta)
                changed_tiers.append(f"P{i+1}(速度{tier['speed']}): "
                                     f"{cur_rs['rcAdvanced'][i]['rc']}→{tier['rc']}")
            # 低速帯で微振動が多い → ブースト過多。反転率×|RC|の積で動的判定
            # （例: 反転率16%でもRC-209なら積33>25で発動。RC-70なら積11で不発）
            elif i <= 1 and tier["rc"] < 0 and br * abs(tier["rc"]) > 25:
                ease = max(15, min(60, int(br * abs(tier["rc"]))))
                tier["rc"] = _clamp_rc(tier["rc"] + ease)
                changed_tiers.append(f"P{i+1}(速度{tier['speed']}): "
                                     f"{cur_rs['rcAdvanced'][i]['rc']}→{tier['rc']}"
                                     f"（微操作域の反転率{br*100:.0f}%×強RCブーストの過多を緩和）")
        # 実測速度分布からRC速度段の境界を最適化
        q = m.get("speedQuantiles")
        new_speeds = _optimize_rc_speeds(adv, q)
        audit.append({"rule": "RC速度段最適化(右)", "value": None,
                      "threshold": "実測分位点との乖離>=8",
                      "fired": bool(new_speeds),
                      "action": (f"速度段を {[t['speed'] for t in adv]} → {new_speeds} に変更"
                                 if new_speeds else "変更なし（実測分布と概ね一致）")})
        if new_speeds:
            for i, s in enumerate(new_speeds):
                adv[i]["speed"] = s
            reasons.append(f"実測スティック速度分布（分位点 {q}）に基づき、右スティックの"
                           f"RC速度段を {new_speeds} に再配置（各段が実際の操作速度帯を均等にカバー）")
        rs["rcAdvanced"] = adv
        if changed_tiers:
            if over > 0.35:
                reasons.append(f"オーバーシュート傾向（率 {over*100:.0f}%）: RCフィルターは負値ほど"
                               f"入力変化をブーストするため、高速帯の負値を0方向へ縮小 → "
                               + " / ".join(changed_tiers))
            elif under > 0.40 or m.get("fastMoveRatio", 1) < 0.02:
                reasons.append(f"アンダーシュート/初動遅れ傾向: 低速帯（微操作〜追いエイム域）の"
                               f"RC負値を強めて初動応答を改善 → " + " / ".join(changed_tiers))
            else:
                reasons.append("速度帯域別のRC調整 → " + " / ".join(changed_tiers))
    else:
        # ベーシック: 全域RC強度を一括調整
        cur_rc = cur_rs["rcStrength"]
        if over > 0.35:
            rs["rcStrength"] = _clamp_rc(cur_rc + int(20 + 60 * over))
            reasons.append(f"オーバーシュート傾向（率 {over*100:.0f}%）→ 全域RC強度を "
                           f"{cur_rc} → {rs['rcStrength']}（負値ブーストを縮小し行き過ぎを抑制）")
        elif under > 0.40:
            rs["rcStrength"] = _clamp_rc(cur_rc - int(20 + 50 * under))
            reasons.append(f"アンダーシュート傾向 → 全域RC強度を {cur_rc} → {rs['rcStrength']}"
                           f"（負値を強め初動応答を改善）")

    # ============================================================
    # カスタムカーブ（P1〜P8 出力値）の調整
    # ============================================================
    if cur_rs.get("curvePreset") == "カスタム":
        low_delta = 0
        if band_rev and band_rev[0] > 0.08:
            low_delta = -min(60, int(band_rev[0] * 200))   # 微操作域の過敏を沈める
        mid_delta = int(under * 80) - int(over * 80)
        high_delta = 0
        if over > 0.35:
            high_delta = -int(over * 50)                   # フリック行き過ぎを抑制
        elif m.get("fastMoveRatio", 1) < 0.02 and under > 0.30:
            high_delta = +25                               # 速い旋回の立ち上がりを補助
        if any((low_delta, mid_delta, high_delta)):
            rs["curvePoints"] = _adjust_curve_regions(
                cur_rs["curvePoints"], low_delta, mid_delta, high_delta)
            reasons.append(f"カスタムカーブを領域別に最適化: 微操作域{low_delta:+d} / "
                           f"追いエイム域{mid_delta:+d} / フリック域{high_delta:+d}"
                           f"（単調増加・0〜1000を維持、入力座標はDZ連動のため出力のみ調整）")
        audit.append({"rule": "カーブ領域別最適化(右)", "value": None, "threshold": "-",
                      "fired": bool(any((low_delta, mid_delta, high_delta))),
                      "action": f"low={low_delta:+d} mid={mid_delta:+d} high={high_delta:+d}"})
    else:
        if over > 0.35:
            rs["curveAdjust"] = _clamp("curveAdjust", cur_rs["curveAdjust"] - 1)
            reasons.append(f"カーブ調整を {rs['curveAdjust']} に一段緩和（オーバーシュート抑制）")
            if cur_rs["curvePreset"] in ("クイック", "ダイナミック"):
                rs["curvePreset"] = "精密"
                reasons.append("高速寄りプリセット使用中のため、カーブプリセットを「精密」に変更を提案")
        elif under > 0.40:
            rs["antiDZ"] = _clamp("antiDZ", cur_rs["antiDZ"] + 2)
            reasons.append(f"アンチデッドゾーンを {rs['antiDZ']}% に増加して初動の立ち上がりを改善")

    # --- スナップバック → サンプリングプリセット ---
    sb = m.get("snapbackRate", 0)
    if check("スナップバック率", sb, ">", 0.8, "サンプリング安定寄り/P5RC緩和"):
        order = ["Extreme", "Excellent", "Good", "Robust"]
        cur = current.get("stickSampling", "Excellent")
        idx = min(len(order) - 1, order.index(cur) + 1) if cur in order else 2
        rec["stickSampling"] = order[idx]
        reasons.append(f"スナップバック {sb:.1f}回/100サンプル検出 → スティックサンプリングを "
                       f"「{cur}」→「{rec['stickSampling']}」へ（安定寄り）")
        if use_adv:
            adv = rs["rcAdvanced"]
            old = adv[4]["rc"]
            adv[4]["rc"] = _clamp_rc(old + 30)
            reasons.append(f"スナップバック対策としてP5(最高速帯)のRCも {old}→{adv[4]['rc']} に緩和")
    elif sb < 0.1 and current.get("stickSampling") == "Robust":
        rec["stickSampling"] = "Good"
        reasons.append("スナップバック未検出 → サンプリングを「Good」に戻し応答性を回復")

    # --- 外周飽和 → 外周デッドゾーン ---
    sat = m.get("saturationRatio", 0)
    if check("外周飽和率", sat, ">", 0.25, "外周DZ拡大"):
        rs["outerDZ"] = _clamp("outerDZ", cur_rs["outerDZ"] + 3)
        reasons.append(f"最大倒し込み使用率 {sat*100:.0f}% → 外周デッドゾーンを "
                       f"{rs['outerDZ']}% に拡大し最大旋回へ早く到達")

    # --- 水平偏差の偏り ---
    hb = vm.get("horizontalBias", 0)
    if check("水平偏差バイアス", abs(hb), ">", 0.15, "再キャリブレーション提案"):
        side = "右" if hb > 0 else "左"
        reasons.append(f"クロスヘア偏差が{side}に偏っています。HyperStrike Hubには角度補正項目が"
                       f"ないため、キャリブレーションウィザードの「中心キャリブレーション"
                       f"（4方向サンプリング）」の再実行を推奨します")

    # --- 微細精度 → 高度サンプリング ---
    if check("アンダーシュート(中程度→アンチDZ)", under, ">", 0.25,
             "アンチDZ+1で初動改善") and over < 0.20 and rs["antiDZ"] == cur_rs["antiDZ"]:
        rs["antiDZ"] = _clamp("antiDZ", cur_rs["antiDZ"] + 1)
        if rs["antiDZ"] != cur_rs["antiDZ"]:
            reasons.append(f"中程度のアンダーシュート({under*100:.0f}%) → アンチデッドゾーンを "
                           f"{cur_rs['antiDZ']}% → {rs['antiDZ']}% に微増して初動の立ち上がりを改善")

    if check("アンダーシュート(高度サンプリング)", vm.get("undershootRatio", 0), ">", 0.30,
             "高度サンプリング14bit") and current.get("advSampling") == "オフ":
        rec["advSampling"] = "14bit"
        reasons.append("微調整の精度不足の兆候 → 高度サンプリングレベルを「14bit」に"
                       "（内部分解能向上、操作感はやや重くなります）")

    # ============================================================
    # リコイル制御診断（射撃区間の実測: 下方向保持の安定性）
    # ============================================================
    if m.get("firingSegmented") and m.get("recoilHoldJitter") is not None:
        rj = m["recoilHoldJitter"]
        wrec = ctx["recoil"]
        if check("リコイル制御ジッター(射撃中)", rj, ">", 0.12,
                 "微操作域RC/カーブの再確認を提案"):
            reasons.append(f"射撃中のリコイル制御に震え（σ={rj:.2f}, 平均下入力"
                           f"{m.get('recoilHoldMean', 0):+.2f}, 武器縦リコイル{wrec['vert']:.1f}）→ "
                           f"微操作域のRCブースト過多か、カーブ低域が敏感すぎる可能性。"
                           f"P1/P2のRC値とカーブP1〜P3を今回の提案値で検証してください")

    # ============================================================
    # 左スティック（移動）: 左軸の実測メトリクスで独立に最適化
    # ============================================================
    lm = m.get("ls") or {}
    ls = rec["ls"]
    cur_ls = current["ls"]
    lj = lm.get("jitterRatio", 0)
    if check("中心ジッター率(左)", lj, ">", 0.15, "左:中心DZ拡大"):
        ls["centerDZ"] = _clamp("centerDZ", cur_ls["centerDZ"] + max(1, round(lj * 10)))
        reasons.append(f"左スティックのニュートラル時ジッター率 {lj*100:.0f}% → "
                       f"中心デッドゾーンを {cur_ls['centerDZ']}% → {ls['centerDZ']}% に拡大")
    elif lj < 0.03 and cur_ls["centerDZ"] > 2:
        ls["centerDZ"] = _clamp("centerDZ", cur_ls["centerDZ"] - 1)
        reasons.append(f"左スティックにドリフト兆候なし → 中心DZを {ls['centerDZ']}% に縮小")

    lsat = lm.get("saturationRatio", 0)
    if check("外周飽和率(左/最高速到達)", lsat, ">", 0.50, "左:外周DZ拡大"):
        ls["outerDZ"] = _clamp("outerDZ", cur_ls["outerDZ"] + 3)
        reasons.append(f"左スティックの最大倒し込み率 {lsat*100:.0f}% → 外周DZを "
                       f"{ls['outerDZ']}% に拡大し最高移動速度へ早く到達")

    if cur_ls.get("rcMode") == "アドバンスド" and cur_ls.get("rcEnabled"):
        ladv = [dict(p) for p in cur_ls["rcAdvanced"]]
        l_band = lm.get("bandReversal", [0] * 5)
        l_changed = []
        for i in range(2):
            br = l_band[i] if i < len(l_band) else 0
            if ladv[i]["rc"] < 0 and br * abs(ladv[i]["rc"]) > 25:
                ease = max(15, min(60, int(br * abs(ladv[i]["rc"]))))
                old_rc = ladv[i]["rc"]
                ladv[i]["rc"] = _clamp_rc(old_rc + ease)
                l_changed.append(f"P{i+1}(速度{ladv[i]['speed']}): {old_rc}→{ladv[i]['rc']}")
            check(f"低速帯ブースト過多(左) P{i+1}(反転率×|RC|)",
                  br * abs(cur_ls["rcAdvanced"][i]["rc"]), ">", 25,
                  f"左P{i+1}のRC負値を0方向へ緩和")
        lsb = lm.get("snapbackRate", 0)
        if check("スナップバック率(左)", lsb, ">", 0.8, "左:P5のRC緩和"):
            old_rc = ladv[4]["rc"]
            ladv[4]["rc"] = _clamp_rc(old_rc + 30)
            l_changed.append(f"P5: {old_rc}→{ladv[4]['rc']}（スナップバック対策）")
        lq = lm.get("speedQuantiles")
        l_speeds = _optimize_rc_speeds(ladv, lq)
        audit.append({"rule": "RC速度段最適化(左)", "value": None,
                      "threshold": "実測分位点との乖離>=8",
                      "fired": bool(l_speeds),
                      "action": (f"速度段を {[t['speed'] for t in ladv]} → {l_speeds}"
                                 if l_speeds else "変更なし")})
        if l_speeds:
            for i, s in enumerate(l_speeds):
                ladv[i]["speed"] = s
            reasons.append(f"左スティックの実測速度分布（分位点 {lq}）に基づき、"
                           f"RC速度段を {l_speeds} に再配置")
        if l_changed:
            reasons.append("左スティックの速度RC調整 → " + " / ".join(l_changed))
        ls["rcAdvanced"] = ladv

    reasons.extend(sens_recommendations(over, under, apex, ctx))

    if not reasons:
        reasons.append("大きな問題は検出されませんでした。現行設定の維持を推奨します。")
    return rec, reasons, audit
