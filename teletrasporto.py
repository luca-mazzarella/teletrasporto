#!/usr/bin/env python3
"""
Teletrasporto (Shunkan Idō) — effetto "Instant Transmission" di Goku sulla webcam.

Porta indice e medio uniti alla fronte (il gesto di Goku): dopo un attimo
sparisci con un'animazione a sfarfallio. Ripeti il gesto per riapparire.

Tasti:
  q / ESC  esci
  b        cattura lo sfondo (esci dall'inquadratura e premi b: risultato migliore)
  spazio   attiva l'effetto manualmente
  d        mostra/nascondi debug (punti della mano, fronte, barra di carica)
  h        mostra/nascondi l'help a schermo
"""
import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_BASE = "https://storage.googleapis.com/mediapipe-models/"
MODELS = {
    "hand": ("hand_landmarker.task", "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"),
    "face": ("blaze_face_short_range.tflite", "face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"),
    "seg": ("selfie_segmenter.tflite", "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"),
}

# Durate dell'effetto (secondi)
VANISH_TIME = 0.55
APPEAR_TIME = 0.45
HOLD_TIME = 0.25      # quanto tenere il gesto prima che scatti
COOLDOWN = 0.8        # pausa minima tra un effetto e l'altro

GLOW_BGR = np.array([255, 240, 205], np.float32)  # bianco-azzurro "ki"


def ensure_model(key):
    name, path = MODELS[key]
    os.makedirs(MODEL_DIR, exist_ok=True)
    dest = os.path.join(MODEL_DIR, name)
    if not os.path.exists(dest):
        print(f"Scarico il modello {name}...")
        urllib.request.urlretrieve(MODEL_BASE + path, dest)
    return dest


# --------------------------------------------------------------------------- #
# Rilevamento: mani, volto, segmentazione persona
# --------------------------------------------------------------------------- #
class Tracker:
    def __init__(self):
        base = mp_python.BaseOptions
        video = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=ensure_model("hand")), running_mode=video,
            num_hands=2, min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_path=ensure_model("face")), running_mode=video,
            min_detection_confidence=0.5))
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base(model_asset_path=ensure_model("seg")), running_mode=video,
            output_confidence_masks=True, output_category_mask=False))
        self.last_ts = -1

    def process(self, frame_bgr, ts_ms):
        ts_ms = max(int(ts_ms), self.last_ts + 1)  # MediaPipe vuole timestamp crescenti
        self.last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hands = self.hands.detect_for_video(img, ts_ms)
        faces = self.face.detect_for_video(img, ts_ms)
        masks = self.seg.segment_for_video(img, ts_ms).confidence_masks
        if len(masks) == 1:
            mask = masks[0].numpy_view()
        else:  # modelli multiclasse: canale 0 = sfondo
            mask = 1.0 - masks[0].numpy_view()
        mask = np.squeeze(mask).astype(np.float32)
        h, w = frame_bgr.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h))
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return hands, faces, mask

    def close(self):
        self.hands.close()
        self.face.close()
        self.seg.close()


def forehead_from_faces(faces, w, h):
    """Punto della fronte e distanza tra gli occhi (scala) del volto più grande."""
    if not faces.detections:
        return None
    det = max(faces.detections, key=lambda d: d.bounding_box.width)
    kp = det.keypoints  # 0 occhio dx, 1 occhio sx, 2 naso, 3 bocca, 4-5 orecchie
    re = np.array([kp[0].x * w, kp[0].y * h])
    le = np.array([kp[1].x * w, kp[1].y * h])
    mouth = np.array([kp[3].x * w, kp[3].y * h])
    eye_mid = (re + le) / 2
    d = max(np.linalg.norm(re - le), 1.0)
    up = eye_mid - mouth
    n = np.linalg.norm(up)
    up = up / n if n > 1e-3 else np.array([0.0, -1.0])
    return eye_mid + up * d * 0.75, d


def detect_gesture(hands, forehead, w, h):
    """True se una mano ha indice+medio tesi e uniti, anulare+mignolo piegati,
    con le punte delle dita appoggiate alla fronte."""
    if forehead is None or not hands.hand_landmarks:
        return False, []
    fh, d = forehead
    for lm in hands.hand_landmarks:
        p = np.array([[l.x * w, l.y * h] for l in lm])
        wrist = p[0]
        dist = lambda i: np.linalg.norm(p[i] - wrist)
        extended = lambda tip, pip: dist(tip) > dist(pip) * 1.12
        folded = lambda tip, pip: dist(tip) < dist(pip) * 1.08
        if not (extended(8, 6) and extended(12, 10)):
            continue
        if not (folded(16, 14) and folded(20, 18)):
            continue
        tips_mid = (p[8] + p[12]) / 2
        if np.linalg.norm(p[8] - p[12]) > d * 0.9:      # dita unite
            continue
        if np.linalg.norm(tips_mid - fh) > d * 1.1:      # sulla fronte
            continue
        return True, [p[8], p[12]]
    return False, []


# --------------------------------------------------------------------------- #
# Sfondo pulito (per mostrare cosa c'è "dietro" quando sparisci)
# --------------------------------------------------------------------------- #
class Background:
    def __init__(self):
        self.bg = None
        self.known = None
        self._filled = None
        self._dirty = True
        self._counter = 0

    def update(self, frame, mask):
        free = cv2.dilate((mask > 0.15).astype(np.uint8), np.ones((25, 25), np.uint8)) == 0
        f = frame.astype(np.float32)
        if self.bg is None:
            self.bg = f.copy()
            self.known = free.copy()
        else:
            self.bg[free] = self.bg[free] * 0.85 + f[free] * 0.15
            self.known |= free
        self._dirty = True

    def capture(self, frame):
        self.bg = frame.astype(np.float32)
        self.known = np.ones(frame.shape[:2], bool)
        self._dirty = True

    def filled(self):
        """Sfondo con le zone mai viste ricostruite (inpainting a bassa risoluzione)."""
        self._counter += 1
        if self._filled is not None and not (self._dirty and self._counter % 10 == 0):
            return self._filled
        self._dirty = False
        if self.known.all():
            self._filled = self.bg.copy()
            return self._filled
        h, w = self.known.shape
        sw, sh = max(w // 4, 1), max(h // 4, 1)
        small = cv2.resize(np.clip(self.bg, 0, 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((~self.known).astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small = cv2.inpaint(small, hole, 7, cv2.INPAINT_TELEA)
        up = cv2.GaussianBlur(cv2.resize(small, (w, h)), (21, 21), 0).astype(np.float32)
        self._filled = np.where(self.known[..., None], self.bg, up)
        return self._filled


# --------------------------------------------------------------------------- #
# L'effetto vero e proprio
# --------------------------------------------------------------------------- #
def ease_in(t):
    return t * t * t


def render(frame, mask, bg, p, tick, rng):
    """p = 0 completamente visibile, p = 1 sparito."""
    if p <= 0.0:
        return frame
    f = frame.astype(np.float32)
    h, w = mask.shape
    # zona da "cancellare": maschera allargata per non lasciare aloni
    hole = cv2.GaussianBlur(cv2.dilate(mask, np.ones((15, 15), np.uint8)), (21, 21), 0)[..., None]
    base = f + (bg - f) * hole
    if p >= 1.0:
        return np.clip(base, 0, 255).astype(np.uint8)

    binm = (mask > 0.5).astype(np.uint8)
    mo = cv2.moments(binm, binaryImage=True)
    if mo["m00"] < 200:
        return np.clip(base if p > 0.5 else f, 0, 255).astype(np.uint8)
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    rows = np.flatnonzero(binm.any(axis=1))
    y_top, y_bot = int(rows[0]), int(rows[-1])

    e = ease_in(p)
    # 1) la figura si stringe in orizzontale e si allunga un po' in verticale
    sx = max(1.0 - e, 0.015)
    sy = 1.0 + 0.18 * e
    M = np.float32([[sx, 0, cx * (1 - sx)], [0, sy, cy * (1 - sy)]])
    person = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pm = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

    # 2) glitch: bande orizzontali spostate a caso
    if p > 0.08:
        for _ in range(int(3 + 14 * p)):
            y0 = int(rng.integers(0, h))
            bh = int(rng.integers(2, max(3, int(h * 0.035))))
            shift = int(rng.normal(0, 8 + 60 * p))
            person[y0:y0 + bh] = np.roll(person[y0:y0 + bh], shift, axis=1)
            pm[y0:y0 + bh] = np.roll(pm[y0:y0 + bh], shift, axis=1)

    # 3) aberrazione cromatica
    off = int(2 + 10 * p)
    person[..., 0] = np.roll(person[..., 0], off, axis=1)
    person[..., 2] = np.roll(person[..., 2], -off, axis=1)

    # 4) la figura diventa luce
    glow = min(1.0, p * 1.3) * 0.75
    person = person * (1 - glow) + GLOW_BGR * glow

    # 5) sfarfallio (il classico "tremolio" prima di sparire)
    alpha = 1.0 - 0.85 * e
    if p > 0.06 and tick % 2 == 0:
        alpha *= 0.3
    a = (pm * alpha)[..., None]
    out = base + (person - base) * a

    # 6) alone luminoso attorno al contorno
    edge = cv2.dilate(pm, np.ones((9, 9), np.uint8)) - cv2.erode(pm, np.ones((9, 9), np.uint8))
    edge = cv2.GaussianBlur(edge, (0, 0), 6)[..., None] * (0.9 * np.sin(np.pi * p))
    out = out + GLOW_BGR * edge

    # 7) linee verticali di velocità + scia finale
    if p > 0.45:
        lines = np.zeros((h, w), np.float32)
        k = (p - 0.45) / 0.55
        top = max(0, y_top - 40)
        bot = min(h - 1, y_bot + 40)
        for _ in range(int(6 + 18 * k)):
            x = int(cx + rng.normal(0, 25 + 120 * (1 - k)))
            y1 = int(rng.integers(top, bot))
            y2 = int(min(bot, y1 + rng.integers(h // 10, h // 2)))
            cv2.line(lines, (x, y1), (x, y2), 1.0, int(rng.integers(1, 3)))
        streak_w = max(1, int(14 * (1 - k)) + 2)
        cv2.line(lines, (int(cx), top), (int(cx), bot), 1.0, streak_w)
        lines = cv2.GaussianBlur(lines, (0, 0), 2.5)
        out = out + GLOW_BGR * (lines * 0.9 * np.sin(np.pi * k))[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Macchina a stati
# --------------------------------------------------------------------------- #
class Teleport:
    VISIBLE, VANISHING, HIDDEN, APPEARING = "visibile", "sparizione", "nascosto", "ricomparsa"

    def __init__(self):
        self.state = self.VISIBLE
        self.t0 = 0.0
        self.hold_start = None
        self.latched = False      # il gesto va rilasciato prima di poter riscattare
        self.last_end = -10.0

    def trigger(self, now):
        if self.state == self.VISIBLE:
            self.state, self.t0 = self.VANISHING, now
        elif self.state == self.HIDDEN:
            self.state, self.t0 = self.APPEARING, now

    def feed_gesture(self, active, now):
        """Ritorna la percentuale di carica (0..1) per la barra di debug."""
        if not active:
            self.hold_start = None
            self.latched = False
            return 0.0
        if self.latched or self.state in (self.VANISHING, self.APPEARING) or now - self.last_end < COOLDOWN:
            return 0.0
        if self.hold_start is None:
            self.hold_start = now
        charge = (now - self.hold_start) / HOLD_TIME
        if charge >= 1.0:
            self.trigger(now)
            self.latched = True
            self.hold_start = None
        return min(charge, 1.0)

    def progress(self, now):
        if self.state == self.VANISHING:
            p = (now - self.t0) / VANISH_TIME
            if p >= 1:
                self.state, self.last_end = self.HIDDEN, now
                return 1.0
            return p
        if self.state == self.APPEARING:
            p = 1 - (now - self.t0) / APPEAR_TIME
            if p <= 0:
                self.state, self.last_end = self.VISIBLE, now
                return 0.0
            return p
        return 1.0 if self.state == self.HIDDEN else 0.0


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Effetto teletrasporto di Goku sulla webcam")
    ap.add_argument("--camera", default="0", help="indice webcam o percorso di un video (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--proc-width", type=int, default=960,
                    help="larghezza a cui elaborare i frame (più bassa = più fluido)")
    ap.add_argument("--no-mirror", action="store_true", help="non specchiare l'immagine")
    ap.add_argument("--virtual-cam", action="store_true",
                    help="invia l'output a una webcam virtuale (pyvirtualcam) per Zoom/Meet/OBS")
    ap.add_argument("--output", help="salva anche il risultato in un file video (.mp4)")
    ap.add_argument("--headless", action="store_true", help="nessuna finestra (per test)")
    return ap.parse_args()


def main():
    args = parse_args()
    src = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Impossibile aprire la sorgente video: {args.camera}")
    is_file = not isinstance(src, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, frame = cap.read()
    if not ok:
        sys.exit("Nessun frame dalla sorgente video.")
    scale = min(1.0, args.proc_width / frame.shape[1])
    resize = lambda img: img if scale == 1.0 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = resize(frame).shape[:2]

    tracker = Tracker()
    bgm = Background()
    tp = Teleport()
    rng = np.random.default_rng()

    vcam = None
    if args.virtual_cam:
        import pyvirtualcam
        vcam = pyvirtualcam.Camera(width=w, height=h, fps=int(fps))
        print(f"Webcam virtuale attiva: {vcam.device}")
    writer = None
    if args.output:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    show_debug, show_help = False, True
    win = "Teletrasporto - Shunkan Ido"
    t_start = time.monotonic()
    tick = 0
    print("Porta indice e medio alla fronte per teletrasportarti. 'b' = cattura sfondo, 'q' = esci.")

    while True:
        if tick > 0:
            ok, frame = cap.read()
            if not ok:
                break
        frame = resize(frame)
        if not args.no_mirror and not is_file:
            frame = cv2.flip(frame, 1)
        # con un file video il "tempo" segue i frame, con la webcam l'orologio reale
        now = tick / fps if is_file else time.monotonic() - t_start

        hands, faces, mask = tracker.process(frame, now * 1000)
        bgm.update(frame, mask)
        forehead = forehead_from_faces(faces, w, h)
        gesture, tips = detect_gesture(hands, forehead, w, h)
        charge = tp.feed_gesture(gesture, now)
        p = tp.progress(now)

        out = render(frame, mask, bgm.filled(), p, tick, rng) if p > 0 else frame.copy()

        if vcam is not None:
            vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        if writer is not None:
            writer.write(out)

        if not args.headless:
            view = out.copy()
            if show_debug:
                if forehead is not None:
                    cv2.circle(view, tuple(int(v) for v in forehead[0]), int(forehead[1] * 1.1), (0, 200, 255), 2)
                for lm in hands.hand_landmarks:
                    for l in lm:
                        cv2.circle(view, (int(l.x * w), int(l.y * h)), 3, (0, 255, 0), -1)
                for t in tips:
                    cv2.circle(view, tuple(int(v) for v in t), 8, (0, 0, 255), 2)
                cv2.rectangle(view, (20, h - 40), (20 + int(200 * charge), h - 25), (0, 255, 255), -1)
                cv2.rectangle(view, (20, h - 40), (220, h - 25), (255, 255, 255), 1)
                cv2.putText(view, f"stato: {tp.state}  gesto: {'SI' if gesture else 'no'}",
                            (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if show_help:
                cv2.putText(view, "2 dita sulla fronte = teletrasporto | b sfondo | d debug | h help | q esci",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                bgm.capture(frame)
                print("Sfondo catturato.")
            elif key == ord("d"):
                show_debug = not show_debug
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord(" "):
                tp.trigger(now)
        tick += 1

    cap.release()
    if writer is not None:
        writer.release()
    if vcam is not None:
        vcam.close()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
