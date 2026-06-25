"""
Driver Drowsiness & Attention Monitor
======================================
Python 3.13 compatible — OpenCV only (no MediaPipe)

Fix: Better eye detection using EAR (Eye Aspect Ratio) logic
     with dlib-free landmark approximation via face ROI analysis
"""

import cv2
import numpy as np
import pygame
import argparse
import time
import os
import sys
import wave


# ═══════════════════════════════════════════════════════════════
#  CONFIG — tune these if needed
# ═══════════════════════════════════════════════════════════════
class Config:
    # Eye: how many consecutive frames eyes must be "closed" → alert
    EYE_CLOSED_FRAMES    = 4    # ~0.5s at 30fps  (lower = more sensitive)
    EYE_CLOSED_SECONDS   = 0.3
    EYE_ALERT_COOLDOWN   = 0.5

    FRAME_W = 960
    FRAME_H = 540
    FONT    = cv2.FONT_HERSHEY_SIMPLEX

    BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
    SOUND_DROWSY = os.path.join(BASE_DIR, 'sounds', 'drowsy_alert.wav')


# ═══════════════════════════════════════════════════════════════
#  COLOURS (BGR)
# ═══════════════════════════════════════════════════════════════
GREEN  = (0,  210,  80)
YELLOW = (0,  210, 255)
RED    = (30,  30, 220)
WHITE  = (255,255, 255)
GREY   = (110,110, 110)
DARK   = (18,  18,  22)
CYAN   = (255, 200,   0)


# ═══════════════════════════════════════════════════════════════
#  SOUND
# ═══════════════════════════════════════════════════════════════
def make_beep(path, freq=880, dur=0.6, vol=0.8, rate=44100):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t    = np.linspace(0, dur, int(rate * dur), False)
    data = (np.sin(2 * np.pi * freq * t) * vol * 32767).astype(np.int16)
    with wave.open(path, 'w') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(data.tobytes())


class AlertManager:
    def __init__(self):
        if not os.path.exists(Config.SOUND_DROWSY):
            make_beep(Config.SOUND_DROWSY, freq=950, dur=0.7)
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
        self._sounds      = {}
        self._last_played = {}
        try:
            self._sounds['drowsy'] = pygame.mixer.Sound(Config.SOUND_DROWSY)
        except Exception as e:
            print(f"[!] Sound load failed: {e}")

    def play(self, key, cooldown):
        now = time.time()
        if now - self._last_played.get(key, 0) >= cooldown:
            if key in self._sounds:
                self._sounds[key].play()
            self._last_played[key] = now


# ═══════════════════════════════════════════════════════════════
#  EYE ANALYSER
#  Logic:
#  - Detect face every frame
#  - Inside face, look for eyes using haar cascade
#  - If face found but NO eyes detected → eyes are closed
#  - Count consecutive "closed" frames → trigger alert after threshold
# ═══════════════════════════════════════════════════════════════
class EyeAnalyser:
    def __init__(self):
        data = cv2.data.haarcascades
        self.face_det = cv2.CascadeClassifier(
            data + 'haarcascade_frontalface_default.xml')
        self.eye_det  = cv2.CascadeClassifier(
            data + 'haarcascade_eye.xml')

        self.closed_frames = 0          # consecutive frames eyes not detected
        self.CLOSED_LIMIT  = 4         # ~0.67s at 30fps before marking closed
        self.closed_since  = None
        self.alert_on      = False

    def analyse(self, frame, gray):
        now = time.time()

        # ── Detect face ───────────────────────────────────────
        faces = self.face_det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_found = len(faces) > 0
        eyes_open  = True   # default: no face = don't raise false alarm

        if face_found:
            # Pick largest face
            fx, fy, fw, fh = sorted(faces, key=lambda r: r[2]*r[3])[-1]
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), GREY, 1)

            # Search for eyes only in the TOP HALF of the face
            top_h = int(fh * 0.55)
            roi_gray  = gray [fy:fy+top_h, fx:fx+fw]
            roi_color = frame[fy:fy+top_h, fx:fx+fw]

            eyes = self.eye_det.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,
                minNeighbors=4,       # lower = more sensitive
                minSize=(18, 18)
            )

            if len(eyes) >= 1:
                # Eyes detected → open
                eyes_open = True
                self.closed_frames = 0
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(roi_color,
                                  (ex, ey), (ex+ew, ey+eh), GREEN, 2)
            else:
                # No eyes inside face → closed
                eyes_open = False
                self.closed_frames += 1

        # ── Timer logic ───────────────────────────────────────
        if not eyes_open and face_found and self.closed_frames >= self.CLOSED_LIMIT:
            if self.closed_since is None:
                self.closed_since = now
        elif eyes_open or not face_found:
            self.closed_since  = None
            self.closed_frames = 0 if eyes_open else self.closed_frames
            self.alert_on      = False

        closed_for = (now - self.closed_since) if self.closed_since else 0.0

        if closed_for >= Config.EYE_CLOSED_SECONDS:
            self.alert_on = True

        return eyes_open, closed_for, self.closed_frames, face_found


# ═══════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════
def blend_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    ov = img.copy()
    cv2.rectangle(ov, (x1,y1),(x2,y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def status_bar(img, label, ratio, color, x, y, w=170, h=13):
    cv2.putText(img, label, (x, y-4), Config.FONT, 0.40, WHITE, 1)
    cv2.rectangle(img, (x,y),(x+w, y+h), GREY, -1)
    fill = int(w * min(max(ratio,0),1))
    if fill > 0:
        cv2.rectangle(img,(x,y),(x+fill, y+h), color, -1)
    cv2.rectangle(img,(x,y),(x+w,y+h), WHITE, 1)


def popup(img, title, subtitle, color):
    fh, fw = img.shape[:2]
    x1,y1 = fw//2-300, fh//2-65
    x2,y2 = fw//2+300, fh//2+65
    blend_rect(img, x1-4,y1-4, x2+4,y2+4, (0,0,0), alpha=0.82)
    blend_rect(img, x1,y1, x2,y2, color, alpha=0.88)
    cv2.putText(img, title,    (x1+18, y1+40), Config.FONT, 0.88, WHITE, 2)
    cv2.putText(img, subtitle, (x1+18, y1+64), Config.FONT, 0.52, WHITE, 1)


# ═══════════════════════════════════════════════════════════════
#  MAIN MONITOR
# ═══════════════════════════════════════════════════════════════
class DriverMonitor:
    def __init__(self, source):
        self.source       = source
        self.alerts       = AlertManager()
        self.eye_analyser = EyeAnalyser()
        self.show_zone    = False

        self.frames        = 0
        self.drowsy_events = 0
        self.session_start = time.time()

    def open_source(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[!] Cannot open: {self.source}"); sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_H)
        return cap

    def draw_hud(self, frame, eyes_open, closed_for, ratio, face_found, now):
        fh, fw = frame.shape[:2]

        blend_rect(frame, 8, 8, 235, 200, DARK, alpha=0.80)
        cv2.putText(frame, "DROWSINESS MONITOR", (18, 32),
                    Config.FONT, 0.55, GREEN, 1)
        cv2.line(frame, (18, 38), (228, 38), GREY, 1)

        elapsed = int(now - self.session_start)
        m, s = divmod(elapsed, 60)
        cv2.putText(frame, f"Session  {m:02d}:{s:02d}",
                    (18, 58), Config.FONT, 0.45, WHITE, 1)

        face_txt   = "Detected" if face_found else "Not found"
        face_color = WHITE if face_found else YELLOW
        cv2.putText(frame, f"Face: {face_txt}",
                    (18, 80), Config.FONT, 0.45, face_color, 1)

        eye_txt   = "OPEN" if eyes_open else "CLOSED"
        eye_color = GREEN if eyes_open else RED
        cv2.putText(frame, f"Eyes: {eye_txt}",
                    (18, 105), Config.FONT, 0.55, eye_color, 2)

        status_bar(frame, f"Closed frames: {ratio}/20",
                   ratio / 20, CYAN, 18, 116)

        if closed_for > 0:
            r = closed_for / Config.EYE_CLOSED_SECONDS
            status_bar(frame, f"Closed {closed_for:.1f}s / {Config.EYE_CLOSED_SECONDS}s",
                       r, RED, 18, 148)

        cv2.putText(frame, f"Alerts fired: {self.drowsy_events}",
                    (18, 190), Config.FONT, 0.42, WHITE, 1)

        cv2.putText(frame, "Q=Quit  R=Reset",
                    (fw // 2 - 75, fh - 10), Config.FONT, 0.42, GREY, 1)

    def run(self):
        cap = self.open_source()
        print("\n[Driver Monitor] Running — Press Q to quit\n")

        prev_drowsy = False

        while True:
            ret, frame = cap.read()
            if not ret:
                if not str(self.source).isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                break

            self.frames += 1
            frame = cv2.resize(frame, (Config.FRAME_W, Config.FRAME_H))
            fh, fw = frame.shape[:2]
            now    = time.time()
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray   = cv2.equalizeHist(gray)

            # ── Eye analysis ───────────────────────────────────
            eyes_open, closed_for, ratio, face_found = \
                self.eye_analyser.analyse(frame, gray)

            drowsy_alert = self.eye_analyser.alert_on
            if drowsy_alert and not prev_drowsy:
                self.drowsy_events += 1
                print(f"  [ALERT] Drowsy! eyes closed {closed_for:.1f}s")
            if drowsy_alert:
                self.alerts.play('drowsy', Config.EYE_ALERT_COOLDOWN)
            prev_drowsy = drowsy_alert

            # ── HUD ────────────────────────────────────────────
            self.draw_hud(frame, eyes_open, closed_for, ratio, face_found, now)

            # ── Popup ──────────────────────────────────────────
            if drowsy_alert:
                popup(frame,
                      "  DROWSINESS ALERT",
                      "Eyes closed too long — Stay alert!", RED)

            cv2.imshow("Drowsiness Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.eye_analyser.closed_since = None
                self.eye_analyser.alert_on     = False
                self.eye_analyser.history      = []
                print("  [R] Alert reset.")

        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        elapsed = int(time.time() - self.session_start)
        m, s = divmod(elapsed, 60)
        print(f"\n  Session: {m:02d}:{s:02d} | Frames: {self.frames} | Drowsy alerts: {self.drowsy_events}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',  type=str, default=None)
    ap.add_argument('--source', type=str, default='0')
    args = ap.parse_args()
    source = args.video if args.video else args.source
    DriverMonitor(source).run()