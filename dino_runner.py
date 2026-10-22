"""
DINO RUNNER — the Chrome dinosaur game, controlled by BLINKING
==============================================================
Blink = jump. Everything else (jaw clenches, eyebrow raises, head movement)
is ignored, because every candidate is checked three ways before it counts:

  1. WHERE  — only a forehead sensor (AF7/AF8), right above the eyes, and only
              a spike in the same direction as your calibration blinks.
  2. SHAPE  — the spike must match the shape of YOUR blink (template matching,
              correlation >= 0.8). Eyebrow raises are slower and look different.
  3. MUSCLE — no burst of fast muscle noise on the ear sensors (TP9/TP10),
              which is where jaw clenches show up.

  1. Run `muselsl stream` in a terminal first.
  2. Run this script.
  3. Calibration:
       a. Relax for a few seconds (baseline noise)
       b. Blink 6 times, whenever you like — each one is detected live
     The script learns your blink's size, direction and shape.
  4. Play. Blink to jump over cactuses.
     Watch out for HIGH birds — don't blink, run under them!

No Muse? The game still starts in keyboard mode (SPACE to jump).

KEYS:
  SPACE  — jump (fallback; also advances through calibration screens)
  R      — restart after game over
  C      — recalibrate
  Q      — quit
"""

import sys
import time
import random
import threading

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, sosfilt, sosfilt_zi

# ─── EEG CONFIG ──────────────────────────────────────────────────────────────
EEG_CHANNELS    = [(0, "TP9"), (1, "AF7"), (2, "AF8"), (3, "TP10")]
FRONTAL         = [(1, "AF7"), (2, "AF8")]   # forehead, right above the eyes — where blinks are
JAW_MUSCLE      = [0, 3]                     # TP9, TP10 — where jaw clenches show up
SRATE           = 256
BLINK_BAND      = (1.0, 8.0)          # Hz — blinks live here; drift, muscle noise and mains hum don't
MUSCLE_BAND     = (20.0, 45.0)        # Hz — muscle twitches (kept below 50 Hz mains hum)
MUSCLE          = 4                   # filtered arrays are [blink band x4 | muscle band x4]
ACT_WIN         = int(0.15 * SRATE)   # calibration activity = peak-to-peak over 150 ms

# Blink check (see docstring)
SHAPE_PRE       = int(0.30 * SRATE)   # blink shape window: this long before the peak...
SHAPE_POST      = int(0.35 * SRATE)   # ...and this long after (so a jump comes ~0.4 s after a blink)
SHAPE_MIN_CORR  = 0.8                 # how closely a spike must match your blink's shape (1 = identical)
MIN_TRIGGER_X   = 5.0                 # never trigger on spikes smaller than this many times still-noise

# Calibration parameters
CALIB_BASELINE_SEC = 4.0
CALIB_NUM_MOVES    = 6
BLINK_MIN_RATIO    = 3.5     # a calibration blink counts once it's this many times above the still-noise
BLINK_TAIL_SEC     = 0.7     # keep recording this long after a blink starts, to catch all of it
BLINK_EPOCH_SEC    = 1.4     # how much signal to keep per calibration blink
CALIB_GOT_IT_SEC   = 0.6     # how long the "got it" flash stays on screen
CALIB_HINT_SEC     = 10.0    # show fit tips if no blink is seen for this long
THRESH_POS         = 0.2     # game trigger sits this far from still-noise (0) to your weakest blink (1)

# ─── GAME CONFIG ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 960, 400
FPS             = 60
GROUND_Y        = HEIGHT - 70
GRAVITY         = 0.8
JUMP_VEL        = -15.5
START_SPEED     = 4.8                 # a little slower than Chrome's: blink checks add ~0.4 s
MAX_SPEED       = 10.5
SPEED_UP        = 0.0010              # added to speed every frame
DINO_X          = 90
BIRDS_AFTER     = 250                 # score at which birds start appearing

# ─── COLORS ──────────────────────────────────────────────────────────────────
SKY         = (247, 247, 247)
INK         = (83,  83,  83)
INK_LIGHT   = (180, 180, 180)
CLOUD       = (225, 225, 225)
UI_WHITE    = (255, 255, 255)
UI_RED      = (220,  60,  60)
UI_GREEN    = (40, 170,  80)
UI_AMBER    = (230, 150,  20)
UI_BLUE     = (50, 120, 220)
DARK_BG     = (32,  33,  36)

# ─── SHARED EEG STATE ────────────────────────────────────────────────────────
jump_event = threading.Event()

# Filled in after calibration, read by HUD + live detection thread
detect_config = {
    "model":        None,   # blink model from compute_calibration()
    "live":         0.0,    # current blink-direction signal on the forehead sensor (for HUD)
    "ignored_at":   0.0,    # when a spike was last rejected as "not a blink" (for HUD)
    "ignored_why":  "",
}
detect_lock = threading.Lock()
stop_eeg    = threading.Event()


# ─── NOISE FILTERS + ACTIVITY MEASURE (shared by calibration + live detection) ─

class EEGFilter:
    """
    Live filters for the 4 EEG channels, fed chunk by chunk. Returns [N, 8]:
      columns 0-3: blink band (BLINK_BAND) — slow drift, muscle twitches and
                   50/60 Hz electrical hum removed, so small blinks stand out
      columns 4-7: muscle band (MUSCLE_BAND) — used to spot jaw clenches
    """
    def __init__(self):
        self.sos = [butter(2, BLINK_BAND, btype='band', fs=SRATE, output='sos'),
                    butter(2, MUSCLE_BAND, btype='band', fs=SRATE, output='sos')]
        self.zi = None

    def __call__(self, chunk):
        """chunk: [N, 4+] raw samples -> [N, 8] filtered."""
        x = np.asarray(chunk, dtype=np.float64)[:, :4]
        if self.zi is None:
            # Start "settled" at the first sample so the Muse's big DC offset doesn't ring
            self.zi = [sosfilt_zi(s)[:, :, None] * x[0][None, None, :] for s in self.sos]
        out = []
        for k, s in enumerate(self.sos):
            y, self.zi[k] = sosfilt(s, x, axis=0, zi=self.zi[k])
            out.append(y)
        return np.hstack(out)


def window_activity(sig):
    """Peak-to-peak over every ACT_WIN-sample sliding window of a 1-D signal."""
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < ACT_WIN:
        return np.zeros(0)
    return np.ptp(np.lib.stride_tricks.sliding_window_view(sig, ACT_WIN), axis=1)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


# ─── BLINK DETECTOR (post-calibration) ───────────────────────────────────────

class BlinkDetector:
    """
    Fed one filtered sample at a time. When the forehead signal spikes past the
    trigger in the blink direction, it waits for the spike to finish, then checks
    SHAPE and MUSCLE. Returns "blink", a reason string if rejected, or None.
    """
    def __init__(self, model):
        self.m = model
        self.win = SHAPE_PRE + SHAPE_POST
        self.blink  = np.zeros(self.win)                    # rolling, oldest first
        self.muscle = np.zeros((self.win, len(JAW_MUSCLE)))
        self.n = 0
        self.peak = None          # biggest value of the spike being tracked
        self.since_peak = 0
        self.armed = True         # wait for the signal to settle between spikes

    def push(self, row):
        m = self.m
        v = m["sign"] * row[m["idx"]]
        self.blink  = np.roll(self.blink, -1);     self.blink[-1] = v
        self.muscle = np.roll(self.muscle, -1, 0); self.muscle[-1] = [row[MUSCLE + c] for c in JAW_MUSCLE]
        self.n += 1

        if not self.armed:
            if v < 0.5 * m["amp_thr"]:
                self.armed = True
            return None
        if self.peak is None:
            if v > m["amp_thr"]:
                self.peak, self.since_peak = v, 0
            return None
        if v > self.peak:
            self.peak, self.since_peak = v, 0
            return None
        self.since_peak += 1
        if self.since_peak < SHAPE_POST:
            return None

        # The spike peaked SHAPE_POST samples ago — the window is centred on it. Check it.
        self.peak, self.armed = None, False
        if self.n < self.win:
            return None
        z = (self.blink - self.blink.mean()) / (self.blink.std() + 1e-9)
        corr = float(np.dot(z, m["template"]) / self.win)
        if corr < SHAPE_MIN_CORR:
            return f"wrong shape ({corr:.2f})"
        if max(rms(self.muscle[:, k]) for k in range(len(JAW_MUSCLE))) > m["muscle_limit"]:
            return "muscle noise"
        return "blink"


def eeg_detect_thread(inlet):
    with detect_lock:
        model = detect_config["model"]
    det  = BlinkDetector(model)
    filt = EEGFilter()

    print(f"[EEG] Live detection on {model['name']}: trigger {model['amp_thr']:.0f}, "
          f"shape >= {SHAPE_MIN_CORR}, muscle <= {model['muscle_limit']:.1f}")

    while not stop_eeg.is_set():
        raw, _ = inlet.pull_sample(timeout=0.5)
        if raw is None:
            continue
        row = filt([raw])[0]
        result = det.push(row)
        with detect_lock:
            detect_config["live"] = max(0.0, model["sign"] * row[model["idx"]])
            if result and result != "blink":
                detect_config["ignored_at"], detect_config["ignored_why"] = time.time(), result
        if result == "blink":
            jump_event.set()
        elif result:
            print(f"[EEG] ignored: {result}")


# ─── CALIBRATION HELPERS ────────────────────────────────────────────────────

def drain_chunk(inlet, filt=None):
    """Pull all queued samples from the inlet without blocking.
    Returns np.array [N, 5+], or filtered [N, 8] if an EEGFilter is given."""
    samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=1024)
    if samples:
        return filt(samples) if filt else np.array(samples)
    return None


def noise_floors(baseline_arr):
    """Per-channel blink-band activity level while sitting still: {idx: floor}."""
    floors = {}
    for idx, _ in EEG_CHANNELS:
        act = window_activity(baseline_arr[:, idx])
        if len(act):
            # 99th percentile ignores a stray blip; the 3x-median cap stops an
            # accidental blink during the "relax" step from inflating the floor
            floors[idx] = float(min(np.percentile(act, 99), 3 * np.median(act)))
    return floors


def blink_strength(epoch_arr, floors):
    """How far above still-noise the biggest bump in this epoch is (best channel), e.g. 5.0 = 5x."""
    best = 0.0
    for idx, floor in floors.items():
        act = window_activity(epoch_arr[:, idx])
        if len(act):
            best = max(best, float(np.max(act)) / max(floor, 1e-3))
    return best


def compute_calibration(baseline_arr, move_epochs):
    """
    Learn the user's blink from the relaxed baseline + calibration blinks (all filtered [N, 8]):
      - which forehead sensor sees it best, and which direction it spikes
      - trigger level: THRESH_POS of the way from still-noise to the weakest blink
      - shape template: the average blink, centred on its peak
      - muscle limit: how much jaw-muscle noise a real blink comes with

    Returns a model dict, or None if no usable blink was found.
    """
    model = None
    for idx, name in FRONTAL:
        peaks = [ep[:, idx][np.argmax(np.abs(ep[:, idx]))] for ep in move_epochs]
        sign  = 1.0 if np.median(peaks) > 0 else -1.0
        amps  = [abs(p) for p in peaks if np.sign(p) == sign]
        noise = float(np.percentile(np.abs(baseline_arr[:, idx]), 99))
        if not amps:
            continue
        snr = float(np.median(amps)) / max(noise, 1e-3)
        if model is None or snr > model["snr"]:
            weakest = float(np.percentile(amps, 20))
            amp_thr = max(noise + THRESH_POS * (weakest - noise), MIN_TRIGGER_X * noise)
            model = dict(idx=idx, name=name, sign=sign, noise=noise, snr=snr,
                         amp_thr=min(amp_thr, 0.8 * weakest))
    if model is None:
        return None

    idx, sign = model["idx"], model["sign"]
    shapes, blink_muscle = [], []
    for ep in move_epochs:
        s = sign * ep[:, idx]
        p = int(np.argmax(s))
        if p - SHAPE_PRE < 0 or p + SHAPE_POST > len(s):
            continue
        w = s[p - SHAPE_PRE:p + SHAPE_POST]
        shapes.append((w - w.mean()) / (w.std() + 1e-9))
        blink_muscle.append(max(rms(ep[p - SHAPE_PRE:p + SHAPE_POST, MUSCLE + c]) for c in JAW_MUSCLE))
    if not shapes:
        return None
    model["template"] = np.mean(shapes, axis=0)

    win = SHAPE_PRE + SHAPE_POST
    still_muscle = [max(rms(baseline_arr[i:i + win, MUSCLE + c]) for c in JAW_MUSCLE)
                    for i in range(0, len(baseline_arr) - win, SHAPE_POST)] or [0.0]
    model["muscle_limit"] = max(3.0 * float(np.median(still_muscle)), 1.5 * float(np.max(blink_muscle)))
    return model


# ─── GAME OBJECTS ────────────────────────────────────────────────────────────

class Dino:
    W, H = 44, 47

    def __init__(self):
        self.x = DINO_X
        self.y = GROUND_Y - self.H
        self.vel = 0.0
        self.on_ground = True
        self.step = 0

    def jump(self):
        if self.on_ground:
            self.vel = JUMP_VEL
            self.on_ground = False
            return True
        return False

    def update(self):
        if not self.on_ground:
            self.vel += GRAVITY
            self.y += self.vel
            if self.y >= GROUND_Y - self.H:
                self.y = GROUND_Y - self.H
                self.vel = 0.0
                self.on_ground = True
        self.step += 1

    def draw(self, surf, dead=False):
        x, y = int(self.x), int(self.y)
        # Tail, body, neck, head
        pygame.draw.polygon(surf, INK, [(x, y + 18), (x + 8, y + 22), (x + 8, y + 32)])
        pygame.draw.rect(surf, INK, (x + 6, y + 18, 24, 18))
        pygame.draw.rect(surf, INK, (x + 20, y + 4, 10, 18))
        pygame.draw.rect(surf, INK, (x + 20, y, 24, 15))
        pygame.draw.rect(surf, INK, (x + 30, y + 15, 10, 4))      # jaw
        pygame.draw.rect(surf, INK, (x + 30, y + 24, 6, 3))       # arm
        # Eye
        if dead:
            pygame.draw.line(surf, SKY, (x + 25, y + 3), (x + 30, y + 8), 2)
            pygame.draw.line(surf, SKY, (x + 30, y + 3), (x + 25, y + 8), 2)
        else:
            pygame.draw.rect(surf, SKY, (x + 26, y + 3, 4, 4))
        # Legs (alternate while running)
        if not self.on_ground or dead:
            pygame.draw.rect(surf, INK, (x + 10, y + 36, 5, 11))
            pygame.draw.rect(surf, INK, (x + 21, y + 36, 5, 11))
        elif (self.step // 6) % 2:
            pygame.draw.rect(surf, INK, (x + 10, y + 36, 5, 11))
            pygame.draw.rect(surf, INK, (x + 21, y + 36, 5, 5))
        else:
            pygame.draw.rect(surf, INK, (x + 10, y + 36, 5, 5))
            pygame.draw.rect(surf, INK, (x + 21, y + 36, 5, 11))

    def get_rect(self):
        # A bit smaller than the drawing — forgiving hitbox
        return pygame.Rect(self.x + 8, self.y + 4, self.W - 14, self.H - 6)


class Cactus:
    def __init__(self, x):
        self.x = x
        self.parts = []
        count = random.choice([1, 1, 1, 2, 2, 3])
        tall = random.random() < 0.4
        off = 0
        for _ in range(count):
            w = random.randint(14, 18) if not tall else random.randint(18, 24)
            h = random.randint(30, 40) if not tall else random.randint(44, 52)
            self.parts.append((off, w, h))
            off += w + 4
        self.width = off - 4

    def update(self, speed):  self.x -= speed
    def off_screen(self):     return self.x + self.width < 0

    def rects(self):
        return [pygame.Rect(self.x + o + 3, GROUND_Y - h + 3, w - 6, h - 3)
                for o, w, h in self.parts]

    def collides(self, rect):
        return any(r.colliderect(rect) for r in self.rects())

    def draw(self, surf):
        for o, w, h in self.parts:
            x = int(self.x + o)
            top = GROUND_Y - h
            pygame.draw.rect(surf, INK, (x + w // 2 - 4, top, 8, h), border_radius=3)
            arm_y = top + h // 3
            pygame.draw.rect(surf, INK, (x, arm_y, 4, h // 3), border_radius=2)
            pygame.draw.rect(surf, INK, (x, arm_y + h // 3 - 4, w // 2, 4))
            pygame.draw.rect(surf, INK, (x + w - 4, arm_y - 6, 4, h // 3), border_radius=2)
            pygame.draw.rect(surf, INK, (x + w // 2, arm_y + h // 3 - 10, w // 2, 4))


class Bird:
    """LOW birds must be jumped. HIGH birds fly over you — jumping into them kills you."""
    W, H = 42, 26

    def __init__(self, x):
        self.x = x
        self.high = random.random() < 0.5
        bottom = GROUND_Y - 62 if self.high else GROUND_Y - 12
        self.y = bottom - self.H
        self.width = self.W
        self.flap = 0

    def update(self, speed):
        self.x -= speed + 0.8
        self.flap += 1

    def off_screen(self):
        return self.x + self.W < 0

    def collides(self, rect):
        return pygame.Rect(self.x + 6, self.y + 6, self.W - 12, self.H - 10).colliderect(rect)

    def draw(self, surf):
        x, y = int(self.x), int(self.y)
        pygame.draw.rect(surf, INK, (x + 8, y + 10, 28, 8))            # body
        pygame.draw.polygon(surf, INK, [(x, y + 12), (x + 10, y + 8), (x + 10, y + 16)])  # beak
        pygame.draw.rect(surf, SKY, (x + 10, y + 11, 2, 2))            # eye
        if (self.flap // 10) % 2:
            pygame.draw.polygon(surf, INK, [(x + 16, y + 10), (x + 28, y + 10), (x + 22, y)])
        else:
            pygame.draw.polygon(surf, INK, [(x + 16, y + 18), (x + 28, y + 18), (x + 22, y + 26)])


class Cloud:
    def __init__(self, x=None):
        self.x = x if x is not None else WIDTH + random.randint(0, 200)
        self.y = random.randint(30, 140)

    def update(self, speed):
        self.x -= speed * 0.2
        if self.x < -80:
            self.__init__()

    def draw(self, surf):
        x, y = int(self.x), self.y
        pygame.draw.ellipse(surf, CLOUD, (x, y + 6, 70, 18))
        pygame.draw.ellipse(surf, CLOUD, (x + 16, y, 36, 22))


# ─── DRAW HELPERS ───────────────────────────────────────────────────────────

def draw_ground(surf, offset, specks):
    pygame.draw.line(surf, INK, (0, GROUND_Y - 2), (WIDTH, GROUND_Y - 2), 2)
    for sx, sy, sw in specks:
        x = int((sx - offset) % (WIDTH + 40)) - 20
        pygame.draw.line(surf, INK_LIGHT, (x, GROUND_Y + sy), (x + sw, GROUND_Y + sy), 2)


def draw_centered(surf, font, text, y, col=INK):
    s = font.render(text, True, col)
    surf.blit(s, (WIDTH // 2 - s.get_width() // 2, y))


def draw_hud_play(surf, fonts, score, high):
    txt = f"HI {high:05d}   {score:05d}" if high else f"{score:05d}"
    s = fonts['score'].render(txt, True, INK)
    surf.blit(s, (WIDTH - s.get_width() - 20, 16))

    with detect_lock:
        model = detect_config["model"]
        val   = detect_config["live"]
        ignored_at, why = detect_config["ignored_at"], detect_config["ignored_why"]
    if model is None:
        surf.blit(fonts['tiny'].render("KEYBOARD MODE", True, INK_LIGHT), (12, 10))
        return
    surf.blit(fonts['tiny'].render(f"EEG ●  {model['name']}", True, UI_GREEN), (12, 10))
    # Bar is scaled so the trigger sits at the middle tick
    bar_max, thr, x = 40, model["amp_thr"], 12
    bar_h = int(min(1.0, val / (2 * thr)) * bar_max)
    col = UI_RED if val > thr else UI_BLUE
    pygame.draw.rect(surf, (230, 230, 230), (x, 28, 26, bar_max))
    pygame.draw.rect(surf, col, (x, 28 + bar_max - bar_h, 26, bar_h))
    pygame.draw.line(surf, UI_AMBER, (x - 2, 28 + bar_max // 2), (x + 28, 28 + bar_max // 2), 2)
    # Show when a spike was ignored, so the player can see the filter at work
    if time.time() - ignored_at < 1.2:
        surf.blit(fonts['tiny'].render(f"ignored: {why}", True, UI_AMBER), (46, 46))


# ─── CALIBRATION SCREENS ────────────────────────────────────────────────────

def calib_screen_intro(screen, fonts):
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "DINO RUNNER", 50, UI_WHITE)
    draw_centered(screen, fonts['med'], "Let's learn what your blink looks like", 120, UI_WHITE)
    draw_centered(screen, fonts['small'], "1.  Relax for a few seconds, eyes open", 180, (150, 200, 255))
    draw_centered(screen, fonts['small'], f"2.  Blink {CALIB_NUM_MOVES} times, whenever you like — a normal, clear blink", 210, (150, 200, 255))
    draw_centered(screen, fonts['big'], "Press SPACE to start", 290, (110, 230, 140))


def calib_screen_baseline(screen, fonts, t):
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "Relax", 70, (255, 200, 100))
    draw_centered(screen, fonts['small'], "look at the dot, eyes open, try not to blink", 125, UI_WHITE)
    # Gently breathing dot — something calm to look at
    r = int(26 + 8 * np.sin(t * 2.5))
    pygame.draw.circle(screen, (255, 200, 100), (WIDTH // 2, HEIGHT // 2 + 40), r)


def draw_blink_meter(screen, fonts, level):
    """Live signal bar; a blink counts when it crosses the white line."""
    bw, bh = 400, 14
    bx, by = WIDTH // 2 - bw // 2, HEIGHT - 40
    scale = BLINK_MIN_RATIO * 2
    fill_col = (110, 230, 140) if level >= BLINK_MIN_RATIO else (100, 180, 255)
    pygame.draw.rect(screen, (70, 70, 80), (bx, by, bw, bh), border_radius=4)
    pygame.draw.rect(screen, fill_col, (bx, by, int(bw * min(1.0, level / scale)), bh), border_radius=4)
    pygame.draw.line(screen, UI_WHITE, (bx + bw // 2, by - 4), (bx + bw // 2, by + bh + 4), 2)
    screen.blit(fonts['tiny'].render("your signal", True, (140, 140, 150)), (bx, by - 18))


def calib_screen_blinks(screen, fonts, count, flash, level, show_hint):
    """
    count: blinks captured so far   flash: 0..1, how fresh the last "got it" is
    """
    screen.fill(DARK_BG)

    # Progress: one dot per blink, filled when captured
    gap = 46
    x0 = WIDTH // 2 - gap * (CALIB_NUM_MOVES - 1) // 2
    for i in range(CALIB_NUM_MOVES):
        pos = (x0 + i * gap, 50)
        if i < count:
            pygame.draw.circle(screen, (110, 230, 140), pos, 14)
        else:
            pygame.draw.circle(screen, (90, 90, 110), pos, 14, 3)

    cy = HEIGHT // 2 - 10
    if flash > 0:
        pygame.draw.circle(screen, (110, 230, 140), (WIDTH // 2, cy), int(105 + 25 * flash), 6)
        draw_centered(screen, fonts['big'], "got it!", cy - 24, (110, 230, 140))
    else:
        draw_centered(screen, fonts['big'], "Blink", cy - 40, UI_WHITE)
        left = CALIB_NUM_MOVES - count
        draw_centered(screen, fonts['small'], f"{left} more — take your time", cy + 14, (140, 140, 150))

    if show_hint:
        draw_centered(screen, fonts['small'], "Not picking it up? Press the forehead sensors flat on your skin,",
                      HEIGHT - 110, (255, 200, 100))
        draw_centered(screen, fonts['small'], "or blink a little more clearly.   C = start over",
                      HEIGHT - 86, (255, 200, 100))

    draw_blink_meter(screen, fonts, level)


def calib_screen_results(screen, fonts, model):
    screen.fill(DARK_BG)
    best = model["snr"]
    if best >= 5:
        verdict, col = "Great signal", (110, 230, 140)
    elif best >= 2:
        verdict, col = "Good enough", (255, 200, 100)
    else:
        verdict, col = "Weak signal — jumps may be unreliable", UI_RED

    draw_centered(screen, fonts['big'], "ALL SET!", 60, (110, 230, 140))
    draw_centered(screen, fonts['med'], verdict, 130, col)
    draw_centered(screen, fonts['small'],
                  f"your blink is {best:.1f}x stronger than when you're relaxed", 170, (150, 200, 255))
    draw_centered(screen, fonts['tiny'],
                  f"using forehead sensor {model['name']} — jaw clenches & eyebrow raises are ignored",
                  200, (140, 140, 150))
    if best < 2:
        draw_centered(screen, fonts['small'], "Adjust the headband fit and press C to redo.", 240, UI_RED)
    draw_centered(screen, fonts['med'], "SPACE = play     C = redo", HEIGHT - 70, (110, 230, 140))


# ─── CALIBRATION RUNNER ─────────────────────────────────────────────────────

def run_calibration(screen, clock, fonts, inlet):
    """Returns the blink model (see compute_calibration)."""
    filt = EEGFilter()     # one filter for the whole session, fed every sample in order

    def handle_keys():
        """Quit on Q/ESC. Returns True if C (start over) was pressed."""
        restart = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                restart = True
        return restart

    def wait_for_space(draw):
        """Show a screen until SPACE (returns True) or C (returns False)."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                                 and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    return False
            draw()
            drain_chunk(inlet, filt)
            pygame.display.flip()
            clock.tick(FPS)

    def collect_blinks(floors):
        """Wait for the user to blink CALIB_NUM_MOVES times. Returns epochs, or None on restart."""
        recent  = np.zeros((0, 8))
        keep_n  = int(2 * SRATE)                  # rolling 2 s of signal
        look_n  = int(0.3 * SRATE)                # "is there a blink right now?" window
        epochs  = []
        seen_at = None                            # when the current blink started
        armed   = True                            # wait for the signal to settle between blinks
        flash_until = 0.0
        last_progress = time.time()

        while len(epochs) < CALIB_NUM_MOVES or time.time() < flash_until:
            if handle_keys():
                return None
            chunk = drain_chunk(inlet, filt)
            if chunk is not None:
                recent = np.vstack([recent, chunk])[-keep_n:]
            level = blink_strength(recent[-look_n:], floors) if len(recent) >= look_n else 0.0
            now = time.time()

            if len(epochs) < CALIB_NUM_MOVES:
                if seen_at is None and armed and level >= BLINK_MIN_RATIO:
                    seen_at = now
                elif seen_at is not None and now - seen_at >= BLINK_TAIL_SEC:
                    epochs.append(recent[-int(BLINK_EPOCH_SEC * SRATE):].copy())
                    print(f"[CALIB] blink {len(epochs)}: {blink_strength(epochs[-1], floors):.1f}x")
                    seen_at, armed = None, False
                    flash_until = now + CALIB_GOT_IT_SEC
                    last_progress = now
                elif not armed and seen_at is None and level < BLINK_MIN_RATIO * 0.5:
                    armed = True

            flash = max(0.0, (flash_until - now) / CALIB_GOT_IT_SEC)
            show_hint = now - last_progress > CALIB_HINT_SEC
            calib_screen_blinks(screen, fonts, len(epochs), flash, level, show_hint)
            pygame.display.flip()
            clock.tick(FPS)
        return epochs

    while True:
        wait_for_space(lambda: calib_screen_intro(screen, fonts))

        # ── Phase 1: a few relaxed seconds, to measure the noise floor ──
        drain_chunk(inlet, filt)
        baseline_samples = []
        start = time.time()
        while time.time() - start < CALIB_BASELINE_SEC:
            handle_keys()
            chunk = drain_chunk(inlet, filt)
            if chunk is not None:
                baseline_samples.append(chunk)
            calib_screen_baseline(screen, fonts, time.time() - start)
            pygame.display.flip()
            clock.tick(FPS)
        baseline_arr = np.vstack(baseline_samples) if baseline_samples else np.zeros((0, 8))
        floors = noise_floors(baseline_arr)

        # ── Phase 2: blinks, whenever the user is ready ──
        move_epochs = collect_blinks(floors)
        if move_epochs is None:
            continue

        # ── Compute & show results ──
        model = compute_calibration(baseline_arr, move_epochs)
        if model is None:
            print("[CALIB] Couldn't learn a blink from the forehead sensors — redoing.")
            continue
        print(f"[CALIB] {model['name']}: blink {model['snr']:.1f}x noise, trigger {model['amp_thr']:.0f}, "
              f"muscle limit {model['muscle_limit']:.1f}")

        if wait_for_space(lambda: calib_screen_results(screen, fonts, model)):
            return model


# ─── MAIN GAME LOOP ─────────────────────────────────────────────────────────

def run_game_loop(screen, clock, fonts, can_recalibrate, high):
    """Plays until the user presses C (returns the high score) or quits."""
    specks = [(random.randint(0, WIDTH + 40), random.randint(6, 30), random.randint(2, 6))
              for _ in range(30)]
    clouds = [Cloud(random.randint(0, WIDTH)) for _ in range(3)]

    STATE_WAITING, STATE_PLAYING, STATE_DEAD = "waiting", "playing", "dead"

    def new_game():
        # dino, obstacles, distance, speed, frames-until-next-obstacle, state
        return Dino(), [], 0.0, START_SPEED, 60, STATE_WAITING

    dino, obstacles, distance, speed, next_gap, state = new_game()
    ground_off = 0.0
    dead_at = 0.0

    while True:
        clock.tick(FPS)
        jumped = jump_event.is_set()
        if jumped:
            jump_event.clear()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    jumped = True
                if event.key == pygame.K_r and state == STATE_DEAD:
                    dino, obstacles, distance, speed, next_gap, state = new_game()
                if event.key == pygame.K_c and can_recalibrate:
                    return high

        score = int(distance / 10)

        if state == STATE_WAITING:
            if jumped:
                state = STATE_PLAYING
                dino.jump()

        elif state == STATE_PLAYING:
            if jumped:
                dino.jump()
            dino.update()
            speed = min(MAX_SPEED, speed + SPEED_UP)
            distance += speed
            ground_off += speed
            for c in clouds:
                c.update(speed)

            # Spawn: gap scales with speed so there is always ~1 s to react
            next_gap -= 1
            if next_gap <= 0:
                if score >= BIRDS_AFTER and random.random() < 0.3:
                    obstacles.append(Bird(WIDTH + 20))
                else:
                    obstacles.append(Cactus(WIDTH + 20))
                next_gap = random.randint(70, 130)

            for o in obstacles:
                o.update(speed)
                if o.collides(dino.get_rect()):
                    state = STATE_DEAD
                    dead_at = time.time()
                    high = max(high, score)
            obstacles = [o for o in obstacles if not o.off_screen()]

        elif state == STATE_DEAD:
            # A blink restarts too (after a short pause so the death-blink doesn't)
            if jumped and time.time() - dead_at > 1.0:
                dino, obstacles, distance, speed, next_gap, state = new_game()
                state = STATE_PLAYING
                dino.jump()

        # Draw
        screen.fill(SKY)
        for c in clouds:
            c.draw(screen)
        draw_ground(screen, ground_off, specks)
        for o in obstacles:
            o.draw(screen)
        dino.draw(screen, dead=(state == STATE_DEAD))
        draw_hud_play(screen, fonts, score, high)

        hint = "C = recalibrate   Q = quit" if can_recalibrate else "SPACE = jump   Q = quit"
        if state == STATE_WAITING:
            draw_centered(screen, fonts['med'], "Blink (or SPACE) to start", 120)
            draw_centered(screen, fonts['small'], "Jump cactuses and low birds • don't blink under HIGH birds", 165, INK_LIGHT)
            draw_centered(screen, fonts['small'], hint, 195, INK_LIGHT)
        elif state == STATE_DEAD:
            draw_centered(screen, fonts['big'], "G A M E   O V E R", 110)
            draw_centered(screen, fonts['small'], "blink or R to restart   •   " + hint, 175, INK_LIGHT)

        pygame.display.flip()


# ─── ENTRY POINT ────────────────────────────────────────────────────────────

def main():
    pygame.init()
    pygame.display.set_caption("DINO RUNNER")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock  = pygame.time.Clock()

    fonts = {
        'big':   pygame.font.SysFont("consolas", 40, bold=True),
        'med':   pygame.font.SysFont("consolas", 26, bold=True),
        'small': pygame.font.SysFont("consolas", 18),
        'tiny':  pygame.font.SysFont("consolas", 13),
        'score': pygame.font.SysFont("consolas", 24, bold=True),
    }

    # ── Splash while we connect ──
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "DINO RUNNER", 140, UI_WHITE)
    draw_centered(screen, fonts['small'], "Looking for Muse stream…", 210, (150, 200, 255))
    pygame.display.flip()

    print("[EEG] Searching for stream…")
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if not streams:
        print("[EEG] No stream found — keyboard mode. Run `muselsl stream` to use the Muse.")
        run_game_loop(screen, clock, fonts, can_recalibrate=False, high=0)
        return
    inlet = StreamInlet(streams[0], max_buflen=4)
    print(f"[EEG] Connected: {inlet.info().name()}")

    eeg_thread_handle = None
    high = 0

    while True:
        # If detection thread is running from a previous round, stop it before recalibrating
        if eeg_thread_handle and eeg_thread_handle.is_alive():
            stop_eeg.set()
            eeg_thread_handle.join(timeout=2.0)
            stop_eeg.clear()

        model = run_calibration(screen, clock, fonts, inlet)
        with detect_lock:
            detect_config["model"] = model
            detect_config["live"]  = 0.0

        eeg_thread_handle = threading.Thread(target=eeg_detect_thread,
                                             args=(inlet,), daemon=True)
        eeg_thread_handle.start()

        # Run game until user presses C to recalibrate
        high = run_game_loop(screen, clock, fonts, can_recalibrate=True, high=high)
        jump_event.clear()


if __name__ == "__main__":
    main()
