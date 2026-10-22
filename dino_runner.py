"""
DINO RUNNER — the Chrome dinosaur game, controlled by ANY facial movement
=========================================================================
The Muse is great at telling "moving" from "still", but bad at telling WHICH
movement it was. So this game needs exactly one input: move = jump.
Jaw clench works best (strong signal, you won't do it by accident).

  1. Run `muselsl stream` in a terminal first.
  2. Run this script.
  3. Calibration:
       a. Stay STILL for 5 seconds (baseline noise)
       b. CLENCH your jaw 6 times when prompted (or blink — just keep it the same move)
     The script picks the 2 channels that see YOUR move best and sets thresholds.
  4. Play. Clench to jump over cactuses.
     Watch out for HIGH birds — stay still and run under them!

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

# ─── EEG CONFIG ──────────────────────────────────────────────────────────────
EEG_CHANNELS    = [(0, "TP9"), (1, "AF7"), (2, "AF8"), (3, "TP10")]
SRATE           = 256
ACT_WIN         = int(0.15 * SRATE)   # activity = peak-to-peak over the last 150 ms
JUMP_COOLDOWN   = 0.35                # seconds between jumps
REARM_FRAC      = 0.6                 # activity must fall below thr*this before next jump
REARM_TIMEOUT   = 2.0                 # ...or re-arm anyway after this long (a held clench = 1 jump)

# Calibration parameters
CALIB_BASELINE_SEC = 5.0
CALIB_NUM_MOVES    = 6
CALIB_PRE_CUE_SEC  = 1.2
CALIB_CAPTURE_SEC  = 1.2
CALIB_REST_SEC     = 0.6

# ─── GAME CONFIG ─────────────────────────────────────────────────────────────
WIDTH, HEIGHT   = 960, 400
FPS             = 60
GROUND_Y        = HEIGHT - 70
GRAVITY         = 0.8
JUMP_VEL        = -15.5
START_SPEED     = 5.5
MAX_SPEED       = 12.0
SPEED_UP        = 0.0012              # added to speed every frame
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
    "channels":     [],     # list of (idx, name, threshold)
    "live_act":     {},     # per-channel current activity (for HUD)
}
detect_lock = threading.Lock()
stop_eeg    = threading.Event()


# ─── ACTIVITY MEASURE (shared by calibration + live detection) ──────────────

def window_activity(sig):
    """Peak-to-peak over every ACT_WIN-sample sliding window of a 1-D signal."""
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < ACT_WIN:
        return np.zeros(0)
    return np.ptp(np.lib.stride_tricks.sliding_window_view(sig, ACT_WIN), axis=1)


# ─── LIVE DETECTION THREAD (post-calibration) ────────────────────────────────

def eeg_detect_thread(inlet):
    with detect_lock:
        chans = list(detect_config["channels"])
    bufs = {idx: np.zeros(ACT_WIN) for idx, _, _ in chans}
    ptr  = 0
    armed = True
    last_jump = 0.0

    print(f"[EEG] Live detection on: {[(n, int(t)) for _, n, t in chans]}")

    while not stop_eeg.is_set():
        sample, _ = inlet.pull_sample(timeout=0.5)
        if sample is None:
            continue
        for idx, _, _ in chans:
            bufs[idx][ptr % ACT_WIN] = sample[idx]
        ptr += 1
        if ptr < ACT_WIN:
            continue

        over, calm = False, True
        with detect_lock:
            for idx, name, thr in chans:
                act = float(np.ptp(bufs[idx]))
                detect_config["live_act"][name] = act
                if act > thr:
                    over = True
                if act > thr * REARM_FRAC:
                    calm = False

        now = time.time()
        # One jump per movement: after firing, wait until the signal settles again
        if not armed and (calm or now - last_jump > REARM_TIMEOUT):
            armed = True
        if over and armed and (now - last_jump) > JUMP_COOLDOWN:
            jump_event.set()
            last_jump = now
            armed = False


# ─── CALIBRATION HELPERS ────────────────────────────────────────────────────

def drain_chunk(inlet):
    """Pull all queued samples from the inlet without blocking. Returns np.array [N, 5+]."""
    samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=1024)
    if samples:
        return np.array(samples)
    return None


def compute_calibration(baseline_arr, move_epochs):
    """
    For each channel, compare the activity while still with the peak activity of each
    cued movement. Pick channels where the weakest move clearly beats the noise.
    Threshold = midway between them.

    Returns: (top_channels [(idx, name, thr)], full_results [(name, snr, thr)])
    """
    full = []
    for idx, name in EEG_CHANNELS:
        noise_act = window_activity(baseline_arr[:, idx])
        if len(noise_act) == 0:
            continue
        noise_floor = float(np.percentile(noise_act, 99))   # ignore a single stray blip

        move_peaks = []
        for ep in move_epochs:
            act = window_activity(ep[:, idx])
            if len(act):
                move_peaks.append(float(np.max(act)))
        if not move_peaks:
            continue
        weakest_move = float(np.percentile(move_peaks, 20))
        median_move  = float(np.median(move_peaks))

        snr = median_move / max(noise_floor, 1e-3)
        if weakest_move > noise_floor:
            thr = (noise_floor + weakest_move) / 2.0
        else:
            thr = noise_floor * 1.3       # moves barely beat noise — stay just above it
        full.append((idx, name, snr, thr))

    full_sorted = sorted(full, key=lambda x: -x[2])
    reliable = [r for r in full_sorted if r[2] > 2.0] or full_sorted[:2]

    top = reliable[:2]
    top_chans   = [(idx, name, thr) for idx, name, snr, thr in top]
    full_report = [(name, snr, thr) for _, name, snr, thr in full_sorted]
    return top_chans, full_report


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
        chans = list(detect_config["channels"])
        live  = dict(detect_config["live_act"])
    if not chans:
        surf.blit(fonts['tiny'].render("KEYBOARD MODE", True, INK_LIGHT), (12, 10))
        return
    surf.blit(fonts['tiny'].render("EEG ●", True, UI_GREEN), (12, 10))
    # Bars are scaled so the threshold sits at the middle tick
    bar_max = 40
    for i, (idx, name, thr) in enumerate(chans):
        val = live.get(name, 0)
        bar_h = int(min(1.0, val / (2 * thr)) * bar_max) if thr > 0 else 0
        x = 12 + i * 40
        col = UI_RED if val > thr else UI_BLUE
        pygame.draw.rect(surf, (230, 230, 230), (x, 28, 26, bar_max))
        pygame.draw.rect(surf, col, (x, 28 + bar_max - bar_h, 26, bar_h))
        pygame.draw.line(surf, UI_AMBER, (x - 2, 28 + bar_max // 2), (x + 28, 28 + bar_max // 2), 2)
        surf.blit(fonts['tiny'].render(name, True, INK), (x, 72))


# ─── CALIBRATION SCREENS ────────────────────────────────────────────────────

def draw_progress_bar(surf, x, y, w, h, frac, col):
    pygame.draw.rect(surf, (70, 70, 80), (x, y, w, h), border_radius=4)
    pygame.draw.rect(surf, col, (x, y, int(w * min(1.0, frac)), h), border_radius=4)


def calib_screen_intro(screen, fonts):
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "DINO RUNNER", 60, UI_WHITE)
    draw_centered(screen, fonts['med'], "Quick calibration (~20 s)", 130, UI_WHITE)
    draw_centered(screen, fonts['small'], f"Step 1: stay completely still for {int(CALIB_BASELINE_SEC)} s", 185, (150, 200, 255))
    draw_centered(screen, fonts['small'], f"Step 2: clench your jaw {CALIB_NUM_MOVES} times on the green cue", 215, (150, 200, 255))
    draw_centered(screen, fonts['small'], "(any move works — blink, eyebrows — but use the SAME one in the game)", 245, (140, 140, 150))
    draw_centered(screen, fonts['big'], "Press SPACE", 310, (110, 230, 140))


def calib_screen_baseline(screen, fonts, frac):
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "STAY STILL", 100, (255, 200, 100))
    draw_centered(screen, fonts['med'], "relax your face, don't blink if you can", 170, UI_WHITE)
    draw_progress_bar(screen, WIDTH // 2 - 200, 250, 400, 20, frac, (255, 200, 100))


def calib_screen_move(screen, fonts, move_num, phase, phase_frac):
    """phase: 'ready' (amber), 'cue' (green CLENCH!), 'capture' (blue), 'rest' (gray)"""
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['med'], f"Move  {move_num} / {CALIB_NUM_MOVES}", 30, UI_WHITE)

    cx, cy = WIDTH // 2, HEIGHT // 2 - 10
    if phase == 'ready':
        col, msg, radius = (255, 200, 100), "get ready", 70 + int(15 * phase_frac)
    elif phase == 'cue':
        col, msg, radius = (110, 230, 140), "CLENCH!", 110
    elif phase == 'capture':
        col, msg, radius = (100, 180, 255), "and relax", 90
    else:
        col, msg, radius = (90, 90, 110), "wait…", 70

    pygame.draw.circle(screen, col, (cx, cy), radius)
    pygame.draw.circle(screen, DARK_BG, (cx, cy), radius - 12)
    draw_centered(screen, fonts['big'], msg, cy - 24, col)


def calib_screen_results(screen, fonts, top_chans, all_chans):
    screen.fill(DARK_BG)
    draw_centered(screen, fonts['big'], "CALIBRATED", 20, (110, 230, 140))
    draw_centered(screen, fonts['small'], "How much louder your move is than staying still (SNR):", 85, (150, 200, 255))

    y = 120
    for name, snr, thr in all_chans:
        col = (110, 230, 140) if any(name == n for _, n, _ in top_chans) else (140, 140, 160)
        draw_centered(screen, fonts['med'], f"{name:<5}  SNR {snr:5.1f}x   thr {thr:5.0f} µV", y, col)
        y += 36

    best = all_chans[0][1] if all_chans else 0
    if best < 2.0:
        draw_centered(screen, fonts['small'], "Weak signal — press C to redo, or adjust the headband fit.", y + 10, UI_RED)
    else:
        draw_centered(screen, fonts['small'], "Green channels will make the dino jump.", y + 10, (255, 200, 100))
    draw_centered(screen, fonts['med'], "SPACE = play     C = redo", HEIGHT - 50, (110, 230, 140))


# ─── CALIBRATION RUNNER ─────────────────────────────────────────────────────

def run_calibration(screen, clock, fonts, inlet):
    """Returns (top_channels, all_channels_report)."""

    def handle_quit():
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN
                                             and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                pygame.quit()
                sys.exit()

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
            drain_chunk(inlet)
            pygame.display.flip()
            clock.tick(FPS)

    def timed_phase(secs, draw, keep=None):
        """Run a screen for `secs` seconds, optionally collecting samples into `keep`."""
        start = time.time()
        while time.time() - start < secs:
            handle_quit()
            chunk = drain_chunk(inlet)
            if keep is not None and chunk is not None:
                keep.append(chunk)
            draw((time.time() - start) / secs)
            pygame.display.flip()
            clock.tick(FPS)

    while True:
        wait_for_space(lambda: calib_screen_intro(screen, fonts))

        # ── Phase 1: baseline ──
        drain_chunk(inlet)
        baseline_samples = []
        timed_phase(CALIB_BASELINE_SEC,
                    lambda f: calib_screen_baseline(screen, fonts, f), baseline_samples)
        baseline_arr = np.vstack(baseline_samples) if baseline_samples else np.zeros((0, 5))

        # ── Phase 2: cued movements ──
        move_epochs = []
        for n in range(1, CALIB_NUM_MOVES + 1):
            timed_phase(CALIB_PRE_CUE_SEC, lambda f: calib_screen_move(screen, fonts, n, 'ready', f))
            drain_chunk(inlet)   # clear backlog right before cue
            epoch = []
            timed_phase(0.35, lambda f: calib_screen_move(screen, fonts, n, 'cue', f), epoch)
            timed_phase(CALIB_CAPTURE_SEC - 0.35,
                        lambda f: calib_screen_move(screen, fonts, n, 'capture', f), epoch)
            if epoch:
                move_epochs.append(np.vstack(epoch))
            timed_phase(CALIB_REST_SEC, lambda f: calib_screen_move(screen, fonts, n, 'rest', f))

        # ── Compute & show results ──
        top_chans, full_report = compute_calibration(baseline_arr, move_epochs)
        print(f"[CALIB] Top channels: {[(n, int(t)) for _, n, t in top_chans]}")
        print(f"[CALIB] Full SNR: {[(n, round(s, 2)) for n, s, _ in full_report]}")

        if top_chans and wait_for_space(
                lambda: calib_screen_results(screen, fonts, top_chans, full_report)):
            return top_chans, full_report


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
            # A blink/clench restarts too (after a short pause so the death-move doesn't)
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
            draw_centered(screen, fonts['med'], "Clench (or SPACE) to start", 120)
            draw_centered(screen, fonts['small'], "Jump cactuses and low birds • stay still under HIGH birds", 165, INK_LIGHT)
            draw_centered(screen, fonts['small'], hint, 195, INK_LIGHT)
        elif state == STATE_DEAD:
            draw_centered(screen, fonts['big'], "G A M E   O V E R", 110)
            draw_centered(screen, fonts['small'], "clench or R to restart   •   " + hint, 175, INK_LIGHT)

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

        top_chans, _ = run_calibration(screen, clock, fonts, inlet)
        with detect_lock:
            detect_config["channels"] = top_chans
            detect_config["live_act"] = {n: 0 for _, n, _ in top_chans}

        eeg_thread_handle = threading.Thread(target=eeg_detect_thread,
                                             args=(inlet,), daemon=True)
        eeg_thread_handle.start()

        # Run game until user presses C to recalibrate
        high = run_game_loop(screen, clock, fonts, can_recalibrate=True, high=high)
        jump_event.clear()


if __name__ == "__main__":
    main()
