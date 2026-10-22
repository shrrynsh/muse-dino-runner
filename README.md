# Muse Dino Runner

The Chrome dinosaur game, controlled with a **Muse 2** EEG headband.
**Blink and the dino jumps.**

The Muse is good at telling *moving* from *still*, but not at telling *which* movement it was,
so the game needs only one input: **blink = jump**. High birds add a twist: you have to
hold your blink and run under them.

## Setup (Windows, cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Play

1. Turn the Muse on (hold the button until the lights scroll) and close the Muse phone app.
2. **Terminal 1** (leave it open):
   ```cmd
   .venv\Scripts\activate.bat
   muselsl stream
   ```
3. **Terminal 2**:
   ```cmd
   .venv\Scripts\activate.bat
   python dino_runner.py
   ```
4. Calibrate: relax for a few seconds, then blink 6 times whenever you like.
   Each blink is detected live (a dot fills in), so there's no rush.
   The game picks the 2 sensors that see your blinks best.

With no Muse connected, the game starts in keyboard mode.

| Key | Action |
|---|---|
| SPACE / ↑ | jump (fallback) |
| R | restart |
| C | recalibrate |
| Q | quit |

## Tips

- Blink in calibration the same way you'll blink in the game. A normal, clear blink is enough.
- Too many random jumps? Raise `THRESH_POS` in `dino_runner.py` (e.g. 0.5).
  Have to blink too hard? Lower it (e.g. 0.3).
- If the calibration screen shows SNR below 2×, adjust the fit (sensors touching skin,
  hair out of the way, slightly damp sensors) and press C to redo it.
- The bars at the top-left show live activity. Crossing the amber line makes the dino jump.
