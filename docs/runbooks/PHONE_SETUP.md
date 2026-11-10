# Playing on a physical Android phone (TODO item P0)

The bot watches the phone through **scrcpy**, which mirrors the phone's screen
into a window on the Windows PC, and taps it back through **adb**. Nothing is
installed on the phone and nothing needs root.

```
phone --USB or Wi-Fi--> adb + scrcpy --> "ClashPhone" window on the PC
                                           |  Windows.Graphics.Capture
                                           v
               perception (detector, readers) -> decision loop -> adb `input tap`
```

**Status, 2026-11-10:** the pipeline is built and unit-tested, including a
check that capture and taps share one geometry. It has **not yet run against
a real phone**. Steps 5-7 are where that first happens, and they are written
so that a wrong setup only watches and never taps.

**Play only where automation is allowed.** Supercell's terms of service forbid
bots in online play, and an account can be banned. Use Training Camp or
friendly battles, on an account you can afford to lose.

The PC side is Windows (live capture uses Windows.Graphics.Capture). Use the
perception venv for every Python command:
`perception/.venv/Scripts/python.exe` (Python 3.11; see `.claude/CLAUDE.md`).

---

## 1. Prepare the phone (once)

1. **Enable Developer options.** Settings > About phone, tap **Build number**
   seven times. On Samsung it is under Software information.
2. **Enable USB debugging.** Settings > System > Developer options > USB
   debugging.
3. Optional, for **Wi-Fi** later: in the same menu, enable **Wireless
   debugging** (Android 11+).
4. **Keep the screen awake** while plugged in (Developer options > Stay
   awake), or pass `--stay-awake` to scrcpy below.
5. In Clash Royale, use the **default** display settings. A changed display
   size or font scaling moves the HUD the calibration measures.

## 2. Install the PC tools (once)

1. **Android SDK Platform-Tools** (adb):
   https://developer.android.com/tools/releases/platform-tools. Unzip, e.g. to
   `C:\platform-tools`, and either add it to PATH or set
   `CLASH_ADB=C:\platform-tools\adb.exe`.
   - Do not use BlueStacks' `HD-Adb.exe` for the phone. Two adb versions fight
     over port 5037 and kill each other's connections. Close BlueStacks while
     using the phone.
2. **scrcpy**: https://github.com/Genymobile/scrcpy (releases page,
   `scrcpy-win64-*.zip`). It bundles its own adb; point it at the same one
   with `set ADB=C:\platform-tools\adb.exe` so only one adb server runs.

## 3. Connect

**USB (start here):**

```bash
adb devices -l
```

The first time, the phone asks **"Allow USB debugging?"**: tick *Always allow*
and accept. The device must show as `device`, not `unauthorized`.

**Wi-Fi (optional, after USB works).** Android 11+: Developer options >
Wireless debugging > *Pair device with pairing code*, then run these two:

```bash
adb pair 192.168.1.20:37000
```

```bash
adb connect 192.168.1.20:41234
```

The pairing port and the connect port are different numbers, both shown on
the phone. Wi-Fi adds latency and jitter; measure it in step 5 before
preferring it.

## 4. Mirror the screen with scrcpy

```bash
scrcpy --window-title ClashPhone --max-size 1280 --max-fps 30 --video-bit-rate 8M --no-audio --stay-awake
```

- `--window-title ClashPhone`: the name `mvp_loop --device` captures by
  default.
- `--max-size 1280`: mirrors a 1080x2400 phone at 576x1280. The loop crops
  the game area and resizes it to 549x976, the frame size every reader was
  measured at, so there is nothing to gain from a bigger window.
- Add `--serial <serial>` if several devices are attached.
- **Do not minimise the window.** Covering it with other windows is fine; a
  minimised window paints nothing and capture stops (`capture/window.py`).
- Do not resize it to a different aspect ratio. Black bars are handled; a
  stretched mirror is not.

## 5. Preflight and a device profile

With scrcpy running and the phone unlocked:

```bash
perception/.venv/Scripts/python.exe perception/tools/phone_check.py --save-profile myphone
```

This checks adb, the device and its authorization, reads the display size,
times adb round trips, and writes `perception/config/devices/myphone.json`.
The profile's `game_rect` is a **guess**: full width, 9:16, centred
vertically. It is marked `"calibrated": false`.

## 6. Calibrate (the step that makes taps land)

Open a **Training Camp** match on the phone, then:

```bash
perception/.venv/Scripts/python.exe perception/tools/phone_check.py --profile myphone --overlay phone_overlay.png
```

Open `phone_overlay.png`:
- the **cyan grid** must sit on the arena's own tiles, edge to edge;
- the four **yellow rings** must sit on the four hand cards;
- the **magenta box** is what the detector sees.

If they are off, edit `game_rect` in `perception/config/devices/myphone.json`.
It is `[x, y, width, height]` as fractions of the whole screen: raise `y` to
move everything down, raise `height` to stretch it. Re-run the overlay until
both line up. Check the EDGES, not the centre: a scale error is zero in the
middle and largest at the sides. Then set `"calibrated": true`.

Optional: measure tap latency at a harmless point. This taps the phone, so do
it on a screen where a tap does nothing:

```bash
perception/.venv/Scripts/python.exe perception/tools/phone_check.py --profile myphone --tap-at 360 60 --taps 5
```

## 7. Run: watch first, then act

Watch only. Nothing is tapped; read the summary at the end:

```bash
perception/.venv/Scripts/python.exe perception/live/mvp_loop.py --device myphone --policy teacher --seconds 240
```

Check:
- decision rate near 1.0 Hz;
- board age p95 under 2,000 ms;
- `mirror, last build fabricated:` showing few unmappable units;
- `tower HP:` maxima that are plausible for your towers;
- the hand read correctly (the `hand` column).

Only then act, in Training Camp:

```bash
perception/.venv/Scripts/python.exe perception/live/mvp_loop.py --device myphone --policy teacher --seconds 240 --act --confirm-log phone_run.jsonl
```

`--act` is refused while the profile is uncalibrated. `--ensure-match` is
emulator-only for now, so start the match by hand. The placement summary at
the end (and `phone_run.jsonl`) says whether each tap landed. On the emulator
the bar was 0 of 28 placements refused.

## 8. Teach it your hand (recommended)

The stock CRBAB icon reader is the weakest reader in the pipeline. The
deck-restricted reader (`live/deck_hand.py`) needs a template pool cut from
footage of YOUR deck on YOUR screen:

1. Record a Training Camp match through the phone pipeline:

   ```bash
   perception/.venv/Scripts/python.exe perception/tools/record_match.py --device myphone --seconds 240
   ```

2. Build the pool: cluster the hand crops into eight groups, look at the
   montage, then name the clusters in order:

   ```bash
   perception/.venv/Scripts/python.exe perception/tools/build_icon_templates.py --frames <recorded dir> --cluster
   ```

   ```bash
   perception/.venv/Scripts/python.exe perception/tools/build_icon_templates.py --frames <recorded dir> --label "Hog Rider,Musketeer,..."
   ```

3. Run with `--hand deck_pool --hand-pool <the new pool dir>`.

## Troubleshooting

| symptom | cause |
|---|---|
| `adb not found` | platform-tools not on PATH; set `CLASH_ADB` |
| device `unauthorized` | accept the prompt on the phone (unlock it first) |
| `no frame from 'ClashPhone'` | scrcpy not running, a different `--window-title`, or the window minimised |
| connections dropping | two adb servers (BlueStacks' and platform-tools'); close one, `adb kill-server` |
| taps land off by a row or a column | `game_rect` is off; redo step 6 and check the EDGES |
| `--act refused ... not calibrated` | by design; finish step 6 |
