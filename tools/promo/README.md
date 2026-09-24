# Promo-video tools

Renderers for raw footage of the engine, for short videos (YouTube Shorts, X,
Reddit). Output goes to `tools/promo/out/`, which git ignores. Nothing here is
imported by training.

| script | makes |
|---|---|
| `mosaic.py` | 64 matches at once in a grid, opening zoomed in on one |
| `export_viewer.py` | any replay, rendered by `web/viewer.html` itself, as an MP4 |
| `scenes.py` | the video's story beats (bugs, the reward loophole) as labelled clips |
| `ghost_trails.py` | the futures the lookahead search compared, drawn as trails |

## One-time setup

1. **Install ffmpeg** (it turns frames into an MP4). In any terminal:

   ```
   winget install --id Gyan.FFmpeg -e
   ```

   The scripts find a winget install even before you restart the terminal. If
   ffmpeg lives somewhere else, pass `--ffmpeg C:\path\to\ffmpeg.exe`.

2. **Use the training venv's Python.** The engine is built for Python 3.11 only,
   so plain `python` fails. Every command below starts with
   `python_ai/venv/Scripts/python.exe`, run from the repo folder.

3. `export_viewer.py` and `scenes.py` also need **Chrome or Edge** installed;
   they open the viewer in an invisible (headless) window.

Every script takes `--still` to write a PNG instead of a video: fast, and a good
way to check a shot before rendering it.

## mosaic.py: the hook shot

```
python_ai/venv/Scripts/python.exe tools/promo/mosaic.py
```

- The first run plays 160 matches (64 boards, 2 matches each so a finished
  board starts a new game instead of freezing, plus spares). About 2 minutes;
  they are cached, so later runs skip this step.
- Rendering takes about 1 minute. The result is `tools/promo/out/mosaic.mp4`:
  8 seconds, 1080x1920, 30 fps, H.264.
- The clip opens full-screen on the most active match, then pulls back to reveal
  all 64. When a match ends, its border flashes the winner's colour.
- A board where a unit shakes in place at a bridge mouth is swapped for a spare.
  That was a real engine defect (`perception/UPSTREAM_REQUESTS.md` item 31,
  fixed 2026-09-24); the check stays as a guard.
- `mosaic.boards.json`, written next to the video, lists which match, decks and
  ticks each board shows.

Each side is the scripted UtilityTeacher playing a real meta deck from
`python_ai/opponents/decks/meta_decks.json`. It is not the neural agent, so
caption this shot as the engine, not as the AI.

| option | does |
|---|---|
| `--seed 1` | a different set of matches (re-simulates) |
| `--focus 12` | open on board 12 instead of the busiest one |
| `--size landscape` | 1920x1080 (also `square`, or `WxH`) |
| `--seconds 10` | clip length |
| `--speed 4 --speed-end 30` | start at 4x game speed and ramp to 30x after the zoom: a "fast-forward" feel |
| `--no-zoom` | show the whole grid from the first frame |
| `--zoom-hold 1.5 --zoom-seconds 3` | time on the single board, and length of the pull-back |
| `--rung 3` | teacher difficulty 0-10 (default: 10, the strongest) |

## export_viewer.py: any replay as a video

```
python_ai/venv/Scripts/python.exe tools/promo/export_viewer.py replays/replay_ep115422.json --start 600 --end 900
```

Opens `web/viewer.html` headless and screenshots every frame, so the footage is
exactly what the viewer shows. Motion is smooth at any speed: the viewer
interpolates between the replay's ticks for export.

- `--layout board` (default): vertical 1080x1920, the arena alone.
- `--layout full`: the whole viewer UI at 1920x1080, including the agent's
  "brain" gauge, hands, towers and event log. Add `--sim-view` on a teacher
  replay (`make_replays.py --teacher-debug`) to show the Simulation View.
- `--crop 7,8,17,25`: zoom to a region, in board tiles (x0,y0,x1,y1). It is
  re-rendered at up to 8x, so a close-up stays sharp.
- `--speed 2`: game seconds per video second.
- `--select ID`: ring one entity in yellow (ids are in the replay JSON).
- `--label "Recreated in the engine"`: a small burned-in note.

Replays to try: `replays/*.json` (the agent against the built-in bot, with its
"brain" value per tick) and `python_ai/replays_e3/` after `make_replays.py`.

## scenes.py: the story beats

```
python_ai/venv/Scripts/python.exe tools/promo/scenes.py
```

Stages each scene in the engine, checks it shows what its caption will claim
(and stops with an error if not), and renders it. About 8 minutes for all six;
name scenes to render only those. Clips land in `tools/promo/out/scenes/`.

| scene | shows | label |
|---|---|---|
| `cannon_corner` | an enemy Hog runs from the bridge; at the last moment a Cannon in the middle would still stop it, the Cannon lands behind the King instead, and the Hog hits the tower | Recreated in the engine |
| `giant_walk_now` | a Giant from the bridge to the tower at today's speed (~8.6 s) | After the fix |
| `giant_walk_bug` | the same walk at the pre-fix speed: 3.5 s | Recreated: pre-fix speed |
| `giant_stuck` | a Fireball knocks a Giant off the bridge and it vibrates in place | Recorded before the fix |
| `mortar_r` | the King and a Mortar both drawn as "R", the letter the engine once used to find the King | Recreated in the engine |
| `fireball_one` | a 4-elixir Fireball that catches one troop | Illustration |

`cannon_corner` is timed for a music cut. The drop moment is measured, not
picked: the scene tries a Cannon in the middle of the arena at every tick of
the Hog's run and drops the real one on the last tick that would still have
saved the tower. That comes about 0.7 s after the Hog leaves the bridge, so the
clip plays in slow motion (0.3x) from the Hog's first stride to just after the
drop; set `SLOW_MO = 1.0` in `scenes.py` for real time.
`out/scenes/cannon_corner.beats.json` gives the clip time of the drop (for the
sound effect) and of the Hog's first hit on the tower.

`giant_stuck` is the one real recording: `recordings/giant_stuck_at_bridge_pre_fix.json`
was captured on 2026-09-24 before the engine fix. A fixed engine cannot
reproduce it, so do not delete that file.

Keep the labels in the edit. "Recreated" footage is honest only while it says so.

## ghost_trails.py: the lookahead

```
python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py
```

Plays one match with the network plus 1-ply search against the built-in bot at
1.5x elixir, the setting of the "0.625 -> 0.944 win rate" result. It picks the
decisions where search overruled the network and renders a clip for each, about
9 seconds:

1. 2 s of the real match;
2. the board freezes, and every candidate move's 4-second future grows as a
   coloured trail from where it would be placed, ending in the critic's score;
3. the best one lights up ("BEST"), the others fade, and the real match plays
   on with the move the search chose.

Clips land in `tools/promo/out/ghost_trails/`. The console lists each decision:
what the network wanted, what search chose and every option's score, so you can
pick the clearest one. `--seed 3` plays a different match; `--clips 5` renders
more decisions.

The search runs as it was measured, with at most 7 candidates.
`--wide-proposals` switches to today's widened sweep (~97 candidates), which is
not what the win rate was measured with and is unreadable on one board.
