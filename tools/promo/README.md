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
| `edit/edit_short.py` | the finished Short: every clip cut to your voiceover, with captions, zooms and sound |

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

To find the most telling decision rather than take the first ones, sweep:

```
python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --sweep 40
```

It plays 40 matches (about 12 s each) and ranks every decision as a showcase,
writing the ranking to `out/ghost_trails/sweep.json`. A decision ranks high
when:
- search overruled the network and chose `--want` (default Hog Rider);
- the options are different KINDS of play (a spell, a building, different
  troops, not one card at two cells);
- the board is busy, with enemies on your side;
- the winner wins clearly and the futures end far apart.

Render one of them with `--seed S --tick T`. `--one-per-card` draws one future
per card (its best cell), so every trail is a different play. Add `--think 2.5` to hold the
frozen "thinking" part longer (the futures growing, the scores, the pick), so
it sits under a long voice line without being slowed down in the edit.

The search runs as it was measured, with at most 7 candidates.
`--wide-proposals` switches to today's widened sweep (~97 candidates), which is
not what the win rate was measured with and is unreadable on one board.

### A staged board

A real match rarely offers a board that is both busy and a clear choice between
different KINDS of play. The measured search takes its options from the
network's top cards, and this network almost never proposes Cannon, The Log or
Fireball. `--board` stages one instead:

```
python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --board tools/promo/boards/triple_elixir.json --lead 4 --think 3
```

The board file ([`boards/triple_elixir.json`](boards/triple_elixir.json), hand-editable) sets:
- the clock;
- both decks;
- tower HP;
- our hand and elixir;
- the troops, where they stand `--lead` seconds before the decision (the engine plays the lead);
- the plays to compare. `"aim"` sends a spell where it does the most damage;
  `"net"` puts a card where the network itself would.

What stays real:
- The network reads the board and names its own pick, which is always one of
  the options, at its own cell.
- Every play is rolled forward in the engine and scored by the critic exactly
  as the search scores its candidates.
- Each score is shown against doing nothing.

Each future also shows what it does to the enemy:
- a spell's blast;
- the troops a building pulls off course;
- the troops it kills.

The clip carries "Staged in the engine" throughout. `--still` writes two PNGs:
every option scored, then the pick lit up. `--horizon 6` looks further ahead than
the measured search's 4 seconds; if you use it, say so in the voiceover.

A staged board is only worth showing if the lookahead's pick really is the
better move. Check it by playing every option out to the end from the frozen
board, not just by its 4-second score: the first board staged for this scene
scored the Hog Rider best and lost with it in 24 games of 24.

## edit/edit_short.py: the finished Short

Cuts the clips above into one vertical video timed to your voiceover:
- captions of 1–3 words that pop in, with highlighted keywords;
- slow push-ins and punch-in zooms, screen shakes, flashes;
- call-outs, and a counter that ticks up;
- a whoosh on every cut;
- your music, ducked under your voice.

Everything it does is in [`edit/short.json`](edit/short.json): change a word or a
timing there and run it again.

### One-time setup

The editor has its own venv, because Whisper must not go into the training venv:

```
py -3.11 -m venv tools/promo/edit/.venv
tools/promo/edit/.venv/Scripts/python.exe -m pip install -r tools/promo/edit/requirements.txt
```

The first voiceover it times downloads Whisper's `base.en` model (about 150 MB,
once). ffmpeg is the same one the other tools use.

### Making the video

1. **Render the footage** (training venv, about 15 minutes in all):

   ```
   python_ai/venv/Scripts/python.exe tools/promo/mosaic.py --seconds 12
   python_ai/venv/Scripts/python.exe tools/promo/scenes.py --no-label --no-select
   python_ai/venv/Scripts/python.exe tools/promo/ghost_trails.py --board tools/promo/boards/triple_elixir.json --lead 4 --think 3
   python_ai/venv/Scripts/python.exe tools/promo/export_viewer.py replays/replay_ep115422.json --start 600 --end 900 --speed 1.5
   ```

   `--no-label` leaves the "Recreated in the engine" notes off the scene clips,
   because the editor draws them on top, where a zoom cannot crop them.
   `--no-select` leaves off the viewer's yellow selection box, since the
   editor's rings mark the key unit instead. The ghost-trails command renders
   the staged lookahead board the edit uses; `short.json`'s lookahead shots
   name its file, and their `_found` note gives the scores.
2. **Add your files** to `tools/promo/edit/assets/`:
   - `voiceover.mp3` (or .wav, .m4a);
   - `music.mp3` (optional);
   - `Montserrat-Black.ttf` (optional; Arial Black stands in until it's there).
3. **Check the timing**. This prints when every word, shot and effect lands, in
   seconds:

   ```
   tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/edit_short.py --timings
   ```

4. **Draft** at half size, fast:

   ```
   tools/promo/edit/.venv/Scripts/python.exe tools/promo/edit/edit_short.py --draft
   ```

   Add `--from 20 --to 32` to render only a stretch, or use `--still 14.5` for
   one frame as a PNG.
5. **Render** the full-size video to `tools/promo/out/short.mp4`: the same
   command without `--draft`, about 2–3 minutes.

With no voiceover yet, the edit is timed at a normal speaking pace and rendered
silent, so you can check the cut before recording. `--voiceover take2.wav` and
`--music other.mp3` try a file without editing the config.

### Editing short.json

**Times.** Any time can be seconds or a moment in your voiceover:

| written | means |
|---|---|
| `"cannon_drop"` | the line with that id starts |
| `"cannon_drop.end"` | it ends |
| `"cannon_drop.behind"` | the word "behind" is said (`"hook.the#2"`: the second "the") |
| `"end"` | the video ends |
| any of those `+0.3` / `-0.2` | shifted by that many seconds |

Because shots and effects name words rather than seconds, a new take re-times
the whole edit by itself: Whisper finds the words again.

**`lines`**: what you say, in order.
- `*word*` is highlighted in the caption, and ` / ` forces a caption break.
- `"caption": false` speaks a line without a caption.
- `"pause_before": 2.5` guarantees at least 2.5 s of silence before the line.
  Read straight through when you record: where your take is shorter, the edit
  inserts the silence into the recording and moves everything after it. The
  music rises `swell_db` (4 dB) there, so it carries the moment. The template
  pauses before "My AI put its Cannon…" and before "Hog Rider."
- Re-word freely. If Whisper can't find a line in your recording, `--timings`
  says so.

**`shots`**: which clip plays when. Each shot runs until the next one starts.

| field | does |
|---|---|
| `at` | when the shot starts (the first is always at 0) |
| `clip` | a file in `tools/promo/out/` |
| `in` | seconds into the clip; `"continue"` picks up where the previous shot of the same clip left off (a jump cut to a new framing) |
| `speed` | 0.5 is half speed |
| `zoom` | `1.2`, or `[1.0, 1.1]` for a slow push across the shot |
| `focus` | the point to zoom toward, `[x, y]` as fractions of the frame; `[[x, y], [x, y]]` pans |
| `overscan` | how far past the clip's edge the camera may look (a fraction of the frame), to slide something low in the clip up from under YouTube's buttons. The strip it reveals is black. |
| `caption_y` | this shot's caption height, for a shot whose action sits where the captions would |
| `sync` | lands a moment of the clip on a moment of your voice: `{"beat": "cannon_drop", "to": "cannon_drop.behind"}`. Beats come from the clip's `.beats.json` (the Cannon drop; the lookahead's `freeze`, `best` and `resume`; the Fireball). `"clip": 3.2` names a clip time instead. It shifts `in`; `"fit": "stretch"` changes `speed` instead. |
| `label` | the corner note. It comes from the clip ("Recreated in the engine"); keep it. |
| `sfx` | the cut's sound (`null` for none) |

**`effects`**: moments on top of the shots. Each has `at`, and most take
`duration` or `until`.

| type | fields |
|---|---|
| `punch` | a quick zoom-in: `scale`, `focus`, `"hold": true` to stay zoomed in until `until` |
| `shake` | `strength` in pixels |
| `flash` | `color` |
| `freeze` | the picture holds while you keep talking |
| `callout` | big text: `text`, `color`, `size`, `y` |
| `counter` | a number counting up: `from`, `to`, `format` (`"{:.0f}%"`), `hold` |
| `mark` | a clean graphic on a target: `shape` `ring` (locks onto it as it fades in) or `arrow` (slides in, then bobs toward it), `pos` (`[x, y]` in the clip's frame; an arrow's tip), `from` (an arrow's tail), `size`, `color`. It stays on its target through zooms and shakes. |
| `sfx` | just a sound: `sound`, `sfx_db` |

Any effect also takes `sfx` (a sound name, or `null` for none) and `sfx_db`.

**Finding a position** for a mark or a focus: render `--still` at that moment
and read the target's place as a fraction of the frame's width and height.

**Sounds.** `whoosh`, `pop`, `impact`, `ding`, `wrong` (a buzzer) and `right`
(a chime) are built in. A file in
`assets/sfx/` with the same name (`whoosh.wav`) replaces one, and any other name
(`scratch.wav`, used as `"sound": "scratch"`) adds one.

**Levels.** Levels are measured, not guessed:
- your voice is set to −16 LUFS;
- the music sits `below_voice_db` under it, and ducks while you speak;
- the final mix is normalised to −14 LUFS, YouTube's playback level.

**Style.** Sizes are pixels on a 1080-wide frame, and `*_y` values are fractions
of the height. YouTube's buttons cover the bottom ~20% and the right edge, so
keep captions above 0.75.

**Warnings.** Every run ends by listing what needs attention:
- a clip that isn't rendered yet (a placeholder card stands in);
- a shot that runs past the end of its clip (the last frame holds);
- a sync the clip cannot reach;
- a line Whisper didn't hear.
