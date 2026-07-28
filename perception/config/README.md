# Calibration profiles

One `profile_<device>_<width>x<height>.json` per capture configuration. A
resolution change invalidates **every** pixel constant in a profile, which is
why the frame size is part of the identity and is asserted on load
(`calib.homography.homography_from_profile`) rather than left as a field that
can quietly disagree with the frames being fed in.

Nothing here yet — profiles are produced from a recording, and none exists.
See `perception/README.md`, *What to record*.

## Shape

```json
{
  "name": "bluestacks_1080x1920",
  "frame_width": 1080,
  "frame_height": 1920,
  "homography": [9 floats, row-major 3x3, screen pixels -> board tiles],
  "anchors_screen": {
    "own_princess_left":  [x, y],
    "own_princess_right": [x, y],
    "opp_princess_left":  [x, y],
    "opp_princess_right": [x, y]
  },
  "rois": {
    "elixir_bar":  [x, y, w, h],
    "clock":       [x, y, w, h],
    "hand_slot_0": [x, y, w, h],
    "hand_slot_1": [x, y, w, h],
    "hand_slot_2": [x, y, w, h],
    "hand_slot_3": [x, y, w, h],
    "next_card":   [x, y, w, h],
    "own_king": [x, y, w, h], "own_princess_left": [x, y, w, h], "...": []
  },
  "reprojection_error_tiles": 0.31
}
```

## The two things most likely to go wrong

**Anchors must be the tower BASE, not its centre.** Towers are tall 3D
models; only the point where a tower meets the ground lies on the plane the
homography models. Anchoring on the visual centre produces an error that
grows with distance from the camera axis — and looks fine in the middle of
the board, which is what makes it hard to notice.

**`reprojection_error_tiles` is measured on held-out landmarks**, never on
the four anchors. A four-point solve has zero residual by construction no
matter how badly the points were picked, so the fit tells you nothing. The
bridges and King towers are excluded from the solve for exactly this reason —
see `geometry.calibration_anchors` / `validation_landmarks`. Stage 0's bar is
`< 0.5` tiles on that held-out set.

`templates/` (digit and card-icon crops) is gitignored: it is cropped from one
specific recording at one specific resolution, regenerable from that
recording, and meaningless without it.
