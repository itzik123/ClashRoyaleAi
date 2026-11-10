# Device profiles

One JSON per physical phone, written by
`perception/tools/phone_check.py --save-profile <name>` and read by
`mvp_loop.py --device <name>`. The fields are documented on
`live/device.DeviceProfile`. `game_rect` is `[x, y, width, height]`, as
fractions of the screen, of the region CRBAB's 720x1280 frame covers. Capture
crops to it and taps map through it. Calibrate it with `--overlay`
(`docs/runbooks/PHONE_SETUP.md`, step 6) before setting `"calibrated": true`.
