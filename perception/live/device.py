"""A physical Android phone as the live loop's target: adb, geometry, taps.

Everything in `live/` was built against the BlueStacks emulator, whose display
is exactly CRBAB's 720x1280. A phone is not: a modern one is 1080x2400 (20:9),
so its screen is taller than the 9:16 space every CRBAB pixel constant
(`DISPLAY_*`, `TILE_*`, `CARD_*`) is written in. Two things must agree, and
they agree only if both come from ONE definition:

  capture   the part of the phone screen the detector sees, resized to its
            368x652 (9:16) input frame
  taps      where a CRBAB 720x1280 coordinate lands on the phone

That definition is `DeviceProfile.game_rect`: the region of the phone screen,
as FRACTIONS of the screen, that corresponds to CRBAB's 720x1280 frame.
Capture crops to it (`crop_box`); taps map through it (`to_device`). A wrong
rect is wrong for both at once, and the overlay tool
(`tools/phone_check.py --overlay`) shows it on a real screenshot.

**The default rect is a guess and is marked as one.** For a 9:16 screen it is
the whole screen. For a taller screen it is full width, centred vertically, at
9:16. How Clash Royale actually lays out a tall phone (extended background,
moved HUD, a taller arena) has NOT been measured on any device here. Calibrate
it per phone before trusting a tap: `calibrated` in the profile says whether
anyone has.

Profiles live in `config/devices/<name>.json`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: CRBAB's coordinate space (`clashroyalebuildabot.constants.DISPLAY_*`).
#: Restated rather than imported so this module stays importable without the
#: vendored bot; `test_device.py` pins it against the real constants.
CRBAB_DISPLAY = (720, 1280)

DEVICES_DIR = Path(__file__).resolve().parent.parent / "config" / "devices"

# The BlueStacks adb the emulator path used. Last resort only: on a machine
# driving a phone, platform-tools' adb is the one to use, and two adb servers
# of different versions fight over port 5037.
_BLUESTACKS_ADB = Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe")


class DeviceError(RuntimeError):
    pass


# --- finding adb -----------------------------------------------------------

def find_adb(explicit: str | os.PathLike | None = None) -> Path:
    """The adb to use, in order: an explicit path, $CLASH_ADB, `adb` on PATH,
    $ANDROID_HOME / $ANDROID_SDK_ROOT platform-tools, BlueStacks' own.

    Raises with the install hint rather than returning a path that does not
    exist: a missing adb otherwise surfaces as a FileNotFoundError on the first
    tap, mid-match.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CLASH_ADB"):
        candidates.append(Path(os.environ["CLASH_ADB"]))
    on_path = shutil.which("adb")
    if on_path:
        candidates.append(Path(on_path))
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(var)
        if root:
            for name in ("adb.exe", "adb"):
                candidates.append(Path(root) / "platform-tools" / name)
    candidates.append(_BLUESTACKS_ADB)
    for path in candidates:
        if path.is_file():
            return path
    raise DeviceError(
        "adb not found. Install Android SDK Platform-Tools "
        "(https://developer.android.com/tools/releases/platform-tools), then "
        "put it on PATH, or set CLASH_ADB to the adb executable, or pass "
        "--adb. Tried: " + ", ".join(str(p) for p in candidates))


def _adb(adb: Path, serial: str | None, *args: str) -> list[str]:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    return cmd + list(args)


def run_adb(adb: Path, serial: str | None, *args: str,
            timeout: float = 15.0) -> str:
    done = subprocess.run(_adb(adb, serial, *args), capture_output=True,
                          text=True, timeout=timeout)
    if done.returncode != 0:
        raise DeviceError(f"adb {' '.join(args)} failed ({done.returncode}): "
                          f"{(done.stderr or done.stdout).strip()}")
    return done.stdout


# --- parsing adb's answers ---------------------------------------------------

@dataclass(frozen=True)
class AdbDevice:
    serial: str
    state: str
    """"device" when usable; "unauthorized" until the phone's USB-debugging
    prompt is accepted; "offline" while the connection is coming up."""
    model: str = ""


def parse_devices(output: str) -> list[AdbDevice]:
    """`adb devices -l` -> one entry per device line."""
    out = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("List of devices") or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        model = next((p.split(":", 1)[1] for p in parts[2:]
                      if p.startswith("model:")), "")
        out.append(AdbDevice(serial=parts[0], state=parts[1], model=model))
    return out


_SIZE = re.compile(r"(Physical|Override) size:\s*(\d+)x(\d+)")


def parse_wm_size(output: str) -> tuple[int, int]:
    """`adb shell wm size` -> (width, height) in PORTRAIT.

    An "Override size" (set by the user or a display-scaling setting) wins over
    "Physical size": it is what apps render at and what `input tap` addresses.
    Reported landscape on some devices; Clash Royale is portrait, so the
    smaller side is the width.
    """
    found = {kind: (int(w), int(h)) for kind, w, h in _SIZE.findall(output)}
    size = found.get("Override") or found.get("Physical")
    if size is None:
        raise DeviceError(f"could not read a display size from: {output!r}")
    w, h = size
    return (min(w, h), max(w, h))


# --- the profile ---------------------------------------------------------------

def default_game_rect(display: tuple[int, int]) -> tuple[float, float, float, float]:
    """(fx, fy, fw, fh) of the screen that CRBAB's 9:16 frame covers, as a
    GUESS: the whole screen at 9:16, else full width at 9:16 centred
    vertically (a wider-than-9:16 screen gets full height, centred
    horizontally)."""
    w, h = display
    target = CRBAB_DISPLAY[0] / CRBAB_DISPLAY[1]
    aspect = w / h
    if abs(aspect - target) < 1e-3:
        return (0.0, 0.0, 1.0, 1.0)
    if aspect < target:                       # taller than 9:16
        fh = (w / target) / h
        return (0.0, (1.0 - fh) / 2.0, 1.0, fh)
    fw = (h * target) / w                     # wider than 9:16
    return ((1.0 - fw) / 2.0, 0.0, fw, 1.0)


@dataclass
class DeviceProfile:
    """One phone, as the live loop needs to know it."""

    name: str
    display: tuple[int, int]
    """(width, height) in device pixels, portrait: what `input tap` takes."""

    serial: str | None = None
    """adb serial ("R5CT..." over USB, "192.168.1.20:5555" over Wi-Fi). None
    means "the only device attached"; adb refuses if there are several."""

    game_rect: tuple[float, float, float, float] | None = None
    """(fx, fy, fw, fh) of the SCREEN that CRBAB's 720x1280 frame covers.
    None means `default_game_rect(display)`, which is a guess."""

    calibrated: bool = False
    """True once someone has checked `game_rect` against a real screenshot
    (`tools/phone_check.py --overlay`) and the overlay sat on the arena."""

    touch_device: str | None = None
    """A `/dev/input/eventN` node for raw-evdev taps, which need ROOT on a
    phone. None (the default) uses `input tap`: no root, ~0.4 s per tap."""

    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.display = (int(self.display[0]), int(self.display[1]))
        if self.display[0] <= 0 or self.display[1] <= 0:
            raise DeviceError(f"display must be positive, got {self.display}")
        if self.game_rect is None:
            self.game_rect = default_game_rect(self.display)
        fx, fy, fw, fh = (float(v) for v in self.game_rect)
        if not (fw > 0 and fh > 0 and fx >= -1e-9 and fy >= -1e-9
                and fx + fw <= 1 + 1e-9 and fy + fh <= 1 + 1e-9):
            raise DeviceError(f"game_rect {self.game_rect} is not inside the "
                              "screen (fractions in [0, 1])")
        self.game_rect = (fx, fy, fw, fh)

    # -- the one geometry --------------------------------------------------
    def game_rect_px(self) -> tuple[float, float, float, float]:
        """`game_rect` in device pixels: (x, y, w, h)."""
        fx, fy, fw, fh = self.game_rect
        W, H = self.display
        return (fx * W, fy * H, fw * W, fh * H)

    def to_device(self, x: float, y: float) -> tuple[int, int]:
        """A CRBAB 720x1280 coordinate -> the device pixel `input tap` takes."""
        gx, gy, gw, gh = self.game_rect_px()
        dx = gx + x * gw / CRBAB_DISPLAY[0]
        dy = gy + y * gh / CRBAB_DISPLAY[1]
        W, H = self.display
        return (int(round(min(max(dx, 0.0), W - 1))),
                int(round(min(max(dy, 0.0), H - 1))))

    def crop_box(self, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
        """`game_rect` as (x, y, w, h) inside a captured image of the WHOLE
        screen at any scale (scrcpy mirrors at --max-size, not native)."""
        fx, fy, fw, fh = self.game_rect
        x = int(round(fx * screen_w))
        y = int(round(fy * screen_h))
        w = max(1, int(round(fw * screen_w)))
        h = max(1, int(round(fh * screen_h)))
        return (x, y, min(w, screen_w - x), min(h, screen_h - y))

    @property
    def screen_aspect(self) -> tuple[int, int]:
        return self.display

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else DEVICES_DIR / f"{self.name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["display"] = list(self.display)
        data["game_rect"] = list(self.game_rect)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, name_or_path: str | os.PathLike) -> "DeviceProfile":
        path = Path(name_or_path)
        if not path.suffix:
            path = DEVICES_DIR / f"{name_or_path}.json"
        if not path.is_file():
            raise DeviceError(
                f"no device profile at {path}. Create one with "
                "`python perception/tools/phone_check.py --save-profile "
                "<name>` while the phone is connected.")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["display"] = tuple(data["display"])
        if data.get("game_rect") is not None:
            data["game_rect"] = tuple(data["game_rect"])
        return cls(**data)

    @classmethod
    def probe(cls, name: str, adb: Path, serial: str | None = None) -> "DeviceProfile":
        """Read a connected phone's display size and build a default (guessed,
        uncalibrated) profile for it."""
        size = parse_wm_size(run_adb(adb, serial, "shell", "wm", "size"))
        return cls(name=name, display=size, serial=serial)


def require_device(adb: Path, serial: str | None) -> AdbDevice:
    """The one usable device, or a DeviceError naming what to fix."""
    devices = parse_devices(run_adb(adb, None, "devices", "-l"))
    if serial:
        devices = [d for d in devices if d.serial == serial]
        if not devices:
            raise DeviceError(f"device {serial!r} is not attached. Check the "
                              "cable or `adb connect <ip>:<port>`.")
    if not devices:
        raise DeviceError("no Android device attached. Enable USB debugging "
                          "(Settings > Developer options), connect the phone, "
                          "and accept the prompt on its screen.")
    if len(devices) > 1:
        raise DeviceError("several devices attached ("
                          + ", ".join(d.serial for d in devices)
                          + "); pass --serial.")
    dev = devices[0]
    if dev.state == "unauthorized":
        raise DeviceError(f"{dev.serial} is unauthorized: unlock the phone "
                          "and accept the 'Allow USB debugging?' prompt.")
    if dev.state != "device":
        raise DeviceError(f"{dev.serial} is {dev.state!r}, not ready.")
    return dev
