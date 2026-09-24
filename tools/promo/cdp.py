"""A minimal Chrome DevTools Protocol client on the standard library.

Just enough to drive web/viewer.html headless: launch Chrome (or Edge) with a
throwaway profile, open one page, send commands and read their results. The
WebSocket framing (RFC 6455) is written out here so the promo tools need
nothing beyond Pillow and ffmpeg.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request

BROWSERS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def find_browser(explicit=None):
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"browser not found at {explicit}")
        return explicit
    for path in BROWSERS:
        if os.path.isfile(path):
            return path
    for name in ("chrome", "msedge", "chromium", "google-chrome"):
        hit = shutil.which(name)
        if hit:
            return hit
    raise SystemExit("Neither Chrome nor Edge was found. Install Chrome, or pass "
                     "--browser C:\\path\\to\\chrome.exe")


def _mask(data, key):
    n = len(data)
    stream = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "little") ^ int.from_bytes(stream, "little")).to_bytes(n, "little")


class _WebSocket:
    """Client side of RFC 6455, text frames only, which is all CDP uses."""

    def __init__(self, url):
        rest = url.split("://", 1)[1]
        hostport, path = rest.split("/", 1)
        host, port = hostport.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n").encode())
        self.buf = bytearray()
        while b"\r\n\r\n" not in self.buf:
            self._fill()
        end = self.buf.index(b"\r\n\r\n")
        status = bytes(self.buf[:end]).split(b"\r\n")[0]
        del self.buf[:end + 4]
        if b" 101" not in status:
            raise ConnectionError(f"DevTools refused the WebSocket: {status!r}")

    def _fill(self):
        chunk = self.sock.recv(1 << 20)
        if not chunk:
            raise ConnectionError("DevTools connection closed")
        self.buf.extend(chunk)

    def _take(self, n):
        while len(self.buf) < n:
            self._fill()
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def send(self, text):
        data = text.encode()
        n = len(data)
        head = bytearray([0x81])  # FIN, text
        if n < 126:
            head.append(0x80 | n)
        elif n < 1 << 16:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        key = os.urandom(4)  # clients must mask
        self.sock.sendall(bytes(head) + key + _mask(data, key))

    def recv(self):
        parts = []
        while True:
            b1, b2 = self._take(2)
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._take(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._take(8))[0]
            key = self._take(4) if b2 & 0x80 else None
            data = self._take(n)
            if key:
                data = _mask(data, key)
            opcode = b1 & 0x0F
            if opcode == 0x8:
                raise ConnectionError("DevTools closed the connection")
            if opcode == 0x9:  # ping
                continue
            parts.append(data)
            if b1 & 0x80:
                return b"".join(parts).decode()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class Browser:
    """One headless page at a fixed CSS viewport and device scale factor."""

    def __init__(self, width, height, scale=1.0, browser=None):
        exe = find_browser(browser)
        self.profile = tempfile.mkdtemp(prefix="promo-browser-")
        self.proc = subprocess.Popen(
            [exe, "--headless=new", "--remote-debugging-port=0",
             f"--user-data-dir={self.profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions", "--mute-audio",
             "--hide-scrollbars", f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port = self._port()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
            targets = json.load(r)
        page = next(t for t in targets if t.get("type") == "page")
        self.ws = _WebSocket(page["webSocketDebuggerUrl"])
        self._id = 0
        self.set_viewport(width, height, scale)

    def _port(self, timeout=20.0):
        path = os.path.join(self.profile, "DevToolsActivePort")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise SystemExit("the browser exited during start-up")
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    line = fh.readline().strip()
                if line:
                    return int(line)
            except OSError:
                pass
            time.sleep(0.05)
        raise SystemExit("the browser did not open its DevTools port")

    def call(self, method, **params):
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:  # anything else is an event; ignored
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def eval(self, expression):
        r = self.call("Runtime.evaluate", expression=expression,
                      returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            detail = r["exceptionDetails"]
            text = detail.get("exception", {}).get("description") or detail.get("text")
            raise RuntimeError(f"page error: {text}")
        return r.get("result", {}).get("value")

    def set_viewport(self, width, height, scale):
        """Takes effect for window.devicePixelRatio on the NEXT page load."""
        self.scale = scale
        self.call("Emulation.setDeviceMetricsOverride", width=int(width),
                  height=int(height), deviceScaleFactor=float(scale), mobile=False)

    def open(self, url, ready, timeout=30.0):
        """Navigate and wait until the JS expression `ready` is truthy."""
        self.call("Page.navigate", url=url)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.eval(ready):
                    return
            except RuntimeError:
                pass
            time.sleep(0.1)
        raise SystemExit(f"page not ready after {timeout:.0f}s: {url}")

    def screenshot(self, clip=None):
        """PNG bytes. `clip` is (x, y, w, h) in CSS pixels; the image comes back
        at the device scale factor."""
        params = {"format": "png"}
        if clip:
            x, y, w, h = clip
            params["clip"] = {"x": x, "y": y, "width": w, "height": h, "scale": 1}
        return base64.b64decode(self.call("Page.captureScreenshot", **params)["data"])

    def close(self):
        try:
            self.call("Browser.close")
        except Exception:
            pass
        self.ws.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
