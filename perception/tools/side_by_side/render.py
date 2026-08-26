"""Inline the payload into the viewer template."""
import sys
from pathlib import Path

tpl = Path(sys.argv[1]).read_text(encoding="utf-8")
# The JSON sits inside <script type="application/json">, so the one sequence
# that can break out of it is a literal "</". Escaping the slash keeps the HTML
# parser inside the element; JSON.parse is unaffected, because \/ is a valid
# JSON escape for /. JSON structure contains no "<" of its own and the base64
# alphabet has none either, so this can only ever hit string content.
payload = Path(sys.argv[2]).read_text(encoding="utf-8").replace("</", "<" + chr(92) + "/")
out = Path(sys.argv[3])
out.write_text(tpl.replace("__PAYLOAD__", payload), encoding="utf-8")
print(f"{out}: {out.stat().st_size / 1e6:.1f} MB")
