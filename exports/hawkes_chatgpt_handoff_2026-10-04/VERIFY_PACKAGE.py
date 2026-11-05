from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    path = root / entry["path"]
    assert path.is_file(), entry["path"]
    payload = path.read_bytes()
    assert len(payload) == entry["bytes"], entry["path"]
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["path"]
print(f"Verified {len(manifest['files'])} bundled files.")
