"""
Download the Chinook database and verify its checksum.
Usage: python scripts/fetch_db.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
SHA256 = "bdf635be69850bd3be09c9a2dbeef7ddfb80036bd3ef3381383cd03b61e4a61a"
DEST = Path(__file__).resolve().parent.parent / "data" / "Chinook_Sqlite.sqlite"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> int:
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists() and sha256(DEST) == SHA256:
        print(f"ok: {DEST} already present and verified")
        return 0
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    actual = sha256(DEST)
    if actual != SHA256:
        DEST.unlink()
        print(f"checksum mismatch: expected {SHA256}, got {actual}", file=sys.stderr)
        return 1
    print(f"ok: {DEST} verified")
    return 0

if __name__ == "__main__":
    sys.exit(main())
