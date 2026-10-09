"""Fetch only audited public aggregates, EFF vocabulary, and a fixed public panel."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "linkedin.zip": ("https://ndownloader.figshare.com/files/14027465", "fc831d2e524913d4692c297a356018f15a75d2a56520fc66576ddff0004faba7"),
    "yahoo.tar.gz": ("https://ndownloader.figshare.com/files/3636168", "061137ea3cc129c7d9f501295cb194e0c6fa158acac702f893cba3cfd5f44efe"),
    "eff_large_wordlist.txt": ("https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt", None),
}
# These are the user's eight already public teaching examples, never user secrets.
PUBLIC_PANEL = ("password", "Password", "password1", "Password1", "password!", "Password!", "Password1!", "P@ssword1!")

def fetch(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent":"PasswordLengthAggregateResearch/1.0", **(headers or {})})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()

def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, (url, expected) in SOURCES.items():
        path = raw / filename
        payload = path.read_bytes() if path.exists() and path.stat().st_size else fetch(url + "?audit=" + str(int(time.time())))
        digest = hashlib.sha256(payload).hexdigest()
        if not payload or (expected and digest != expected):
            raise ValueError(f"source checksum validation failed: {filename}")
        path.write_bytes(payload)
        manifest.append({"filename":filename, "url":url, "sha256":digest, "bytes":len(payload)})
    for name, article in (("linkedin",7350287),("yahoo",2057937)):
        path = raw / (name + "_metadata.json")
        if not path.exists():
            path.write_bytes(fetch(f"https://api.figshare.com/v2/articles/{article}"))
        info = json.loads(path.read_text())
        if info["license"]["name"] != "CC0":
            raise ValueError("source terms changed; audit required")
        archive = raw / ("linkedin.zip" if name=="linkedin" else "yahoo.tar.gz")
        if hashlib.md5(archive.read_bytes()).hexdigest() != info["files"][0]["computed_md5"]:
            raise ValueError("metadata MD5 mismatch")
    (raw / "source_manifest.json").write_text(json.dumps({"retrieved_at_utc":datetime.now(timezone.utc).isoformat(), "files":manifest},indent=2)+"\n")
    panel = []
    cache = {}
    queried_at = datetime.now(timezone.utc).isoformat()
    for example in PUBLIC_PANEL:
        digest = hashlib.sha1(example.encode("utf-8")).hexdigest().upper()
        prefix, suffix = digest[:5], digest[5:]
        if prefix not in cache:
            payload = fetch("https://api.pwnedpasswords.com/range/"+prefix, {"Add-Padding":"true"}).decode("ascii")
            cache[prefix] = {}
            for line in payload.splitlines():
                hashed_suffix, count = line.split(":")
                if int(count)>0:
                    cache[prefix][hashed_suffix] = int(count)
        panel.append({"example":example, "length":len(example), "corpus_occurrences":cache[prefix].get(suffix,0)})
    # No API response containing other suffix hashes is saved.
    (raw / "hibp_public_panel.json").write_text(json.dumps({"queried_at_utc":queried_at, "method":"SHA1 first five hex characters over HTTPS; Add-Padding:true; local suffix matching", "scope":"eight public illustrative examples from request; no active credentials", "prefix_requests":len(cache), "rows":panel},indent=2)+"\n")
    print("Validated archives; saved eight-example HIBP count snapshot.")

if __name__ == "__main__": main()
