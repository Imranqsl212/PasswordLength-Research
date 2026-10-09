"""Verify every delivered ZIP member against its embedded file manifest.

The ZIP's own digest is written adjacent to it to avoid a circular hash claim.
No archive is extracted, uploaded or treated as a public repository deposit.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(path):
    path = Path(path).resolve()
    manifest_member = ROOT.name + "/results/statistics/artifact_manifest.json"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        manifest_bytes = archive.read(manifest_member)
        manifest = json.loads(manifest_bytes)
        expected = {ROOT.name + "/" + name for name in manifest["files"]} | {manifest_member}
        if set(names) != expected:
            raise ValueError("ZIP inventory differs from its manifest")
        for name, expected_hash in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Nonrelative manifest member")
            member = ROOT.name + "/" + name
            actual = hashlib.sha256(archive.read(member)).hexdigest()
            if actual != expected_hash:
                raise ValueError("ZIP member digest differs: " + name)
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Local file changed after archive creation: " + name)
        if manifest_bytes != (ROOT / "results/statistics/artifact_manifest.json").read_bytes():
            raise ValueError("Archive manifest differs from the delivered local manifest")
        if any(part in (".git", ".venv", "__pycache__") for name in names for part in Path(name).parts):
            raise ValueError("Excluded runtime/version-control directory in archive")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"status": "PASSED", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "archive": path.name, "archive_bytes": path.stat().st_size,
              "archive_sha256": digest, "members_verified": len(names),
              "manifest_file_digests_verified": len(manifest["files"]),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "document": "paper/submission/Imran_Zakirov_Password_Length_Manuscript.docx",
              "checks": manifest["checks"], "public_repository_or_deposit": "NOT ASSIGNED"}
    Path(str(path) + ".sha256").write_text(digest + "  " + path.name + "\n")
    Path(str(path) + ".audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=Path,
                        default=ROOT.parent / "password-research-deliverables.zip")
    args = parser.parse_args()
    verify(args.archive)
