"""Check publication links/provenance and optionally bundle the safe artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_digest(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Stale digest: {path.relative_to(ROOT)}")


def audit():
    figures = json.loads((ROOT / "results/figures/figure_metadata.json").read_text())
    require_digest(ROOT / "experiments/make_figures.py", figures["source_sha256"])
    for name, digest in figures["input_sha256"].items():
        require_digest(ROOT / "results/tables" / name, digest)
    for name, digest in figures["output_sha256"].items():
        require_digest(ROOT / "results/figures" / name, digest)
    require_digest(ROOT / "results/figures/captions.md", figures["captions_sha256"])
    run = json.loads((ROOT / "results/statistics/run_metadata.json").read_text())
    for name, digest in run["source_sha256"].items():
        require_digest(ROOT / name, digest)
    model_metadata = json.loads((ROOT / "results/statistics/nonlinear_metadata.json").read_text())
    require_digest(ROOT / "experiments/nonlinear_models.py", model_metadata["source_sha256"])
    require_digest(ROOT / "config.json", model_metadata["config_sha256"])
    for field in ("input_sha256", "output_sha256"):
        for name, digest in model_metadata[field].items():
            require_digest(ROOT / name, digest)
    dp_metadata = json.loads((ROOT / "results/statistics/dp_metadata.json").read_text())
    for field in ("source_sha256", "input_sha256", "output_sha256"):
        for name, digest in dp_metadata[field].items():
            require_digest(ROOT / name, digest)
    configuration_digest = hashlib.sha256(json.dumps(dp_metadata["configuration"], sort_keys=True).encode()).hexdigest()
    if configuration_digest != dp_metadata["configuration_sha256"]:
        raise ValueError("DP configuration digest is stale")
    check = json.loads((ROOT / "results/statistics/reproduction_check.json").read_text())
    if check["changed_tables"]:
        raise ValueError("The last frozen-input reproduction has changed tables")
    actual_names = {path.name for path in (ROOT / "results/tables").glob("*.csv")}
    if actual_names != set(check["table_sha256"]):
        raise ValueError("Reproduction does not cover every current table")
    for name, digest in check["table_sha256"].items():
        require_digest(ROOT / "results/tables" / name, digest)

    manuscript = (ROOT / "paper/manuscript.md").read_text()
    results_text = re.split(r"## \d+\. Results\n", manuscript, maxsplit=1)[1]
    results_text = re.split(r"## \d+\. Discussion\n", results_text, maxsplit=1)[0]
    for number in range(1, 10):
        if not re.search(rf"\bFigure {number}\b", results_text):
            raise ValueError(f"Figure {number} lacks an explicit Results reference")
    if "**DATA UNAVAILABLE**" not in manuscript:
        raise ValueError("Missing explicit unavailable human-structure status")
    # The journal extension has separate provenance and never rewrites the
    # original numeric baseline or timing measurements.
    journal = json.loads((ROOT / "results/statistics/journal_extensions_metadata.json").read_text())
    require_digest(ROOT / "analysis/journal-analysis-plan.json", journal["plan_sha256"])
    for field in ("source_sha256", "input_sha256", "output_sha256"):
        for name, digest in journal[field].items():
            require_digest(ROOT / name, digest)
    hashing = json.loads((ROOT / "results/statistics/hashing_uncertainty_metadata.json").read_text())
    require_digest(ROOT / hashing["derivation_source_relative_path"], hashing["derivation_source_sha256"])
    require_digest(ROOT / hashing["output_relative_path"], hashing["output_sha256"])
    for name, digest in hashing["input_sha256"].items():
        require_digest(ROOT / name, digest)
    journal_figures = json.loads((ROOT / "results/statistics/journal_figures_metadata.json").read_text())
    require_digest(ROOT / "experiments/make_journal_figures.py", journal_figures["source_sha256"])
    for name, digest in journal_figures["input_sha256"].items():
        require_digest(ROOT / "results/tables" / name, digest)
    for name, digest in journal_figures["output_sha256"].items():
        require_digest(ROOT / "results/journal_figures" / name, digest)
    require_digest(ROOT / "results/journal_figures/captions.json", journal_figures["captions_sha256"])
    bibliography = json.loads((ROOT / "paper/references-verified.json").read_text())
    require_digest(ROOT / "paper/references.bib", bibliography["source_sha256"])
    document = json.loads((ROOT / "results/statistics/docx_audit.json").read_text())
    require_digest(ROOT / document["docx"], document["sha256"])
    require_digest(ROOT / "paper/manuscript.md", document["source_sha256"])
    require_digest(ROOT / "experiments/build_docx.py", document["builder_sha256"])
    for item in document["figures"]:
        require_digest(ROOT / item["file"], item["sha256"])
    if document["figures_embedded"] != 12 or document["references"] != 34 or document["local_external_relationships"]:
        raise ValueError("Incomplete or nonportable Word document")
    integrity = json.loads((ROOT / "results/statistics/journal_integrity_audit.json").read_text())
    if integrity["status"] != "PASSED":
        raise ValueError("Final document-integrity checks did not pass")
    require_digest(ROOT / "paper/manuscript.md", integrity["source_sha256"])
    require_digest(ROOT / document["docx"], integrity["docx_sha256"])
    require_digest(ROOT / "experiments/journal_integrity_audit.py", integrity["audit_source_sha256"])
    links_checked = 0
    for markdown_file in ROOT.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^\n)]+)\)", markdown_file.read_text()):
            target = target.strip("<>").split("#", 1)[0]
            if not target or re.match(r"[a-zA-Z]+://", target):
                continue
            target = re.sub(r":\d+$", "", target)
            path = Path(target)
            if not path.is_absolute():
                path = markdown_file.parent / path
            if not path.exists():
                raise ValueError(f"Broken local link in {markdown_file.relative_to(ROOT)}: {target}")
            links_checked += 1
    return {"tests_passed": check.get("tests_passed"), "tables_verified": len(actual_names), "plots_verified": len(figures["output_sha256"]),
            "plot_inputs_verified": len(figures["input_sha256"]), "local_links_verified": links_checked,
            "results_figure_references": list(range(1, 10)),
            "journal_plot_exports_verified": len(journal_figures["output_sha256"]),
            "docx_embedded_figures": document["figures_embedded"], "docx_references": document["references"],
            "docx_layout_verification": document["layout_render"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    checks = audit()
    excluded = {"__pycache__", ".venv", ".git"}
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in excluded for part in p.relative_to(ROOT).parts)
             and p.name != "artifact_manifest.json"]
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
                "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
                "scope": "aggregate research and controlled synthetic experiments; unavailable analyses explicitly marked"}
    manifest_path = ROOT / "results/statistics/artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.bundle:
        files.append(manifest_path)
        archive = ROOT.parent / "password-research-deliverables.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(files):
                bundle.write(path, str(Path(ROOT.name) / path.relative_to(ROOT)))
        checks.update(archive=str(archive), archive_bytes=archive.stat().st_size)
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
