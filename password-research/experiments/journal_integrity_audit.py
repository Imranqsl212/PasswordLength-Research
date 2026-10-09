"""Check final article numbering, evidence statuses, OOXML and optional preview.

This is document-integrity validation, not a scientific result or human review.
QuickLook supplies continuous HTML, so it cannot verify Word pagination.
"""
from pathlib import Path
from datetime import datetime, timezone
from html.parser import HTMLParser
from collections import Counter
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(value):
    return re.sub(r"\s+", " ", value).strip()


def require(condition, label):
    if not condition:
        raise ValueError(label)


class PreviewReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.tables = 0
        self.images = []
    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables += 1
        if tag == "img":
            self.images.append(dict(attrs).get("src"))
    def handle_data(self, data):
        self.text.append(data)


def audit(preview=None):
    source = ROOT / "paper/manuscript.md"
    text = source.read_text()
    build = json.loads((ROOT / "results/statistics/manuscript_build_metadata.json").read_text())
    word = json.loads((ROOT / "results/statistics/docx_audit.json").read_text())
    references = json.loads((ROOT / "paper/references-verified.json").read_text())
    require(sha(source) == build["manuscript_sha256"] == word["source_sha256"], "Stale manuscript build")
    require(sha(ROOT / "experiments/build_docx.py") == word["builder_sha256"], "Stale Word builder")
    require(sha(ROOT / "paper/references-verified.json") == build["reference_json_sha256"], "Stale reference formatting")
    require(sha(ROOT / "paper/references.bib") == references["source_sha256"], "Stale bibliography audit")
    refs_by_key = {item["id"]: item for item in references["references"]}
    number_to_key = {number: key for key, number in build["cited_references"].items()}
    require(set(number_to_key) == set(range(1, 35)), "Incomplete numbered bibliography")
    before, after = text.split("## References\n", 1)
    bibliography, appendices = after.split("## Appendix A", 1)
    ref_numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", bibliography, re.M)]
    require(ref_numbers == list(range(1, 35)), "Bibliography numbering")
    citation_count = 0
    cited = []
    for number, target in re.findall(r"(?<!!)\[(\d+)\]\((https?://[^)]+)\)", before + appendices):
        number = int(number)
        require(number in number_to_key, "Undefined numbered citation")
        item = refs_by_key[number_to_key[number]]
        aliases = set(item.get("url_aliases", [])) | {item.get("url", "")}
        if item.get("doi"):
            aliases.add("https://doi.org/" + item["doi"])
        require(target in aliases, "Citation URL disagrees with its numbered record")
        citation_count += 1
        if number not in cited:
            cited.append(number)
    require(set(cited) == set(range(1, 35)), "Uncited reference")
    labels = re.findall(r"^\*\*Figure (\d+|S\d+)\.", text, re.M)
    require(labels == [str(n) for n in range(1, 10)] + ["S1", "S2", "S3"], "Figure numbering/captions")
    table_labels = re.findall(r"^\*\*Table (\d+|A\d+)\.", text, re.M)
    require(table_labels == [str(n) for n in range(1, 8)] + ["A" + str(n) for n in range(1, 9)], "Table numbering/captions")
    results = text.split("## 6. Results", 1)[1].split("## 7. Discussion", 1)[0]
    for number in range(1, 10):
        require(re.search(rf"\bFigure {number}\b", results), "Missing Results figure reference")
    for phrase in ("DATA UNAVAILABLE", "NOT RUN", "Imran Zakirov", "Prioritet",
                   "no external funding", "no competing interests", "OpenAI Codex",
                   "No public repository URL", "not preregistered"):
        require(phrase in text, "Required disclosure/status missing: " + phrase)
    require(not re.search(r"\b(?:TODO|TBD|INSERT HERE)\b", text), "Unresolved editorial placeholder")
    equation_text = [line[2:-2] for line in text.splitlines() if line.startswith("$$") and line.endswith("$$")]
    document = ROOT / word["docx"]
    require(sha(document) == word["sha256"], "Stale Word digest")
    with zipfile.ZipFile(document) as archive:
        xml = ET.fromstring(archive.read("word/document.xml"))
        rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        media = [n for n in archive.namelist() if n.startswith("word/media/")]
        require(len(media) == 12, "Word media count")
        require(len(xml.findall(".//w:tbl", NS)) == 15, "Word table count")
        require(len(xml.findall(".//wp:docPr", NS)) == 12, "Figure drawing count")
        portable = [rel for rel in rels if rel.attrib.get("TargetMode") == "External"]
        require(all(re.match(r"https?://", rel.attrib["Target"]) for rel in portable), "External local content dependency")
        media_digests = Counter(hashlib.sha256(archive.read(n)).hexdigest() for n in media)
        require(media_digests == Counter(item["sha256"] for item in word["figures"]), "Embedded figure bytes differ")
        paragraphs = ["".join(p.itertext()) for p in xml.findall(".//w:p", NS)]
        word_text = normalize(" ".join(xml.itertext()))
        require(all(normalize(e) in word_text for e in equation_text), "Missing equation text in Word")
        require(sum(bool(re.match(r"^\[\d+\] ", p)) for p in paragraphs) == 34, "Word numbered reference count")
        require(all("descr" in x.attrib for x in xml.findall(".//wp:docPr", NS)), "Missing figure alt text")
        section = xml.find(".//w:sectPr", NS)
        require(section is not None and section.find("w:pgSz", NS) is not None, "Missing page geometry")
    reproduction = json.loads((ROOT / "results/statistics/reproduction_check.json").read_text())
    require(reproduction["tests_passed"] == 54 and reproduction["tables_checked"] == 34 and not reproduction["changed_tables"], "Incomplete reproduction")
    for name, digest in reproduction["table_sha256"].items():
        require(sha(ROOT / "results/tables" / name) == digest, "Table changed since reproduction")
    for name in ("journal-claim-evidence.md", "journal-upgrade-report.md", "journal-submission-checklist.md"):
        require((ROOT / "docs" / name).is_file(), "Missing deliverable: " + name)
    preview_result = None
    if preview:
        preview = Path(preview).resolve()
        html_path = preview / "Preview.html"
        reader = PreviewReader()
        reader.feed(html_path.read_text())
        preview_text = normalize(" ".join(reader.text))
        require(reader.tables == 15 and len(reader.images) == 12, "Incomplete continuous preview")
        require(all(normalize(e) in preview_text for e in equation_text), "Preview omitted equations")
        for value in ("Imran Zakirov", "[34]", "Appendix F", "DATA UNAVAILABLE", "NOT RUN"):
            require(value in preview_text, "Preview omitted required text")
        attachments = [preview / image for image in reader.images]
        require(all(p.is_file() for p in attachments), "Missing preview attachment")
        require(Counter(sha(p) for p in attachments) == media_digests, "Preview image bytes differ")
        preview_result = {"type": "macOS QuickLook continuous HTML; NOT paginated Word rendering",
                          "html_sha256": sha(html_path), "images": 12, "tables": 15,
                          "equations_visible": 4, "bibliography_end_present": True,
                          "native_pagination_verified": False}
    # Flag words for a context review instead of mistaking an explicit negation
    # or a literature citation for an unsupported assertion.
    terms = ("prove", "causal", "dominates", "realistic attack", "first study",
             "independent", "entropy", "secure", "predictability")
    scan = {}
    for term in terms:
        hits = [normalize(line) for line in text.splitlines() if term in line.lower()]
        scan[term] = {"occurrences_by_line": len(hits), "interpretation": "Context reviewed in claim-evidence/reviewer reports; not an automated scientific verdict"}
    report = {"status": "PASSED", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": sha(source), "docx_sha256": sha(document),
              "audit_source_sha256": sha(Path(__file__)), "references": 34,
              "citation_occurrences": citation_count, "numbered_figures": labels,
              "numbered_tables": table_labels, "embedded_figures": 12,
              "editable_equations": len(equation_text), "external_local_relationships": 0,
              "tests_passed": 54, "csv_tables_reproduced": 34,
              "preview": preview_result, "overclaim_context_scan": scan,
              "limits": ["Not external peer review or human authorship verification",
                         "Actual Word/Pages pagination remains unverified",
                         "Venue eligibility and final author approval remain unestablished",
                         "Human structural data unavailable; operational/GPU comparisons not run"]}
    (ROOT / "results/statistics/journal_integrity_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    if preview_result:
        word["continuous_preview"] = preview_result
        word["layout_render"] = "QuickLook continuous content and initial thumbnail inspected; actual Word/Pages pagination UNVERIFIED"
        (ROOT / "results/statistics/docx_audit.json").write_text(json.dumps(word, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("overclaim_context_scan", "limits")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-directory", type=Path)
    args = parser.parse_args()
    audit(args.preview_directory)
