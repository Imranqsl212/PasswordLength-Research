"""Offline reanalysis of frozen safe inputs; preserves original timing measurements."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import unittest
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--without-figures",action="store_true")
    args=parser.parse_args()
    tables=ROOT/"results/tables"
    before={p.name:digest(p) for p in tables.glob("*.csv")}
    test_count=unittest.defaultTestLoader.discover(str(ROOT/"tests")).countTestCases()
    commands=[("unit and frozen-result tests",["-m","unittest","discover","-s","tests","-v"]),
              ("aggregate analysis",["-m","src.analysis"]),
              ("frozen synthetic controls and HIBP snapshot",["experiments/synthetic_controls.py"]),
              ("supplementary tables",["experiments/supplement_tables.py"]),
              ("DP mechanism and aggregate sensitivity diagnostics",["experiments/dp_sensitivity.py"]),
              ("exploratory nonlinear length models",["experiments/nonlinear_models.py"]),
              ("training-only thresholds, long grid and separate-source comparison",["experiments/journal_extensions.py","--root","."]),
              ("uncertainty from original five-value timing summaries",["experiments/hashing/derive_uncertainty.py","--root","."])]
    if not args.without_figures:
        commands.extend([("original figures",["experiments/make_figures.py","--root","."]),
                         ("journal main and appendix plates",["experiments/make_journal_figures.py"])])
    for label,command in commands:
        print("Running: "+label,flush=True)
        subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    after={p.name:digest(p) for p in tables.glob("*.csv")}
    changed=[name for name in sorted(before.keys() | after.keys()) if before.get(name)!=after.get(name)]
    report={"completed_at_utc":datetime.now(timezone.utc).isoformat(),"mode":"offline frozen-input reanalysis",
            "tests_passed":test_count,"tables_checked":len(before),"table_sha256":after,"changed_tables":changed,
            "hashing":"original measured run preserved; not rerun",
            "note":"deterministic tables compare byte hashes; plot metadata/timestamps may change without changing visual content"}
    (ROOT/"results/statistics/reproduction_check.json").write_text(json.dumps(report,indent=2)+"\n")
    if changed:
        raise RuntimeError("Table hashes changed: "+", ".join(changed)+". Inspect differences before accepting a new baseline.")
    print(f"Reproduced {len(after)} table CSVs; no table hash changed.")

if __name__=="__main__":main()
