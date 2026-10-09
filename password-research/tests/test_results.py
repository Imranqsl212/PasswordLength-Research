from pathlib import Path
import hashlib
import json
import unittest
import pandas as pd
from src.analysis import read_sources
from src.core import mass,size

ROOT=Path(__file__).resolve().parents[1]

class FrozenResultTests(unittest.TestCase):
    def test_source_mass_and_archive_integrity(self):
        lengths,full,yahoo=read_sources()
        self.assertEqual(len(lengths),64)
        self.assertEqual((mass(full),size(full)),(174292189,57431283))
        self.assertEqual((mass(yahoo),size(yahoo)),(69301337,33895873))
        self.assertEqual(sum(mass(h) for h in lengths.values()),174296614)
    def test_monotonic_supported_curves(self):
        table=pd.read_csv(ROOT/"results/tables/coverage_curves.csv")
        for _,group in table.groupby("group"):
            ordered=group.sort_values("guesses")
            self.assertTrue((ordered.guesses<=ordered.anonymous_bins).all())
            self.assertTrue(ordered.account_coverage.between(0,1).all())
            self.assertTrue((ordered.account_coverage.diff().dropna()>=-1e-12).all())
    def test_tie_bounds_and_total_mass(self):
        table=pd.read_csv(ROOT/"results/tables/length_distribution.csv")
        for k in (100,1000,10000,100000,1000000):
            e=table[f"pooled_top{k}_account_pct"]
            self.assertTrue((e>=table[f"pooled_top{k}_tie_min_pct"]-1e-12).all())
            self.assertTrue((e<=table[f"pooled_top{k}_tie_max_pct"]+1e-12).all())
            pooled=pd.read_csv(ROOT/"results/tables/coverage_curves.csv")
            target=pooled[(pooled.group=="LinkedIn_pooled_lengths")&(pooled.guesses==k)].account_coverage.iloc[0]
            weighted=(e*table.released_account_mass).sum()/table.released_account_mass.sum()/100
            self.assertAlmostEqual(weighted,target,places=12)
    def test_frozen_synthetic_features_no_strings(self):
        df=pd.read_csv(ROOT/"data/derived/synthetic_features.csv")
        self.assertEqual(len(df),26700)
        self.assertFalse(any(c in df.columns for c in ("password","secret","verifier","email","username")))
        self.assertTrue((df[df.category.str.contains("26|52|62|94")].length==df[df.category.str.contains("26|52|62|94")].utf8_bytes).all())
        for _,g in df[df.category!="eff_uniform_words"].groupby(["category","length","replicate"]):
            self.assertEqual(len(g),100)
    def test_original_hashing_run_export_digest(self):
        info=json.loads((ROOT/"results/statistics/hashing_metadata.json").read_text())
        self.assertEqual(info["cells_completed"],36)
        self.assertEqual(info["output_sha256"],hashlib.sha256((ROOT/"results/tables/hashing_cost.csv").read_bytes()).hexdigest())

if __name__=="__main__":unittest.main()
