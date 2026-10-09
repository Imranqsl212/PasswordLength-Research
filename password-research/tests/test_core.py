import unittest
import numpy as np
from scipy.stats import spearmanr
from src.core import *

class CoreTests(unittest.TestCase):
    def test_parse_aggregation(self):
        h = parse_histogram("3 1\n2 1\n1 2\n1 1\n")
        self.assertEqual(h, [(3,1),(2,1),(1,3)])
        self.assertEqual(mass(h),8)
        self.assertEqual(size(h),5)
        self.assertEqual(coverage(h,2),5/8)
        self.assertEqual(guesses_for_mass(h,.5),2)
    def test_invalid(self):
        for t in ("0 1", "2 -1", "password 2", "1 2 3", ""):
            with self.assertRaises(ValueError): parse_histogram(t)
    def test_ties(self):
        a=[(2,2)]; pooled=[(4,1),(2,4)]
        self.assertEqual(rank_map(pooled)[2],(2,5,3.5))
        e,lo,hi,d=pooled_tie_coverage(a,pooled,2)
        self.assertEqual((e,lo,hi,d),(.25,0,.5,.25))
    def test_uniform(self):
        self.assertEqual(uniform_baseline(2,3)["space"],8)
        self.assertEqual(uniform_baseline(2,3)["expected_guesses"],4.5)
        self.assertEqual(uniform_baseline(2,3)["entropy_bits"],3)
        self.assertEqual(uniform_coverage(2,3,2),.25)
        self.assertEqual(uniform_coverage(2,3,20),1)
    def test_features_unicode_and_whitespace(self):
        self.assertEqual(features("é")["length"],1)
        self.assertEqual(features("e\u0301")["length"],2)
        self.assertEqual(features(" a ")["length"],3)
        self.assertEqual(features("é")["utf8_bytes"],2)
        self.assertEqual(features("漢字")["symbols"],0)
        self.assertEqual(features("漢字")["ends_symbol"],0)
        self.assertEqual(features("Ab2026!")["year_1900_2099"],1)
        self.assertEqual(features("121212")["whole_string_repeated_substring"],1)
        self.assertEqual(features("a1a1a1")["alternating_character_classes"],1)
        self.assertEqual(features("aa1b2")["alternating_character_classes"],0)
    def test_compressed_statistics(self):
        x=[6,6,8,10]; y=[5,1,2,2]; w=[2,3,4,1]
        expected=spearmanr(np.repeat(x,w),np.repeat(y,w)).statistic
        self.assertAlmostEqual(weighted_spearman(x,y,w),expected)
        self.assertEqual(weighted_quantile([1,4,9],[5,2,3],.5),1)
        self.assertEqual(weighted_quantile([1,4,9],[5,2,3],.9),9)

if __name__ == "__main__": unittest.main()
