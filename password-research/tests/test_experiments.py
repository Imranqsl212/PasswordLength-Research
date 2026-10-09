from pathlib import Path
import importlib.util
import unittest
import numpy as np
from src.analysis import fit_length_model, merge_histograms, remove_top
from src.core import mass, size

P=Path(__file__).resolve().parents[1]/"experiments/synthetic_controls.py"
spec=importlib.util.spec_from_file_location("synthetic_controls",P)
controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)

class ExperimentTests(unittest.TestCase):
    def test_secure_generator_alphabet_length(self):
        for n in (6,7,16,32):
            value=controls.secure_characters("abc123!",n)
            self.assertEqual(len(value),n)
            self.assertTrue(set(value)<=set("abc123!"))
    def test_word_length_dynamic_program(self):
        # Four equiprobable tuples: a-a, a-bb, bb-a, bb-bb.
        self.assertEqual(controls.word_length_counts(["a","bb"],2),{3:1,4:2,5:1})
    def test_pool_and_remove_tie(self):
        h={6:[(4,1),(2,2)],8:[(2,2)]}
        pool=merge_histograms(h.values())
        self.assertEqual(pool,[(4,1),(2,4)])
        after=remove_top(h[6],pool,2)
        self.assertEqual(after,[(2,1.5)])
        self.assertEqual(mass(after),3)
    def test_pilot_constant_length_fold(self):
        h={6:[(5,1),(1,100)],8:[(3,1),(1,100)]}
        output,cv=fit_length_model(h,merge_histograms(h.values()),2)
        self.assertEqual(len(cv),2)
        self.assertTrue(np.isfinite(cv.log_loss).all())
        self.assertEqual(output["structural_model_status"],"DATA UNAVAILABLE")

if __name__=="__main__":unittest.main()
