"""Secure local controls and word-length mathematics; never a guessing engine."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import secrets
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import ALPHABETS, features

ROOT=Path(__file__).resolve().parents[1]

def secure_characters(alphabet,length):
    return "".join(secrets.choice(alphabet) for _ in range(length))

def word_length_counts(words,word_count):
    """Number of unambiguous word tuples conditional on joined character length."""
    one=Counter(map(len,words));counts={0:1}
    for _ in range(word_count):
        new=Counter()
        for a,ca in counts.items():
            for b,cb in one.items():new[a+b]+=ca*cb
        counts=new
    return {length+word_count-1:n for length,n in counts.items()}

def run(regenerate=False):
    cfg=json.loads((ROOT/"config.json").read_text())
    path=ROOT/"data/derived/synthetic_features.csv"
    vocabulary=(ROOT/"data/raw/eff_large_wordlist.txt").read_text().splitlines()
    words=[line.split()[1] for line in vocabulary]
    assert len(words)==7776 and len(set(words))==7776 and all(" " not in w for w in words)
    if regenerate or not path.exists():
        rows=[];digest=hashlib.sha256()
        def add(value,category,replicate,entropy,word_count=0):
            b=value.encode();digest.update(len(b).to_bytes(4,"big"));digest.update(b)
            rows.append({"category":category,"replicate":replicate,"word_count":word_count,
                         "entropy_bits":entropy,**features(value)})
        for rep in range(cfg["synthetic_replicates"]):
            for length in cfg["random_lengths"]:
                for name,alphabet in ALPHABETS.items():
                    for _ in range(cfg["synthetic_samples_per_cell"]):
                        add(secure_characters(alphabet,length),name,rep,length*math.log2(len(alphabet)))
                for _ in range(cfg["synthetic_samples_per_cell"]):
                    unit=secure_characters(ALPHABETS["lowercase26"],2)
                    add((unit*math.ceil(length/2))[:length],"synthetic_repeat2",rep,2*math.log2(26))
                    root=secure_characters(ALPHABETS["lowercase26"],4)
                    add(root+"1!"+"a"*(length-6),"synthetic_fixed_suffix",rep,4*math.log2(26))
            for w in cfg["passphrase_word_counts"]:
                for _ in range(cfg["synthetic_samples_per_cell"]):
                    value=" ".join(secrets.choice(words) for _ in range(w))
                    add(value,"eff_uniform_words",rep,w*math.log2(len(words)),w)
        df=pd.DataFrame(rows);df.to_csv(path,index=False)
        metadata={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"generator":"secrets.choice; independent OS randomness, intentionally not seedable",
                  "analysis_seed":cfg["analysis_seed"],"secure_generator_seed":"not applicable",
                  "temporary_string_bundle_sha256":digest.hexdigest(),"features_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                  "n":len(df),"config":cfg,"wordlist_sha256":hashlib.sha256((ROOT/"data/raw/eff_large_wordlist.txt").read_bytes()).hexdigest(),
                  "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "interpretation":"synthetic controls are not human observations; frozen features are inputs to reproducible analysis; text not retained"}
        (ROOT/"results/statistics/synthetic_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    # Always analyze the serialized input, including on its first generation.
    # round_trip prevents parser rounding from changing float means or entropy.
    df=pd.read_csv(path,float_precision="round_trip")
    summary=[]
    for (cat,length,rep),g in df[df.category!="eff_uniform_words"].groupby(["category","length","replicate"]):
        summary.append({"category":cat,"length":length,"replicate":rep,"n":len(g),"entropy_bits":float(g.entropy_bits.iloc[0]),
                        "mean_diversity":float(g.diversity.mean()),"adjacent_repeat_fraction":float(g.adjacent_repeat.mean())})
    pd.DataFrame(summary).to_csv(ROOT/"results/tables/synthetic_summary.csv",index=False)
    phrases=[]
    for (w,rep),g in df[df.category=="eff_uniform_words"].groupby(["word_count","replicate"]):
        phrases.append({"word_count":w,"replicate":rep,"n":len(g),"median_length":float(g.length.median()),
                        "q10_length":float(g.length.quantile(.1)),"q90_length":float(g.length.quantile(.9)),"entropy_bits":w*math.log2(len(words))})
    pd.DataFrame(phrases).to_csv(ROOT/"results/tables/passphrase_summary.csv",index=False)
    conditional=[]
    for w in cfg["passphrase_word_counts"]:
        counts=word_length_counts(words,w)
        assert sum(counts.values())==len(words)**w
        for length,count in counts.items():
            conditional.append({"word_count":w,"length":length,"word_tuple_count":str(count),
                                "conditional_entropy_bits":math.log2(count),"unconditional_entropy_bits":w*math.log2(len(words)),
                                "length_probability":count/len(words)**w})
    pd.DataFrame(conditional).to_csv(ROOT/"results/tables/passphrase_conditional.csv",index=False)
    panel=json.loads((ROOT/"data/raw/hibp_public_panel.json").read_text())
    transformations=pd.DataFrame(panel["rows"])
    transformations["count_ratio_to_base"]=transformations.corpus_occurrences/transformations.corpus_occurrences.iloc[0]
    transformations["evidence_type"]="empirical corpus count; selected public panel"
    transformations["global_rank"]="UNAVAILABLE"
    transformations.to_csv(ROOT/"results/tables/transformations.csv",index=False)
    print(f"Synthetic feature snapshot: {len(df)} samples; EFF words={len(words)}; fixed HIBP panel={len(transformations)}.")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--regenerate",action="store_true")
    run(p.parse_args().regenerate)
