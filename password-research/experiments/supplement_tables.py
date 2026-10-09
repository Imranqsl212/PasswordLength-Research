from pathlib import Path
import json
import math
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.analysis import read_sources,merge_histograms
from src.core import mass,size,coverage
ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"results/tables"

def main():
    lengths,full,yahoo=read_sources()
    inventory=[
        {"source":"LinkedIn v2 full","evidence":"DP empirical aggregate","account_mass":mass(full),"anonymous_bins":size(full),"length_groups":0,"terms":"CC0","strings":"none"},
        {"source":"LinkedIn v2 summed length lists","evidence":"DP empirical aggregate","account_mass":sum(mass(h) for h in lengths.values()),"anonymous_bins":sum(size(h) for h in lengths.values()),"length_groups":len(lengths),"terms":"CC0","strings":"none"},
        {"source":"Yahoo v1 all","evidence":"DP empirical aggregate","account_mass":mass(yahoo),"anonymous_bins":size(yahoo),"length_groups":0,"terms":"CC0","strings":"none"},
        {"source":"HIBP fixed public panel","evidence":"API empirical counts","account_mass":None,"anonymous_bins":8,"length_groups":3,"terms":"official API documentation","strings":"eight user-provided teaching examples"},
        {"source":"Local synthetic controls","evidence":"controlled synthetic observations","account_mass":None,"anonymous_bins":26700,"length_groups":None,"terms":"project license","strings":"not retained"},
        {"source":"EFF vocabulary","evidence":"generation vocabulary","account_mass":None,"anonymous_bins":7776,"length_groups":None,"terms":"CC BY 4.0 EFF policy","strings":"ordinary words, not credentials"}]
    pd.DataFrame(inventory).to_csv(T/"dataset_characteristics.csv",index=False)
    df=pd.read_csv(T/"length_distribution.csv")
    matched=[]
    for _,r in df[(df.length>=6)&(df.length<=16)].iterrows():
        for n in (26,52,62,94):
            baseline=10000/n**int(r.length)
            human=r.conditional_top10000_account_pct/100
            matched.append({"nominal_length":int(r.length),"uniform_alphabet":n,"budget":10000,
                            "conditional_released_human_coverage":human,"theoretical_uniform_coverage":baseline,
                            "log10_coverage_ratio":math.log10(human/baseline),"released_mass":int(r.released_account_mass),
                            "comparison":"oracle human distribution vs uniform theory; source length semantics unspecified"})
    pd.DataFrame(matched).to_csv(T/"random_vs_human.csv",index=False)
    threat=[]
    for rate in (1,10,100):
        threat.append({"model":"online hypothetical", "configuration":f"{rate}/second", "rate_per_second":rate,
                       "checks_per_hour":rate*3600,"checks_per_day":rate*86400,"seconds_for_10000":10000/rate,
                       "limitations":"continuous attempts; real throttling, lockout, MFA, latency and detection alter budget"})
    hashdf=pd.read_csv(T/"hashing_cost.csv")
    for algorithm,g in hashdf[hashdf.cost_level==3].groupby("algorithm"):
        latency=g.verify_median_ms.median()/1000;rate=1/latency
        threat.append({"model":"local serial verification illustration", "configuration":algorithm+" level3",
                       "rate_per_second":rate,"checks_per_hour":rate*3600,"checks_per_day":rate*86400,"seconds_for_10000":10000/rate,
                       "limitations":"reciprocal measured median; no accelerators, parallelism, batch optimization or real attack"})
    pd.DataFrame(threat).to_csv(T/"threat_models.csv",index=False)
    tail=[]
    for l,h in lengths.items():
        counts=dict(h)
        tail.append({"length":l,"frequency1_distinct_fraction":counts.get(1,0)/size(h),
                     "frequency2_distinct_fraction":counts.get(2,0)/size(h),
                     "even_frequency_distinct_fraction":sum(m for f,m in h if f%2==0)/size(h),
                     "released_mass":mass(h),"anonymous_bins":size(h)})
    pd.DataFrame(tail).to_csv(T/"tail_diagnostics.csv",index=False)
    wide={}
    for l,h in lengths.items():wide.setdefault("6-7" if l<=7 else "8-9" if l<=9 else "10-11" if l<=11 else "12-14" if l<=14 else "15-16" if l<=16 else "17+",[]).append(h)
    pd.DataFrame([{"bin":k,"released_mass":mass(merge_histograms(v)),"anonymous_bins":size(merge_histograms(v)),
                   "conditional_top10000_pct":coverage(merge_histograms(v),10000)*100} for k,v in wide.items()]).to_csv(T/"alternate_length_bins.csv",index=False)
    long=[]
    for minimum in (16,20,24,32):
        sub=df[df.length>=minimum];total=sub.released_account_mass.sum()
        for k in (100,1000,10000,100000,1000000):
            early=float((sub.released_account_mass*sub[f"pooled_top{k}_account_pct"]/100).sum())
            long.append({"minimum_source_length":minimum,"budget":k,"released_mass":int(total),
                         "expected_early_mass":early,"account_early_pct":100*early/total,
                         "status":"descriptive privatized counts; tie expectations, no plaintext examples"})
    pd.DataFrame(long).to_csv(T/"long_early_membership.csv",index=False)
    statuses=[
        ("A exhaustive search","RUN_THEORETICAL","N^L, expected guesses and uniform coverage, no actual enumeration"),
        ("D frequency informed","RUN_AGGREGATE","anonymous conditional/pooled frequency oracle; no candidate strings"),
        ("B dictionary order","NOT RUN","strings and independently validated training/test corpus unavailable"),
        ("C transformation order","NOT RUN","no candidate generator; HIBP panel measures occurrence counts only"),
        ("E statistical guessing","NOT RUN","no appropriate plaintext training/test corpus; literature only"),
        ("Length-only descriptive model","RUN_EMPIRICAL_AGGREGATE","distinct-bin top10k membership; leave-length-out calibration"),
        ("Length+structure human model","DATA UNAVAILABLE","anonymous histograms have no structural covariates"),
        ("Synthetic characters/passphrases","RUN_CONTROLLED","CSPRNG generation and feature validation, theoretical entropy"),
        ("Local password hashing","RUN_EMPIRICAL","36 cells;5 reps;3 algorithms;no credential cracking")]
    pd.DataFrame(statuses,columns=["analysis","status","scope"]).to_csv(T/"experiment_status.csv",index=False)
    print("Saved dataset, matched-distribution, threat-model, tail, long-password and status tables.")

if __name__=="__main__":main()
