"""Reproducible aggregate-spectrum analysis, without plaintext or attack code."""
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import platform
import zipfile
import tarfile
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.core import *

ROOT = Path(__file__).resolve().parents[1]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def read_sources():
    expected={"linkedin.zip":"fc831d2e524913d4692c297a356018f15a75d2a56520fc66576ddff0004faba7",
              "yahoo.tar.gz":"061137ea3cc129c7d9f501295cb194e0c6fa158acac702f893cba3cfd5f44efe"}
    for filename,digest in expected.items():
        if sha(ROOT/"data/raw"/filename)!=digest:
            raise ValueError(f"audited input checksum mismatch: {filename}")
    lengths = {}
    with zipfile.ZipFile(ROOT / "data/raw/linkedin.zip") as archive:
        full = parse_histogram(archive.read("linkedin_dataset/full.out").decode("ascii"))
        for name in archive.namelist():
            match = re.search(r"/length(\d+)\.out$",name)
            if match:
                lengths[int(match[1])] = parse_histogram(archive.read(name).decode("ascii"))
    with tarfile.open(ROOT / "data/raw/yahoo.tar.gz","r:gz") as archive:
        name = next(x for x in archive.getnames() if x.endswith("/yahoo-all.txt"))
        yahoo = parse_histogram(archive.extractfile(name).read().decode("ascii"))
    return dict(sorted(lengths.items())), full, yahoo

def merge_histograms(histograms):
    counts=Counter()
    for hist in histograms:
        for f,m in hist: counts[f]+=m
    return sorted(counts.items(),reverse=True)

def summarize(label,hist,pool,budgets):
    ranks = rank_map(pool)
    f = [x[0] for x in hist]; m = [x[1] for x in hist]
    w = [a*b for a,b in hist]
    row={"length":label,"released_account_mass":mass(hist),"anonymous_distinct_bins":size(hist),
         "distinct_median_frequency":weighted_quantile(f,m,.5),
         "distinct_q90_frequency":weighted_quantile(f,m,.9),
         "distinct_mean_frequency":mass(hist)/size(hist),
         "account_weighted_median_frequency":weighted_quantile(f,w,.5),
         "account_weighted_pooled_midrank_median":weighted_quantile([ranks[x][2] for x in f],w,.5),
         "conditional_g50":guesses_for_mass(hist,.5), "conditional_g90":guesses_for_mass(hist,.9)}
    for k in budgets:
        if k <= size(pool):
            e,lo,hi,d=pooled_tie_coverage(hist,pool,k)
            row.update({f"pooled_top{k}_account_pct":100*e,f"pooled_top{k}_tie_min_pct":100*lo,
                        f"pooled_top{k}_tie_max_pct":100*hi,f"pooled_top{k}_distinct_pct":100*d})
        row[f"conditional_top{k}_account_pct"] = 100*coverage(hist,k) if k<=size(hist) else None
    return row

def correlations(lengths):
    x=[];f=[];m=[]
    for length,hist in lengths.items():
        for frequency,multiplicity in hist:
            x.append(length);f.append(frequency);m.append(multiplicity)
    accounts=np.asarray(f)*np.asarray(m)
    return {"rho_length_frequency_distinct":weighted_spearman(x,f,m),
            "rho_length_frequency_account":weighted_spearman(x,f,accounts),
            "rho_length_frequency_midrank_distinct":-weighted_spearman(x,f,m),
            "rho_length_frequency_midrank_account":-weighted_spearman(x,f,accounts),
            "anonymous_distinct_bins":float(sum(m)),"released_account_mass":float(sum(accounts)),
            "length_groups":len(lengths),"inference":"descriptive exact weighted statistic on released histogram; no p-value or sampling CI"}

def fit_length_model(lengths,pool,k):
    """Descriptive bin-membership model; no password inputs or structure inference."""
    ranks=rank_map(pool); poolcounts=dict(pool)
    x=[];y=[];w=[];group=[]
    for length,hist in lengths.items():
        for f,m in hist:
            first,_,_=ranks[f]
            q=max(0,min(poolcounts[f],k-first+1))/poolcounts[f]
            for label,weight in ((1,m*q),(0,m*(1-q))):
                if weight:
                    x.append(length);y.append(label);w.append(weight);group.append(length)
    x=np.asarray(x,dtype=float);y=np.asarray(y);w=np.asarray(w);group=np.asarray(group)
    # Constant rescaling stabilizes optimization while preserving bin weighting.
    def fit(ix):
        mean=np.average(x[ix],weights=w[ix]);sd=math.sqrt(np.average((x[ix]-mean)**2,weights=w[ix])) or 1.0
        model=LogisticRegression(C=1e6,max_iter=1000,tol=1e-10)
        model.fit(((x[ix]-mean)/sd).reshape(-1,1),y[ix],sample_weight=w[ix]/w[ix].sum())
        return model,mean,sd
    model,mean,sd=fit(np.ones(len(x),dtype=bool))
    coef=float(model.coef_[0,0]/sd); intercept=float(model.intercept_[0]-coef*mean)
    losses=[];rows=[]
    for held in sorted(lengths):
        train=group!=held;test=~train
        fold,mu,sigma=fit(train)
        p=fold.predict_proba(((x[test]-mu)/sigma).reshape(-1,1))[:,1]
        base=np.average(y[train],weights=w[train])
        loss=np.average(-(y[test]*np.log(p)+(1-y[test])*np.log1p(-p)),weights=w[test])
        base_loss=np.average(-(y[test]*math.log(base)+(1-y[test])*math.log1p(-base)),weights=w[test])
        rows.append({"held_length":held,"observed_distinct_early_fraction":float(np.average(y[test],weights=w[test])),
                     "predicted_fraction":float(p[0]),"constant_training_fraction":float(base),
                     "log_loss":float(loss),"constant_log_loss":float(base_loss),
                     "test_anonymous_bins":float(w[test].sum())})
    cv=pd.DataFrame(rows)
    total=float(w.sum())
    return {"model":"length_only_logistic","budget":k,"coefficient_per_character":coef,
            "odds_ratio_per_character":math.exp(coef),"intercept":intercept,
            "held_length_log_loss":float(np.average(cv.log_loss,weights=cv.test_anonymous_bins)),
            "held_length_constant_log_loss":float(np.average(cv.constant_log_loss,weights=cv.test_anonymous_bins)),
            "account_vs_distinct":"distinct-bin weighted", "anonymous_bins":total,
            "structural_model_status":"DATA UNAVAILABLE",
            "validation":"leave-length-out descriptive calibration diagnostic; target cutoff uses full pool, not prospective prediction"},cv

def remove_top(hist,pool,k):
    ranks=rank_map(pool);count=dict(pool);out=[]
    for f,m in hist:
        start,_,_=ranks[f]; fraction=max(0,min(count[f],k-start+1))/count[f]
        remaining=m*(1-fraction)
        if remaining>0: out.append((f,remaining))
    return out

def run(pilot=False):
    config=json.loads((ROOT/"config.json").read_text())
    lengths,full,yahoo=read_sources()
    base=ROOT/("results/pilot" if pilot else "results")
    for d in ("tables","statistics"): (base/d).mkdir(parents=True,exist_ok=True)
    if pilot:
        lengths={l:lengths[l] for l in (6,8)}
    pool=merge_histograms(lengths.values())
    rows=[summarize(l,h,pool,config["coverage_budgets"]) for l,h in lengths.items()]
    pd.DataFrame(rows).to_csv(base/"tables/length_distribution.csv",index=False)
    curve=[]
    curve_k=sorted(set(config["coverage_budgets"]+[int(x) for x in np.geomspace(1,size(pool),100)]))
    for label,hist in [(str(l),h) for l,h in lengths.items()]+[("LinkedIn_full",full),("Yahoo_all",yahoo),("LinkedIn_pooled_lengths",pool)]:
        for k in curve_k:
            if k<=size(hist):
                curve.append({"group":label,"guesses":k,"account_coverage":coverage(hist,k),
                              "denominator_mass":mass(hist),"anonymous_bins":size(hist)})
    pd.DataFrame(curve).to_csv(base/"tables/coverage_curves.csv",index=False)
    grouped={}
    for l,h in lengths.items(): grouped.setdefault(str(l) if l<16 else "16+",[]).append(h)
    bins=[summarize(l,merge_histograms(hs),pool,config["coverage_budgets"]) for l,hs in grouped.items()]
    pd.DataFrame(bins).to_csv(base/"tables/length_bins.csv",index=False)
    primary={l:h for l,h in lengths.items() if l in config["primary_lengths"]}
    corr={"all_lengths":correlations(lengths),"primary_6_to_16":correlations(primary)}
    corr["consistency"]={"full_account_mass":mass(full),"length_account_mass":mass(pool),
                         "difference":mass(pool)-mass(full),"full_distinct_bins":size(full),"length_distinct_bins":size(pool),
                         "yahoo_account_mass":mass(yahoo),"yahoo_distinct_bins":size(yahoo)}
    robustness=[]
    for name,subset in [("all",primary),("frequency_at_least_2",{l:[(f,m) for f,m in h if f>=2] for l,h in primary.items()}),
                        ("frequency_at_least_5",{l:[(f,m) for f,m in h if f>=5] for l,h in primary.items()}),
                        ("exclude_length15_diagnostic",{l:h for l,h in primary.items() if l!=15}),
                        ("restrict_lengths6_to14_diagnostic",{l:h for l,h in primary.items() if l<=14}),
                        ("remove_pooled_top100",{l:remove_top(h,pool,100) for l,h in primary.items()})]:
        summary=correlations(subset);robustness.append({"condition":name,**summary})
    pd.DataFrame(robustness).to_csv(base/"tables/robustness.csv",index=False)
    adjacent=[];random=[]
    for l in config["random_lengths"]:
        for alphabet,chars in ALPHABETS.items():
            u=uniform_baseline(len(chars),l)
            random.append({"alphabet":alphabet,"length":l,"alphabet_size":len(chars),"search_space":str(u["space"]),
                           "expected_guesses":u["expected_guesses"],"entropy_bits":u["entropy_bits"],
                           "coverage_at_10000":uniform_coverage(len(chars),l,10000)})
    pd.DataFrame(random).to_csv(base/"tables/theoretical_baseline.csv",index=False)
    for l in range(6,16):
        if l in lengths and l+1 in lengths:
            a,b=lengths[l],lengths[l+1]
            adjacent.append({"from_length":l,"to_length":l+1,
                             "conditional_g50_ratio":guesses_for_mass(b,.5)/guesses_for_mass(a,.5),
                             "conditional_top10000_change_pp":100*(coverage(b,10000)-coverage(a,10000)),
                             "from_mass":mass(a),"to_mass":mass(b),"uniform_lowercase_search_space_ratio":26,
                             "causality":"different observed source groups, not an intervention"})
    pd.DataFrame(adjacent).to_csv(base/"tables/marginal_length.csv",index=False)
    lengthmodel,cv=fit_length_model(primary,pool,config["primary_budget"])
    pd.DataFrame([lengthmodel]).to_csv(base/"tables/model_results.csv",index=False)
    cv.to_csv(base/"tables/model_validation.csv",index=False)
    corr["length_model"]=lengthmodel
    (base/"statistics/summary.json").write_text(json.dumps(corr,indent=2)+"\n")
    feature_rows=[{"length":l,"frequency":f,"multiplicity":m} for l,h in lengths.items() for f,m in h]
    if not pilot:
        pd.DataFrame(feature_rows).to_csv(ROOT/"data/derived/length_frequency_histogram.csv",index=False)
    metadata={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"pilot":pilot,"config":config,
              "input_sha256":{p.name:sha(p) for p in (ROOT/"data/raw").iterdir() if p.is_file() and p.stat().st_size},
              "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/"src").glob("*.py")},
              "config_sha256":sha(ROOT/"config.json"),"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
              "scikit_learn":importlib.metadata.version("scikit-learn"),"scipy":importlib.metadata.version("scipy"),
              "notes":"released mass, not exact distinct people; anonymous frequency oracles, not operational attacks"}
    (base/"statistics/run_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    assert all(0<=x["account_coverage"]<=1 for x in curve)
    print(json.dumps({"pilot":pilot,"length_groups":len(lengths),"account_mass":mass(pool),"distinct_bins":size(pool),"statistics":corr},indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--pilot",action="store_true")
    run(parser.parse_args().pilot)
