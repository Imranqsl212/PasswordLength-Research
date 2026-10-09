"""Publication plates from frozen aggregate tables; no candidate generation."""
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prissm-journal-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "prissm-journal-font-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results/tables"
OUT = ROOT / "results/journal_figures"
C = ["#176B87", "#CB6A31", "#238673", "#845CA2", "#BA4752", "#667D43", "#454D66"]
INPUTS = ["theoretical_baseline.csv", "length_distribution.csv", "coverage_curves.csv",
          "synthetic_summary.csv", "journal_train_threshold_validation.csv", "journal_long_early.csv",
          "passphrase_summary.csv", "passphrase_conditional.csv", "hashing_cost.csv", "hashing_uncertainty.csv",
          "journal_cross_dataset.csv", "transformations.csv", "nonlinear_model_validation.csv", "threat_models.csv"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.labelsize":11,
        "axes.titlesize":11.5, "legend.fontsize":9, "axes.spines.top":False,
        "axes.spines.right":False, "axes.grid":True, "grid.alpha":.18,
        "pdf.fonttype":42, "svg.fonttype":"none", "savefig.facecolor":"white"})
    tables = {p:pd.read_csv(TABLES/p) for p in INPUTS}
    length = tables["length_distribution.csv"].query("6 <= length <= 16").sort_values("length")
    captions = []; outputs = []

    def plate(identifier, title, panels=1, height=4.8):
        fig, axes = plt.subplots(1, panels, figsize=(8.2,height), squeeze=False)
        fig.suptitle(f"Figure {identifier}. {title}", fontsize=13, weight="bold", x=.02, ha="left")
        fig.subplots_adjust(left=.1, right=.98, bottom=.15, top=.84, wspace=.32)
        return fig, axes.ravel()

    def save(fig, identifier, stem, caption):
        for ext in ("png","pdf","svg"):
            path = OUT/f"{stem}.{ext}"; fig.savefig(path,dpi=300,bbox_inches="tight",pad_inches=.1)
            outputs.append(path.name)
        plt.close(fig)
        captions.append({"number":identifier,"stem":stem,"caption":caption,"path":f"../results/journal_figures/{stem}.png"})

    theory = tables["theoretical_baseline.csv"]
    fig,(ax,) = plate("1","Uniform search-space baseline")
    for i,(key,label) in enumerate([( "lowercase26","Lowercase, N=26"),("letters52","Letters, N=52"),
                                  ("alphanumeric62","Letters/digits, N=62"),("printable94","Visible ASCII, N=94")]):
        d=theory[theory.alphabet==key]; ax.plot(d.length,[math.log10(int(n)) for n in d.search_space],"o-",ms=3,color=C[i],label=label)
    ax.set(xlabel="ASCII character length",ylabel="log₁₀ of uniform support Nᴸ")
    ax.legend(frameon=False); ax.set_xticks([6,8,12,16,20,24,32])
    save(fig,"1","jfig01_uniform","Theoretical support under independent uniform draws from four specified ASCII alphabets. N=94 excludes space. No empirical sample denominator or source-password entropy is implied.")

    fig,(ax,bx)=plate("2","Released frequency and conditional G50",2)
    for key,label,style,col in [("account_weighted_median_frequency","Account Mass","o-",C[0]),("distinct_median_frequency","Distinct bins","s--",C[1])]:
        ax.plot(length.length,length[key],style,color=col,label=label)
    ax.set(yscale="log",xlabel="Source-defined length",ylabel="Median released frequency"); ax.legend(frameon=False)
    bx.plot(length.length,length.conditional_g50,"o-",color=C[2]); bx.set(yscale="log",xlabel="Source-defined length",ylabel="Conditional G50 (oracle positions)")
    for a in (ax,bx):a.set_xticks([6,8,10,12,14,16])
    save(fig,"2","jfig02_frequency","All primary groups 6–16: 173,904,254 units of released Account Mass and 57,238,505 Distinct Anonymous Bins. G50 orders each known-length group separately. Medians use the stated distinct-bin or mass weights; these are perturbed historical spectra, not population estimates.")

    fig,(ax,)=plate("3","Length-conditioned oracle concentration")
    curves=tables["coverage_curves.csv"]
    for l in range(6,17):
        d=curves[curves.group.astype(str)==str(l)].sort_values("guesses")
        ax.plot(d.guesses,100*d.account_coverage,label=f"L={l}",color=plt.cm.viridis((l-6)/10),lw=2.5 if l==15 else 1.4,ls="--" if l==15 else "-")
    ax.set(xscale="log",xlabel="Anonymous oracle positions K",ylabel="Conditional released Account Mass (%)",ylim=(0,102))
    ax.legend(ncol=3,frameon=False,loc="upper left")
    save(fig,"3","jfig03_conditional","All eleven primary source-length groups, each divided by its own released Account Mass. Length 15 is dashed. Curves stop at the available positive-bin support; lines connect evaluated budgets. The oracle presumes length and the popularity order are known. No operational guessing or account recovery was measured.")

    fig,(ax,bx)=plate("4","Nominal length and generator assumptions",2,5.3)
    ax.plot(length.length,length.conditional_top10000_account_pct/100,"o-",color=C[0],label="Released corpus oracle")
    for key,label,col in [("lowercase26","Uniform N=26",C[2]),("printable94","Uniform N=94",C[1])]:
        d=theory[(theory.alphabet==key)&theory.length.between(6,16)];ax.plot(d.length,d.coverage_at_10000,"--",color=col,label=label)
    ax.set(yscale="log",xlabel="Nominal source length",ylabel="Coverage probability at K=10,000",ylim=(1e-29,1))
    ax.set_yticks([10.**i for i in [0,-7,-14,-21,-28]]);ax.legend(frameon=False,loc="lower left",fontsize=8.5)
    syn=tables["synthetic_summary.csv"]
    for key,label,col,style in [("lowercase26","Uniform lowercase",C[2],"-"),("printable94","Uniform visible ASCII",C[1],"-"),
                               ("synthetic_repeat2","Repeated two-letter unit",C[3],"--"),("synthetic_fixed_suffix","Four letters + fixed padding",C[4],"--")]:
        d=syn[syn.category==key].groupby("length").entropy_bits.first();bx.plot(d.index,d.values,style,color=col,label=label)
    bx.set(xlabel="ASCII character length",ylabel="Known generator entropy (bits)");bx.legend(frameon=False,loc="upper left",fontsize=8)
    for a in (ax,bx):a.set_xticks([6,12,16,24,32] if a is bx else [6,8,12,16])
    save(fig,"4","jfig04_matched","Left: empirical released conditional concentration versus mathematical min(K/Nᴸ,1), matched only by nominal length; source encoding and generation method are unknown. Right: entropy from four explicit injective uniform generators, checked using 16,800 frozen feature records. Equal local ASCII length permits different known generator entropy. This is not a matched human/random attack trial.")

    fig,(ax,bx)=plate("5","Models with training-only frequency thresholds",2,5.2)
    valid=tables["journal_train_threshold_validation.csv"]
    observed=valid[valid.model=="linear_logistic"].sort_values("held_length")
    for a in (ax,bx):
        a.plot(observed.held_length,observed.observed_distinct_fraction*100,"o-",color=C[0],label="Observed fraction")
        a.set(yscale="log",xlabel="Held source length",ylabel="Distinct-bin threshold membership (%)");a.set_xticks([6,8,10,12,14,16])
    specs=[("linear_logistic","Linear",ax,1),("quadratic_logistic","Quadratic",ax,2),("cubic_logistic","Cubic",ax,3),
           ("intercept_only","Fold constant",bx,1),("bspline_cubic_4knots","Spline, 4 knots",bx,2),("bspline_cubic_5knots","Spline, 5 knots",bx,3)]
    for key,label,a,i in specs:
        d=valid[valid.model==key].sort_values("held_length");a.plot(d.held_length,d.predicted_fraction*100,"--",color=C[i],label=label)
    for a in (ax,bx):a.legend(frameon=False,fontsize=8.2,loc="upper right")
    save(fig,"5","jfig05_train_threshold","Six fixed exploratory models, eleven held-length folds. Each threshold and boundary-tie fraction uses only the other ten primary groups and is frozen before application to the held spectrum. Outcome thresholds differ across folds; weighted log loss is an aggregate diagnostic, not prospective user/source prediction. The held denominators total 57,238,505 Distinct Anonymous Bins. Human structural adjustment remains DATA UNAVAILABLE.")

    fig,(ax,)=plate("6","Long-group exposure across all fixed thresholds")
    long=tables["journal_long_early.csv"]
    for i,(minimum,d) in enumerate(long.groupby("minimum_source_length",sort=True)):
        ax.plot(d.budget,d.account_fraction*100,"o-",ms=4,color=C[i],label=f"Source length ≥{minimum}")
    upper=max(1, float(long.account_fraction.max())*100*1.25)
    ax.set(xscale="log",xlabel="Pooled anonymous oracle positions K",ylabel="Eligible released Account Mass (%)",ylim=(0,upper))
    ax.legend(ncol=2,frameon=False,loc="upper left")
    save(fig,"6","jfig06_long_thresholds","Complete exploratory grid: seven minimum lengths × six budgets = 42 cells. Each curve divides selected pooled-oracle mass by the released mass of all supplied lengths meeting its threshold. The original ≥16/top10,000 cell is retained. Sharp tie bounds and distinct-bin denominators are exported separately. Threshold sensitivity does not identify individual strings, cause, original population fractions or real attack success.")

    fig,(ax,bx)=plate("7","Random-word length and conditional entropy",2,5.3)
    phrases=tables["passphrase_summary.csv"]; conditional=tables["passphrase_conditional.csv"]
    for i,(w,g) in enumerate(phrases.groupby("word_count")):
        median=g.median_length.median(); low=g.q10_length.median(); high=g.q90_length.median(); entropy=float(g.entropy_bits.iloc[0])
        ax.errorbar(median,entropy,xerr=[[median-low],[high-median]],fmt="o",color=C[i],capsize=3,label=f"{int(w)} words")
        d=conditional[conditional.word_count==w].sort_values("length");bx.plot(d.length,d.conditional_entropy_bits,color=C[i],label=f"{int(w)} words")
        bx.axhline(entropy,color=C[i],ls="--",lw=.7,alpha=.7)
    ax.set(xlabel="Joined character length (spaces included)",ylabel="Unconditional generator entropy (bits)");ax.legend(frameon=False,fontsize=8.5)
    bx.set(xlabel="Known joined length",ylabel="Conditional tuple entropy (bits)");bx.legend(frameon=False,fontsize=8.5)
    save(fig,"7","jfig07_words","Independent uniform draws with replacement from 7,776 EFF words, joined by one space. Left: 1,500 frozen phrases; markers are medians of three replicate medians and bars are medians of replicate 10th–90th length quantiles, not confidence intervals. Right: exact convolution counts conditional on known word count and joined length; dashed lines are unconditional entropy. No ordinary human-sentence entropy is estimated.")

    fig,axes=plt.subplots(2,2,figsize=(8.2,6.6));fig.suptitle("Figure 8. Local verifier cost and retained timing variation",fontsize=13,weight="bold",x=.02,ha="left")
    fig.subplots_adjust(left=.1,right=.98,bottom=.1,top=.86,wspace=.3,hspace=.45)
    hashing=tables["hashing_cost.csv"]
    for ax,alg in zip(axes.ravel(),["pbkdf2_sha256","argon2id","bcrypt"]):
        for level,g in hashing[hashing.algorithm==alg].groupby("cost_level"):
            g=g.sort_values("length_bytes");ax.errorbar(g.length_bytes,g.verify_median_ms,yerr=[g.verify_median_ms-g.verify_q25_ms,g.verify_q75_ms-g.verify_median_ms],fmt="o-",ms=3,color=C[int(level)-1],label=f"Level {int(level)}")
        ax.set(title=alg,xlabel="ASCII input length (bytes)",ylabel="Matching verification median (ms)",yscale="log");ax.legend(frameon=False,fontsize=8)
    ax=axes.ravel()[3]
    unc=tables["hashing_uncertainty.csv"];r=unc[(unc.algorithm=="argon2id")&(unc.cost_level==2)&(unc.length_bytes==8)&(unc.operation=="hash")].iloc[0]
    ax.plot([1,2,3,4,5],[r.min_ms,r.q25_ms,r.median_ms,r.q75_ms,r.max_ms],"o-",color=C[4])
    ax.set(title="Retained Argon2id hash outlier",xlabel="Sorted observation (n=5)",ylabel="Hash latency (ms)",xticks=[1,2,3,4,5])
    save(fig,"8","jfig08_hashing","36 cells, one warmup and five timed calls per operation/cell. Upper/left panels show matching-verification medians and descriptive quartile spans. Levels: PBKDF2 100k/300k/600k; Argon2id 19/32/64 MiB, t=2/2/3, p=1; bcrypt cost 8/10/12. Lower-right retains all five sorted hash times for Argon2id 32 MiB, 8 bytes, including 299.644 ms. Sorted times are recovered exactly from n=5 quantiles; temporal order was lost. Unequal parameter settings and one host do not rank security or attacker throughput.")

    fig,(ax,bx)=plate("9","Separate-source whole-spectrum concentration",2)
    cross=tables["journal_cross_dataset.csv"]
    ax.plot(cross.budget,cross.linkedin_full_coverage*100,"o-",color=C[0],label="LinkedIn full")
    ax.plot(cross.budget,cross.yahoo_all_coverage*100,"s-",color=C[1],label="Yahoo all")
    ax.set(xscale="log",xlabel="Anonymous oracle positions K",ylabel="Released Account Mass (%)");ax.legend(frameon=False)
    bx.plot(cross.budget,cross.yahoo_minus_linkedin_percentage_points,"o-",color=C[2]);bx.set(xscale="log",xlabel="Common oracle positions K",ylabel="Yahoo − LinkedIn (percentage points)")
    save(fig,"9","jfig09_cross_source","Separately analyzed historical spectra at six fixed common budgets. Denominators: LinkedIn full 174,292,189; Yahoo all 69,301,337 units of released Account Mass. They are not pooled. Differences describe overall concentration under their own frequency orders. Yahoo has no length covariate, so this cannot replicate the length association or establish a temporal trend.")

    # Auxiliary evidence stays in the appendix rather than displacing core results.
    fig,(ax,)=plate("S1","Fixed public transformation illustration")
    trans=tables["transformations.csv"];ax.barh(range(8),trans.corpus_occurrences,color=C[0])
    ax.set_yticks(range(8),trans.example);ax.invert_yaxis();ax.set(xscale="log",xlabel="HIBP historical corpus occurrences")
    save(fig,"S1","jfigS1_transformations","Eight deliberately selected public teaching examples, frozen HIBP snapshot of 5 October 2026. Counts are corpus occurrences, not current use, population prevalence, global guess ranks or causal transformation effects. Every selected example is reported.")
    fig,(ax,bx)=plate("S3","Fixed-budget mathematical illustrations",2)
    for rate in (1,10,100):
        k=np.logspace(1,6,60);ax.plot(k,k/rate,label=f"Assumed {rate}/s")
    for alg,g in hashing[hashing.cost_level==3].groupby("algorithm"):
        proxy=g.verify_median_ms.median()/1000;bx.plot(k,k*proxy,label=alg.replace("_sha256",""))
    for a in (ax,bx):a.set(xscale="log",yscale="log",xlabel="Hypothetical checks K",ylabel="Constant-cost proxy time (seconds)");a.legend(frameon=False,fontsize=8)
    save(fig,"S3","jfigS3_budget","Online rates are assumed admitted checks; offline illustrations multiply a chosen local matching-verification median proxy by K. This is constant-cost arithmetic, not expected stochastic runtime, optimized attacker speed or recovery time. Lockout, throttling and MFA are separate conditions. No real system was tested.")
    fig,(ax,bx)=plate("S2","Original full-pool calibration diagnostic",2)
    v=tables["nonlinear_model_validation.csv"].query("fit_scope == 'held_length'")
    for a,modelnames in [(ax,["linear_logistic","quadratic_logistic","cubic_logistic"]),(bx,["intercept_only","bspline_cubic_4knots","bspline_cubic_5knots"])]:
        d=v[v.model=="linear_logistic"].sort_values("held_length");a.plot(d.held_length,d.observed_distinct_early_fraction*100,"o-",color=C[0],label="Observed")
        for i,name in enumerate(modelnames,1):
            d=v[v.model==name].sort_values("held_length");a.plot(d.held_length,d.predicted_fraction*100,"--",color=C[i],label=name.replace("_logistic","").replace("bspline_cubic_","spline ").replace("intercept_only","constant"))
        a.set(yscale="log",xlabel="Held source length",ylabel="Full-pool early distinct bins (%)");a.legend(frameon=False,fontsize=8)
    save(fig,"S2","jfigS2_original_models","Preserved original diagnostic with labels and cutoff from the full 64-length pool, including held groups. This differs from Figure 5's training-only threshold target and is nonprospective. All six fixed specifications and all eleven primary groups are retained; human structure is DATA UNAVAILABLE.")
    (OUT/"captions.json").write_text(json.dumps(captions,indent=2)+"\n")
    (OUT/"captions.md").write_text("# Journal figure captions\n\n"+"\n\n".join(f"## Figure {c['number']}\n\n{c['caption']}" for c in captions)+"\n")
    metadata={"source_sha256":sha(Path(__file__)),"input_sha256":{p:sha(TABLES/p) for p in INPUTS},"output_sha256":{p:sha(OUT/p) for p in outputs},"captions_sha256":sha(OUT/"captions.json"),"figures":len(captions),"main_figures":9,"appendix_figures":3,"dpi":300}
    (ROOT/"results/statistics/journal_figures_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Rendered {len(captions)} journal plates / {len(outputs)} PNG, PDF and SVG files.")


if __name__=="__main__":main()
