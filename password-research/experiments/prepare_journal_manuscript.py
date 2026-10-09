"""Integrate the audited revision, safe new results and numbered bibliography."""
from pathlib import Path
import hashlib
import json
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"paper"
BASE=(PAPER/"manuscript_pre_journal_upgrade.md").read_text()
REFS=json.loads((PAPER/"references-verified.json").read_text())["references"]
BY_ID={r["id"]:r for r in REFS}
T=ROOT/"results/tables"


def section(start,end):
    return BASE.split(start+"\n",1)[1].split(end+"\n",1)[0].strip()


FIGMAP={"1":"1","2":"2","3":"3","4":"4","5":"S1","6":"S2","7":"S3","8":"8","9":"7"}
TABMAP={"1":"1","2":"2","3":"A1","4":"3","5":"A3","6a":"A4","6b":"A5","7":"A6","8":"A7"}


def clean(text):
    text=re.sub(r"\bFigure ([1-9])\b",lambda m:"Figure "+FIGMAP[m[1]],text)
    text=re.sub(r"\bTable (6a|6b|[1-8])\b",lambda m:"Table "+TABMAP[m[1]],text)
    text=text.replace("Section 5.4", "Section 5.7")
    text=text.replace("requested five operational guessing strategies", "five operational guessing strategies")
    text=text.replace("supplied in the project request", "selected as a fixed illustrative panel")
    text=text.replace("already supplied in the request", "selected for illustration")
    text=text.replace("the request", "the fixed teaching panel")
    text=text.replace("human popularity order", "source-popularity order")
    text=text.replace("released human oracle spectrum", "released corpus oracle spectrum")
    text=text.replace("adjacent human length groups", "adjacent source-length groups")
    text=text.replace("historical human-password group", "historical source-length group")
    text=text.replace("The comparison was requested after", "The comparison was added after")
    text=text.replace("securely generated", "CSPRNG-generated")
    text=text.replace("significant whitespace", "literal whitespace")
    text=text.replace(r"$$S=N^L,\qquad H=L\log_2N,\qquad E[G]=(N^L+1)/2,\qquad C(K)=\min(K/N^L,1).$$", "$$S = Nᴸ;   H = L log₂ N;   E[G] = (Nᴸ + 1)/2;   C(K) = min(K/Nᴸ, 1).$$")
    text=text.replace(r"$$C_{\ell}(K)=\frac{\sum_{i=1}^{K}f_{\ell,i}}{A_{\ell}}.$$", "$$Cℓ(K) = (∑ᵢ₌₁ᴷ fℓ,ᵢ) / Aℓ.$$")
    text=text.replace(r"$$\left[\max\left(0,\frac{C-E}{U+E}\right),\;\min\left(1,\frac{C+E}{U-E}\right)\right],\qquad U>E.$$", "$$[max(0, (C − E)/(U + E)), min(1, (C + E)/(U − E))],  U > E.$$")
    return text


def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join("---" for _ in headers)+" |\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)


def part_table(text):
    start=text.index("| ")
    match=re.search(r"\n\n",text[start:])
    end=start+match.start() if match else len(text)
    return text[start:end]


def bib_link(key):
    return f"[{BY_ID[key]['title']}]({BY_ID[key]['url']})"


def main():
    results=pd.read_csv(T/"journal_train_threshold_results.csv").set_index("model")
    cross=pd.read_csv(T/"journal_cross_dataset.csv")
    long=pd.read_csv(T/"journal_long_early.csv")
    uncertainty=pd.read_csv(T/"hashing_uncertainty.csv")
    captions=json.loads((ROOT/"results/journal_figures/captions.json").read_text())
    cap={c["number"]:c for c in captions}
    def figure(number):
        c=cap[str(number)]
        return f"![Figure {number}: {c['stem']}]({c['path']})\n\n**Figure {number}.** {c['caption']}"
    labels={"intercept_only":"Fold constant","linear_logistic":"Linear logistic","quadratic_logistic":"Quadratic logistic","cubic_logistic":"Cubic logistic","bspline_cubic_4knots":"Cubic spline, 4 knots","bspline_cubic_5knots":"Cubic spline, 5 knots"}
    gain=100*(1-results.loc["cubic_logistic","held_log_loss"]/results.loc["linear_logistic","held_log_loss"])
    model_table=table(["Length-only specification","Held log loss","Binary Brier loss","L15 observed/predicted","L16 observed/predicted"],
        [[labels[k],f"{r.held_log_loss:.9f}",f"{r.held_brier_loss:.9f}",f"{r.length15_observed_to_predicted_ratio:.3f}",f"{r.length16_observed_to_predicted_ratio:.3f}"] for k,r in results.iterrows()])
    cross_table=table(["Oracle positions K","LinkedIn full mass (%)","Yahoo all mass (%)","Yahoo − LinkedIn (pp)"],
        [[f"{int(r.budget):,}",f"{r.linkedin_full_coverage*100:.3f}",f"{r.yahoo_all_coverage*100:.3f}",f"{r.yahoo_minus_linkedin_percentage_points:.3f}"] for r in cross.itertuples()])
    hash_rows=[]
    settings={"pbkdf2_sha256":["100k iterations","300k iterations","600k iterations"],"argon2id":["19 MiB, t=2, p=1","32 MiB, t=2, p=1","64 MiB, t=3, p=1"],"bcrypt":["cost 8","cost 10","cost 12"]}
    for algorithm in ["pbkdf2_sha256","argon2id","bcrypt"]:
        for level in [1,2,3]:
            g=uncertainty.query("algorithm == @algorithm and cost_level == @level and operation == 'verify'")
            hash_rows.append([algorithm.replace("_sha256","-SHA256"),settings[algorithm][level-1],f"{g.median_ms.median():.3f}",f"{g.iqr_ms.min():.3f}–{g.iqr_ms.max():.3f}",f"{g.sample_variance_ms2.min():.3f}–{g.sample_variance_ms2.max():.3f}"])
    hash_table=table(["Algorithm","Parameters","Median of cell medians (ms)","Cell IQR range (ms)","Cell variance range (ms²)"],hash_rows)
    source_result=clean(section("### 5.1 Source consistency and length distribution","### 5.2 Coverage, rank association, and long early membership"))
    coverage_result=clean(section("### 5.2 Coverage, rank association, and long early membership","### 5.3 Nominal matched-length comparison and random-word controls"))
    matched=clean(section("### 5.3 Nominal matched-length comparison and random-word controls","### 5.4 Adjacent lengths and sensitivity"))
    words=matched.split("Uniform selection of 2–6",1)[1]
    matched=matched.split("Uniform selection of 2–6",1)[0].strip()
    sensitivity=clean(section("### 5.4 Adjacent lengths and sensitivity","### 5.5 Public transformation observations"))
    sensitivity=sensitivity.replace("| Sensitivity, lengths", "**Table 5. Descriptive sensitivity of the primary association.**\n\n| Sensitivity, lengths")
    # The mathematical DP scenarios belong with robustness methods; their
    # conditional numerical illustrations are retained in the same subsection.
    dp=sensitivity.split("**Mechanism-specific mathematics.**",1)[1]
    sensitivity=sensitivity.split("**Mechanism-specific mathematics.**",1)[0].strip()
    original_model=clean(section("### 5.6 Descriptive length model","### 5.7 Verification cost and mathematical threat scenarios"))
    cost=clean(section("### 5.7 Verification cost and mathematical threat scenarios","## 6. Discussion"))
    cost_scenarios=cost.split("**Table A6.",1)[1].split("**Table A7.",1)[0].strip()
    hardware=cost.split("**Table A7.",1)[1].strip()
    intro=section("## 1. Introduction","## 2. Background").split("The research questions are:",1)[0].strip()
    sources=clean(section("### 4.1 Data sources and ethical scope","### 4.2 Preprocessing and validation"))
    preprocessing=clean(section("### 4.2 Preprocessing and validation","### 4.3 Operational definitions of predictability"))
    definitions=clean(section("### 4.3 Operational definitions of predictability","### 4.4 Statistics and model diagnostic"))
    stats=clean(section("### 4.4 Statistics and model diagnostic","### 4.5 Controlled generators, public examples, and local hashing"))
    controls=clean(section("### 4.5 Controlled generators, public examples, and local hashing","## 5. Results"))
    entropy=clean(section("### 2.1 Entropy and exhaustive search","### 2.2 Guessing models and ranks"))
    models=clean(section("### 2.2 Guessing models and ranks","### 2.3 Verification cost and threat models"))
    related=clean(section("## 3. Related Work","## 4. Methodology"))
    discussion=clean(section("## 6. Discussion","## 7. Limitations"))
    limitations=clean(section("## 7. Limitations","## 8. Security and Ethical Considerations"))
    ethics=clean(section("## 8. Security and Ethical Considerations","## 9. Conclusion"))
    transforms=clean(section("### 5.5 Public transformation observations","### 5.6 Descriptive length model"))
    supported_coverage=coverage_result.split("**Empirical.**",1)[1]
    rho=supported_coverage.split("At K=10,000",1)[0].strip()
    original_coverage_table=coverage_result.split("**Empirical.**",1)[0]
    original_coverage_table=original_coverage_table.replace("Figure 3 displays supported coverage curves.","Figure 3 displays conditional curves; Figure 9 gives separate-source comparators.")
    original_coverage_table=original_coverage_table.replace("Figure 2 displays frequency medians and G50; ","")
    text=f'''# Password Length and Frequency Concentration in Differentially Private Releases

**A Reproducible Reanalysis with Controlled Generation Baselines**

**Imran Zakirov**

UVK “Prioritet”

Study snapshot: 5 October 2026. Manuscript revision: 6 October 2026.

## Abstract

Length expands the support of independently random passwords, but length alone does not identify their generation distribution. Published anonymous password spectra permit concentration analysis while withholding the identities and structures needed to evaluate realistic guessing. We reanalyzed the LinkedIn differentially private release, comprising 64 source-defined length groups with 174,296,614 units of released Account Mass, and a separate Yahoo spectrum with 69,301,337 units. Exact weighted ranks, conditional and pooled frequency-oracle curves, tie bounds, tail diagnostics and mechanism-specific conditional sensitivity calculations were combined with 26,700 frozen synthetic feature records and 36 local verifier-cost cells. Across LinkedIn lengths 6–16, Spearman association between length and released frequency was −0.335 under account-mass weighting and +0.041 under distinct-bin weighting; the latter reversed under tail filtering. Conditional top-10,000 coverage was 35.65%, 9.39% and 31.09% at lengths 6, 13 and 16. A revised diagnostic excluded each held length before defining its training frequency threshold; cubic logistic reduced held-length mean log loss by only {gain:.3f}% relative to linear logistic. Pooled top-10,000 exposure among source lengths at least 16 was 13.12%. These results describe concentration under a released-spectrum oracle, not observed compromises or causal benefits from adding characters. Uniform and explicit patterned generators show why equal length permits different known entropy. Historical recovery, unresolved length coding and the length-15 anomaly, privacy perturbation, and unavailable human structural covariates limit generalization. Independent replication of the length association and operational guessing remain unavailable.

**Keywords:** password length; frequency concentration; differential privacy; anonymous frequency spectrum; controlled generation; verifier cost; reproducibility.

## 1. Introduction

{clean(intro)}

The primary contribution is an auditable secondary analysis of a published differentially private release: contrasting two weightings, retaining nonmonotonic groups, exposing threshold sensitivity and separating identified release statistics from missing behavioral or attack evidence. The study does not claim a new password-guessing algorithm or population estimator. Its secondary controls illustrate how generator assumptions and verification costs change the question being answered.

## 2. Research Questions and Hypotheses

1. **RQ1 — Released concentration:** How does source-defined length associate with released frequency and conditional/pooled oracle concentration, under Account Mass and Distinct Anonymous Bin weighting?
2. **RQ2 — Generation assumptions:** At equal nominal length, how do released conditional concentration and explicitly uniform or patterned generation distributions differ?
3. **RQ3 — Sensitivity and marginal length:** How do observed adjacent groups and long-group thresholds differ, and what marginal gain follows analytically from adding an independent random choice?
4. **RQ4 — Descriptive transformations:** What occurrence differences exist in the fixed public teaching panel, and which population or causal conclusions remain unidentified?
5. **RQ5 — Source and evaluation scope:** What does a separate historical spectrum corroborate, and which structural, operational-guessing or temporal evaluations can the inputs support?

The original question about whether human structure dominates length after adjustment is not answerable with these inputs. Human length-plus-structure and interaction models are **DATA UNAVAILABLE**. The focused questions are descriptive or analytic; no confirmatory hypothesis test, participant-level p-value or population confidence interval is justified. An expectation of lower concentration in longer groups is explored against all observed groups, including counterexamples. Uniform-generator entropy identities are mathematical statements, not empirical human hypotheses. New revision analyses were fixed before their own execution but after the original results were known; they are exploratory and not preregistered.

## 3. Related Work

{related}

Composition-policy studies and empirical metric studies already examine interactions between policy, structure and measured strength. University-password work provides a real institutional context, while probabilistic and neural-model studies show that training distribution and evaluation assumptions affect measured ranks. These are prior findings, not experiments performed in this reanalysis. {bib_link('komanduri2011people')}, {bib_link('dellamico2010strength')}, {bib_link('weir2010metrics')}, {bib_link('mazurek2013university')}, {bib_link('zou2025llm')}

The closest statistical prior, Blocki and Liu's rigorous analysis of empirical datasets, distinguishes empirical frequency curves from population-aware guessing quantities and derives bounds under stated data assumptions. The present descriptive DP-release calculations do not replace that work or identify an unobserved population oracle. The contribution is a reproducible source-specific audit and controlled diagnostic extension, rather than a claim of methodological priority. {bib_link('blocki2023rigorous')}

The accompanying novelty matrix compares datasets, methods, findings and the limited incremental contribution of this study. A literature search is not a proof of priority; no “first study” claim is made.

## 4. Data and Ethical Scope

{sources}

An additional bounded search examined researcher-published human-study resources, the CUPS/Password Guessability Service ecosystem, and public research repository metadata. Candidate provenance, consent/access, license, plaintext/identifier presence, frequency, feature availability, representativeness and redistribution were checked. Public documentation or repository visibility alone was not treated as permission to process credentials. The inspected alternatives comprised document-only/meter-score studies, qualitative strategy tables, assigned-random typing records, or access-controlled resources; none verified a reusable human-choice feature table joined to genuine empirical frequency or independent operational outcomes. This search does not establish that no suitable dataset exists. For this revision, the required structural validation dataset remains **DATA UNAVAILABLE**. Candidate-level evidence and exact retrieval records accompany the dataset audit.

No raw breached credential corpus was acquired. No restricted human-password dataset was locally analyzed or redistributed. Yahoo is retained as a separate overall-concentration comparator; it supplies no length covariate. The LinkedIn full and length-partitioned releases are related outputs from the same underlying source and are not independent validation datasets.

## 5. Methodology

### 5.1 Data sources and analysis configuration

The original source snapshot and analysis configuration are preserved. Primary lengths are 6–16; all 64 supplied lengths are retained for pooled spectra and extended descriptive checks. Fixed oracle budgets are 10, 100, 1,000, 10,000, 100,000 and 1,000,000 wherever supported. The original “long and early” definition is length at least 16 and pooled top 10,000. The post-result revision reports the complete grid of minimum lengths 12, 14, 15, 16, 20, 24 and 32 at all six budgets, retaining the original cell. These are sensitivity scenarios, not optimized thresholds.

### 5.2 Preprocessing

{preprocessing}

### 5.3 Operational definition of predictability

{definitions}

“Predictability” here means released frequency concentration under stated oracle assumptions. An early anonymous bin is not an identified recovered password. The reported length/frequency units are compressed source cells with multiplicity; dependence, duplicates and transformation families cannot be traced without identities. No participant bootstrap is used to manufacture independent samples from those counts.

### 5.4 Attack models and leakage control

{models}

Uniform brute-force coverage is computed analytically; enumeration is not executed. Frequency ordering is an identityless descriptive oracle for the same release. Training-based frequency, dictionary, rule-based and statistical operational guessing are **NOT RUN**. The input cannot separate exact duplicates, near variants or transformation families into independently identified train/test corpora. No source-level operational generalization is therefore claimed. A future consent-created study would need family/source-separated evaluation and independently trained ordering before any recovery curve could be interpreted as attack performance.

### 5.5 Synthetic controls and public observations

{entropy}

{controls}

For the injective uniform controlled generators, Shannon entropy and min-entropy coincide: H = H∞ = log₂ S. Fixed repetition or padding creates no additional random choices. For word count w and joined length L, the conditional word-tuple distribution is uniform over its counted support, so both conditional entropies equal log₂ Cw(L). The analysis does not estimate empirical human entropy from a small synthetic sample, nor train a structure detector to predict a label defined by that same detector.

### 5.6 Statistical analysis and revised model diagnostic

{stats}

The new diagnostic removes held-group participation in cutoff construction. For each held length h, only the other ten primary length histograms define the frequency boundary for 10,000 training bins. The frozen rule qh(f) equals one above that cutoff, the training tie fraction at it, and zero below it. It is then applied to training and held released frequencies. Distinct-bin-weighted positive/negative mass gives exact sufficient statistics for the length-only binary objective. Scaling, polynomial bases and uniform spline knots use training lengths only; the six original fixed specifications and C=10⁶ are retained without held-fold tuning.

$$qₕ(f) = 1 if f > tₕ;   qₕ(f) = αₕ if f = tₕ;   qₕ(f) = 0 if f < tₕ.$$

The outcome is membership above a training-defined released-frequency threshold, not membership among a known string order or the original full-pool top 10,000. Cutoffs differ across folds. Distinct-bin-weighted binary log loss, binary Brier loss, group calibration error and observed/predicted ratios are reported against each fold's training-constant baseline. Binary Brier expectation is q(1−p)²+(1−q)p², retaining q(1−q); it differs from merely squaring group prevalence error. Equal-group loss is also exported. AUC and precision/recall are not pooled across changing fold targets with only one predicted score per held group. Eleven aggregate groups do not constitute independent-user or independent-source validation. Endpoint lengths 6 and 16 require extrapolation. The original diagnostic is preserved in Appendix C, with its full-pool target limitation explicit.

### 5.7 Robustness and privacy perturbation

{sensitivity}

**Mechanism-specific mathematics.** {dp}

### 5.8 Verifier-cost measurement and threat model

The saved benchmark measures complete local matching-verification and hashing calls. Incorrect-input checks establish correctness and were not timed. Variance is reconstructed from the special n=5 linear-quantile identity: the saved minimum, q25, median, q75 and maximum are exactly the five sorted observations at saved precision. Their mean and sample variance (divisor n−1) are recoverable, but temporal order, paired covariance, scheduling and tail stability are not. No timing rerun or observation was invented. Function specifications are documented by RFC 8018 (PBKDF2) and RFC 9106 (Argon2). {bib_link('rfc8018')}, {bib_link('rfc9106')}

For an online service, define B(T) as its admitted-request budget over horizon T and r(t) as a stipulated admitted-rate ceiling in the same account/service scope. A symbolic constraint is Kon(T) ≤ min(B(T), floor(∫₀ᵀ r(t) dt)). Throttling, lockout and delays alter this admitted budget. Authentication requires both a correct password event P and other required conditions F, giving Pr(authentication) = Pr(P) Pr(F given P); factor independence is not assumed.

For a purely hypothetical constant-cost offline model with w equal always-busy workers and complete-call cost c, finite-budget time is ceil(K/w)c. The measured experiment supplies only serial matching-call latency for w=1. A full-memory model also requires w Mcall ≤ Mavailable, excluding additional overhead; this necessary memory condition does not predict throughput. Argon2 lanes p are not independent workers w. GPU/ASIC capacity, parallel cost and nonmatching latency are **NOT RUN**. Multiplying a selected median by K is a constant-cost illustration, not the expected stochastic sum or password-recovery time. The full symbolic assumptions are retained in the hashing supplement and Figure S3.

## 6. Results

### 6.1 Length and released frequency

{source_result}

**Empirical.** {rho}

Figure 2 retains every primary group. Appendix Table A1 includes complete conditional and pooled coverage/G50 values; no length-15 observation is silently removed.

{figure(2)}

### 6.2 Nominal length-matched corpus/generator comparison

{matched}

For the repeated two-letter generator, adding displayed repetition beyond its two random positions adds zero entropy; the four-letter-prefix construction behaves likewise beyond its fixed random choices. A new independent EFF word adds log₂(7776) = 12.9248 unconditional bits, but an arbitrary additional displayed character in a phrase has no comparable universal multiplier. These are generator-specific statements, not measured human append-character effects.

{figure(1)}

{figure(4)}

### 6.3 Length models and missing structural adjustment

**Human length × structure: DATA UNAVAILABLE.** There is no estimated Model B (length plus human structure) or Model C (human interactions). Synthetic literal features do not substitute for this experiment. Table 4 reports the feasible revised length-only calibration diagnostic.

**Table 4. Training-only-threshold length-model comparison.** Eleven folds; 57,238,505 held Distinct Anonymous Bins in total. Fold-specific binary targets differ; all six fixed specifications are reported. Values are descriptive without population intervals.

{model_table}

Linear held-length log loss is {results.loc['linear_logistic','held_log_loss']:.9f}, versus {results.loc['intercept_only','held_log_loss']:.9f} for fold constants. Cubic reduces it to {results.loc['cubic_logistic','held_log_loss']:.9f}: an absolute reduction of {results.loc['linear_logistic','held_log_loss']-results.loc['cubic_logistic','held_log_loss']:.9f}, or {gain:.3f}% relative to linear. Its mean binary Brier loss is {results.loc['cubic_logistic','held_brier_loss']:.9f}, versus {results.loc['intercept_only','held_brier_loss']:.9f} for constants. The small loss differences reflect severe class imbalance; they do not imply near-perfect classification. Linear underpredicts length 15/16 by {results.loc['linear_logistic','length15_observed_to_predicted_ratio']:.2f}/{results.loc['linear_logistic','length16_observed_to_predicted_ratio']:.2f}, while cubic ratios remain {results.loc['cubic_logistic','length15_observed_to_predicted_ratio']:.2f}/{results.loc['cubic_logistic','length16_observed_to_predicted_ratio']:.2f}. Both splines worsen aggregate loss relative to linear. No convergence warnings were recorded across 66 new fits. Figure 5 displays all specifications. The results cannot be directly compared with the original full-pool-target loss as if the target had remained fixed.

{figure(5)}

### 6.4 Oracle curves and long-group sensitivity

Figure 3 reports conditional curves for all eleven primary lengths with each group's own denominator. At 10,000 positions, conditional mass concentration ranges nonmonotonically across source groups; values at lengths 6, 13 and 16 are 35.65%, 9.39% and 31.09%. Pooled and conditional ranks answer different questions. Dictionary/rule/statistical train/test recovery curves are **NOT RUN**.

{figure(3)}

The revision's complete 42-cell long-threshold grid is shown in Figure 6. At pooled K=10,000, exposure among source lengths at least 12, 14, 15, 16, 20, 24 and 32 is respectively {', '.join(f'{100*long[(long.minimum_source_length==v)&(long.budget==10000)].account_fraction.iloc[0]:.2f}%' for v in [12,14,15,16,20,24,32])}. The original ≥16 cell is 131,591 / 1,003,139 = 13.12% of eligible released Account Mass. These changing denominators explain why a threshold change is not a within-person length effect. Length 15's unusual spectrum contributes to changes around that boundary. Zero exposure at a fixed budget does not establish universal strength. Expected tie allocation, sharp tie bounds and distinct-bin fractions for every cell accompany the table export.

{figure(6)}

### 6.5 Complexity transformations

The complete fixed eight-example HIBP panel appears in Appendix B, Table A3 and Figure S1. All examples were observed, with materially different historical occurrence counts. Password1! has more occurrences than Password! in the frozen snapshot despite its greater length. This auxiliary illustration does not establish transformation prevalence, global rank or causal resistance gain. Naturally occurring transformation-family analysis and family-separated prediction remain **DATA UNAVAILABLE**.

### 6.6 Passphrases

Uniform selection of 2–6 {words}

{figure(7)}

### 6.7 Hashing and verifier cost

All 36 cells completed in the recorded Darwin 24.1.0 arm64/Python 3.12.14 environment, using OpenSSL 3.5.8, argon2-cffi 25.1.0 and bcrypt 5.0.0. There were 216 hash calls, 216 matching checks and 36 incorrect-input correctness checks including warmups. The serial protocol lasted about 31.58 seconds. Exact CPU/RAM were not saved at that run. A separate later observation identifies Apple M1, MacBookAir10,1, 8 GiB RAM, eight physical/logical cores, macOS 15.1.1 build 24B91; it is context, not independently verified historical machine continuity or thermal/load state.

**Table 6. Local matching-verification summaries.** Each configuration comprises four ASCII length cells (8/16/32/64 bytes), each with five timed observations after one warmup. IQR and sample-variance ranges are across those cell-specific values, not a pooled variance or uncertainty interval for the median of medians.

{hash_table}

The 72-row operation-by-cell uncertainty table preserves hash and verification statistics, means and extrema. A retained Argon2id 32 MiB/t=2/p=1 hash cell at eight bytes has median 39.585 ms, IQR 0.710 ms, maximum 299.644 ms and sample variance 13,546.425 ms². Its cause is unknown. This outlier shows why medians and quartile spans cannot summarize tail behavior fully. Figure 8 makes it visible. Five observations do not establish negligible length effects, temporal stability, p95/p99 latency or expected service capacity.

Parameter levels are unequal workloads and do not rank algorithms globally. The largest Argon2id cell has p=1, unlike RFC 9106's p=4 lower-memory option. Low-cost PBKDF2/bcrypt cells are experimental contrasts, not deployment recommendations. bcrypt above 72 bytes was rejected, not benchmarked. No optimized parallel or GPU/ASIC results were measured. {bib_link('provos1999bcrypt')}, {bib_link('rfc9106')}, {bib_link('owasp2026storage')}

{figure(8)}

### 6.8 Cross-dataset comparison and temporal limits

**Table 7. Separate-source overall concentration.** Each spectrum retains its own released mass denominator and frequency order. Difference is Yahoo minus LinkedIn in percentage points; it is not a population estimate or temporal effect.

{cross_table}

Both historical releases show nonuniform concentration: at K=10,000, Yahoo exceeds LinkedIn by 2.312 percentage points, and at K=1,000,000 by 7.341 points. Figure 9 reports all six budgets. Yahoo's missing length covariate makes independent replication of the length relationship **DATA UNAVAILABLE**. The distinct collection/recovery processes, services and periods preclude a temporal policy-era inference. No incompatible sources are pooled, and the linked LinkedIn full/length outputs are not counted as separate validation populations.

{figure(9)}

## 7. Discussion

{discussion}

The training-only-cutoff extension removes a specific held-group construction dependency, while leaving the original observational limits intact. Its endpoint calibration improves under polynomial flexibility, but cubic's {gain:.3f}% relative log-loss improvement is small, splines do not improve loss, and the targets vary by fold. Consequently, the extension supports a diagnostic description rather than a claim that length predicts future users or that structure is unnecessary. The absence of structural data is an explicit boundary on the conclusion, not evidence for or against structural dominance.

Practically, engineers should combine the applicable minimum length with whole-password compromised/common-value screening, admitted-request controls, password-manager-compatible input and MFA appropriate to the context. Selection unpredictability and verifier work factor are separate layers; successful local verification latency does not describe optimized cracking capacity or authentication compromise. These conclusions retain NIST's minimum of 15 characters for centrally verified single-factor passwords and the permitted eight-character minimum when used only as part of MFA. {bib_link('nist2025authentication')}, {bib_link('owasp2026authentication')}

## 8. Threats to Validity

{limitations}

**Distinct uncertainty sources.** Privacy perturbation changes the released spectrum under the documented mechanism. Sampling uncertainty would require a supported population sampling model, which is absent. Historical selection reflects service/policy/demographic context; recovery selection reflects how the LinkedIn subset became available. Source-processing artifacts include unverified length units, normalization, duplication and truncation hypotheses. These sources are not combined into one confidence interval. Conditional L1 scenarios constrain certain indicator counts under their assumptions but do not quantify all five sources or identify the actual unperturbed Spearman association.

The new held diagnostic excludes a length from threshold construction, not a source or family of independently identified passwords. Training and test groups remain components of the same historical release and privacy process. Near-duplicate/family leakage cannot be audited from anonymous bins; no source-generalization claim is made. Its AUC/precision/recall are not replaced by inappropriate pooled metrics. Minima/budgets and nonlinear specifications are post-result exploratory sensitivity choices, fully reported to limit selective presentation.

## 9. Ethical and Security Considerations

{ethics}

### 9.1 Author contributions and AI-use disclosure

The listed author, Imran Zakirov, supplied the research topic and revision directions. OpenAI Codex assisted with research planning, literature searches and organization, source acquisition, Python code, experiment execution, statistical calculations, figures, internal checking and manuscript drafting/formatting. Specialist review reports in the supplement are internal AI-assisted simulations and separate computational checks, not external peer review or independent human scientific approval. Additional independent experimental contributions have not been documented in this workflow. The listed author is responsible for verifying the sources, methods, interpretation, factual accuracy and final approval before submission. AI is not listed as an author. This disclosure does not by itself establish eligibility under a venue's independent-authorship or AI policy.

### 9.2 Funding

This research received no external funding, as confirmed by the author.

### 9.3 Competing interests

The author declares no competing interests, as confirmed in the preparation of this manuscript.

## 10. Reproducibility and Data Availability

**Public sources.** Versioned LinkedIn/Yahoo anonymous aggregate releases and provider metadata are publicly available under the documented CC0 terms. The attributed EFF ordinary vocabulary is publicly available under its published terms. Source URLs, versions, license records, sizes and SHA-256 values accompany the acquisition manifest. The fixed HIBP snapshot is retained for downstream arithmetic; live service counts can change.

**Supplied artifacts.** The accompanying archive contains code, analysis configuration, frozen synthetic feature inputs, derived CSV tables, plot exports, mathematical derivations, reference audit and source manifests. It is computationally reproducible from the supplied frozen artifacts under the recorded environment; no restricted human corpus is required. Synthetic strings were transient and discarded, so secure regeneration is a new sample, not recovery of the original strings. Original timing reruns produce new measurements. No raw breached credentials, identifiers, salts or verifier records are released. No restricted human structural dataset is included; that analysis remains DATA UNAVAILABLE.

**Code availability.** Code is supplied with this manuscript under MIT; original prose and figures use CC BY 4.0, with third-party inputs retaining their terms. No public repository URL, archived repository DOI or Git commit has been assigned at this stage. Source/input/output digests identify the delivered snapshot; the artifact manifest and archive SHA-256 provide a verifiable substitute for this local delivery, without implying a deposited public repository. Publication should accompany a stable repository or archive deposit.

The analysis uses Python 3.12.14 and the pinned environment in requirements.lock (NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2). Analysis/cell-order seed is 20261005; CSPRNG sampling is intentionally unseeded. From a copied project directory, a fresh reader can run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python experiments/reproduce.py
```

The command checks tests/source archives, recomputes the original aggregates and controls, DP/nonlinear diagnostics, training-only-threshold extension, hashing uncertainty and both plot sets, then checks every numerical CSV against its saved byte hash. Expected outputs are the results/tables and results/statistics directories, nine main/three appendix journal plates and the reproduction_check record. Timing calls and live acquisition are excluded from this command. The portable Word build uses requirements-document.txt and `experiments/build_docx.py`; its embedded-figure audit is recorded separately. On another numerical platform, any byte mismatch requires explicit inspection, not silent baseline replacement.

## 11. Conclusion

Source-defined length has a moderately negative Account-Mass-weighted association with released frequency in the analyzed LinkedIn spectrum, while the weak distinct-bin association is fragile to low-frequency and length-15 diagnostics. Conditional oracle concentration is nonmonotonic, and long-group membership changes with the specified minimum and budget. These results answer the descriptive release question; they do not identify a causal per-character resistance multiplier or current-user behavior.

For specified independent uniform generators, each new random choice increases support and entropy according to its alphabet or vocabulary. Fixed repetition/padding adds displayed length without new entropy. The nominal corpus/generator comparison therefore separates distributions and assumptions; it is not an observed human/random attack trial. A separate Yahoo spectrum corroborates overall concentration but cannot replicate the length association. Human structural adjustment, operational guessing and temporal change remain open.

The engineering implication is to retain the applicable NIST minimum length alongside compromised-password screening, throttling, suitable salted storage, manager-compatible inputs and MFA. The results do not weaken NIST SP 800-63B-4's 15-character single-factor requirement or justify treating length as a sufficient guarantee of unpredictable choice. {bib_link('nist2025authentication')}, {bib_link('owasp2026authentication')}

## References

{{REFERENCES}}

## Appendix A. Detailed coverage and status

{original_coverage_table}

**Table A2. Evidence and execution status.** Frequency-oracle concentration is not operational recovery performance.

{table(['Layer','Execution','What it establishes'],[
['Uniform baseline','RUN THEORETICAL','Specified generator support/entropy/coverage'],
['Anonymous frequency oracle','RUN AGGREGATE','Released conditional and pooled concentration'],
['Training-based frequency guesser','NOT RUN','Independent identified train/test order unavailable'],
['Dictionary/rule/statistical guessing','NOT RUN','No operational recovery curves'],
['Length-only diagnostic','RUN AGGREGATE','Exploratory calibration of released-frequency targets'],
['Human length + structure / interaction','DATA UNAVAILABLE','No validated human feature/outcome join'],
['Cross-source length replication / temporal change','DATA UNAVAILABLE','Yahoo supplies overall spectrum only'],
['Controlled generators / local verifier costs','RUN CONTROLLED','Explicit synthetic feature and local timing observations']])}

## Appendix B. Auxiliary transformation observations

{transforms}

{figure('S1')}

## Appendix C. Original full-pool model diagnostic

{original_model}

{figure('S2')}

## Appendix D. Mathematical budget illustrations and hardware context

**Table A6. {cost_scenarios}

These numerical illustrations use a selected constant-cost proxy; they are not expectations of random elapsed times, measurements of admitted online traffic or optimized offline rates.

{figure('S3')}

**Table A7. {hardware}

## Appendix E. Supplementary material index

- Current-paper audit, missing/completed experiments and mission-status mapping: docs/journal-current-paper-audit.md and docs/journal-upgrade-report.md.
- Candidate dataset/access/ethics audit: docs/journal-dataset-audit.md and its saved metadata.
- Statistical methods, all sensitivity cells and internal statistical review: docs/journal-statistical-report.md, docs/journal-reviewer-2.md, docs/dp-sensitivity.md and docs/nonlinear-models.md.
- Hashing protocol, exact uncertainty reconstruction and internal practitioner review: docs/journal-hashing-report.md, docs/hashing-method.md and docs/journal-reviewer-3.md.
- Citation support, metadata caveats and novelty matrix: docs/journal-reference-audit.md and docs/journal-novelty-matrix.md.
- Internal cybersecurity reviewer and editor simulations: docs/journal-reviewer-1.md and docs/journal-editor-simulation.md. These are AI-assisted internal assessments.
- Claim-evidence matrix and final checklist: docs/journal-claim-evidence.md and docs/journal-submission-checklist.md.
- Configuration/source/version hashes: config.json, analysis/journal-analysis-plan.json and results/statistics/*metadata.json.
- Original and journal plots: results/figures and results/journal_figures; PNG/PDF/SVG exports and full captions.
'''
    # Existing links to local supplements become portable text, not user-specific
    # external filesystem relationships. Numbered source citations are verified
    # against exact URLs/aliases and DOI records, then ordered by first mention.
    lookup={}
    def norm(url):return url.strip().rstrip('/').split('#')[0].lower().replace('http://','https://')
    for r in REFS:
        for url in [r['url'],*r.get('url_aliases',[])]:lookup[norm(url)]=r['id']
        if r.get('doi'):lookup[norm('https://doi.org/'+r['doi'])]=r['id']
    for alias,key in {
        'https://www.eff.org/dice':'eff2016words',
        'https://www.usenix.org/system/files/conference/usenixsecurity15/sec15-paper-ur.pdf':'ur2015biases',
        'https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_melicher.pdf':'melicher2016neural',
        'https://arxiv.org/pdf/2002.01513':'harsha2021bicycle',
    }.items():lookup[norm(alias)]=key
    numbers={}; unmatched=[]
    def link(match):
        label,url=match.groups()
        if not url.startswith(('https://','http://')):
            if match.start() and text[match.start()-1]=='!':return match.group(0)
            return label
        key=lookup.get(norm(url))
        if key is None:
            unmatched.append((label,url));return match.group(0)
        if key not in numbers:numbers[key]=len(numbers)+1
        return f"[{numbers[key]}]({BY_ID[key]['url']})"
    text=re.sub(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)",link,text)
    if unmatched:raise ValueError(f'Unmapped source citations: {unmatched}')
    bibliography=[]
    for key,n in numbers.items():
        r=BY_ID[key]; venue=r.get('journal') or r.get('booktitle') or r.get('institution') or r.get('publisher','')
        bits=[f"[{n}] {r['author_display']}.",f"({r.get('year','n.d.')}).",r['title']+'.']
        if venue:bits.append(venue+'.')
        if r.get('volume'):bits.append('Vol. '+r['volume']+'.')
        if r.get('pages'):
            bits.append(('Article ' if r.get('journal') and r['pages'].isdigit() and len(r['pages'])>=5 else 'pp. ')+r['pages'].replace('--','–')+'.')
        if r.get('number'):bits.append(str(r['number'])+'.')
        if r.get('doi'):bits.append(f"[doi:{r['doi']}](https://doi.org/{r['doi']}).")
        else:bits.append(f"[Source]({r['url']}).")
        if r['type'] in ['misc','techreport']:bits.append('Accessed 6 October 2026.')
        bibliography.append(' '.join(bits))
    # A references section contains cited entries, not a dump of unused records.
    text=text.replace('{REFERENCES}','\n\n'.join(bibliography))
    text=text.replace('The model diagnostic’s cutoff uses held-group information, has severe class imbalance, and fails at longer groups.', 'The preserved original Appendix C diagnostic uses held-group information in cutoff construction. The revised diagnostic excludes that construction dependency but remains an aggregate within-source evaluation with severe class imbalance and endpoint errors.')
    text=text.replace('Independent privacy perturbation means', 'Separate privacy releases mean')
    text=text.replace('between the independent full release and', 'between the separately sanitized full release and')
    text=text.replace('data. DP diagnostic supplement', 'data (see the supplementary DP diagnostic report).')
    text=text.replace('Continuous attempts before service controls', 'Assumed constant admitted rate; no lockout/delay modeled')
    text=text.replace('Same mathematical assumption', 'Same admitted-rate assumption')
    text=text.replace('Local PBKDF2 illustration |', 'PBKDF2-SHA256, 600k |').replace('Local Argon2id illustration |', 'Argon2id, 64 MiB/t=3/p=1 |').replace('Local bcrypt illustration |', 'bcrypt, cost 12 |')
    text=text.replace('One local serial worker', 'Serial median proxy over four length-cell medians')
    text=text.replace('This project contributes', 'This study contributes')
    text=re.sub(r"The model diagnostic['’]s cutoff uses held-group information, has severe class imbalance, and fails at longer groups\.", "The preserved original Appendix C diagnostic uses held-group information in cutoff construction. The revised diagnostic excludes that dependency but remains an aggregate within-source evaluation with severe class imbalance and endpoint errors.", text)
    dp_start=text.index('**Conditional sensitivity example, length 15.**')
    dp_end=text.index('### 5.8 Verifier-cost measurement and threat model', dp_start)
    radius_block=text[dp_start:dp_end].strip()
    radius_block=radius_block.replace('| Assumed true mass |', '**Table A8. Conditional radius illustrations for length 15.** These are assumption-dependent envelopes, not release-specific confidence intervals.\n\n| Assumed true mass |')
    text=text[:dp_start]+'Conditional numerical radius illustrations are reported in Appendix E, Table A8.\n\n'+text[dp_end:]
    text=text.replace('## Appendix E. Supplementary material index','## Appendix E. Conditional privacy-sensitivity illustrations\n\n'+radius_block+'\n\n## Appendix F. Supplementary material index')

    text=re.sub(r'\n{3,}', '\n\n', text)

    # Move robustness result table from methods into Results without moving the
    # DP mechanism and conditional radius derivation.
    sens_start=text.index('**Table 5.')
    sens_end=text.index('**Mechanism-specific mathematics.**',sens_start)
    sens_block=text[sens_start:sens_end].strip()
    text=text[:sens_start]+"Association robustness varies weighting, f≥2/f≥5 retention, top-100 removal, and exploratory length exclusions. Table 5 in Results reports every specified primary comparison.\n\n"+text[sens_end:]
    anchor='Figure 2 retains every primary group.'
    text=text.replace('### 6.4 Oracle curves and long-group sensitivity',sens_block+'\n\n### 6.4 Oracle curves and long-group sensitivity')
    text=text.replace(figure(1),'').replace('### 5.6 Statistical analysis and revised model diagnostic',figure(1)+'\n\n### 5.6 Statistical analysis and revised model diagnostic')
    text=text.replace(figure(3),'').replace(figure(2),figure(2)+'\n\n'+figure(3))
    (PAPER/'manuscript.md').write_text(text)
    (PAPER/'references-formatted.md').write_text('# Formatted references\n\n'+'\n\n'.join(bibliography)+'\n')
    (ROOT/'results/statistics/manuscript_build_metadata.json').write_text(json.dumps({
        'manuscript_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'reference_json_sha256':hashlib.sha256((PAPER/'references-verified.json').read_bytes()).hexdigest(),
        'cited_references':numbers,'unused_bibliography_keys':sorted(set(BY_ID)-set(numbers)),
        'author':'Imran Zakirov','affiliation':'UVK “Prioritet”','main_figures':9,'appendix_figures':3,
        'title_scope':'descriptive reanalysis of DP spectra; no human structural or operational guessing claim',
        'funding_and_conflicts':'user confirmed no','ai_disclosure':'broad research/code/execution/analysis/drafting disclosed'},indent=2)+'\n')
    print(f"Prepared journal manuscript: {len(text.split())} words; {len(numbers)} cited references; 9 main +3 appendix figures.")


if __name__=='__main__':main()
