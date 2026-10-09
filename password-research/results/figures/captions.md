# Figure captions

Each figure is saved as 300 dpi PNG, PDF and editable SVG. Titles and footers are part of the export; captions provide the full interpretation and denominator.

## Figure 1

**Evidence type:** Theoretical.

Mathematical search spaces N^L for four explicitly defined alphabets and all configured lengths. The vertical axis is log10 search-space size; the spacing therefore represents orders of magnitude. Printable ASCII uses 94 visible characters, excluding space. This calculation assumes independent uniform choices and does not estimate entropy of observed human passwords. No empirical denominator applies.

Files: [fig01_theoretical_search_space.png](fig01_theoretical_search_space.png), [fig01_theoretical_search_space.pdf](fig01_theoretical_search_space.pdf), [fig01_theoretical_search_space.svg](fig01_theoretical_search_space.svg).

## Figure 2

**Evidence type:** Empirical aggregate-spectrum analysis.

Left: released-frequency medians using either one vote per anonymous distinct bin or released account mass as the weight. Right: the minimum anonymous frequency-oracle budget sufficient to cover half of the mass within each known-length group (conditional G50). All eleven source groups 6–16 are included; no anomalous group is removed. Across these groups the released mass is 173,904,254 and the number of anonymous bins is 57,238,505. Neither weighting estimates a representative population of all users. Values inherit differential-privacy release distortion, length-policy/source effects and unknown password identities.

Files: [fig02_frequency_and_g50.png](fig02_frequency_and_g50.png), [fig02_frequency_and_g50.pdf](fig02_frequency_and_g50.pdf), [fig02_frequency_and_g50.svg](fig02_frequency_and_g50.svg).

## Figure 3

**Evidence type:** Empirical aggregate-spectrum analysis.

Left: conditional anonymous frequency-oracle coverage for six evenly spaced displayed length groups; each denominator is its own group's released mass. Right: whole-release comparator curves, with denominators LinkedIn full release: 174,292,189; Yahoo release: 69,301,337; LinkedIn pooled length releases: 174,296,614. Every curve ends within its available number of anonymous bins. The pooled length releases and full LinkedIn release differ in mass because they are separate noisy releases; they are not independent corpora. Lines connect evaluated budgets for readability. Frequency spectra determine an idealized oracle upper bound, not an executable ordering of identified passwords or an actual account-recovery experiment.

Files: [fig03_oracle_coverage.png](fig03_oracle_coverage.png), [fig03_oracle_coverage.pdf](fig03_oracle_coverage.pdf), [fig03_oracle_coverage.svg](fig03_oracle_coverage.svg).

## Figure 4

**Evidence type:** Empirical and theoretical, explicitly distinguished.

Left: observed released account-password corpus mass covered by an oracle knowing the source-defined length at budget K=10,000, alongside the mathematical probabilities min(K/N^L,1) for independently uniform N=26 and N=94 strings. Corpus coverage uses each released length group's account mass; combined mass is 173,904,254. Generation mechanisms of source bins are unobserved; this is not a verified human-generated cohort. This comparison matches nominal source lengths: source encoding and normalization are unavailable, so these are not exact matched-length samples. The random curves are calculated distributions, not observed recovery of synthetic passwords and not a cross-corpus operational guessing test. Right: known entropy of four synthetic generators at matching ASCII code-point/byte lengths. Repeating a random two-letter unit preserves 2 log2(26) bits; a random four-letter prefix followed by fixed '1!' and fixed padding preserves 4 log2(26) bits. There are 16,800 generated records across these four categories, 100 per cell and three independent secure-random replicates. Entropy is derived from generator definitions, not estimated from these samples. Fixed padding is a construction example, not an estimate of human behavior.

Files: [fig04_same_length_comparison.png](fig04_same_length_comparison.png), [fig04_same_length_comparison.pdf](fig04_same_length_comparison.pdf), [fig04_same_length_comparison.svg](fig04_same_length_comparison.svg).

## Figure 5

**Evidence type:** Empirical fixed-example observations.

Historical Pwned Passwords corpus-occurrence counts for eight selected public examples queried through the k-anonymity range API. Counts are displayed on a logarithmic axis and each label includes observed byte/ASCII character length. The panel's sample size is eight deliberately selected examples, not a random sample of human passwords. Global guessing ranks and a representative population denominator are unavailable. Counts establish corpus presence and relative counts among this panel; they do not prove current use, identify any person, estimate a causal benefit of a transformation or compare passwords against an implemented attack. Query provenance is retained in the dataset audit.

Files: [fig05_fixed_transformations.png](fig05_fixed_transformations.png), [fig05_fixed_transformations.pdf](fig05_fixed_transformations.pdf), [fig05_fixed_transformations.svg](fig05_fixed_transformations.svg).

## Figure 6

**Evidence type:** Empirical descriptive model diagnostic.

All six fixed exploratory models are evaluated on the same eleven leave-one-length-out folds: intercept-only, linear/quadratic/cubic logistic, and cubic B-splines with four/five uniformly spaced knots. The left panel shows polynomial fits; the right shows spline fits and the constant baseline, with observed fractions repeated for reference. Models are trained on ten released length groups and predict the eleventh. Transformations and knots use only training lengths; spline extrapolation is linear beyond training knots. No held-fold tuning is performed. The outcome is membership among the pooled top 10,000 anonymous bins, with fractional membership at the tied boundary and one weight per distinct bin. The cutoff and outcome labels use the full pool, including the held-out group, so this is not prospective validation. The held-group denominator totals 57,238,505 Distinct Anonymous Bins. Vertical values are percentages of distinct bins, not Account Mass coverage. The comparison was added after inspecting the original linear miscalibration and remains exploratory. This exercise diagnoses aggregate interpolation/extrapolation, not independent prediction of individuals or password strings. Human length plus structure remains DATA UNAVAILABLE.

Files: [fig06_length_model_diagnostic.png](fig06_length_model_diagnostic.png), [fig06_length_model_diagnostic.pdf](fig06_length_model_diagnostic.pdf), [fig06_length_model_diagnostic.svg](fig06_length_model_diagnostic.svg).

## Figure 7

**Evidence type:** Mathematical scenarios informed by local measurement.

Time to execute a fixed hypothetical budget K is K/r for online rates of 1, 10 and 100 checks/s, or K times local successful-verification latency for the highest configured hashing level. Offline illustration uses the median across four length-cell verification medians: pbkdf2_sha256: 62.472 ms; argon2id: 120.636 ms; bcrypt: 234.146 ms. These are single-process CPU measurements, not optimized GPU/ASIC attacker throughput, concurrency, universal offline rates or expected time to recover any password. Online rates are hypothetical and omit account-specific controls; rate limiting, detection and MFA can materially change the scenario. No authentication service is contacted. The fixed-budget arithmetic has no empirical population denominator.

Files: [fig07_threat_model_budget_time.png](fig07_threat_model_budget_time.png), [fig07_threat_model_budget_time.pdf](fig07_threat_model_budget_time.pdf), [fig07_threat_model_budget_time.svg](fig07_threat_model_budget_time.svg).

## Figure 8

**Evidence type:** Empirical controlled local benchmark.

All algorithm–cost–length cells of the local benchmark are shown with a shared log-latency scale. Solid lines/circles are successful verification medians; dashed lines/crosses are hashing medians, including secure salt generation. Error bars are verification IQRs from five timed repetitions after one warmup; they are not confidence intervals. PBKDF2 levels use 100k/300k/600k iterations. Argon2id levels use 19/32/64 MiB and passes 2/2/3, always parallelism 1. bcrypt uses log2 rounds 8/10/12; level 1 is a historical comparison. There are 36 cells and 180 timed observations per operation, plus excluded warmups and incorrect-input checks. All inputs are fresh synthetic ASCII strings; bcrypt inputs longer than 72 bytes are rejected and not benchmarked. The 64 MiB Argon2 level differs from RFC 9106's p=4 option. Differences do not rank algorithm security, establish equivalence or estimate attacker throughput. Actual versions and host metadata are in hashing_metadata.json.

Files: [fig08_hashing_latency.png](fig08_hashing_latency.png), [fig08_hashing_latency.pdf](fig08_hashing_latency.pdf), [fig08_hashing_latency.svg](fig08_hashing_latency.svg).

## Figure 9

**Evidence type:** Empirical synthetic lengths and theoretical entropy.

Left: 1,500 synthetic phrases from uniform independent sampling with replacement from the 7,776-word EFF vocabulary, 100 phrases per word-count cell in each of three independent CSPRNG replicates. Individual dots are replicate median lengths; the central marker is their median; horizontal bars are medians of the replicate 10th–90th length quantiles and are descriptive, not confidence intervals. Words are joined with one space, including spaces in character length, so hyphen-containing vocabulary entries remain unambiguous. The vertical coordinate w log2(7776) is a theoretical generator property. Right: log2 of exact counts of word tuples with each total length, computed by convolution of vocabulary word-length multiplicities, conditional on known word count and length. Dashed lines show unconditional entropy. Theoretical rare lengths are retained; no human-sentence model, observed guessing resistance or empirical entropy estimate is asserted.

Files: [fig09_random_word_passphrases.png](fig09_random_word_passphrases.png), [fig09_random_word_passphrases.pdf](fig09_random_word_passphrases.pdf), [fig09_random_word_passphrases.svg](fig09_random_word_passphrases.svg).
