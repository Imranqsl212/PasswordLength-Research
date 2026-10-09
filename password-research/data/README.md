# Data inventory and attribution

The input archives contain anonymous frequency histograms, not credential dumps. Keep them separate from derived outputs.

| File | Provenance | Redistribution terms |
| --- | --- | --- |
| raw/linkedin.zip | Figshare article 7350287 v2, file 14027465; https://doi.org/10.6084/m9.figshare.7350287.v2 | CC0 |
| raw/yahoo.tar.gz | Joseph Bonneau, Figshare article 2057937 v1, file 3636168; https://doi.org/10.6084/m9.figshare.2057937.v1 | CC0 |
| raw/eff_large_wordlist.txt | Electronic Frontier Foundation, 2016 Long Wordlist; https://www.eff.org/files/2016/07/18/eff_large_wordlist.txt | CC BY 4.0 under https://www.eff.org/copyright; original list unchanged |
| raw/hibp_public_panel.json | Exactly eight public illustrative strings supplied in request; range API snapshot 5 Oct 2026 | Provider documentation; cite https://haveibeenpwned.com/API/v3#PwnedPasswords |
| raw/*metadata.json | Public repository metadata, including license/version/MD5 | Source metadata |
| derived/length_frequency_histogram.csv | LinkedIn source-defined length, released frequency, multiplicity | Derived anonymous counts |
| derived/synthetic_features.csv | Frozen features from 26,700 newly generated specimens | Project-generated synthetic data; no strings retained |

Each aggregate row is frequency followed by multiplicity. Account mass is the sum of frequency multiplied by multiplicity. Summing frequency alone gives an incorrect denominator. Anonymous bins are not identified passwords or necessarily unperturbed distinct strings. Histograms are already privacy-sanitized.

Archives are read directly without extraction. SHA-256 and repository MD5 values validate inputs. raw/source_manifest.json records source URLs and hashes. The two source releases describe historical observations; neither is representative of all 2026 users. See ../docs/dataset-audit.md for collection methods, discrepancies, and privacy/recovery/length uncertainties.

The HIBP panel contains only the already public teaching examples. No personal input, usernames, emails, identity-linked password hashes, salts, or verifier records are present. Counts cannot establish current use or global guess ranks.

Only secure random generator feature snapshots are kept. The temporary-string digest identifies a generation run but cannot reconstruct its texts. Reuse the frozen snapshot to reproduce statistics exactly; fresh CSPRNG samples intentionally differ. EFF separators are spaces because a few unchanged words contain hyphens.
