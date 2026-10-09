# Figure index

The final manuscript embeds the nine strongest main plates and three appendix plates from `results/journal_figures/` (PNG at 300 dpi, PDF and editable SVG). Full captions are in that directory. The new plots use saved numerical tables; no strings or candidate orders are inferred from source spectra.

The original nine figures remain in `results/figures/` with their original source/input/output digests. They document the earlier analysis and are not silently substituted for the journal plates.

Recreate both sets with the offline entry point:

```sh
python experiments/reproduce.py
```

Journal plates alone: `python experiments/make_journal_figures.py`.
