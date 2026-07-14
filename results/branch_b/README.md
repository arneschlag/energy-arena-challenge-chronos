# Branch B experiment results

This directory contains saved experiment outputs for Branch B, separated from the
training/evaluation code in `experiments/`.

Contents:

- `out_v2/`: original Branch-B zero-shot suite.
- `out_ft_v2/`: original Branch-B LoRA finetuning suite.
- `out_repro/`: zero-shot runs for the three Branch-A-inspired repro configs.
- `out_ft_repro/`: LoRA finetuning runs for the three Branch-A-inspired repro configs.
- `out_v2_run2/`: second zero-shot run for the selected G1 configs.
- `out_ft_v2_run2/`: second LoRA run for selected G1 configs, strict runs only.
- `out_ft_v2_run2_pad2/`: Pad2 ROCm recovery for `G1_C2.1` and `G1_C2.2`.
- `out_repro_run2/`: second zero-shot run for the three repro configs.
- `out_ft_repro_run2/`: second LoRA run for the three repro configs.

Each run folder stores Chronos samples as Parquet plus an aggregated `results.csv`.
The code that generated and scores these artifacts lives in `experiments/`.

Note: `out_ft_v2_run2_pad2` uses two constant zero known covariates as a ROCm
compatibility workaround. Treat those rows as recovery results, not as strict
bit-for-bit repeats of the non-padded configuration.
