# dental-rad-cli

Chairside dental radiograph documentation aid. Takes bitewing (BW) or
periapical (PA) X-rays, produces structured findings JSON, an annotated
PNG, and a note-draft text — suitable for copy-paste into clinical
notes. Two findings in v0: alveolar bone loss (severity + horizontal /
vertical pattern) and interproximal caries (with depth staging).

**This is a documentation aid, not a diagnostic device.** Every finding
is a candidate the doctor verifies before it enters a chart. No FDA
path; no autonomous claims.

## Status

**Pre-v0 prototype.** Hour-0 in progress: training pipeline cleanroom-
reimplemented from arxiv 2506.20522 methodology, models training on
RTX 4090 (hostname: `pickles`), first signal expected on a held-out
set of scrubbed bitewings.

Design doc (decisions + rationale):
`../work-root/output/proposals/2026-05-11-dental-rad-cli-v0-design.md`

## Scope

- **In scope (v0):** tooth detection + FDI numbering, CEJ / bone-crest /
  apex keypoints, tooth + bone segmentation, per-site % bone loss,
  horizontal / vertical pattern classification, AAP staging summary,
  interproximal caries detection (with depth staging if dataset supports),
  annotated PNG render, structured JSON output, template note-draft text.
- **Out of scope (v0):** periapical lesions, overhanging restorations,
  calculus, furcation, resorption, pulp findings, impactions, panoramic
  films, multi-image visit aggregation, web UI, Curve integration,
  per-doctor phrasing customization, FDA path.

## Hardware

- Training: NVIDIA RTX 4090 (host `pickles` over Tailscale)
- Inference at chairside: any CUDA box or M-series Mac (MPS path TBD post-hour-5)

## Datasets

- **Bone loss:** [DenPAR v3](https://zenodo.org/records/16645076) — 1,000
  annotated intra-oral periapical radiographs, CC-BY 4.0, official
  Training / Validation / Testing splits
- **Caries:** Roboflow public BW datasets (specific dataset decided
  during hour-0 inspection)

## Methodology reference

Cleanroom-reimplemented from arxiv [2506.20522](https://arxiv.org/abs/2506.20522)
(Wimalasiri et al., Scientific Reports 2026). The upstream repo
([chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss](https://github.com/chathurawimalasiri/analysis-in-detecting-alveolar-bone-loss))
has no LICENSE file, so this project follows a methodology-reference
discipline: read the upstream code to understand the approach; write
fresh code in this repo; do not copy or port directly.

## License

MIT. See [LICENSE](LICENSE).

## Layout

```
dental-rad-cli/
├── data/                  # gitignored — DenPAR + Roboflow datasets
├── weights/               # gitignored — trained checkpoints
├── docs/                  # PHI scrub recipe, hour-5 gate criteria
├── scripts/               # data download + training entrypoints
├── src/dental_rad_cli/    # pipeline modules, CLI entrypoint
├── tests/                 # rule-layer unit tests
└── examples/
    ├── input/             # synthetic samples for smoke tests
    └── eval/              # 3-5 scrubbed BWs/PAs for hour-0 evaluation
```

## Run (post-hour-0)

```bash
uv sync
bash scripts/download_denpar.sh
bash scripts/train_all.sh         # ~3-5 hours on RTX 4090
uv run dental-rad-cli analyze examples/eval/bw01.jpg --out ./results/
```

Cold-pickup install < 30 min once `weights/` is populated.
