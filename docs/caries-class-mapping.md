# Caries class mapping — ICCMS 6-tier → 3-class model → schema depth

## Source dataset

Baasils ICCMS Dental Caries on Roboflow
(<https://universe.roboflow.com/baasils-workspace/iccms-dental-caries-etomb>):
537 source bitewing radiographs, Public Domain license, polygon-
annotated against the ICCMS 6-tier radiographic scale
(RA1, RA2, RA3, RB4, RC5, RC6). Roboflow v3 augments to 1455 images
(3× with flip + rotate ±15° + brightness ±15%).

The earlier v0-considered dataset (Renielaz `dental-caries-x-ray`) was
abandoned at hour-0 due to a structurally corrupted class list with
only 13 RC6 samples. See `v0.5-caries-remediation.md` for the forensic.
Baasils is the validated replacement, confirmed clean by direct REST-API
probe of the Roboflow project metadata + the Salehizeinabadi 2025
paper's per-class mAP50 results (RC6 = 0.80 implies trainable
distribution).

Real ICCMS distribution per the Roboflow project metadata:

| ICCMS class | Annotations |
|-------------|------------:|
| RA1 | 84 |
| RA2 | 405 |
| RA3 | 167 |
| RB4 | 121 |
| RC5 | 124 |
| RC6 | 97 |
| **Total** | **998** |

## Why a 3-class collapse

The Roboflow project metadata shows real ICCMS counts of 84/405/167/
121/124/97 across the 6 tiers. A direct 6-class YOLOv8s model could
plausibly train at this size, but the imbalance between RA2 (405) and
RA1 (84) is large; pooling enamel tiers smooths it. The deepest tier
(RC6 = 97 samples) is preserved as a separately auditable class
because deep-caries recall is the load-bearing clinical signal.

```
initial   = RA1 + RA2 + RA3   (enamel through enamel-dentin junction)
moderate  = RB4 + RC5         (outer + middle dentin)
deep      = RC6               (inner dentin / pulp-near)
```

Internal YOLO class indices (written by
`data/caries_adapter.py::build_yolo_caries_dataset`):
`0=initial`, `1=moderate`, `2=deep`.

## Mapping the 3-class output to the schema's `CariesDepth`

`schema.CariesFinding.depth` is the ICDAS-style 5-tier literal
`"E1" | "E2" | "D1" | "D2" | "D3"`. The inference helper
(`pipeline/caries_inference.py::detect_caries`) maps the 3-class model
output to the schema as:

| Model class | ICCMS tiers pooled | Schema `depth` |
|------------:|--------------------|---------------:|
| `0` initial | RA1 + RA2 + RA3 | `"E1"` |
| `1` moderate | RB4 + RC5 | `"D1"` |
| `2` deep | RC6 | `"D3"` |

The intermediate `"E2"` and `"D2"` literals are reserved for a future
higher-resolution model trained on a larger corpus; v0.5 emits only
`"E1"`, `"D1"`, and `"D3"`.
