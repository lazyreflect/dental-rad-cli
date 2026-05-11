# Caries class mapping — ICCMS 6-tier → 3-class model → schema depth

## Source dataset

Renielaz Dental Caries X-ray on Roboflow
(<https://universe.roboflow.com/renielaz/dental-caries-x-ray>):
586 bitewing radiographs, CC-BY 4.0, polygon-annotated against the
ICCMS 6-tier radiographic scale (RA1, RA2, RA3, RB4, RC5, RC6).

## Why a 3-class collapse

A direct 6-class model trained on 586 images averages ~98 images per
class. The deepest tier (RC6, inner-dentin / pulp-near) is also the
rarest in the source distribution, so the effective sample count for
RC6 drops well below that average and starves training. A 5-class
collapse (merging just RA1+RA2+RA3) helps but still leaves RC5 thin.
The 3-class collapse below pools enamel and middle-dentin tiers and
keeps RC6 isolated (preserving deep-caries recall as a separately
auditable class), bringing per-class sample counts into a range YOLOv8s
can fit at this corpus size:

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
higher-resolution model trained on a larger corpus; v0 emits only
`"E1"`, `"D1"`, and `"D3"`.
