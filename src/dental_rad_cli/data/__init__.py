"""DenPAR v3 dataset adapter + PyTorch dataset classes.

This package bridges the on-disk DenPAR v3 layout (Zenodo record
16645076) and the training/inference code in this repo. See
`denpar_adapter.py` for the v3 schema (verified by direct inspection)
and conversion to YOLO / COCO-keypoint formats; see `denpar_dataset.py`
for PyTorch `Dataset` classes used by the keypoint trainer.
"""
