# PHI Scrub Recipe — Hour-0 Eval Set (3–5 Bitewings)

> **Discipline note.** Scrubbed copies go in `examples/eval/`. Originals
> NEVER enter the repo. This recipe is for 3–5 images at hour-0; do NOT
> scale to 30+ without revisiting whether to automate (auto-scrub
> pipeline + audit log + per-image hash record).

This recipe handles two PHI vectors:

1. **Metadata sidecar identifiers** — DICOM headers, EXIF tags,
   filename patterns like `SMITH_JOHN_19550412.dcm`. Mechanical strip.
2. **Burned-in pixel identifiers** — patient name + DOB rendered into
   the radiograph corner by Curve's image viewer. Visual inspection +
   manual crop. Joseph must eyeball each image first.

---

## Pre-flight checklist

- [ ] Closed Slack / email / any screen-share tool. This procedure
      touches PHI; treat the laptop as quarantined for the next ~20 min.
- [ ] On the M4 MacBook (dev box), not the RTX 4090 desktop. The 4090
      box never sees originals — it only sees `examples/eval/` after
      sync.
- [ ] Tools installed: `pydicom` (`pip install pydicom`), `pillow`
      (`pip install pillow`), `exiftool` (`brew install exiftool`),
      `imagemagick` (`brew install imagemagick`).
- [ ] Workspace dir exists OUTSIDE the repo:
      `mkdir -p ~/tenant-data/dental-rad-eval/originals ~/tenant-data/dental-rad-eval/staging`
- [ ] Confirm `~/tenant-data/` is in your global gitignore (it is by
      workspace convention; verify with `git check-ignore -v ~/tenant-data/foo`).
- [ ] Originals destination is `~/tenant-data/dental-rad-eval/originals/`.
      Repo destination is `~/repos/work/dental-rad-cli/examples/eval/`.

---

## Steps

1. **Export 3–5 bitewings from Curve.** Save as PNG or JPEG to
   `~/tenant-data/dental-rad-eval/originals/`. Skip DICOM unless that's
   the only export option — fewer headers to scrub.

2. **Snapshot the file inventory before touching anything.**
   `ls -la ~/tenant-data/dental-rad-eval/originals/ > ~/tenant-data/dental-rad-eval/inventory-pre.txt`

3. **Visually inspect each image for burned-in PHI.** Open in Preview.
   Look at all four corners + bottom edge for patient name / DOB /
   chart number / date strings. Note the pixel region (e.g.,
   "bottom-left, ~0–180px wide, ~0–40px tall") in a scratchpad.

4. **Crop the burned-in region.** For each image with burned-in PHI:
   ```
   magick ~/tenant-data/dental-rad-eval/originals/IMG_001.png \
     -fill black -draw "rectangle 0,0 180,40" \
     ~/tenant-data/dental-rad-eval/staging/staged_001.png
   ```
   Adjust coords per image. If a corner has nothing burned in, copy
   straight through with `cp`. **Do not skip this step on the
   assumption that "Curve doesn't burn in PHI" — it sometimes does,
   depending on viewer settings.**

5. **Strip EXIF/metadata from staged files.**
   `exiftool -all= -overwrite_original ~/tenant-data/dental-rad-eval/staging/*.png`

6. **If any source was DICOM**, strip DICOM headers with pydicom:
   ```
   python -c "import pydicom, sys; ds = pydicom.dcmread(sys.argv[1]); \
   ds.remove_private_tags(); \
   [setattr(ds, t, '') for t in ['PatientName','PatientID','PatientBirthDate','PatientSex','StudyDate','InstitutionName','ReferringPhysicianName']]; \
   ds.save_as(sys.argv[2])" original.dcm staged.dcm
   ```
   Then export to PNG with a DICOM-to-PNG tool of choice.

7. **Rename to anonymous IDs.** Move staged files into the repo with
   sequential IDs:
   ```
   cd ~/tenant-data/dental-rad-eval/staging
   i=1; for f in *.png; do cp "$f" \
     ~/repos/work/dental-rad-cli/examples/eval/bw$(printf '%02d' $i).png; \
     i=$((i+1)); done
   ```

8. **Re-verify metadata is clean on the renamed files.**
   `exiftool ~/repos/work/dental-rad-cli/examples/eval/bw*.png | grep -iE "name|date|patient|institution|author"`
   Expected output: empty. If anything matches, stop and re-scrub.

9. **Final visual pass.** Open each `bw*.png` in Preview at 200% zoom.
   Scan corners + edges once more. If anything looks like text, crop
   harder and re-run from step 5.

10. **Commit only the scrubbed copies.** `git status` in the repo should
    show only `examples/eval/bw*.png` as untracked. Originals stay in
    `~/tenant-data/` forever.

11. **Cleanup staging.** `rm ~/tenant-data/dental-rad-eval/staging/*`.
    Originals remain at `~/tenant-data/dental-rad-eval/originals/` for
    re-derivation if needed.

---

## What this recipe does NOT cover

- Annotation files (JSON with bounding boxes). If those reference
  patient context in any field, scrub separately.
- Multi-patient batch exports. This recipe assumes one image per
  patient and Joseph eyeballs each.
- The auto-scrub pipeline. Build that when N > 30, not before.
