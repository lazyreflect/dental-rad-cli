# Hour-5 Gate — Perception Layer Go/No-Go

This gate exists because the cheapest moment to abandon a wrong
architecture is hour 5, not hour 50. The hour-0 prototype is a
perception-layer probe — does a DenPAR-v3-trained model produce
landmark + lesion output that's actually usable on Joseph's own
bitewings, not the curated test split? If the answer is "kind of," the
sunk-cost gravity at hour 50 will be much harder to escape than the
gravity at hour 5. Decide honestly now.

---

## The eval set

The 3–5 bitewings Joseph scrubbed per `phi-scrub-recipe.md`, sitting at
`examples/eval/bw*.png`. These are real clinical bitewings from real
patients seen in Joseph's practice — not the DenPAR test split, not
synthetic, not academic-clean. The point of this gate is that the
prototype must work on Joseph's own image distribution, which is the
only one that matters for the downstream product.

---

## Pass criteria — observable, eyeball-able

Mark each green / yellow / red at hour 5. Two or more reds = no-go.
Three or more yellows = no-go. Pure greens or "all green plus one
yellow" = go.

1. **CEJ landmark placement.** On at least 3 of 5 bitewings, the
   cemento-enamel junction markers land within ~5 pixels (eyeballed,
   at 200% zoom) of where Joseph would place them. Not "near the
   tooth" — at the actual CEJ.

2. **Alveolar crest landmark placement.** On at least 3 of 5
   bitewings, the alveolar crest markers track the bone height
   correctly on both mesial and distal of each visible tooth.
   "Correctly" = Joseph would not need to move the marker more than
   ~5 pixels.

3. **Bone-loss percentage output sanity.** For teeth with obvious
   bone loss visible in the image, the model's reported percentage
   loss is within ~10 percentage points of Joseph's clinical read.
   Teeth with no visible bone loss should report <15% loss.

4. **No catastrophic failures.** Zero of 5 bitewings show the model
   placing landmarks on the wrong tooth, flipping mesial/distal, or
   outputting landmarks in soft tissue / image background.

5. **Caries head sanity check** (if the caries head is wired up at
   hour 5; defer this row if not). On at least 2 of 5 bitewings with
   a frank interproximal caries lesion, the caries head fires on the
   correct tooth surface (correct quadrant, correct surface — not
   just "somewhere in the image").

6. **The usability test.** Look at each output (landmark overlay +
   bone-loss table + caries flags). For each of the 5 bitewings, ask
   honestly: **"Would I paste this output into a patient's Curve
   chart note today?"** Not "is it directionally right?" — would you
   actually paste it as your clinical documentation? Pass = at least
   3 of 5 are pasteable as-is or with one trivial edit.

---

## No-go protocol

If the gate fails, do not start the CLI scaffold. Take 30 minutes to
diagnose which failure mode fired, then choose:

- **Training-data mismatch (BW vs PA).** If landmarks are sane on PA
  images but not BW (DenPAR v3 is primarily PA-weighted), re-evaluate
  whether a BW-specific fine-tune is the next experiment. Probably
  yes — cheap to test.

- **Hyperparameter / inference-settings issue.** If outputs look
  noisy or inconsistent, try one round of inference-side tweaks
  (input resolution, normalization, NMS threshold) before declaring
  architectural failure. Cap at 1 hour.

- **Architecture mismatch.** If two-stage detection-then-regression
  is producing tooth-misassignment errors, evaluate the
  one-model-two-heads alternative (flagged as T-B in the
  methodology notes). This is a re-implementation, not a tweak —
  decide if it's worth another 5–10 hour spike.

- **CV perception is not the right tool.** If multiple failure modes
  fire simultaneously and no cheap fix is in sight, consider whether
  a commercial dental-radiograph API (Pearl, Overjet,
  Videa-Health-equivalent) gives a 90% answer for the integration
  layer while the perception research moves to a side project.
  Honest exit ramp.

Update `README.md` and any in-flight session prompts with the
no-go finding before closing the laptop. Future-Joseph at hour 0 of
the next spike must not have to re-derive what failed.

---

## Yes-go protocol

If the gate passes:

1. Write CLI scaffold (`dental-rad` entrypoint, `--input` / `--output`
   flags, dry-run mode).
2. Add a tiny test suite that runs the perception layer on the 5
   eval bitewings + asserts the output structure (not the values —
   the eval set is too small to assert numeric thresholds).
3. Wire the caries head if not already wired.
4. Sketch the NoteBrusher integration shape: does this become a
   subprocess NoteBrusher shells out to, or an HTTP endpoint, or a
   pure Python import? Decide based on where the GPU runs.
5. Schedule the next gate: hour-20 gate on a 15-image expanded eval
   set, with the same six criteria above tightened.

---

## Resisting wishful thinking at hour 5

Joseph at hour 5 will be tired, will have spent ~$X on GPU time, and
will want this to work. The gate criteria above are deliberately
phrased to resist that pull. If you find yourself rationalizing why
criterion 6 ("would I paste this into Curve?") should be relaxed to
"would I paste this with significant edits?" — that's the gate
working. Mark it yellow, count the yellows, follow the protocol.
