# Held-out touches log

Append-only. Every read of `denpar_held_out.txt` (whether via
`benchmark_eval.py --split=held-out` or any other path) must be logged
here BEFORE the read.

The held-out set is one-shot. Touching it twice is a discipline failure
that invalidates the final-eval claim. If you find yourself wanting a
second touch, the right move is usually to do more dev-set work
instead.

## Schema

Each entry:

```
## YYYY-MM-DD HH:MM — <session-id or human name>
**Reason for touch:**
**What number is being read:**
**Why this is a justified end-of-development touch (not iterative tuning):**
**Resulting number:**
```

## Touches

_(none yet — split locked 2026-05-12)_
