# Severe-bucket dev sites — bone-mask extent dissection (BR9)

Filtered: GT bone-loss >= 6.0 mm. n = 34 sites across 20 stems.

Tolerance for 'reaches deep': ±30 px of GT bone-y, sampled within ±10 px of the CEJ x-column. Eroded mask (5 px) used for category decision.

## Category breakdown

| category | n | meaning |
|---|---|---|
| mask_reaches_deep   | 21 | bone-mask extends within tol of GT bone-y → algorithm-side fix possible (rule mis-picks within an adequate mask) |
| mask_bimodal        | 1 | mask reaches deep AND has coronal pixels → 907-style contamination; algorithm-side fix needed |
| mask_short          | 4 | bone-mask max-apical extent stays well coronal to GT bone-y → training-side issue (mask itself can't support correct landmark) |
| no_bone_at_cej_x    | 8 | no bone-mask pixels within ±10 px of CEJ x → segmentation didn't fire near the CEJ column |

## Headline interpretation

**Algorithm-side fix space exists.** Most severe sites have bone masks that *do* extend to the deep crest; the landmark-selection rule is choosing the wrong y. The BRneg-1 CEJ-x sampling attempt didn't deliver because (hypothesis) the apical-extent isn't necessarily at the CEJ x specifically — may need a different rule like 'most apical bone-on-tooth pixel within X% of the CEJ x', or weighted median, or training a small head.

- mask_short 4/34 (12%)
- mask_reaches_deep + mask_bimodal 22/34 (65%)
- no_bone_at_cej_x 8/34 (24%)

## Per-site detail

| stem | surface | gt_mm | cej_y | gt_bone_y | mask y_min | y_median | y_max | apical_extent | category |
|---|---|---|---|---|---|---|---|---|---|
| 887 | mesial | 13.67 | 434 | 832 | None | None | None | None | no_bone_at_cej_x |
| 125 | mesial | 11.78 | 303 | 674 | None | None | None | None | no_bone_at_cej_x |
| 1227 | mesial | 10.75 | 616 | 282 | 227 | 260 | 294 | 227 | mask_reaches_deep |
| 804 | mesial | 10.07 | 369 | 701 | 690 | 701 | 713 | 713 | mask_reaches_deep |
| 896 | distal | 9.49 | 728 | 468 | 358 | 380 | 475 | 358 | mask_reaches_deep |
| 54 | distal | 9.18 | 356 | 617 | 564 | 590 | 612 | 612 | mask_reaches_deep |
| 881 | mesial | 9.11 | 481 | 746 | 650 | 664 | 675 | 675 | mask_short |
| 907 | distal | 9.05 | 224 | 518 | 264 | 278 | 296 | 296 | mask_short |
| 881 | distal | 8.98 | 496 | 757 | 745 | 761 | 777 | 777 | mask_reaches_deep |
| 804 | mesial | 8.96 | 362 | 658 | 635 | 660 | 682 | 682 | mask_reaches_deep |
| 108 | distal | 8.83 | 366 | 645 | 635 | 647 | 660 | 660 | mask_reaches_deep |
| 413 | mesial | 8.78 | 534 | 795 | 776 | 799 | 822 | 822 | mask_reaches_deep |
| 108 | mesial | 8.60 | 359 | 631 | None | None | None | None | no_bone_at_cej_x |
| 907 | mesial | 8.51 | 231 | 507 | 264 | 280 | 296 | 296 | mask_short |
| 804 | distal | 8.41 | 369 | 646 | None | None | None | None | no_bone_at_cej_x |
| 108 | mesial | 8.32 | 392 | 655 | 633 | 647 | 660 | 660 | mask_reaches_deep |
| 413 | distal | 8.31 | 523 | 770 | 730 | 746 | 762 | 762 | mask_reaches_deep |
| 558 | mesial | 8.07 | 744 | 511 | None | None | None | None | no_bone_at_cej_x |
| 804 | distal | 7.97 | 400 | 663 | 655 | 672 | 686 | 686 | mask_reaches_deep |
| 44 | mesial | 7.94 | 924 | 683 | 657 | 682 | 706 | 657 | mask_reaches_deep |
| 887 | distal | 7.88 | 390 | 620 | None | None | None | None | no_bone_at_cej_x |
| 54 | mesial | 7.68 | 334 | 552 | 530 | 547 | 560 | 560 | mask_reaches_deep |
| 109 | mesial | 7.31 | 577 | 416 | 400 | 412 | 424 | 400 | mask_reaches_deep |
| 896 | distal | 7.08 | 739 | 545 | None | None | None | None | no_bone_at_cej_x |
| 892 | mesial | 7.04 | 783 | 548 | 530 | 544 | 558 | 530 | mask_reaches_deep |
| 907 | mesial | 6.80 | 244 | 465 | 295 | 448 | 503 | 503 | mask_bimodal |
| 87 | distal | 6.68 | 358 | 527 | 460 | 472 | 489 | 489 | mask_short |
| 1231 | mesial | 6.58 | 763 | 565 | 553 | 565 | 581 | 553 | mask_reaches_deep |
| 44 | mesial | 6.50 | 795 | 597 | 578 | 596 | 609 | 578 | mask_reaches_deep |
| 695 | mesial | 6.49 | 453 | 313 | None | None | None | None | no_bone_at_cej_x |
| 539 | distal | 6.41 | 305 | 493 | 484 | 500 | 515 | 515 | mask_reaches_deep |
| 580 | mesial | 6.39 | 424 | 599 | 571 | 601 | 627 | 627 | mask_reaches_deep |
| 611 | distal | 6.08 | 829 | 612 | 594 | 613 | 633 | 594 | mask_reaches_deep |
| 109 | mesial | 6.03 | 536 | 403 | 392 | 404 | 417 | 392 | mask_reaches_deep |