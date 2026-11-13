# SeqLayers Checkpoint - 2026-11-12

## Repository State

- Branch: `main`, 212 commits ahead of `origin/main`.
- Verification: `julia --project=. test/runtests.jl` passed 2898/2898 tests in 3m54s.
- Hygiene edit in this checkpoint: `.gitignore` now filters generated run-control noise under `experiments/runs/`:
  `autostop.out`, `done*`, `nohup*.out`, `watch*.out`, and `terms-*.tsv`.
- Remaining visible local state after the ignore edit:
  `.gitignore` modified; `experiments/runs/trace2/notused-about.log` was already modified; a small number of watch scripts,
  trace logs, `prodw/dump_tune.log`, and `tuples/void-closure-bug/` remain untracked because they look more intentional than
  generic run noise.

## Current Baselines

Best comparable full decode on the 2,899-word all-treebank dev slice:

| run | note | token F1 | UAS | LAS | sentence F1 | build s | decode s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261111-202339 | decode2_wc3ctl | 0.9928 | 0.7453 | 0.6885 | 0.9582 | 700.9 | 4047.5 |
| 20261112-032817 | decode2_wc3clu | 0.9928 | 0.7450 | 0.6906 | 0.9503 | 722.2 | 12411.7 |

Current perfect-context ceiling/control on the same 2,899-word slice:

| run | note | gap acc | UAS | LAS | head lost | coverage | score s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261112-151338 | control_mac | 0.9978 | 0.8865 | 0.8386 | 0.0162 | 1.0 | 568.0 |

Recent variants are effectively neutral or worse:

| run | note | UAS | LAS | reading |
| --- | --- | ---: | ---: | --- |
| 20261112-120109 | confirm_new_min50 | 0.8858 | 0.8379 | neutral |
| 20261112-114953 | confirm_new_min20 | 0.8855 | 0.8379 | neutral |
| 20261112-150116 | prodw_tune1 | 0.8834 | 0.8368 | worse than control |
| 20261112-085953 | genred_coarser | 0.8875 | 0.8386 | tiny UAS uptick, no LAS gain |

Approximate attachment miss accounting on 2,899 words:

- Best full decode: about 738 wrong attachments.
- Perfect-context control: about 329 wrong attachments.
- Decode-order/context gap: about 409 attachments.

That gap is now the main target. Feature tweaks inside the perfect-context ceiling are moving too little.

## Ceiling Error Shape

For `control_mac`, attachment errors in the ceiling are mostly over-short attachments:

| kind | n | share |
| --- | ---: | ---: |
| too_near | 188 | 0.571 |
| wrong_side | 77 | 0.234 |
| too_far | 64 | 0.195 |

Distance-specific weakness is concentrated on longer left arcs:

| gold distance | n | UAS | LAS |
| --- | ---: | ---: | ---: |
| L6+ | 261 | 0.479 | 0.383 |
| L3-5 | 310 | 0.774 | 0.674 |
| R6+ | 80 | 0.738 | 0.700 |
| R3-5 | 309 | 0.890 | 0.848 |

Head-property predictions are uneven. `head_xpos` is the weakest at 0.625, while `head_dir` is strong at 0.902.
The weak middle is `head_dist`, `head_trank`, and the kin properties, which is exactly where the early examples fail.

## Early Examples

Command used:

```sh
julia --project=experiments experiments/ceiling_explain.jl model=chain train_sents=500 dev_sents=5 select=maxdev leftovers=greedy minctx=2 greedyby=odds leftsmooth=evidence relstrikes=2 relpool=false ctxmode=accum rootabsent=true projective=true headtrank=true kinitems=true headkin=left,right neighbor2=true redundant=true poolsigma=0.1 poolmarginal=drop poolkeepgen=true poolscale=logodds gate=0.5 relevance=0.25 workers=1 docs=3 nsent=2 maxerr=2 maxlen=16 explaintop=1 nopunct=true detail=false
```

Two early ceiling misses:

- `"How about right here ?"`: `about` should attach to `here`, but predicts `How`.
  The gold candidate starts with strong `kin.right=sibling` evidence, but accum leftovers pull it negative.
  The predicted candidate gets highly positive leftovers from local/neighbor form cues and wins despite head properties
  saying the gold head is to the right at distance 2 while the predicted head is left/distance 1.
- `"No no don't play with that ."`: first `No` should attach to second `no`, but predicts `play`.
  This is a too-far/structural-generalization miss: head properties strongly overpredict distance `3-5`, tree rank `4+`,
  and `kin.right=sibling`, and the predicted `play` candidate receives a very large positive leftover stack.

Both examples fit the aggregate pattern: the model is too willing to accept locally plausible structural templates
whose leftovers swamp the head-property/domain evidence.

## Diagnosis

The project is not blocked on syntax, tests, or event plumbing. It is blocked on decode-time exposure and overconfident
attachment scoring:

1. The full decoder still loses about 409 attachments relative to the same model in perfect context.
2. The remaining ceiling errors are mostly too-near, but the worst distance bucket is long left arcs.
3. Recent work has mostly tried to prune or reweight leftover evidence. Those changes have not materially moved the ceiling.
4. The early examples show the attached head properties often contain the right warning signal, but the final candidate score
   can still be dominated by accum leftovers.

## Recommended Next Experiment

Run a paired full-decode diagnostic rather than another feature tweak:

1. Save predictions for the current best full decode with `conllu_out` on the same 2,899-word slice.
2. Add a small comparison script that aligns predicted/gold tokens and labels each wrong attachment by whether the
   perfect-context ceiling gets it right.
3. Report the earliest 20 such avoidable errors by document order with:
   predicted head, gold head, current committed head properties at decision time if available, distance class, deprel,
   and whether the mistake is too-near, too-far, wrong-side, false-root, or missed-root.

If a full decode is too expensive for every iteration, start with `devtb=EWT dev_sents=40` because the recent `fresh_fresh`
run already shows it can surface high-UAS behavior on a smaller slice, then promote only promising changes to all treebanks.
