# Reward Hacking & KL Penalty

This page explains a key failure mode in RLHF training and how the
**KL divergence penalty** prevents the policy from gaming the reward signal.

---

## Why the KL Penalty Matters

Without the KL divergence penalty, the policy learns to game the reward
model by stuffing hedge words with no coherent meaning — **reward hacking**.

**Total reward formula:**

```
R_total(x, y) = r_RM(x, y) − β × KL(π_θ(y|x) ‖ π_SFT(y|x))
```

The KL term penalises incoherent distributions relative to the coherent
SFT reference, keeping the policy from collapsing into repetitive
hedge-word spam.

| Type | Score | Text |
|------|-------|------|
| HACKED (gamed) | 1.000 | May might could possibly potentially uncertain risk volatili... |
| HACKED (gamed) | 0.999 | The stock may could possibly might uncertain possibly may co... |
| GENUINE hedge | 0.878 | The stock may decline if inflation persists beyond the curre... |
| No hedge | 0.492 | The stock rose sharply after earnings were released yesterda... |

The **hacked** examples score high on the heuristic (lots of hedge words)
but are semantically meaningless. The KL penalty prevents this by ensuring
the policy output remains close to the SFT distribution.

---

## Hedge Word Lexicon

Words that **increase** the score (epistemic hedges, uncertainty markers,
conditionality, analytical qualifiers):

`may` · `might` · `could` · `possibly` · `potentially` · `appears to` ·
`seems to` · `tends to` · `suggests` · `uncertain` · `unclear` ·
`subject to` · `risk` · `risks` · `volatility` · `fluctuation` ·
`variability` · `if` · `depending on` · `assuming` · `provided that` ·
`contingent` · `subject to change` · `historically` · `based on` ·
`according to` · `indicates` · `analysis suggests` · `data shows` ·
`evidence suggests`

---

## Speculative Word Lexicon

Words that **decrease** the score (weighted ×1.5):

`will definitely` · `will certainly` · `guaranteed` · `guarantee` ·
`without a doubt` · `absolutely will` · `100%` · `sure to` ·
`skyrocket` · `explode` · `moon` · `to the moon` · `unstoppable` ·
`massive gains` · `huge profits` · `get rich` · `can't lose` ·
`you must buy` · `buy now` · `sell immediately` · `don't miss` ·
`once in a lifetime` · `never been a better time` · `amazing` ·
`incredible returns` · `unbelievable` · `shocking`
