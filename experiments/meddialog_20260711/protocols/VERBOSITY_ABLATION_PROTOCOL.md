# Verbosity-Bias Ablation Protocol

This final protocol was fixed before any ablation quality-judge scores were
generated. An earlier proposed Concise/Padded rewrite design was rejected during
pre-score QA because semantic validators missed removed medical details. Those
prototype variants are not eligible for judging.

## Question

Does the current absolute LLM-judge pipeline award higher content-quality scores
when an answer is made longer solely by repeating information already present?

## Sample

- 30 completed conversations from the five-method, N=100 MedDialog experiment.
- Six conversations per source method, selected with seed `20260712`.
- Within each source method, cross three visible-assistant-length tertiles with
  two prior-score strata (at or below versus above that method's median Total
  /50), selecting one case from each of the six cells.
- Require at least 80 original visible assistant words and 30 distinct underlying
  `source_index` values.
- Selection uses only the original five-method artifact. No ablation scores exist
  at selection time.

## Final conditions

Each selected conversation yields three separately judged conditions:

- Original: byte-identical source conversation.
- Original Repeat: byte-identical to Original but assigned a separate opaque
  condition/cache key to measure judge repeatability.
- Padded: every original user and assistant message remains present verbatim.
  Extra length is added only by appending exact copies of declarative sentence or
  line units already present in the same assistant turn, preceded by a fixed
  nonmedical transition. Units containing a question mark are never repeated.

Padding targets approximately 1.45 times the original total visible assistant
word count and is distributed across available assistant turns. The construction
is deterministic and makes no model calls.

## Deterministic validity requirements

- User messages are byte-identical across conditions.
- Role order and assistant-turn count are identical.
- Original and Original Repeat conversations are byte-identical.
- Every Padded assistant message begins with the complete original assistant
  message verbatim.
- Every appended substantive unit is an exact substring of the corresponding
  original assistant message.
- No question-bearing unit is appended; question-mark count is unchanged.
- The set of numeric expressions and URLs is unchanged. Repetition may increase
  multiplicity but cannot introduce a new value or URL.
- All 30 Padded conversations must be longer than their originals. Actual word
  ratios are reported.

## Blinding and judge configuration

- The existing four evaluator prompts and score-band anchors are unchanged.
- Each condition is evaluated absolutely in a separate call; sibling versions
  never share context.
- Source method and condition names are not inserted into evaluator prompts.
- Original is rescored rather than reusing its earlier five-method score.
- Judge: DeepSeek v4-flash, temperature 0, thinking enabled, reasoning high,
  with API usage logging enabled.

## Prespecified outcomes

The primary verbosity-preference outcome is the content score:

`Anticipation + Discovery + Medical Quality`, range 0-40.

Calibration is excluded from the primary score because its construct explicitly
includes appropriate amount and restraint. A Padded Calibration decrease is
correct rubric behavior, not bias. Total /50 and the four dimensions are
secondary outcomes.

Primary contrast: Padded minus Original. Original Repeat minus Original measures
the judge's noise floor. Padded minus Original Repeat is a sensitivity contrast.

For each outcome report paired mean difference, paired bootstrap 95% and 90%
intervals, median difference, win/tie/loss, paired rank-biserial correlation,
paired Wilcoxon p-value, and Holm correction across the three condition contrasts
within that outcome. Also report Original/Original Repeat exact agreement and
mean absolute difference.

Practical-invariance margins fixed before scoring:

- Anticipation and Discovery: +/-0.5 points.
- Medical Quality: +/-1.0 point.
- Primary content score /40: +/-1.0 point.
- Total /50: +/-1.0 point.

A 90% interval entirely inside a margin supports practical invariance. Failure
to reject a difference is not itself evidence of no bias. N=30 is a screening
pilot; intervals that cross a margin are inconclusive and should motivate N=100.

## Interpretation

- Positive verbosity preference: Padded materially exceeds both Original and
  the repeat-noise baseline on the content score without any new proposition.
- Evidence against positive verbosity preference: Padded is invariant or scores
  lower on content dimensions, while Calibration remains stable or decreases.
- Repeat disagreement bounds how much one-call score movement can be attributed
  to judge nondeterminism.

This stress test can detect rewards for redundant added length. It does not test
whether a semantically lossless concise rewrite is unfairly penalized, production
token cost, clinical correctness against expert gold labels, or how users would
respond to the padded wording.
