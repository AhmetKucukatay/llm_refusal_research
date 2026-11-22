# Arditi et al. (2024) Reproduction Notes

## Scope

This reproduction targets the Qwen 1.8B Chat experiments from:

**Arditi et al., “Refusal in Language Models Is Mediated by a Single Direction,” NeurIPS 2024.**

The implementation separates reusable intervention and evaluation primitives in `src/refusal_toolkit/` from Arditi-specific reproduction code in `reproductions/arditi_2024/`.

The authors' released repository was inspected at commit:

`9d852fae1a9121c78b29142de733cb1340770cc3`

The primary model is `Qwen/Qwen-1_8B-Chat`. The checkpoint was obtained from the official ModelScope distribution because the Hugging Face route was unavailable in our environment.

---

## Reproduced pipeline

The reproduction implements the following stages:

1. harmful and harmless train/validation sampling,
2. baseline refusal-score filtering,
3. residual-stream activation extraction,
4. difference-in-means candidate direction construction,
5. candidate evaluation using:
   - harmful-prompt ablation refusal score,
   - harmless-prompt activation-addition refusal score,
   - harmless-prompt ablation KL divergence,
6. direction selection,
7. directional ablation,
8. positive and negative activation addition,
9. harmful JailbreakBench generation and substring evaluation,
10. harmless generation and substring evaluation,
11. language-model loss evaluation on Pile, Alpaca, and generated harmless completions.

The reproduced intervention wiring follows the authors' released implementation rather than introducing a new intervention scheme.

---

## Direction selection

The paper selects the candidate with minimum harmful-prompt bypass score subject to three constraints described in Appendix C.1:

- activation addition must induce refusal on harmless prompts,
- KL divergence under ablation must remain below `0.1`,
- directions from the final 20% of model layers are excluded.

The final-layer restriction is intended to avoid selecting directions that are too close to direct token-unembedding directions, such as directions that merely suppress refusal-start tokens rather than representing a higher-level refusal feature.

### Our result

For Qwen 1.8B Chat:

| Property | Result |
| --- | --- |
| Selected post-instruction position | `-2` |
| Selected source layer | `15 / 24` |
| Ablation refusal score | approximately `-5.06` |
| Positive steering score | approximately `1.65` |
| Ablation KL divergence | approximately `0.065` |
| Eligible candidates | `12 / 120` |

This exactly matches the **released repository artifact**, which also selects position `-2`, layer `15`.

However, Table 5 of the published paper reports Qwen 1.8B as position `-1`, layer `15 / 24`.

We therefore preserve this discrepancy explicitly:

- **published paper:** `(-1, 15)`
- **released artifact:** `(-2, 15)`
- **our reproduction:** `(-2, 15)`

Our result should consequently be described as matching the released implementation/artifact, not Table 5 exactly.

### Additional selection observations

The candidate at position `-2`, layer `14` produced a slightly stronger harmful-prompt ablation effect than the selected candidate, but its KL divergence exceeded the `0.1` threshold and it was therefore discarded.

The candidate at position `-1`, layer `15` remained eligible and induced slightly stronger refusal under positive steering, but its harmful-prompt ablation score was slightly worse than that of `(-2, 15)`. Because the final ranking minimizes the harmful-prompt ablation refusal score after filtering, `(-2, 15)` wins.

There is also an interesting token-position pattern in Qwen 1.8B. The five post-instruction positions are:

```text
-5  <|im_end|>
-4  \n
-3  <|im_start|>
-2  assistant
-1  \n
```

The eligible-candidate counts were:

```text
-5: 0
-4: 2
-3: 0
-2: 5
-1: 5
```

In particular, all candidate directions from positions `-5` and `-3` failed the positive-steering criterion before the late-layer cutoff. The strongest eligible candidates instead concentrate near the assistant-generation boundary, especially at the `assistant` token and the following newline.

This is only an observation from one model and one reproduction and is **not treated as a general mechanistic claim**. It is nevertheless worth comparing across models and later refusal-direction methods.

---

## Harmful behavioral evaluation

Evaluation uses the 100-example JailbreakBench set used by the released pipeline.

The substring metric follows the authors' released refusal-prefix list. A completion is counted as a substring jailbreak success when none of the known refusal substrings appears.

### Results

| Condition | Our substring ASR | Released substring ASR |
| --- | ---: | ---: |
| Baseline | **0.30** | **0.30** |
| Directional ablation | **0.99** | **0.99** |
| Negative activation addition | **0.97** | **0.97** |

All three aggregate substring results match the released Qwen 1.8B artifacts exactly.

The experiment therefore reproduces the main causal behavioral effect: removing the selected direction, either through directional ablation or negative activation addition, strongly suppresses refusal on harmful prompts.

---

## Harmless behavioral evaluation

The harmless evaluation uses 100 test examples sampled using the same RNG progression as the released full pipeline.

### Results

| Condition | Refusal rate | Non-refusal rate | Released non-refusal rate |
| --- | ---: | ---: | ---: |
| Baseline | **0.03** | **0.97** | **0.97** |
| Positive activation addition | **0.98** | **0.02** | **0.02** |

These aggregate results also match the released artifacts exactly.

Together with the harmful evaluation, this reproduces the bidirectional causal intervention result:

- removing or subtracting the direction suppresses refusal,
- adding the direction induces refusal even on harmless instructions.

---

## Language-model loss evaluation

Loss evaluation follows the released implementation:

- Pile is evaluated as ordinary next-token language modeling,
- Alpaca loss is evaluated only on assistant-completion tokens,
- generated harmless completions are evaluated using the same completion-only masking principle.

The released configuration uses:

```text
batch size = 2
number of batches = 2048
Pile max sequence length = 256
```

### Pile and Alpaca results

| Condition | Dataset | Our CE | Released CE |
| --- | --- | ---: | ---: |
| Baseline | Pile | **2.92109** | **2.92097** |
| Ablation | Pile | **2.93312** | **2.93319** |
| Baseline | Alpaca | **1.77931** | **1.77933** |
| Ablation | Alpaca | **1.78565** | **1.78566** |

The corresponding token counts match the released artifacts exactly:

```text
Pile:    904,444 tokens
Alpaca:  269,234 tokens
```

These results provide a strong validation of the loss masking, dataset order, tokenization, and intervention implementation.

Directional ablation causes only a small increase in Pile and Alpaca cross-entropy, consistent with the claim that removing the refusal direction has relatively limited impact on general language-model behavior.

### Generated-completion loss deviation

The generated harmless-completion loss does not match the released artifact:

```text
our evaluated tokens:       21,018
released evaluated tokens:  24,257
```

Our generated harmless responses are generally shorter than the released completion artifact. The resulting cross-entropy numbers are therefore not directly comparable.

Because Pile and Alpaca match extremely closely, this difference is treated as a **generation-artifact deviation**, not evidence of an error in the loss implementation.

The exact source of the generation-length difference was not established.

The negative-activation-addition loss run was interrupted by an external Hugging Face dataset connection failure and was not rerun, since the independent Pile and Alpaca baseline/ablation checks already validated the loss pipeline sufficiently for this reproduction.

---

## LlamaGuard 2 evaluation

The paper uses two complementary harmful-completion metrics:

1. refusal substring matching,
2. Meta Llama Guard 2 safety classification.

The authors' released Qwen 1.8B artifacts report approximately:

| Condition | Released LlamaGuard 2 jailbreak rate |
| --- | ---: |
| Baseline | `0.20` |
| Directional ablation | `0.82` |
| Negative activation addition | `0.79` |

These numbers are included **only as released-reference results**.

LlamaGuard 2 was not rerun in this reproduction because the released evaluation path requires:

- gated access to `meta-llama/Meta-Llama-Guard-2-8B`,
- a Together API credential,
- the released evaluator's external API path.

The current environment does not have the required gated model access or Together credential.

Therefore, the LlamaGuard 2 values above must **not** be described as reproduced results.

---

## Environment and deviations

### Model distribution source

The intended checkpoint is Qwen 1.8B Chat.

The reproduction uses the official ModelScope distribution rather than the Hugging Face distribution because the Hugging Face route was unavailable.

This is a distribution-source deviation, not an intentional model-family or checkpoint substitution.

Byte-for-byte equivalence with the historical Hugging Face snapshot used by the authors was not established.

### FlashAttention

The released environment includes FlashAttention-related support.

FlashAttention was unavailable in the Windows reproduction environment, so Qwen fell back to PyTorch scaled-dot-product attention.

This changes the execution backend and performance characteristics but does not intentionally change the model architecture or intervention methodology.

### Legacy environment

Arditi-specific model execution is isolated in a separate environment using the important released dependency versions, including:

```text
torch 2.3.0 + CUDA 12.1
transformers 4.44.2
accelerate 0.31.0
numpy 1.26.4
datasets 2.19.2
```

The generic toolkit is developed and tested separately using the main project environment.

---

## Reproduction status

The Qwen 1.8B Arditi reproduction is considered **complete for the intended project scope**.

Strong reproduction evidence includes:

- exact match with the released selected direction,
- exact aggregate harmful substring results,
- exact aggregate harmless substring results,
- near-exact Pile and Alpaca loss values,
- exact Pile and Alpaca evaluated-token counts.

Known limitations are documented rather than hidden:

- published Table 5 and released direction metadata disagree on the selected Qwen token position,
- harmless generated-completion lengths differ from the released artifact,
- negative-addition loss evaluation was not completed after an external dataset connection failure,
- LlamaGuard 2 was not rerun because required external access was unavailable.

These limitations do not alter which results are labeled as reproduced versus reference-only.
