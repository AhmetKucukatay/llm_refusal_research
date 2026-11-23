# Student Ablation Projects on Refusal Mechanisms

## 1. Purpose

This repository contains a completed reproduction of the Qwen-1.8B-Chat experiments from Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.

The goal of the following student projects is **not to reproduce another full paper**. Each project should instead answer one small, well-defined research question by changing one component of the existing Arditi pipeline while keeping everything else fixed.

The four proposed projects are:

| Project | Research question | Difficulty |
|---|---|---|
| 1. Selection-constraint ablation | Which parts of Arditi's direction-selection procedure are actually necessary? | Low |
| 2. Token-position ablation | Why do useful refusal directions concentrate at specific post-instruction tokens? | Low–Medium |
| 3. Matched contrastive pairs | Does Ball-style matched harmful/harmless data produce a cleaner refusal direction than Arditi-style independent datasets? | Medium |
| 4. MLP localization | Is Arditi's global refusal direction concentrated in a small number of MLP neurons? | Medium–High |

All four projects should use the same base model and existing Arditi reproduction unless the experiment explicitly changes one component.

---

# 2. Repository Architecture

The repository is organized around an important distinction:

```text
src/refusal_toolkit/
    generic, paper-independent utilities

reproductions/
    paper-specific implementations

tests/unit/
    tests for generic toolkit code

tests/reproductions/
    tests for reproduction-specific code

data/
    downloaded/generated datasets; generally not committed

artifacts/
    generated model outputs, tensors, experiment results; gitignored

docs/
    documentation and experiment reports
```

The existing Arditi implementation is under:

```text
reproductions/arditi_2024/
```

Students **must not modify the original reproduction logic merely to make their experiment convenient**.

New experiments should instead live under:

```text
reproductions/arditi_2024/ablations/
```

Recommended structure:

```text
reproductions/
└── arditi_2024/
    ├── ...
    └── ablations/
        ├── selection_constraints/
        ├── token_position/
        ├── matched_contrasts/
        └── mlp_localization/
```

For example:

```text
reproductions/arditi_2024/ablations/token_position/
├── __init__.py
├── run.py
└── analysis.py
```

Not every project needs both `run.py` and `analysis.py`. Keep the implementation as small as possible.

Experiment-specific generated files should go under:

```text
artifacts/arditi_2024/ablations/<experiment_name>/
```

Experiment-specific datasets should go under:

```text
data/arditi_2024/ablations/<experiment_name>/
```

Each project should eventually have a short committed report:

```text
docs/ablations/<experiment_name>.md
```

---

# 3. Environment Setup

## 3.1 Two environments

The repository currently uses two Python environments for different purposes.

### Development environment

The normal repository environment is:

```text
.venv
```

Use this for:

- unit tests,
- Ruff,
- generic toolkit development,
- non-legacy utilities.

From the repository root on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run tests with:

```powershell
python -m pytest tests -q
```

Run linting with:

```powershell
python -m ruff check src reproductions tests
```

---

### Arditi runtime environment

The paper-faithful Qwen runtime uses a separate environment:

```text
../.venv-arditi
```

From the repository root:

```powershell
..\.venv-arditi\Scripts\Activate.ps1
```

Important versions include approximately:

```text
Python              3.11
torch               2.3.0+cu121
transformers        4.44.2
accelerate          0.31.0
numpy               1.26.4
datasets            2.19.2
```

Additional packages used by the existing reproduction include `tiktoken`, `einops`, `sentencepiece`, and the Qwen remote-code dependencies.

The project package should be installed in this environment using:

```powershell
python -m pip install -e . --no-deps
```

The `--no-deps` is intentional. The generic project metadata currently targets a newer Torch stack than the legacy Arditi runtime.

Do not allow `pip` to silently upgrade Torch, Transformers, or NumPy in this environment.

---

## 3.2 Model checkpoint

All four experiments should use the exact same model:

```text
Qwen/Qwen-1_8B-Chat
```

The local checkpoint is expected at:

```text
models/qwen-1_8b-chat/
```

Do not replace it with:

- Qwen1.5,
- Qwen2,
- Qwen3,
- a quantized version,
- another chat checkpoint.

The existing reproduction used the official ModelScope distribution because Hugging Face access was unavailable.

This is important because the purpose of these projects is an **ablation of the Arditi reproduction**, not a model-comparison experiment.

---

## 3.3 FlashAttention

FlashAttention is not available in the current Windows runtime.

Qwen therefore falls back to PyTorch scaled-dot-product attention.

This is already a documented reproduction deviation and students do **not** need to fix it.

Do not spend project time trying to install FlashAttention on Windows unless there is a concrete experimental reason.

---

# 4. Import and Coding Rules

Always run experiment modules from the repository root.

Good:

```powershell
python -m reproductions.arditi_2024.ablations.token_position.run
```

Avoid:

```powershell
python reproductions/arditi_2024/ablations/token_position/run.py
```

Use absolute imports.

Good:

```python
from refusal_toolkit.evaluation import collect_last_token_logits
from reproductions.arditi_2024.selection import evaluate_candidate
```

Do not use:

```python
import sys
sys.path.append(...)
```

and do not copy existing utility functions into the experiment folder.

Before implementing a new operation, search the existing repository for it.

---

# 5. Existing Code That Should Be Reused

The following modules already implement most of the Arditi pipeline.

## Qwen prompt and tokenization

```text
reproductions/arditi_2024/qwen.py
```

Important existing utilities include:

```python
format_qwen_prompt(...)
format_qwen_prompts(...)
configure_qwen_tokenizer(...)
get_qwen_post_instruction_token_ids(...)
tokenize_qwen_instructions(...)
```

Do not manually rebuild the Qwen chat template in an ablation.

The reproduction uses the post-instruction sequence corresponding to:

```text
-5   <|im_end|>
-4   \n
-3   <|im_start|>
-2   assistant
-1   \n
```

---

## Activation extraction

```text
reproductions/arditi_2024/extraction.py
```

Existing utilities include:

```python
extract_resid_pre_activations(...)
compute_mean_resid_pre_activations(...)
```

The extracted tensor convention used by the reproduction is:

```text
[batch, position, layer, hidden]
```

Do not silently change tensor ordering.

---

## Refusal scoring

```text
reproductions/arditi_2024/refusal.py
```

The existing reproduction computes the released Qwen refusal score from the refusal-token probabilities.

Reuse the existing scoring implementation instead of introducing another definition of refusal.

---

## Candidate selection

```text
reproductions/arditi_2024/selection.py
```

Important utilities include:

```python
evaluate_candidate(...)
should_discard_candidate(...)
select_best_candidate(...)
```

The original released selection constraints reproduced in this repository are:

```text
KL divergence <= 0.1
positive steering score >= 0
source layer outside the final 20% of the network
```

After filtering, candidates are ranked by harmful-prompt ablation refusal score.

---

## Interventions

Generic intervention primitives are in:

```text
src/refusal_toolkit/interventions.py
```

Arditi/Qwen-specific wiring is in:

```text
reproductions/arditi_2024/interventions.py
```

Important existing functions include:

```python
get_qwen_ablation_hooks(...)
get_qwen_addition_hooks(...)
```

Generic hook management uses:

```python
temporary_hooks(...)
```

Always ensure hooks are removed after each condition.

Never leave hooks attached while running the next experimental condition.

---

## Evaluation

```text
src/refusal_toolkit/evaluation.py
```

Important utilities include:

```python
collect_last_token_logits(...)
compute_kl_divergence_from_logits(...)
compute_masked_next_token_nll(...)
```

---

## Generation

```text
src/refusal_toolkit/generation.py
```

The existing:

```python
generate_completions(...)
```

should be used when behavioral generation is required.

Existing runners under:

```text
reproductions/arditi_2024/
```

such as:

```text
select_direction.py
generate_evaluation_completions.py
generate_harmless_completions.py
evaluate_completions.py
```

are useful examples of how the model, tokenizer, hooks, and artifacts are wired together.

They should be treated as references, not modified independently by each student.

---

# 6. General Experimental Rules

All projects should follow the same basic principles.

## Keep train / validation / test roles separate

The existing pipeline conceptually contains:

```text
training data
    direction construction

validation data
    candidate / hyperparameter selection

test data
    final behavioral evaluation
```

Do not tune a layer, coefficient, threshold, neuron count, or intervention strength using JailbreakBench test results.

If a value is chosen because it performs best on the test set, the test result is no longer a held-out result.

---

## Change one factor at a time

For an ablation to be interpretable:

```text
baseline pipeline
        vs.
baseline pipeline + one controlled change
```

is preferable to simultaneously changing:

```text
dataset
+ layer
+ token position
+ coefficient
+ evaluation metric
```

If several things change together, the experiment cannot tell us which change caused the result.

---

## Use the same evaluation where possible

At minimum, relevant experiments should report some subset of:

```text
harmful ablation refusal score
harmless positive-steering refusal score
harmless KL divergence
harmful held-out jailbreak/refusal behavior
harmless held-out refusal behavior
```

Do not invent a different metric only because the existing result looks inconvenient.

---

## Reproducibility

Every experiment report should state:

```text
model
seed
training data
validation data
test data
token position
layer
direction normalization
intervention type
intervention coefficient
number of examples
```

Where randomness exists, use explicit seeds.

The current Arditi sampling convention uses seed:

```text
42
```

---

# 7. Project 1 — Selection-Constraint Ablation

## Research question

**How important are the constraints used by Arditi's direction-selection procedure?**

The reproduced pipeline evaluates 120 candidate directions:

```text
5 token positions × 24 layers
```

and filters them before choosing the final direction.

The selected Qwen direction in our reproduction is:

```text
position = -2
layer = 15
```

But we already observed an interesting example:

```text
(-2, 14)
```

produces a slightly stronger harmful ablation effect than the selected direction, but is removed because:

```text
KL > 0.1
```

This suggests that the constraints may be preventing the selection procedure from choosing highly disruptive directions.

---

## Main experiment

Keep **all candidate directions and candidate scores fixed**.

Only change the selection policy.

Compare:

```text
A. Original selection
B. No KL constraint
C. No positive-steering constraint
D. No late-layer constraint
```

Optional diagnostic:

```text
E. No constraints
```

For every condition, apply exactly the same final ranking rule:

```text
minimum harmful ablation refusal score
```

The only difference should be which candidates are eligible.

---

## What this project tests

Each constraint has a different scientific meaning.

### KL constraint

Tests whether the direction can suppress refusal without strongly changing normal harmless behavior.

Prediction:

```text
remove KL filter
→ stronger jailbreak effect may be selected
→ but collateral damage may increase
```

### Positive-steering constraint

Arditi wants a direction that works in both directions:

```text
remove direction → less refusal
add direction    → more refusal
```

Without this condition, the selected vector may be good at destroying refusal but not represent a genuine bidirectional behavioral control feature.

### Late-layer constraint

Late layers are close to the model output.

A late-layer candidate might appear excellent simply because it directly affects logits associated with refusal-start tokens rather than representing a higher-level refusal feature.

---

## Existing code to reuse

Primary module:

```python
from reproductions.arditi_2024.selection import (
    should_discard_candidate,
    select_best_candidate,
)
```

If candidate scores need to be recomputed:

```python
evaluate_candidate(...)
```

should be reused.

Do not rewrite:

- refusal score,
- KL calculation,
- steering calculation,
- direction ablation.

---

## New code

Create:

```text
reproductions/arditi_2024/ablations/selection_constraints/
├── __init__.py
├── run.py
└── analysis.py
```

`run.py` should load the existing candidate evaluations and apply alternative filtering policies.

If the current saved artifact does not contain every candidate evaluation, compute them once and save them before comparing policies.

Do **not** separately recompute candidate scores for each condition.

---

## Output

Recommended JSON structure:

```json
{
  "condition": "no_kl_filter",
  "selected_position": -2,
  "selected_layer": 14,
  "validation": {
    "ablation_refusal": 0.0,
    "steering_refusal": 0.0,
    "kl": 0.0
  },
  "test": {
    "harmful_metric": 0.0,
    "harmless_metric": 0.0
  }
}
```

Create one summary table:

| Condition | Position | Layer | Ablation | Steering | KL | Harmful test | Harmless test |
|---|---:|---:|---:|---:|---:|---:|---:|

---

## Main pitfalls

Do not:

- change candidate directions,
- change validation examples,
- tune thresholds on test results,
- select a new threshold after seeing test performance.

The point is to test the **existing constraints**, not optimize a new method.

---

## Definition of done

The project is complete when it can answer:

> What kind of candidate is selected when each original constraint is removed, and what trade-off does that constraint prevent?

---

# 8. Project 2 — Token-Position Ablation

## Research question

**How dependent is the refusal direction on the post-instruction token from which it is extracted?**

Our Qwen reproduction considers:

```text
-5   <|im_end|>
-4   \n
-3   <|im_start|>
-2   assistant
-1   \n
```

A striking observation from our existing run is:

```text
eligible candidates by position

-5 → 0
-4 → 2
-3 → 0
-2 → 5
-1 → 5
```

Thus the best causal directions appear concentrated near the assistant-generation boundary rather than uniformly across the chat-template suffix.

The Apple paper independently observes that its refusal neurons frequently become active around the assistant-turn boundary on harmful prompts.

This project tests whether that boundary localization can be seen directly in the Arditi direction.

---

## Primary experiment

Fix the source layer:

```text
layer = 15
```

This is the layer selected by the original reproduction.

Change only the extraction position.

Evaluate:

```text
direction(-5, 15)
direction(-4, 15)
direction(-3, 15)
direction(-2, 15)
direction(-1, 15)
```

This is important: **do not allow each token position to choose its own best layer in the primary experiment**.

Otherwise token position and layer become confounded.

---

## Why this experiment is especially cheap

The existing candidate tensor already has shape:

```text
[5, 24, 2048]
```

Therefore, if the artifact is available, no activation extraction is necessary.

The five directions can be read directly from the existing candidate tensor.

---

## Measurements

For each position, calculate:

```text
harmful ablation refusal score
harmless steering refusal score
harmless ablation KL
```

Then run held-out behavior with the same intervention settings.

Recommended plots:

```text
token position vs harmful ablation score
token position vs positive steering score
token position vs KL
token position vs held-out harmful effect
```

The x-axis should contain the actual token label, not only `-5`, `-4`, etc.

For example:

```text
<|im_end|>
newline
<|im_start|>
assistant
newline
```

---

## Existing code to reuse

Candidate tensor:

```text
candidate_directions.pt
```

Evaluation:

```python
from reproductions.arditi_2024.selection import evaluate_candidate
```

Qwen token identification:

```python
from reproductions.arditi_2024.qwen import (
    get_qwen_post_instruction_token_ids,
)
```

Behavioral intervention:

```python
get_qwen_ablation_hooks(...)
get_qwen_addition_hooks(...)
```

Generation:

```python
generate_completions(...)
```

---

## New code

Create:

```text
reproductions/arditi_2024/ablations/token_position/
├── __init__.py
├── run.py
└── analysis.py
```

No modification to `selection.py` should be necessary.

---

## Optional extension

Only after the fixed-layer experiment is finished:

```text
positions × layers 13–18
```

can be plotted as a small heatmap.

Do not immediately search all layers for the best token-specific configuration, because that changes the experiment from an ablation into another selection procedure.

---

## Main interpretation question

If `assistant` and the final newline are much stronger than the structural boundary tokens:

```text
<|im_end|>
<|im_start|>
```

then the result supports the idea that the refusal representation becomes causally actionable around the transition into assistant generation.

It does **not** prove that refusal is computed only at those tokens.

The claim must remain local to this model and experiment.

---

## Definition of done

The project should answer:

> Holding the layer fixed, how strongly does the causal quality of the refusal direction depend on the exact post-instruction token?

---

# 9. Project 3 — Arditi Random Contrasts vs. Ball-Style Matched Contrasts

## Research question

**Does constructing harmful/harmless contrasts from matched counterfactual prompts produce a cleaner refusal direction than using independent harmful and harmless datasets?**

This is inspired by Ball et al.

Their harmfulness analysis does not simply draw an unrelated harmless dataset. Instead, harmless prompts are generated by rewriting harmful prompts while preserving as much wording and sentence structure as possible.

The purpose is to reduce nuisance differences such as:

```text
topic
syntax
lexical content
prompt length
writing style
```

while changing the harmful intent.

---

## Important distinction

The original Arditi-style construction is approximately:

\[
r_A =
E[h(x_{\text{harmful}})]
-
E[h(x_{\text{harmless}})].
\]

The harmful and harmless examples come from different prompt distributions.

The Ball-inspired matched construction is:

\[
r_M =
\frac{1}{N}
\sum_i
\left(
h(x_i^{\text{harmful}})
-
h(x_i^{\text{matched harmless}})
\right).
\]

The harmless prompt is a minimally edited counterfactual of the corresponding harmful prompt.

---

## Important mathematical point

Simply shuffling the matched pairs is **not a useful ablation**.

Because:

\[
\frac{1}{N}\sum_i(h_i-g_i)
=
\frac{1}{N}\sum_i h_i
-
\frac{1}{N}\sum_i g_i,
\]

permuting the harmless examples does not change the resulting mean-difference vector.

The important change is therefore **which harmless distribution is constructed**, not the pairing order itself.

---

## Primary experiment

Use exactly the same 128 harmful training prompts in both conditions.

### Condition A — Arditi

Use the existing harmless training set.

### Condition B — Matched

For every harmful training prompt, construct one minimally edited harmless prompt.

The matched rewrite should:

- preserve the general sentence structure when possible,
- preserve the main verb when possible,
- preserve as many words as possible,
- replace only the harmful intent/content,
- remain clearly harmless,
- remain a sensible instruction.

The generated matched dataset must be saved and manually inspected before activation extraction.

---

## Data format

Recommended format:

```json
[
  {
    "id": "0001",
    "harmful": "...",
    "harmless_matched": "...",
    "generator": "...",
    "prompt_version": "v1"
  }
]
```

Save it under:

```text
data/arditi_2024/ablations/matched_contrasts/
```

Do not overwrite the existing Arditi train splits.

---

## Data-quality diagnostics

Before running the model, report:

```text
number of pairs
mean harmful token length
mean matched-harmless token length
mean absolute token-length difference
lexical overlap
number of manually rejected/repaired pairs
```

Use the Qwen tokenizer when measuring token length.

Character length is not sufficient.

---

## Primary controlled comparison

To isolate **data construction**, initially fix the site:

```text
position = -2
layer = 15
```

Construct:

```text
r_Arditi(-2, 15)
r_matched(-2, 15)
```

Do not rerun full candidate selection in the first experiment.

Otherwise two things change simultaneously:

```text
contrast construction
+
selected layer/position
```

---

## Measurements

### 1. Direction similarity

Compute:

\[
\cos(r_A,r_M).
\]

This tells us whether the matched construction discovers essentially the same representation axis.

### 2. Harmful ablation

Does matched direction removal suppress refusal more or less than Arditi's direction?

### 3. Harmless steering

Does adding the matched direction induce refusal?

### 4. Harmless KL

Does the matched direction produce less collateral disturbance?

---

## Very important normalization control

Directional ablation internally removes a normalized projection, so direction magnitude is largely controlled.

Activation addition is different: raw vector magnitude matters.

Therefore steering comparisons should report both:

```text
A. raw extracted vector
B. norm-matched vector
```

For the norm-matched condition:

\[
r'_M
=
r_M
\frac{\|r_A\|}{\|r_M\|}.
\]

This separates:

```text
orientation difference
```

from:

```text
direction norm difference.
```

Do not silently normalize only one method.

---

## Optional third condition

A useful stronger control is:

### Condition C — Length-matched unrelated harmless prompts

Sample unrelated harmless prompts whose Qwen token-length distribution approximately matches the harmful set.

Then compare:

```text
Arditi random
length-matched random
semantic matched
```

This can help separate:

```text
length effects
```

from:

```text
semantic counterfactual matching.
```

This condition is optional.

---

## Existing code to reuse

Dataset loading:

```text
reproductions/arditi_2024/data.py
```

Qwen tokenization:

```text
reproductions/arditi_2024/qwen.py
```

Activation extraction:

```python
extract_resid_pre_activations(...)
compute_mean_resid_pre_activations(...)
```

Candidate/direction logic should reuse the same difference-in-means implementation used by the current reproduction.

Evaluation should reuse:

```python
evaluate_candidate(...)
```

and the existing intervention hooks.

---

## New code

Create:

```text
reproductions/arditi_2024/ablations/matched_contrasts/
├── __init__.py
├── data.py
├── run.py
└── analysis.py
```

`data.py` here should contain only experiment-specific matched-pair loading/validation.

Do not put Ball-specific dataset logic into:

```text
src/refusal_toolkit/
```

unless a second independent project later needs exactly the same generic operation.

---

## Possible outcomes

### Matched direction works better

Possible interpretation:

> Some nuisance variation in the original independent datasets was removed, giving a more selective refusal/harmfulness representation.

Especially interesting if:

```text
similar harmful effect
+
lower harmless KL
```

is observed.

### Arditi works better

Possible interpretation:

> The broader distributional contrast captured by independent harmful/harmless datasets is useful for finding the behavioral refusal control direction.

### They are almost identical

Also meaningful.

If:

```text
cosine ≈ 1
behavior ≈ same
```

then Arditi's direction appears robust to substantial changes in contrast construction.

---

## Definition of done

The project should answer:

> Does semantic counterfactual matching change the geometry or causal quality of the Arditi refusal direction?

---

# 10. Project 4 — Apple-Inspired MLP Localization of the Arditi Direction

## Background

Arditi identifies a vector in the residual stream and removes its component throughout the network.

The newer Apple paper asks a more localized question:

> Can the same safety behavior depend on a much smaller internal component?

Their work identifies individual MLP neurons whose intervention can produce effects comparable to Arditi's global refusal-direction intervention.

Their main intervention pins a single MLP neuron activation to a chosen constant at all token positions.

They report that single-neuron interventions can reach attack-success rates close to the full Arditi direction intervention in their tested Qwen3 and Llama-3.1 models.

This student project should **not reproduce the entire Apple paper**.

Instead it asks a smaller question using the refusal direction we already have.

---

## Research question

**How sparse is the implementation of the Arditi refusal direction inside the MLP at its selected layer?**

The existing Arditi result gives:

```text
selected layer = 15
selected residual direction = r
```

At this layer, every MLP intermediate neuron writes back into the residual stream through the MLP's final output projection.

Each neuron therefore has an associated residual-space write vector:

\[
w_i \in R^{d_{\text{model}}}.
\]

The first experiment asks:

\[
\cos(w_i,r)
\]

for every neuron.

This is directly motivated by the geometric analysis in the Apple paper, where the refusal direction is compared against MLP down-projection neuron vectors.

---

## Phase A — Geometry

This is the required part of the project.

### Step 1: inspect the Qwen-1.8B MLP

Do **not** assume Qwen-1.8B has the same module names as Qwen3.

Before writing any intervention code:

```python
layer = model.transformer.h[15]

print(layer.mlp)

for name, module in layer.mlp.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name, module.weight.shape)
```

Identify the projection mapping:

```text
MLP intermediate dimension
        ↓
residual hidden dimension
```

The module may not be named `down_proj`.

Do not hard-code an Apple/Qwen3 module name into Qwen-1.8B code.

---

## Step 2: verify matrix orientation

For a PyTorch linear layer:

```text
weight.shape = [out_features, in_features]
```

If the projection maps:

```text
d_ff → d_model
```

then:

```text
weight.shape = [d_model, d_ff]
```

and the residual-space vector contributed by intermediate neuron `i` is:

```python
weight[:, i]
```

not:

```python
weight[i, :]
```

Always assert dimensions before computing cosines.

For example:

```python
assert direction.shape == (d_model,)
assert projection.weight.shape[0] == d_model
```

---

## Step 3: rank neurons by alignment

Normalize:

```python
r = direction / direction.norm()
```

Construct all neuron output vectors:

```text
[d_ff, d_model]
```

and calculate signed and absolute cosine similarity.

Save at least:

```text
neuron index
signed cosine
absolute cosine
rank
```

---

## Step 4: random baseline

A high maximum cosine can occur simply because thousands of neuron vectors are being searched.

Therefore the top cosine must be compared against a null.

The simplest project-level null is empirical:

1. Generate many random unit vectors in residual space.
2. For every random vector, calculate the maximum absolute cosine with any MLP neuron vector.
3. Compare the observed refusal-direction maximum against this distribution.

Example:

```text
1000 random directions
```

is sufficient for an initial empirical comparison.

The Apple paper performs a related significance analysis and finds that its observed top alignment exceeds a random-direction expectation across all seven tested models.

---

## Step 5: top-k reconstruction

For:

```text
k = 1, 5, 10, 50, 100
```

ask how well the span of the top-k aligned neuron vectors reconstructs the refusal direction.

Useful metrics:

```text
cosine(original r, reconstructed r)
relative reconstruction error
fraction of direction norm explained
```

Compare to randomly selected neuron sets of the same size.

This gives a direct answer to:

> Is the refusal direction unusually concentrated in a small MLP subspace?

---

## Phase B — Behavioral intervention

This part is optional but highly desirable.

Instead of implementing the complete Apple candidate search, perform a simpler ablation:

```text
top-k aligned neurons
vs.
random-k neurons
```

for:

```text
k = 1
k = 5
k = 10
```

At layer 15.

A forward hook can zero the selected intermediate MLP coordinates before the output projection.

Conceptually:

```python
hidden[..., neuron_indices] = 0
```

The hook must affect only the intended neurons.

Test it first on a toy tensor.

---

## Important distinction from the Apple paper

The Apple paper does **not** simply zero the neuron.

Its main constant intervention sets:

\[
h_i \leftarrow m
\]

and selects the intervention magnitude using validation data.

Therefore our zeroing experiment must be described as:

> an Apple-inspired neuron-localization ablation

not:

> a reproduction of the Apple single-neuron attack.

If zero-ablation produces little behavioral effect, an optional second phase can implement the Apple-style constant intervention for only the top few geometry-selected neurons.

Any constant value must be selected using validation data, not JailbreakBench.

---

## Existing code to reuse

Load the existing:

```text
selected_direction.pt
selection_result.json
```

Reuse:

```text
Qwen model loading
tokenization
generation
refusal evaluation
harmless evaluation
```

from the existing Arditi pipeline.

No new refusal metric should be introduced.

---

## New code

Create:

```text
reproductions/arditi_2024/ablations/mlp_localization/
├── __init__.py
├── geometry.py
├── hooks.py
├── run_geometry.py
└── run_behavior.py
```

Only create `run_behavior.py` if Phase B is attempted.

Tests:

```text
tests/reproductions/test_arditi_2024_mlp_localization.py
```

At minimum, test that the neuron hook:

```text
changes only selected coordinates
does not change unselected coordinates
preserves tensor shape
```

---

## What should remain experiment-specific

Do not immediately add Qwen MLP module-name logic to:

```text
src/refusal_toolkit/
```

The first implementation should stay under the ablation folder.

If later another paper requires the same generic neuron-intervention primitive, then it can be refactored into the toolkit.

---

## Definition of done

The required experiment should answer:

> Is the Arditi refusal direction unusually aligned with a small number of MLP neuron-output vectors at layer 15?

The optional behavioral extension should answer:

> Does suppressing those aligned neurons affect refusal more than suppressing random neurons?

---

# 11. Git Workflow

Each student should work on a separate branch.

Example:

```powershell
git checkout main
git pull
git checkout -b ablation/token-position
```

Suggested branch names:

```text
ablation/selection-constraints
ablation/token-position
ablation/matched-contrasts
ablation/mlp-localization
```

Before committing:

```powershell
python -m pytest tests -q
python -m ruff check src reproductions tests
git status
```

Generated artifacts, model checkpoints, and large datasets should not be committed.

Commit:

```text
source code
unit tests
small configuration files
documentation/report
```

Do not commit:

```text
models/
artifacts/
API keys
large generated completion files
temporary notebooks
cache files
```

---

# 12. Testing Expectations

Every new mathematical or hook utility should receive a small deterministic test.

GPU integration tests are not required for every operation.

For example:

### Direction normalization

Test using tiny vectors.

### Selection filtering

Create synthetic candidate scores and verify which candidate survives.

### Token-position indexing

Use a toy tensor and verify the correct position is selected.

### Neuron hook

Use a small tensor and verify exactly the intended coordinate changes.

The goal of unit tests is to catch implementation mistakes without requiring a Qwen checkpoint.

---

# 13. Common Mistakes to Avoid

## Do not modify the original reproduction to create the ablation

Bad:

```text
change selection.py until it behaves differently
```

Good:

```text
keep selection.py unchanged
implement an alternative policy inside the ablation folder
```

---

## Do not redefine existing metrics

If the repository already has refusal scoring, KL calculation, generation, or substring evaluation, reuse it.

---

## Do not tune on the test set

JailbreakBench is evaluation data.

Do not repeatedly test several coefficients and report whichever gives the largest ASR.

---

## Do not confuse direction norm with direction orientation

This is especially important for activation addition.

If two extraction methods produce different vector norms, raw steering strength is not a pure comparison of direction quality.

Use norm-matched controls where appropriate.

---

## Do not assume paper architecture names match Qwen-1.8B

In particular, Apple uses terminology such as:

```text
down_proj
```

for newer architectures.

The exact Qwen-1.8B module tree must be inspected before attaching neuron hooks.

---

## Do not leave hooks registered

Every intervention should use controlled registration/removal.

A hook accidentally left attached can contaminate every subsequent condition.

---

# 14. Student Deliverable

Each student should eventually submit:

```text
1. research question
2. exact baseline
3. exact variable changed
4. implementation
5. test(s)
6. experiment configuration
7. result table/plot
8. short interpretation
9. limitations
```

The final report should distinguish:

```text
Observation:
What the experiment actually measured.

Interpretation:
What mechanism might explain the observation.

Claim:
What the evidence is strong enough to support.
```

Do not turn a single-model observation into a universal statement about LLM refusal.

---

# 15. Recommended Assignment

If four students are available, a reasonable allocation is:

### Student 1 — Selection constraints

Best for someone who should first become comfortable with the repository and evaluation pipeline.

Main work is controlled experiment design rather than new model internals.

### Student 2 — Token position

Still relatively simple, but requires understanding residual activations, chat-template tokens, and causal evaluation.

### Student 3 — Matched contrasts

Best assigned to the student already reading Ball et al.

The central contribution is the dataset/contrast construction rather than new intervention machinery.

Ball et al. construct harmfulness contrasts using minimally modified harmless versions of harmful prompts, specifically to preserve much of the original wording and structure.

### Student 4 — MLP localization

Best for the strongest student technically.

This student must understand:

```text
residual stream
MLP intermediate activations
linear-layer matrix orientation
forward hooks
geometric cosine analysis
random controls
```

The Apple paper provides the motivation: it asks whether Arditi's distributed residual direction can correspond to a much more localized neuron-level mechanism, and explicitly studies geometric alignment between refusal directions and MLP neuron write vectors.

---

# 16. Overall Scientific Story

Although these are four independent small projects, together they form one coherent investigation.

Starting from the reproduced Arditi direction:

```text
How was it selected?
        ↓
Project 1
Selection constraints

Where in the prompt does it appear?
        ↓
Project 2
Token position

How sensitive is it to the way harmful/harmless
contrasts are constructed?
        ↓
Project 3
Matched contrasts

How localized is its implementation inside the model?
        ↓
Project 4
MLP neurons
```

The overall question is therefore not simply:

> Can we reproduce Arditi?

That part is already done.

The next question is:

> **How robust, localized, and method-dependent is the refusal direction that Arditi identifies?**

Each student attacks one part of that question while sharing the same validated baseline.