# LLM Refusal Research Toolkit

This repository provides reusable tools and paper-specific reproductions for studying refusal representations and interventions in large language models.

The first completed reproduction targets Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.

## Goals

- Reproduce refusal-related representation and intervention methods faithfully.
- Separate reusable utilities from paper-specific implementations.
- Support activation extraction, direction construction, steering, ablation, and evaluation.
- Make future reproductions easy to add without rewriting shared infrastructure.
- Clearly document methodological choices and reproduction deviations.

## Project structure

```text
src/refusal_toolkit/          Reusable, paper-agnostic utilities
reproductions/                Paper-specific reproduction code
tests/unit/                   Unit tests for reusable utilities
tests/reproductions/          Paper-specific correctness tests
docs/                         Reproduction reports and documentation
notebooks/                    Exploratory analysis and visualization
data/                         Local reproduction datasets
artifacts/                    Generated experiment outputs
```

## Reproductions

### Arditi et al. (2024)

The Qwen 1.8B Chat reproduction is complete for the current project scope.

Key reproduced results include:

- the selected refusal direction from the authors' released implementation,
- harmful-prompt refusal suppression through directional ablation and negative activation addition,
- refusal induction on harmless prompts through positive activation addition,
- near-exact Pile and Alpaca language-model loss results.

The reproduction also documents discrepancies between the published paper and released artifacts, as well as environment and evaluation limitations.

See [`docs/arditi_2024.md`](docs/arditi_2024.md) for the full reproduction report.