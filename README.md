# Arditi Refusal Reproduction

This repository contains a careful reproduction and extension of refusal-direction
methods for large language models, beginning with Arditi et al. (2024),
*Refusal in Language Models Is Mediated by a Single Direction*.

## Goals

- Reproduce the core refusal-direction extraction procedure.
- Reproduce steering and ablation interventions.
- Validate results on selected models and datasets.
- Keep the implementation readable, reusable, and easy to extend.

## Project structure

```text
configs/        Experiment configuration files
notebooks/      Exploratory analysis and visualization
scripts/        Executable experiment entry points
src/            Reusable Python implementation
tests/          Unit and integration tests