# SBA 928: Market Research Prompt Engineering and Fine-Tuning

My project shows the ability to train an AI to give more factual and grounded answers, while calling out possible bias caused by the data set.

## Datasets

| Dataset | Source (Kaggle) | Contents |
|---|---|---|
| Shopping behavior | `lakshitmalhotra28/consumer-behavior-and-shopping-habits` | 3,900 customers (one row each): item, gender, age, location, payment, subscription |
| Competitor data | `mexwell/adidas-x-nike` | Adidas (2,625) and Nike (643) India product listings, April 2020, prices in INR |

`kagglehub` downloads both automatically on the first run. No Kaggle account is needed for these public datasets.

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU is recommended (trained on an RTX 4070 SUPER). `pyproject.toml` installs the PyTorch CUDA 13.2 build.
- The first model run downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB).

## Setup

```bash
uv sync --group dev
```

Installs every package, including the dev tools (pytest, ruff), at the exact versions in `uv.lock`.

## Run the pipeline

Run every command from the project root (the folder with `pyproject.toml`), in this order:

| Step | Command | Creates |
|---|---|---|
| 1. Generate + split the dataset | `uv run python -m sba_928_seth_cutright.step_01_generate_dataset` | `data/processed/` train (723) / validation (84) / test (96) + `manifest.json` (split 80/10/10 per topic, seed 42) |
| 2. Prompt engineering (base model, no training) | `uv run python -m sba_928_seth_cutright.step_02_prompt_engineering` | `artifacts/prompt_baseline/` |
| 3. LoRA fine-tuning | `uv run python -m sba_928_seth_cutright.step_03_fine_tune` | `artifacts/models/qwen-sba-928/` |
| 4. Base vs. tuned on the test split | `uv run python -m sba_928_seth_cutright.step_04_compare_models` | `artifacts/evaluation/` |
| 5. Bias audit | `uv run python -m sba_928_seth_cutright.step_05_bias_audit` | `artifacts/bias_audit/` |
| 6. Ask the tuned model a new question | `uv run python -m sba_928_seth_cutright.step_06_custom_inference` | `artifacts/custom_inference/` |

Step 6 accepts `--question "..."` and `--context "..."` (see `--help`).

**Fine-tuned adapter location:** `artifacts/models/qwen-sba-928/final/` (LoRA adapter + tokenizer, ~16 MB). Training settings are in `config/default.json`.

## Tests and code quality

```bash
uv run pytest
uv run ruff check .
```

- `pytest`: split sizes, no record ID in two splits, prompt variants are distinct, and the scoring functions (including regression checks for a fixed scorer bug and for scoring the region/age gender questions).
- `ruff`: style and common Python errors.

## Project structure

```
config/default.json            training settings
data/processed/                train / validation / test JSONL + manifest
src/sba_928_seth_cutright/     package: step_01 ... step_06 + helpers
    load_data.py, prepare_data.py   load + clean the Kaggle data
    modeling.py                     load Qwen (base or + adapter), generate replies
    prompts.py                      prompt variants (minimal / structured / grounded)
    metrics.py                      scoring (numbers quoted, conclusion correct)
tests/test_project.py          pytest tests
artifacts/                     outputs of steps 2-6 (predictions, adapter, evaluation, bias audit)
docs/                          written report
```

## Results

| | Correct conclusions | Percentages quoted |
|---|---|---|
| Best prompt variant, base model (validation, v1 data) | 1 / 12 | 54% |
| Base model (test) | 7 / 96 | 36% |
| **Fine-tuned model (test)** | **77 / 96** | **97%** |

### Bias audit: men vs. women questions, before and after mitigation

The first version (v1, git tag [`v1-before-mitigation`](https://github.com/sethcutright1-sketch/sba_928_seth_cutright/tree/v1-before-mitigation)) never concluded "women are more likely." It had only 9 training examples with that answer. Version 2 adds gender comparisons within each region and age group, which gives many different, naturally balanced examples, and splits train/validation/test separately for each topic (stratified).

| Correct answer | v1 tuned (training examples) | v2 tuned (training examples) |
|---|---|---|
| Women more likely | **0 / 6** (9) | **21 / 24** (210) |
| Men more likely | 3 / 3 (15) | 23 / 27 (204) |
| Similar rate | 3 / 3 (27) | 19 / 27 (186) |
| All test questions | 16 / 24 | 77 / 96 |

Fine-tuning worked much better than prompting. However, fine-tuning can cause problems of its own if not done right. For example, the model says "there are about 2x more subscribers than non-subscribers buying Boots," because that sentence was a constant in the training answers. I plan on fixing this.

Full analysis: see the report in `docs/`.

## Limitations

- Small test groups
- Likely synthetic data
- No competitor test question

## AI assistance

The AI helped me structure code, bug check, and brainstorm. `load_data.py`, the `STATE_TO_REGION` dictionary, and the factual sections of this README were written with AI assistance; I wrote the rest of the code myself.
