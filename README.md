# Radiology Reporting Harness

I built this project to explore turning radiologist dictation into structured reports while preserving unchanged statements in a supplied normal template. The experiments compare Gemini and MedGemma. I fine-tuned MedGemma in Google Colab using the free-tier T4 GPU to accelerate the model’s compute-intensive training.

## Experiments and scores

| Approach | Prompting | Radiology Edit Score |
| --- | --- | ---: |
| Gemini 3.5 Flash-Lite | Zero-shot | **0.44760** |
| MedGemma 1.5 4B IT | Zero-shot | **0.47780** |
| Fine-tuned MedGemma 1.5 4B IT | One-shot | **0.40393** |

Lower Radiology Edit Score is better. The fine-tuned, one-shot MedGemma run scored 0.40393, compared with 0.47780 for zero-shot MedGemma. These runs use different prompting setups, so the scores compare the complete approaches rather than isolating the effect of fine-tuning.

| Fine-tuning setting | Value |
| --- | --- |
| Method | QLoRA with PEFT LoRA adapters and Hugging Face TRL `SFTTrainer` |
| Base model | `google/medgemma-1.5-4b-it` |
| Quantization | 4-bit NF4 with double quantization |
| Compute and storage dtype | BF16 when supported, otherwise FP16; storage uses the compute dtype |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | `all-linear` |
| Learning rate | `2e-4` |
| Epochs | 2 |
| Per-device train / eval batch size | 1 / 1 |
| Gradient accumulation | 8 steps |
| Optimizer | Not explicitly set in the notebook |

Final training loss at the end of epoch 2: **0.100544**.

## A tour of the repo

- `notebooks/gemini/` contains the Gemini zero-shot experiment.
- `notebooks/medgemma/` contains the MedGemma zero-shot and fine-tuned report-generation experiments.
- `notebooks/exploration/` holds an exploratory notebook.
- `scripts/` contains the submission utility and MedGemma preprocessing script.
- `docs/` contains task details and reporting guidance.

I ran experiments in Google Colab and local environments, so each notebook may need some environment-specific setup.

## Running the notebooks

Input data files aren't included in this repository. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Gemini inference needs a `GEMINI_API_KEY`. MedGemma requires Hugging Face access and acceptance of the model terms. Fine-tuning requires a CUDA GPU; the notebook recommends an A100, and notes that a T4 may not have enough memory. Keep credentials, downloaded data, processed data, generated submissions, and checkpoints local. They are excluded from Git on purpose.
