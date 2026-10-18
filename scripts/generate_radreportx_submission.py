#!/usr/bin/env python3
"""Generate ``final_submission.csv`` with the bionlp/RadReportX checkpoint.

Example:
    python generate_radreportx_submission.py --output final_submission.csv

RadReportX is distributed as a Torchtune/PyTorch checkpoint rather than a
Transformers model. The checkpoint is about 16 GB and is downloaded on the
first run. A CUDA GPU is recommended; Apple MPS can also be selected.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download


MODEL_ID = "bionlp/RadReportX"
CHECKPOINT_FILE = "meta_model_0.pt"  # The checkpoint used by the author's inference.py.
TOKENIZER_FILE = "tokenizer.model"
REQUIRED_COLUMNS = {"case_id", "template_content", "dictation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("test.csv"))
    parser.add_argument("--output", type=Path, default=Path("final_submission.csv"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".model_cache") / "RadReportX",
        help="Directory that holds the downloaded checkpoint and tokenizer.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="Use auto to prefer CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Optional Hugging Face token (defaults to the HF_TOKEN environment variable).",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps was requested, but Apple MPS is unavailable.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_prompt(template: str, dictation: str) -> str:
    """Build a template-editing prompt for one competition case."""
    return f"""You are a radiologist preparing a final imaging report.

Edit the supplied normal template using only the dictated findings. Keep all
template statements that are not contradicted by the dictation, preserve the
template field order and wording whenever possible, and place each abnormality
under its matching FINDINGS field. Update IMPRESSION to summarize the important
abnormal findings. Do not infer facts not present in the template or dictation.

Return only the completed report, starting with FINDINGS: and ending with
IMPRESSION:. Do not include an explanation, markdown fences, or a preamble.

NORMAL TEMPLATE:
{template.strip()}

DICTATION:
{dictation.strip()}

COMPLETED REPORT:
"""


def format_report(generation: str) -> str:
    """Remove incidental chat text and guarantee the two required headings."""
    report = generation.strip()
    start = report.find("FINDINGS:")
    if start >= 0:
        report = report[start:]

    # The checkpoint was fine-tuned for disease extraction, so an off-task
    # response can be short. Retain that response but keep the CSV valid.
    if "IMPRESSION:" not in report:
        findings = report.removeprefix("FINDINGS:").strip()
        return f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{findings}".strip()
    if not report.startswith("FINDINGS:"):
        report = f"FINDINGS:\n{report}"
    return report.strip()


def download_model_files(model_id: str, model_dir: Path, token: str | None) -> tuple[Path, Path]:
    """Download only the two RadReportX files needed for inference."""
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id=model_id,
            filename=CHECKPOINT_FILE,
            local_dir=model_dir,
            token=token,
        )
    )
    tokenizer = Path(
        hf_hub_download(
            repo_id=model_id,
            filename=TOKENIZER_FILE,
            local_dir=model_dir,
            token=token,
        )
    )
    return checkpoint, tokenizer


def load_model(checkpoint: Path, tokenizer_path: Path, device: str):
    """Use RadReportX's published Torchtune loading procedure."""
    try:
        from torchtune import utils
        from torchtune.models import convert_weights
        from torchtune.models.llama3 import llama3_8b, llama3_tokenizer
        from torchtune.utils._checkpointing._checkpointer_utils import safe_torch_load
    except ImportError as exc:
        raise RuntimeError(
            "Torchtune is required for RadReportX. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    print("Constructing the Llama 3 8B model ...", flush=True)
    model = llama3_8b()
    state_dict = convert_weights.meta_to_tune(safe_torch_load(str(checkpoint)))
    model.load_state_dict(state_dict)
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    tokenizer = llama3_tokenizer(str(tokenizer_path))
    return model, tokenizer, utils


@torch.inference_mode()
def generate_reports(
    test: pd.DataFrame,
    model,
    tokenizer,
    tune_utils,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    reports: list[str] = []
    for index, row in enumerate(test.itertuples(index=False), start=1):
        prompt_text = make_prompt(row.template_content, row.dictation)
        prompt = torch.tensor(
            tokenizer.encode(prompt_text, add_bos=True, add_eos=False),
            dtype=torch.int,
            device=device,
        )
        output = tune_utils.generate(
            model=model,
            prompt=prompt,
            max_generated_tokens=max_new_tokens,
            temperature=temperature,
            stop_tokens=tokenizer.stop_tokens,
            pad_id=tokenizer.pad_id,
        )
        raw_output = tokenizer.decode(output[0])
        # Torchtune's generator may return either the continuation or prompt +
        # continuation, depending on the installed version.
        continuation = raw_output.split("COMPLETED REPORT:", maxsplit=1)[-1]
        reports.append(format_report(continuation))
        print(f"Generated {index}/{len(test)} reports", flush=True)
    return reports


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")

    test = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS.difference(test.columns)
    if missing:
        raise ValueError(f"{args.input} is missing required columns: {sorted(missing)}")

    device = resolve_device(args.device)
    print(f"Using {device}; downloading/loading {args.model} ...", flush=True)
    checkpoint, tokenizer_path = download_model_files(args.model, args.model_dir, args.hf_token)
    model, tokenizer, tune_utils = load_model(checkpoint, tokenizer_path, device)
    reports = generate_reports(
        test, model, tokenizer, tune_utils, device, args.max_new_tokens, args.temperature
    )

    submission = pd.DataFrame({"case_id": test["case_id"], "report": reports})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
