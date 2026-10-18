"""Prepare radiology report examples for MedGemma supervised fine-tuning.

The labeled competition data is split into train/test JSONL files.  The
competition's unlabeled test set is exported separately for inference and is
never included in the supervised split.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


REQUIRED_TRAIN_COLUMNS = {
	"case_id",
	"modality",
	"body_part",
	"study_description",
	"patient_age_band",
	"patient_sex",
	"template_content",
	"dictation",
	"report",
}
REQUIRED_TEST_COLUMNS = REQUIRED_TRAIN_COLUMNS - {"report"}

SYSTEM_PROMPT = (
	"You are a radiologist writing a structured imaging report. Convert the "
	"dictation into a complete report by editing the supplied normal template. "
	"Return only FINDINGS and IMPRESSION. Route each dictated finding to the "
	"matching template field, replace its normal statement when abnormal, "
	"preserve unmentioned normal statements, and do not infer unsupported "
	"findings. Keep the template wording and field order whenever possible."
)


def _clean_row(row: dict[str, str | None]) -> dict[str, str]:
	"""Normalize CSV headers and values, including a possible UTF-8 BOM."""
	return {
		(key or "").lstrip("\ufeff"): (value or "").strip()
		for key, value in row.items()
	}


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
	with path.open("r", encoding="utf-8-sig", newline="") as handle:
		reader = csv.DictReader(handle)
		columns = {(column or "").lstrip("\ufeff") for column in (reader.fieldnames or [])}
		missing = required_columns - columns
		if missing:
			raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

		rows = [_clean_row(row) for row in reader]

	for row_number, row in enumerate(rows, start=2):
		missing_values = [column for column in required_columns if not row[column]]
		if missing_values:
			raise ValueError(
				f"{path}:{row_number} has empty required values: {sorted(missing_values)}"
			)
	return rows


def _user_prompt(row: dict[str, str]) -> str:
	return (
		"Complete the report using this case information. The demographic and "
		"study fields are context only; the template and dictation are the "
		"sources of report findings.\n\n"
		f"Modality: {row['modality']}\n"
		f"Body part: {row['body_part']}\n"
		f"Study description: {row['study_description']}\n"
		f"Patient age band: {row['patient_age_band']}\n"
		f"Patient sex: {row['patient_sex']}\n\n"
		f"TEMPLATE:\n{row['template_content']}\n\n"
		f"DICTATION:\n{row['dictation']}"
	)


def _record(row: dict[str, str], include_answer: bool) -> dict[str, Any]:
	messages: list[dict[str, str]] = [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": _user_prompt(row)},
	]
	if include_answer:
		messages.append({"role": "assistant", "content": row["report"]})

	return {
		"case_id": row["case_id"],
		"messages": messages,
		"metadata": {
			"modality": row["modality"],
			"body_part": row["body_part"],
			"study_description": row["study_description"],
			"patient_age_band": row["patient_age_band"],
			"patient_sex": row["patient_sex"],
		},
	}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
	with path.open("w", encoding="utf-8") as handle:
		for record in records:
			handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def preprocess(
	train_csv: Path,
	competition_test_csv: Path,
	output_dir: Path,
	test_fraction: float = 0.15,
	seed: int = 42,
) -> dict[str, int]:
	"""Create supervised train/test JSONL and unlabeled inference JSONL."""
	if not 0 < test_fraction < 1:
		raise ValueError("test_fraction must be between 0 and 1")

	labeled_rows = _read_csv(train_csv, REQUIRED_TRAIN_COLUMNS)
	competition_rows = _read_csv(competition_test_csv, REQUIRED_TEST_COLUMNS)
	shuffled_rows = labeled_rows.copy()
	random.Random(seed).shuffle(shuffled_rows)
	test_size = round(len(shuffled_rows) * test_fraction)
	test_rows = shuffled_rows[:test_size]
	train_rows = shuffled_rows[test_size:]

	output_dir.mkdir(parents=True, exist_ok=True)
	_write_jsonl(output_dir / "train.jsonl", [_record(row, True) for row in train_rows])
	_write_jsonl(output_dir / "test.jsonl", [_record(row, True) for row in test_rows])
	_write_jsonl(
		output_dir / "competition_test.jsonl",
		[_record(row, False) for row in competition_rows],
	)

	manifest = {
		"seed": seed,
		"test_fraction": test_fraction,
		"train_rows": len(train_rows),
		"test_rows": len(test_rows),
		"competition_test_rows": len(competition_rows),
		"format": "JSONL chat messages compatible with Hugging Face/TRL SFT preprocessing",
	}
	(output_dir / "manifest.json").write_text(
		json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
	)
	return manifest


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--train-csv", type=Path, default=Path("train.csv"))
	parser.add_argument("--competition-test-csv", type=Path, default=Path("test.csv"))
	parser.add_argument("--output-dir", type=Path, default=Path("medGemma/processed_data"))
	parser.add_argument("--test-fraction", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	manifest = preprocess(
		args.train_csv,
		args.competition_test_csv,
		args.output_dir,
		test_fraction=args.test_fraction,
		seed=args.seed,
	)
	print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
	main()
