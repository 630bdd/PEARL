"""
Question Paraphrasing Pipeline with T5 Model

This script implements a batch processing pipeline for question paraphrasing using pre-trained T5 models.
It supports multiple dataset formats and produces standardized output with original/paraphrased question pairs.
"""

import argparse
import json
import os
from typing import List, Dict, Any, Union

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.utils.loader import load_json_data


def chunk_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into fixed-sized chunks.

    Args:
        input_list: Input list to be split
        chunk_size: Maximum size of each chunk

    Returns:
        List of chunks, each containing up to chunk_size elements

    Raises:
        ValueError: If chunk_size is not positive
    """
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk_size {chunk_size}, must be positive integer")

    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_dataset_keys(dataset: str) -> List[str]:
    """Get required keys for supported datasets.

    Args:
        dataset: Dataset identifier

    Returns:
        List of required field names

    Raises:
        ValueError: For unsupported dataset types
    """
    dataset_config = {
        "CompQ": ["answers", "ori_question"],
        "WebQSP": ["answers", "ori_question"],
        "CWQ": ["answers", "ori_question"],
        "SiQA": ["options", "body", "ori_question"],
        "CSQA": ["options", "ori_question"],
        "SVAMP": ["answers", "body", "ori_question"]
    }

    if dataset not in dataset_config:
        raise ValueError(f"Unsupported dataset: {dataset}. "
                         f"Valid options: {list(dataset_config.keys())}")

    return dataset_config[dataset]


def main():
    # Configuration parsing
    parser = argparse.ArgumentParser(
        description="Question Paraphrasing Pipeline with T5 Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["CompQ", "WebQSP", "CWQ", "SiQA", "SVAMP"],
                        help="Name of the dataset to process")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained T5 model directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for model inference")
    parser.add_argument("--output_dir", type=str, default="paraphrase",
                        help="Output directory for processed results")

    args = parser.parse_args()

    # Data loading and validation
    dataset_keys = get_dataset_keys(args.dataset)

    # Load main dataset
    test_data_path = os.path.join('data', 'test_data', f'{args.dataset}_test_data.json')
    test_data = load_json_data(test_data_path)

    # Load auxiliary data
    aux_data_path = os.path.join('data', 'test_data', f'{args.dataset}_aux_test.json')
    aux_data = load_json_data(aux_data_path)

    # Prepare questions
    questions = [data["ori_question"] for data in test_data]
    batched_questions = chunk_list(questions, args.batch_size)

    # Model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).to(device)

    # Question paraphrasing
    paraphrased_questions = []
    prompt_suffix = ' This sentence should be rewritten as:'

    for batch in tqdm(batched_questions, desc="Processing batches"):
        formatted_batch = [q + prompt_suffix for q in batch]
        inputs = tokenizer(
            formatted_batch,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=256,
            num_beams=5,
            early_stopping=True
        )

        decoded_outputs = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )
        paraphrased_questions.extend(decoded_outputs)

    # Verify output consistency
    if len(paraphrased_questions) != len(test_data):
        raise RuntimeError(
            f"Output count mismatch: {len(paraphrased_questions)} paraphrased vs "
            f"{len(test_data)} original questions"
        )

    # Prepare output structure
    processed_data = []
    for idx, data in enumerate(test_data):
        output_item = {
            "answers": data.get("answers"),
            "options": data.get("options"),
            "body": data.get("body"),
            "ori_question": data["ori_question"],
            "para_question": paraphrased_questions[idx],
            "ice_prompts_list": [data["epr_prompt"]],
            "ori_instruction": data["uprise_instruction"].replace("@@", data["ori_question"]),
            "para_instruction": data["uprise_instruction"].replace("@@", paraphrased_questions[idx])
        }

        # Handle dataset-specific substitutions
        if "body" in dataset_keys:
            for key in ["ori", "para"]:
                output_item[key + '_' + "instruction"] = (
                    output_item[key + '_' + "instruction"]
                    .replace("$$", data["body"])
                )

        processed_data.append(output_item)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.dataset}_paraphrase.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Processing completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()