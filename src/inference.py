# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import fire
import torch
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)

from src.utils.loader import load_model, load_json_data

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_DATASETS = ["CompQ", "WebQSP", "CWQ", "SiQA", "CSQA", "SVAMP"]


def generate_text(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompts: List[str],
        generation_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Generate text from given prompts using the model.

    Args:
        model: Loaded pretrained model
        tokenizer: Loaded tokenizer
        prompts: List of input prompts
        generation_config: Dictionary of generation parameters

    Returns:
        List of generated texts
    """
    default_config = {
        "max_new_tokens": 256,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0
    }
    if generation_config:
        default_config.update(generation_config)

    try:
        batch = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                **default_config,
                pad_token_id=tokenizer.eos_token_id
            )

        return [
            postprocess_response(tokenizer.decode(output, skip_special_tokens=True))
            for output in outputs
        ]
    except Exception as e:
        raise RuntimeError(f"Generation failed: {str(e)}") from e


def postprocess_response(response: str) -> str:
    """Extract the response content after the marker."""
    parts = response.split("### Response:")
    return parts[-1].strip() if len(parts) > 1 else response.strip()


def build_prompt_template(
        dataset: str,
        item: Dict[str, Any],
        use_uprise: bool = False,
        use_epr: bool = False
) -> List[str]:
    """Construct prompts based on dataset and configuration."""
    base_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

    templates = []
    elements = _get_prompt_elements(item, dataset)

    for element in elements:
        if use_uprise:
            prompt = _build_uprise_prompt(base_prompt, element)
        else:
            prompt = _build_standard_prompt(base_prompt, element, dataset, use_epr)
        templates.append(prompt)

    return templates


def _get_prompt_elements(item: Dict[str, Any], dataset: str) -> List[Dict[str, Any]]:
    """Organize prompt elements based on dataset type."""

    def format_options(choices: List[Dict[str, str]]) -> str:
        """Format multiple-choice options as a single string."""
        return ' '.join(f"{choice['label']}: {choice['text']}" for choice in choices)

    base_elements = [
        {
            "question": item["ori_question"],
            "example": item["ice_prompts_list"][0],
            "instruction": item["ori_instruction"]
        },
        {
            "question": item["para_question"],
            "example": item["ice_prompts_list"][0],
            "instruction": item["para_instruction"]
        }
    ]

    if dataset in {"CompQ", "WebQSP", "CWQ"}:
        return base_elements
    elif dataset in {"SiQA", "CSQA"}:
        options_str = format_options(item['options']['choices'])
        for element in base_elements:
            element["options"] = options_str
        if dataset == "SiQA":
            for element in base_elements:
                element["body"] = item["body"]
        return base_elements
    elif dataset == "SVAMP":
        for element in base_elements:
            element["body"] = item["body"]
        return base_elements

    return []


def _build_standard_prompt(
        base: str,
        element: Dict[str, Any],
        dataset: str,
        use_epr: bool
) -> str:
    """Construct standard prompt template."""
    components = [base]

    # Add instruction section
    if dataset in ["SiQA", "CSQA"]:
        components.append(
            "\n\n### Instruction: Select one of the following options for the correct answer to the question.")
    else:
        components.append("\n\n### Instruction: Answer the below question.")

    # Add example if using EPR
    if use_epr:
        components.append(f"\n\n### Example: {element.get('example', '')}")

    # Add question context
    if "body" in element:
        components.append(f"\n\n### Question: {element['body']} {element['question']}")
    else:
        components.append(f"\n\n### Question: {element['question']}")

    # Add options if available
    if "options" in element:
        components.append(f"\n\n### Options: {element['options']}")

    components.append("\n\n### Response:")
    return "".join(components)


def _build_uprise_prompt(base: str, element: Dict[str, Any]) -> str:
    """Construct UPRISE-style prompt."""
    components = [
        base,
        f"\n\n### Instruction: {element['instruction']}"
    ]

    if "options" in element:
        components.append(f"\n\n### Options: {element['options']}")

    components.append("\n\n### Response:")
    return "".join(components)


def evaluate_results(
        dataset: str,
        predictions: List[str],
        ground_truth: Dict[str, Any]
) -> Dict[str, bool]:
    """Evaluate model predictions against ground truth."""
    evaluation = {}

    if dataset in ["CompQ", "WebQSP", "CWQ"]:
        evaluation["original"] = _evaluate_qa(predictions[0], ground_truth["answers"])
        evaluation["paraphrased"] = _evaluate_qa(predictions[1], ground_truth["answers"])
    elif dataset in ["SVAMP"]:
        evaluation["original"] = _evaluate_mwp(predictions[0], ground_truth["answers"])
        evaluation["paraphrased"] = _evaluate_mwp(predictions[1], ground_truth["answers"])
    elif dataset in ["SiQA", "CSQA"]:
        evaluation["original"] = _evaluate_cr(predictions[0], ground_truth)
        evaluation["paraphrased"] = _evaluate_cr(predictions[1], ground_truth)

    return evaluation


def _evaluate_qa(prediction: str, answers: List[str]) -> bool:
    """Check if any answer is present in the prediction."""
    return any(
        str(ans).lower() in prediction.lower()
        for ans in answers if ans is not None
    )


def _evaluate_mwp(prediction: str, answers: List[str]) -> bool:
    """Check if any answer is present in the prediction."""
    return any(
        ' ' + str(ans) in prediction.lower() or ' ' + str(int(float(ans))) in prediction.lower()
        for ans in answers if ans is not None
    )


def _evaluate_cr(prediction: str, data: Dict[str, Any]) -> bool:
    """Check if correct multiple choice answer is predicted."""
    correct_answer = next(
        c["text"] for c in data["options"]["choices"]
        if c["label"] == data["options"]["answerKey"]
    )
    return (
            data["options"]["answerKey"] + ":" in prediction
            or correct_answer.lower() in prediction.lower()
    )


def main(
        dataset: str,
        model_path: str,
        epr: bool = False,
        uprise: bool = False
):
    """Main execution pipeline.

    Args:
        dataset: Name of the dataset to process
        model_path: Path to pretrained model
        epr: Enable example-based prompting
        uprise: Use UPRISE prompt format
    """
    # Validate dataset
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported: {SUPPORTED_DATASETS}")

    # Configure paths
    input_path = Path(f"output/paraphrase/{dataset}_paraphrase.json")
    output_path = Path(f"output/result/{dataset}_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize model and data
    model, tokenizer = load_model(model_path)
    dataset_items = load_json_data(str(input_path))

    results = []
    original_correct = 0
    paraphrased_correct = 0

    for item in tqdm(dataset_items, desc="Processing items"):
        try:
            # Generate prompts
            prompts = build_prompt_template(dataset, item, uprise, epr)

            # Generate responses
            responses = generate_text(model, tokenizer, prompts)

            # Evaluate results
            evaluation = evaluate_results(dataset, responses, item)

            # Record results
            record = {
                "original_question": prompts[0],
                "original_prediction": responses[0],
                "original_correct": evaluation["original"],
                "paraphrased_question": prompts[1],
                "paraphrased_prediction": responses[1],
                "paraphrased_correct": evaluation["paraphrased"]
            }
            results.append({**item, **record})

            # Update counters
            original_correct += int(evaluation["original"])
            paraphrased_correct += int(evaluation["paraphrased"])

        except Exception as e:
            print(f"Error processing item: {str(e)}")
            continue

    # Save results
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    total = len(dataset_items)
    print(f"\nEvaluation Summary ({dataset}):")
    print(f"Original Accuracy: {original_correct / total:.2%}")
    print(f"Paraphrased Accuracy: {paraphrased_correct / total:.2%}")


if __name__ == "__main__":
    fire.Fire(main)
