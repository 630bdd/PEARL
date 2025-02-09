import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

def load_model(
        base_model: str,
        load_in_8bit: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with proper configuration.

    Args:
        base_model: Path or name of the pretrained model
        load_in_8bit: Whether to load the model in 8-bit mode

    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(
            base_model,
            trust_remote_code=True
        )
        config.init_device = DEFAULT_DEVICE

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            config=config
        )
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        List of JSON objects

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file contains invalid JSON
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}") from e