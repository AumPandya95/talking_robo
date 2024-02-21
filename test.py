import os
import re
from pathlib import Path
from llama_cpp import Llama
from typing import List, Optional


# Path
MODEL_PATH = os.path.join(Path(__file__), "model/")

# device = "cuda" # the device to load the model onto
device = "mps"


def find_files(directory: str, file_search_term: Optional[str] = None) -> List[str]:
    """
    Scans a given directory and returns a list of files that match a given search term.
    :param directory: str - The path of the directory to be scanned
    :param file_search_term: Optional[str] - The search term to be matched in file names. If None, all files will be returned
    :return: List[str] - A list of file names matching the search term or all files in the directory if file_search_term is None
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if re.search(file_search_term, filename, re.IGNORECASE):
                file_list.append(filepath)

    return file_list[0]


def tokenize_prompt(prompt: dict) -> str:
    """
    Takes in a dictionary of prompt containing the role and content for the LLM response.

    :param prompt: str - Instruction to be given to the LLM model
    :return: str - Tokenized prompt
    """
    input_prompt = f"<s>[INST] {prompt.get('role')} [/INST]</s> {prompt.get('content')}"

    return input_prompt


def test_instruction_model(
    prompt_input: List[dict] | dict, model: Optional[str] = None
) -> dict:
    """
    Takes model name as input along with the content and questions for the llm.

    :param prompt_input: str - List of system prompts to be passed for the LLM inference
    :param model: Optional[str] - Model to be used for inference

    :return: dict - Output from the LLM in a dictionary
    """
    # Get the model path and load the model
    model_path = find_files("model/", model)
    llm_model = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_batch=512,
        n_threads=7,
        n_gpu_layers=2,
        verbose=True,
        seed=42,
    )
    # LLM inference
    final_output = []
    if isinstance(prompt_input, list):
        output = llm_model.create_chat_completion(prompt_input, max_tokens=4096)
    else:
        # Get tokenized prompts
        tokenized_prompt = tokenize_prompt(prompt=prompt_input)
        output = llm_model.create_completion(
            tokenized_prompt, echo=True, stream=False, max_tokens=4096
        )

    return output


if __name__ == "__main__":

    chat_messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {
            "role": "assistant",
            "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!",
        },
        {"role": "user", "content": "Do you have mayonnaise recipes?"},
    ]

    instruction_message = {
        "system": "Follow the instructions below to complete the task.",
        "user": "Create a python script to scan a directory and return the files of the directory that match a particular case insensitive string 'Mistral'. Add docstring as well.",
    }

    # Get LLM inference for chat
    llm_chat_output = test_instruction_model(
        model="mistral", prompt_input=chat_messages
    )
    # Get LLM inference for instruction
    llm_instruction_output = test_instruction_model(
        model="mistral", prompt_input=instruction_message
    )

    print(llm_chat_output)
    print(llm_instruction_output)
