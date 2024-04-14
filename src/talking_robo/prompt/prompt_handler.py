import yaml
from typing import Optional
from dataclasses import dataclass

from talking_robo.exceptions import LangchainException


@dataclass
class PromptCreator:
    question: str
    model: str
    keep_memory: Optional[bool] = False

    def __post_init__(self):
        self.validate_model()

    def validate_model(
        self
    ):
        """Validate the specified model"""
        if self.model.lower() not in ["gemini", "mistral", "chatgpt"]:
            raise LangchainException(
                errors=f"Model {self.model} will be supported in the future. "
                f"You can select models from Gemini, Mistral and ChatGPT. "
                f"For now, the default model will be set to Gemini."
            )

    def yml_loader(
        self
    ) -> dict:
        """Load the prompt template"""
        with open(file_path) as f:
            try:
                loaded_data = yaml.safe_load(f)
                return loaded_data
            except yaml.YAMLError as e:
                logger.exception(f"Error while loading YAML from file: {e}")


if __name__ == "__main__":
    prompt_class = PromptCreator(
        question="What are your capabilities?",
        model="Claude"
    )