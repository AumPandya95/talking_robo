import os
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
        self.yml_loader()

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
        model_name = self.model.lower()
        file_path = os.path.abspath(f"conf/{model_name}_prompt_template.yml")

        try:
            with open(file_path) as f:
                try:
                    loaded_data = yaml.safe_load(f)
                    return loaded_data
                except yaml.YAMLError as e:
                    raise LangchainException(
                        errors=f"Error while loading YAML from file: {e}"
                    )
        except FileNotFoundError as e:
            raise LangchainException(
                errors=f"It seems the prompt template for {self.model} "
                f"does not exist. Please create one. "
                f"Error message: {e}"
            )


if __name__ == "__main__":
    prompt_class = PromptCreator(
        question="What are your capabilities?",
        model="mistral"
    )
