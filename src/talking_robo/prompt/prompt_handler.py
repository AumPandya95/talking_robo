import os
import yaml
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from talking_robo.exceptions import LangchainException
from talking_robo.prompt.chain_creator import ChainCreator


@dataclass
class PromptHandler:
    user_question: str
    model: str
    hugging_face_pipeline: HuggingFaceEndpoint
    keep_memory: Optional[bool] = False

    def __post_init__(self):
        self.validate_model()
        self.prompt_template = self.yml_loader()
        self.generate_chain()

    def validate_model(
        self
    ) -> Optional[Exception]:
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
        root_path = os.path.join(Path.cwd().parent)
        file_path = os.path.join(
            root_path, f"conf/{model_name}_prompt_template.yml"
        )

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
    
    def load_format_instructions(
        self
    ) -> str:
        """Load format instructions"""
        response_schemas = [
            ResponseSchema(name=name, description=description)
            for name, description in self.prompt_template["prompt"][
                "response_schemas"
            ].items()
        ]
        output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )

        return output_parser


    def generate_chain(
        self
    ) -> str:
        """Generate prompt"""
        prompt_object = ChainCreator(
            user_question=self.user_question,
            template=self.prompt_template,
            pipeline=self.hugging_face_pipeline
        )
        self.generated_chain = prompt_object.chain
        # Get output parser
        self.llm_output_parser = self.load_format_instructions()
        format_instructions = self.llm_output_parser.get_format_instructions()

        # Invoking the llm
        self.response = self.generated_chain.invoke(
            {'question': self.user_question,
             'format_instructions': format_instructions},
        )


if __name__ == "__main__":
    prompt_class = PromptHandler(
        user_question="What are your capabilities?",
        model="gemini",
    )
    # print(prompt_class.generated_prompt)
