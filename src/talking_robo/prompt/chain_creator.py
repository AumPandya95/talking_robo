from dataclasses import dataclass
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint


@dataclass
class ChainCreator:
    """
    Given a prompt template, this module resolves the input variables, 
    generates a prompt and a chain for the model
    """
    user_question: str
    template: str
    pipeline: HuggingFaceEndpoint

    def __post_init__(self):
        self.create_prompt()

    def create_prompt(
        self
    ):
        """
        Create prompts from the given template
        """
        system_prompt = PromptTemplate(
            template=self.template["prompt"]["system"],
            input_variables=["question", "format_instructions"]
        )

        self.chain = LLMChain(
            llm=self.pipeline,
            prompt=system_prompt
        )


if __name__ == "__main__":
    from talking_robo.prompt.prompt_handler import PromptHandler
    query = "How to add two numbers in python?"
    prompt_handler = PromptHandler(
        user_question=query,
        model="Gemini",
        keep_memory=False
    )
    prompt_template = prompt_handler.prompt_template
    print(f"Prompt template -> {prompt_template}")
    prompt_creator = ChainCreator(
        template=prompt_template,
        user_query=query
    )
    # print(prompt_creator.final_prompt)
