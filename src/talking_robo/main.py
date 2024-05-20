import os
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint

from talking_robo.prompt.prompt_handler import PromptHandler


app = FastAPI()

env_path = os.path.join(Path.cwd().parent, ".env")
load_dotenv(env_path)


def load_model():
    llm=HuggingFaceEndpoint(
        repo_id="google/gemma-1.1-2b-it",
        temperature=1,
        max_new_tokens=1000
    )

    return llm

LLM_HUGGINGFACE = load_model()


@app.get("/")
def get_model_inference(
    input_question: str
):
    # Prompt chain
    chain_object = PromptHandler(
        user_question=input_question,
        hugging_face_pipeline=LLM_HUGGINGFACE,
        model="gemini"
    )
    response = chain_object.response
    output_parser = chain_object.llm_output_parser

    return response, output_parser


if __name__ == "__main__":
    user_query = "How do we use decorators in Python?"
    response, parser = get_model_inference(
        input_question=user_query
    )
    # print(response)
