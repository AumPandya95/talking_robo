import os
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer


app = FastAPI()

env_path = os.path.join(Path.cwd().parent, ".env")
load_dotenv(env_path)


def load_model():
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it",
                                              device_map="auto",
                                              token=os.environ.get("hugging_face_token")
                                              )
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it",
                                                 device_map="auto",
                                                 token=os.environ.get("hugging_face_token"),
                                                 use_cache=True
                                                 # quantization_config=quantization_config
                                                 )
    print(f"Model loaded in {time.time() - start_time}s.")
    return tokenizer, model

TOKENIZER, MODEL = load_model()


@app.get("/")
def get_model_inference(
    input_text: str
):
    input_ids = TOKENIZER(input_text, return_tensors="pt").to("mps")
    outputs = MODEL.generate(**input_ids, max_length=1000)

    return TOKENIZER.decode(outputs[0])
