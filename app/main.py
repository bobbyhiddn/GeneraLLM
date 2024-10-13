from fastapi import FastAPI, HTTPException
from transformers import pipeline
import os
from dotenv import load_dotenv
import torch

app = FastAPI()

# Load environment variables
load_dotenv()

# Load models once when the server starts
try:
    token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = os.getenv("MODEL_NAME")  # Get model name from environment variable

    if not model_name:
        raise ValueError("MODEL_NAME environment variable is not set")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable is not set")

    # Load the model using the specified model name and Hugging Face token
    model = pipeline(
        "text-generation",
        model=model_name,
        use_auth_token=token,  # Make sure the token is passed correctly
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.post("/generate/")
async def generate_text(prompt: str, max_length: int = 100):
    try:
        result = model(prompt, max_length=max_length)
        return {"generated_text": result[0]['generated_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
