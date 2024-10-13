from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import os
import torch
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

if not HUGGINGFACE_TOKEN or not MODEL_NAME:
    logger.error("Environment variables HUGGINGFACE_TOKEN and MODEL_NAME must be set.")
    raise Exception("Environment variables HUGGINGFACE_TOKEN and MODEL_NAME must be set.")

# Load the model
try:
    logger.info("Loading model...")
    model = pipeline(
        "text-generation",
        model=MODEL_NAME,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float32,  # Explicitly set dtype for CPU
        device_map="cpu",           # Ensure model loads on CPU
        trust_remote_code=True      # Add if necessary
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Model loading failed")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Define the request model
class GenerateRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        prompt = request.prompt
        logger.info(f"Received prompt: {prompt}")
        output = model(prompt, max_length=100)
        generated_text = output[0]["generated_text"]
        logger.info("Text generation successful.")
        return {"generated_text": generated_text}
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}
