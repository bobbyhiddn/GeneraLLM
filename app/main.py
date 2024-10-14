from flask import Flask, request, jsonify
from transformers import pipeline
import os
import torch
import logging
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

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
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Model loading failed")
    raise Exception(f"Model loading failed: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data['prompt']
        logger.info(f"Received prompt: {prompt}")
        output = model(prompt, max_length=100)
        generated_text = output[0]["generated_text"]
        logger.info("Text generation successful.")
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        logger.exception("Generation failed")
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500

@app.route('/')
def root():
    return jsonify({"message": "Hello from Flask!"})

if __name__ == '__main__':
    app.run(debug=True)