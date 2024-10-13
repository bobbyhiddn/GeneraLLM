#!/bin/bash

# Load .env file
if [ ! -f .env ]; then
  echo ".env file not found!"
  exit 1
fi

# Read the .env file and export variables
export $(grep -v '^#' .env | xargs)

# Set the secrets using flyctl
flyctl secrets set \
  FLASK_SECRET_KEY="$FLASK_SECRET_KEY" \
  HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  MODEL_NAME="$MODEL_NAME"
  # Add any other environment variables here

echo "Secrets set successfully on Fly.io!"

# Deploy to Fly.io
fly deploy
