#!/bin/bash

# FastMind LM Data Preparation Script
# Downloads TinyStories dataset for proper training

set -e

echo "📦 FastMind LM - Data Preparation"
echo "=================================="

# Create data directory
mkdir -p data

# Download TinyStories dataset (2GB total, ~2B tokens)
echo "📥 Downloading TinyStories dataset..."

# Training data
if [ ! -f "data/train.txt" ]; then
    echo "  Downloading training data (1.4GB)..."
    curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
         -o "data/train.txt"
    echo "  ✅ Training data saved to data/train.txt"
else
    echo "  ✅ Training data already exists"
fi

# Validation data
if [ ! -f "data/val.txt" ]; then
    echo "  Downloading validation data (600MB)..."
    curl -L "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \
         -o "data/val.txt"
    echo "  ✅ Validation data saved to data/val.txt"
else
    echo "  ✅ Validation data already exists"
fi

# Count tokens (rough estimate)
echo ""
echo "📊 Dataset Statistics:"
if [ -f "data/train.txt" ]; then
    TRAIN_SIZE=$(wc -c < data/train.txt)
    TRAIN_TOKENS=$((TRAIN_SIZE / 1))  # Rough estimate: ~1 byte per token
    echo "  Train: $(numfmt --to=iec $TRAIN_SIZE) ($(numfmt --to=si $TRAIN_TOKENS) tokens)"
fi

if [ -f "data/val.txt" ]; then
    VAL_SIZE=$(wc -c < data/val.txt)
    VAL_TOKENS=$((VAL_SIZE / 1))
    echo "  Valid: $(numfmt --to=iec $VAL_SIZE) ($(numfmt --to=si $VAL_TOKENS) tokens)"
fi

echo ""
echo "🎯 Ready for training!"
echo "   Use configs/small.toml for proper training"
echo "   Expected: 10,000+ steps, loss < 2.0, ppl < 8.0"
