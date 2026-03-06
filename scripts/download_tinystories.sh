#!/bin/bash

# Download TinyStories dataset for smoke tests
mkdir -p data
cd data

wget -O TinyStories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -O TinyStories-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

echo "TinyStories download complete."
