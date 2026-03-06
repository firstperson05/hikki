# FastMind LM - Download TinyStories Dataset
# Save this file as UTF-8 WITHOUT BOM

Write-Host "FastMind LM - Downloading TinyStories Dataset" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# URLs
$trainUrl = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
$validUrl = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# Create data directory
New-Item -ItemType Directory -Force -Path "data" | Out-Null

# Download train
$trainFile = "data\train.txt"
if (-not (Test-Path $trainFile)) {
    Write-Host "Downloading train data (~1.8GB)..." -ForegroundColor Yellow
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($trainUrl, (Resolve-Path "data").Path + "\train.txt")
        Write-Host "OK: train.txt saved" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        Write-Host "Try manually: curl -L $trainUrl -o data\train.txt" -ForegroundColor Yellow
    }
} else {
    Write-Host "OK: train.txt already exists, skipping" -ForegroundColor Green
}

# Download validation
$valFile = "data\val.txt"
if (-not (Test-Path $valFile)) {
    Write-Host "Downloading validation data (~200MB)..." -ForegroundColor Yellow
    try {
        $wc = New-Object System.Net.WebClient
        $wc.DownloadFile($validUrl, (Resolve-Path "data").Path + "\val.txt")
        Write-Host "OK: val.txt saved" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: $_" -ForegroundColor Red
        Write-Host "Try manually: curl -L $validUrl -o data\val.txt" -ForegroundColor Yellow
    }
} else {
    Write-Host "OK: val.txt already exists, skipping" -ForegroundColor Green
}

# Stats
Write-Host ""
Write-Host "Dataset stats:" -ForegroundColor Cyan

if (Test-Path $trainFile) {
    $trainMB = [math]::Round((Get-Item $trainFile).Length / 1MB, 1)
    Write-Host "  Train : $trainMB MB"
} else {
    Write-Host "  Train : NOT FOUND" -ForegroundColor Red
}

if (Test-Path $valFile) {
    $valMB = [math]::Round((Get-Item $valFile).Length / 1MB, 1)
    Write-Host "  Valid : $valMB MB"
} else {
    Write-Host "  Valid : NOT FOUND" -ForegroundColor Red
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. cargo run --release -- tokenize --train-corpus data\train.txt --vocab-size 4096 --output tokenizer_4k.bpe"
Write-Host "  2. cargo run --release -- train --config configs\medium.toml --data data\train.txt --val data\val.txt --tokenizer tokenizer_4k.bpe"