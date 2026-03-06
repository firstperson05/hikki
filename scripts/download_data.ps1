# Download TinyStories dataset
Write-Host "📦 FastMind LM - Downloading TinyStories Dataset" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

$trainUrl = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
$validUrl = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# Create data directory
New-Item -ItemType Directory -Force -Path "data" | Out-Null

Write-Host "📥 Downloading TinyStories train (~1.8GB)..." -ForegroundColor Yellow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-WebRequest -Uri $trainUrl -OutFile "data/train_full.txt" -UseBasicParsing
$stopwatch.Stop()

Write-Host "✅ Train data downloaded in $($stopwatch.Elapsed.ToString('mm\:ss'))" -ForegroundColor Green

Write-Host "📥 Downloading TinyStories valid (~200MB)..." -ForegroundColor Yellow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
Invoke-WebRequest -Uri $validUrl -OutFile "data/val_full.txt" -UseBasicParsing
$stopwatch.Stop()

Write-Host "✅ Valid data downloaded in $($stopwatch.Elapsed.ToString('mm\:ss'))" -ForegroundColor Green

# Check file sizes
$trainSize = (Get-Item "data/train_full.txt").Length / 1MB
$validSize = (Get-Item "data/val_full.txt").Length / 1MB

Write-Host ""
Write-Host "📊 Dataset Statistics:" -ForegroundColor Cyan
Write-Host "  Train: $([math]::Round($trainSize, 2)) MB"
Write-Host "  Valid: $([math]::Round($validSize, 2)) MB"
Write-Host "  Total: $([math]::Round($trainSize + $validSize, 2)) MB"

Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Green
Write-Host "  cargo run --release -- tokenize --train-corpus data/train_full.txt --vocab-size 4096 --output tokenizer_4k.bpe"
Write-Host "  cargo run --release -- train --config configs/medium.toml --data data/train_full.txt --val data/val_full.txt"
