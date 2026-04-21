# Load environment variables from .env securely
if (Test-Path ".env") {
    Write-Host "Loading variables from .env file..."
    Get-Content .env | Where-Object { $_ -match "^[\w]+=" } | ForEach-Object {
        $name, $value = $_.Split('=', 2)
        [System.Environment]::SetEnvironmentVariable($name, $value)
    }
} else {
    Write-Host "Warning: .env file not found. HF_TOKEN may be missing." -ForegroundColor Yellow
}

# Turn off oneDNN optimizations to prevent annoying warnings
$env:TF_ENABLE_ONEDNN_OPTS="0"

# Assuming you already have 'venv' created in this folder, activate it
if (Test-Path -Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating local Virtual Environment..."
    . .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Warning: Could not find venv. Running in global environment." -ForegroundColor Yellow
}

# Install necessary libraries from requirements file
Write-Host "Checking dependencies..."
pip install -r requirements.txt --no-input

# Run the training script!
Write-Host "Starting train.py locally on Windows..."
python train.py

Write-Host "Done!"
