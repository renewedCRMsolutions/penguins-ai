# File: setup_windows.ps1
# Save this as a PowerShell script

Write-Host "Setting up Penguins AI on Windows..." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..."
& .\penguins_env\Scripts\activate

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Run setup
Write-Host "Downloading NHL data..."
python setup/download_nhl_data.py

# Train model
Write-Host "Training Expected Goals model..."
python train/train_xg_model.py

Write-Host "Setup complete! To start the API, run:" -ForegroundColor Green
Write-Host "uvicorn api.main:app --reload" -ForegroundColor Yellow