#!/bin/bash

# Create folder structure
mkdir -p data/images data/masks models

# Download model checkpoint
echo "Downloading model checkpoint..."
wget -O models/model_swinvit.pt https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt

# Download dataset ZIP from Google Drive (using gdown)
echo "Downloading dataset ZIP from Google Drive..."
gdown '1I1LR7XjyEZ-VBQ-Xruh31V7xExMjlVvi' -O ../data/dataset.zip

# Unzip the dataset
echo "Unzipping dataset..."
unzip -q ../data/dataset.zip -d data/

# Move images and labels into proper folders
echo "Organizing files..."
mv ../data/Task06_Lung/imagesTs/* ../data/images/
mv ../data/Task06_Lung/imagesTr/* ../data/images/
mv ../data/Task06_Lung/labelsTr/* ../data/masks/

# Clean up
rm -r ../data/Task06_Lung
rm ../data/dataset.zip

# Run Python script
echo "Running script..."
python3 main.py