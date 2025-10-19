# CMPT 353 – Indoor/Outdoor Classification Project

This project explores whether a dual-sensor setup (e.g., barometer + magnetometer, or audio + light) can reliably distinguish indoor vs. outdoor environments without relying on GPS. The pipeline includes preprocessing sensor/audio data, extracting features, merging with ground-truth labels, and training a Random Forest classifier to evaluate performance.

# Main Dependencies:
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
librosa (for audio RMS feature extraction)
joblib

Or install them manually:
```
pip install numpy pandas scipy scikit-learn matplotlib seaborn librosa joblib
```
# Repository Structure

project/ 
- main.py 
- analysis.py 
- analysis_2.py 
- audio_process.py 
- feature_merge.py 
- labels.py 
- model_train.py 
- sensor_process.py 
- project_data/ # Raw CSVs, WAV audio, and manual labels 
- analysis_output/ # Generated figures & plots 
- *.csv # Merged datasets and results 

# Scripts Overview

main.py: Entry point that ties together preprocessing, feature extraction, merging, and model training. Run this to reproduce the pipeline.

audio_process.py: Extracts RMS audio features from .wav recordings using librosa.

sensor_process.py: Cleans and filters sensor CSVs (accelerometer, gyroscope, barometer, light, magnetometer). Applies a Butterworth filter.

labels.py: Loads manual ground-truth timestamps (.txt) and aligns them with sensor/audio data.

feature_merge.py: Joins audio features, sensor features, and labels into a synchronized dataset (merged_dataset.csv).

model_train.py: Trains a Random Forest classifier (200 trees, balanced class weights). Evaluates dual-sensor vs. all-sensor setups, outputs results to CSV.

analysis.py / analysis_2.py: Exploratory analysis and visualizations (distributions, filtering, statistical tests, misclassification).

# Usage

Preprocess audio and sensors + Merge features with labels + Train and evaluate models:
```
python main.py
```

Visualize results:
```
python analysis.py
python analysis_2.py
```

# Results

Best performance came from a dual-sensor pair (barometer + magnetometer), which slightly outperformed the all-sensor setup.

Adding more sensors did not always improve accuracy, showing the value of simplicity over complexity.

# Author

Anh Truong – CMPT 353, August 2025