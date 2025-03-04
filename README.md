# Emotion Classification Using Physiological Signals and Deep Learning

## Overview
This project implements a deep learning neural network with dropout layers and batch normalization to classify human emotions based on heart rate, blood oxygen levels (SpO2), and body temperature. The code generates synthetic data, extracts key statistical features, and trains the neural network to recognize emotions such as neutral, happy, sad, and angry.

## Features
- **Synthetic Data Generation**: Simulates physiological data for different emotional states.
- **Feature Extraction**: Computes statistical features from heart rate, SpO2, and body temperature.
- **Neural Network Model**: A multi-layer neural network with dropout and batch normalization.
- **Training & Evaluation**: Implements learning rate scheduling and tracks model accuracy.
- **Visualization**: Generates confusion matrices, ROC curves, and learning curves for model evaluation.

## Data Generation
The project synthesizes labeled datasets based on predefined physiological ranges:
- **Heart Rate (BPM)**: Categorized ranges for neutral, happy, sad, and angry states.
- **SpO2 Levels (%)**: Simulated SpO2 variations linked to emotional states.
- **Body Temperature (°C)**: Generates variations based on mood conditions.

## Model Architecture
- **Input Layer**: Accepts extracted statistical features.
- **Hidden Layers**: Four fully connected layers with batch normalization and dropout.
- **Output Layer**: Predicts one of four emotional states.

## Training & Performance
- **Optimization**: Uses Adam optimizer with CrossEntropy loss.
- **Learning Rate Scheduling**: Adjusts learning rate based on validation loss.
- **Performance Metrics**: Evaluates model accuracy, precision, recall, and F1-score.

## Results
- **Final Accuracy**: Displayed after training and evaluation.
- **Confusion Matrix**: Visualizes model performance across emotion classes.
- **ROC Curves**: Illustrates classification performance per class.
- **Training Curves**: Shows loss and accuracy trends over epochs.

## Usage
Run the python file. It's as simple as that.

## Dependencies
- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---
Developed as part of my biomedical engineering senior design project.
