# Alphabet Soup Charity Deep Learning Challenge

## Overview
This project aims to build and optimize a neural network model that predicts whether applicants funded by Alphabet Soup will succeed in their ventures. The model is built and optimized using TensorFlow, Keras, and Keras Tuner for hyperparameter tuning.

## Project Files

### 1. `AlphabetSoupCharity.ipynb`
This notebook contains the initial implementation of the neural network model used to predict whether an organization will successfully use the funding provided by Alphabet Soup. It includes the data preprocessing, model training, and evaluation steps.

### 2. `AlphabetSoupCharity_Optimization.ipynb`
This notebook is an extension of the initial model, where various optimization techniques are applied, including the addition of more layers, increased epochs, and hyperparameter tuning using Keras Tuner.

### 3. `AlphabetSoupCharity.h5`
This is the saved model in HDF5 format after training the initial neural network. You can load this file to make predictions or further optimize the model.

### 4. `AlphabetSoupCharity_Optimization.h5`
This is the saved model in HDF5 format after applying hyperparameter tuning and optimization to the initial model. You can load this file to use the optimized model for making predictions or further analysis.

### 5. `Alphabet Soup Charity Analysis Report.docx`
This document contains a detailed analysis of the deep learning model's performance, comparing different optimization techniques, such as hyperparameter tuning and model architecture adjustments.

### 6. `tune_dir/`
This directory contains the output files from the hyperparameter tuning process using Keras Tuner. It includes the best model configurations found during the tuning process.

### 7. `checkpoint/` and `checkpoints/`
These directories store model checkpoints created during the training process. The checkpoints can be used to restore the model at a particular point during training or to continue training from a saved state.

### 8. `README.md`
This file provides an overview of the project and explains the contents of each file and folder in the project directory.

## Instructions for Use

1. **Run the Notebooks**:
   - Start with `AlphabetSoupCharity.ipynb` to see the initial model and training steps.
   - Use `AlphabetSoupCharity_Optimization.ipynb` to explore the optimization techniques applied to improve the model's performance.

2. **Load the Models**:
   - You can load the saved models (`AlphabetSoupCharity.h5` or `AlphabetSoupCharity_Optimization.h5`) to make predictions using:
     ```python
     from tensorflow.keras.models import load_model
     model = load_model('AlphabetSoupCharity.h5')  # Or load the optimization model
     ```

3. **Hyperparameter Tuning**:
   - If you want to run hyperparameter tuning, ensure that you have `Keras Tuner` installed and refer to the `tune_dir/` for the best configurations found during tuning.

## Model Summary

The project focuses on building a binary classification model using TensorFlow and Keras. The initial model has been improved through several optimization techniques, such as:
- Binning 'ASK_AMT' column
- Adding more hidden layers
- Increasing the number of epochs
- Hyperparameter tuning using Keras Tuner

The results and observations of the various optimizations are detailed in the **Alphabet Soup Charity Analysis Report.docx** file.
