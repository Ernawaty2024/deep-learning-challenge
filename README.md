# Alphabet Soup Charity Funding Predictor

## Overview
This project aims to create a binary classifier using a deep learning neural network model that can predict whether applicants funded by the nonprofit organization, Alphabet Soup, will succeed in their ventures. The classifier is built using a dataset of over 34,000 organizations that have received funding, and it employs machine learning techniques to predict the success of future applicants.

## Purpose
The nonprofit foundation Alphabet Soup wants a tool to help select the applicants for funding who have the best chance of success. By building a binary classifier, the foundation will be able to make data-driven decisions on funding, improving the overall success rate of their investments.

## Dataset
The dataset contains information on various organizations, including details about the application type, affiliation, classification, income, and special considerations. The main target of the model is the `IS_SUCCESSFUL` column, which indicates whether an organization succeeded after receiving funding.

### Dataset Columns:
- **EIN** and **NAME**: Identification columns (removed during preprocessing).
- **APPLICATION_TYPE**: The type of application submitted.
- **AFFILIATION**: Sector affiliation of the organization.
- **CLASSIFICATION**: Government classification of the organization.
- **USE_CASE**: Reason for funding.
- **ORGANIZATION**: Type of organization.
- **STATUS**: Organization’s current status (active or inactive).
- **INCOME_AMT**: Income classification of the organization.
- **SPECIAL_CONSIDERATIONS**: Special considerations for the application.
- **ASK_AMT**: Amount of funding requested.
- **IS_SUCCESSFUL**: Binary target column indicating whether the funding was successful.

### Preprocess the Data
1. **Data Preprocessing**:
   - Read the data from the `charity_data.csv` file.
   - Remove unnecessary columns such as `EIN` and `NAME`.
   - Identify the target variable (`IS_SUCCESSFUL`) and feature variables.
   - Combine low-frequency categorical values into a new value labeled `Other` to reduce noise.
   - Use `pd.get_dummies()` to convert categorical data into numerical form.
   - Split the dataset into training and testing datasets using `train_test_split`.
   - Apply `StandardScaler` to scale the feature variables for better model performance.

### Compile, Train, and Evaluate the Model
1. **Model Definition**:
   - Use TensorFlow/Keras to design a neural network model for binary classification.
   - Compile the model using the `binary_crossentropy` loss function, the `adam` optimizer, and accuracy as the evaluation metric.
   
2. **Training**:
   - Train the model using the scaled training data.
   - Include a callback to save the model weights every five epochs.
   - Evaluate the model using the test data to determine its accuracy and loss.

3. **Output**:
   - Save the model in HDF5 format with the name `AlphabetSoupCharity.h5`.

### Optimize the Model
1. **Optimization Strategies**:
   - Adjust the input data by dropping columns or binning values differently.
   - Add more neurons or hidden layers to increase the model’s capacity.
   - Experiment with different activation functions for better performance.
   - Increase or decrease the number of epochs to fine-tune the model’s training.

    1.    
2. **Goal**:
   - The goal is to achieve an accuracy of over 75%. The model can be optimized using any of the strategies above. Results are saved as `AlphabetSoupCharity_Optimization.h5`.

### Step 4: Write a Report on the Neural Network Model
1. **Overview**: Briefly explain the purpose of the analysis.
   
2. **Data Preprocessing**:
   - **Target**: The `IS_SUCCESSFUL` column.
   - **Features**: Columns such as `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, etc.
   - **Removed Variables**: Columns like `EIN` and `NAME`, which are not useful for predictions.

3. **Model Training and Evaluation**:
   - Detail the number of layers, neurons, and activation functions chosen for the model.
   - Indicate whether the target performance was achieved.
   - List optimization attempts and their effects on the model's performance.

4. **Summary**:
   - Summarize the overall results.
   - Provide recommendations on how alternative models (e.g., random forest, support vector machines) could potentially improve the prediction accuracy.

### Step 5: Finalize and Submit the Project
1. **Move Files**:
   - Download the Colab notebooks used during the project.
   - Place them in local GitHub repository folder.
   
2. **Push to GitHub**:
   - Push the project files to GitHub for final submission.

## Conclusion
By creating and optimizing this binary classifier, Alphabet Soup can make better decisions on which applicants to fund, maximizing the success rate of its investments. The model created in this project serves as a foundation for future improvements in the decision-making process.

This README serves as a guide to understanding and replicating the model development process for predicting applicant success.
