<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">KNN Classification on Breast Cancer Dataset</div>

# 1. Project Overview
This project demonstrates a comprehensive implementation of K-Nearest Neighbors (KNN) classification using the Breast Cancer Wisconsin dataset. The implementation includes data preprocessing, model training, and extensive evaluation metrics.

# 1.1 Project Structure
- `knn_classification.py`: Main Python script containing the KNN classification implementation
- `PROJECT_QA.md`: Documentation of project questions and answers
- `requirements.txt`: Python package dependencies
- `environment.yml`: Conda environment configuration
- `.gitignore`: Git ignore file for Python projects

# 1.2 Dependencies
- numpy
- pandas
- scikit-learn

# 1.3 Features
## 1.3.1 Breast Cancer Dataset Analysis
- Uses the Breast Cancer Wisconsin dataset (built into scikit-learn)
- Implements data standardization using StandardScaler
- Splits data into training (80%) and testing (20%) sets
- Trains a KNN classifier with k=5
- Provides comprehensive model evaluation metrics including:
  - Classification report
  - Confusion matrix
  - Accuracy, Precision, Recall, and F1 scores

## 1.3.2 Toy Dataset Demonstration
- Creates a synthetic dataset with similar characteristics
- Demonstrates model prediction capabilities
- Shows class probabilities and confidence scores
- Provides clear interpretation of results (Malignant/Benign classification)

# 1.4 Dataset Information
The Breast Cancer Wisconsin dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The dataset includes 569 samples with 30 features each, and the target variable indicates whether the tumor is malignant (1) or benign (0).

# 1.5 How to Run
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```
or using conda:
```bash
conda env create -f environment.yml
conda activate knn-classification
```

2. Run the script:
```bash
python knn_classification.py
```

The script will output:
- Classification report
- Confusion matrix
- Detailed metrics (Accuracy, Precision, Recall, F1 Score)
- Toy dataset predictions with confidence scores
- Feature names used in the analysis

# 1.6 Results Interpretation
- The model provides both classification results and probability scores
- Results are clearly labeled as Malignant or Benign
- Confidence scores are provided as percentages
- All 30 features are documented for reference
