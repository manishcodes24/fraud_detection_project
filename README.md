# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using supervised machine learning algorithms. The implementation includes data preprocessing, feature scaling, model training using Logistic Regression and Random Forest, and evaluation using standard classification metrics.

## Project Structure
fraud_detection_project/
│
├── data/ # Raw and processed datasets
├── models/ # Saved models (pickle format)
├── outputs/ # Evaluation plots and reports
├── src/ # Source scripts for each stage
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── evaluate_model.py
│ └── utils.py
├── main.py # Main script to run the pipeline
├── requirements.txt # Python dependencies
└── README.md # Project documentation


## Dataset

- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions, with 492 labeled as fraud.
- Features:
  - `Time`: Seconds elapsed between the transaction and the first transaction.
  - `Amount`: Transaction amount.
  - `V1` to `V28`: PCA-transformed features.
  - `Class`: Target variable (0 = legitimate, 1 = fraud).

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies:
    pip install -r requirements.txt

3. Download the dataset from Kaggle and place creditcard.csv inside the data/ directory.

4. Run the project:
    python main.py

### Processing Steps
    Data Preprocessing: Loads the raw dataset and checks for missing values.
    Feature Engineering: Standardizes Amount and Time, drops originals.
    Model Training: Trains Logistic Regression and Random Forest on training set.
    Evaluation: Prints classification metrics and saves confusion matrix and ROC curve.

#### Performance
| Model               | Accuracy | ROC AUC | Precision (fraud) | Recall (fraud) | F1-Score (fraud) |
| ------------------- | -------- | ------- | ----------------- | -------------- | ---------------- |
| Logistic Regression | 0.9992   | 0.9573  | 0.83              | 0.64           | 0.72             |
| Random Forest       | 0.9996   | 0.9528  | 0.94              | 0.81           | 0.87             |


##### Outputs
Confusion matrix and ROC curve plots are saved to the outputs/ directory.
Trained models are saved to the models/ directory in .pkl format.