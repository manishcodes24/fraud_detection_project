from src.data_preprocessing import load_and_inspect_data, save_cleaned_data
from src.feature_engineering import scale_features
from src.model_training import split_data, train_logistic_regression, train_random_forest, save_model
from src.evaluate_model import evaluate_model

if __name__ == "__main__":
    input_file = "/Users/manishraghunathareddy/Developer/Projects/fraud_detection_project/data/creditcard.csv"
    output_file = "/Users/manishraghunathareddy/Developer/Projects/fraud_detection_project/data/processed_data.csv"

    df = load_and_inspect_data(input_file)

    df_scaled = scale_features(df)

    save_cleaned_data(df_scaled, output_file)

    # 1. Split the data
    X_train, X_test, y_train, y_test = split_data(df_scaled)

    # 2. Train both models
    log_reg_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 3. Save models
    save_model(log_reg_model, "models/logistic_regression_model.pkl")
    save_model(rf_model, "models/random_forest_model.pkl")

    # 4. Evaluate both models
    evaluate_model(log_reg_model, X_test, y_test, "Logistic Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")

