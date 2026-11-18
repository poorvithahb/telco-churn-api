import os
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from src.data_prep import load_and_prep_data

def train_models(data_path):
    # --- FIX FOR MLFLOW UPDATE & WINDOWS PATH ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # --------------------------------------------

    # 1. Load and prepare the data
    X_train, X_test, y_train, y_test, preprocessor = load_and_prep_data(data_path)
    
    # 2. Set the MLflow experiment name
    mlflow.set_experiment("Telco_Churn_MLOps")

    # 3. Define the models we want to try
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_f1 = 0

    # 4. Loop through the models, train, and log to MLflow
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Create a pipeline with preprocessing + model
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            
            # Train the model
            clf.fit(X_train, y_train)
            
            # Make predictions
            preds = clf.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            # Log parameters and metrics to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            
            # Notice how this line is indented to match the lines above it!
            mlflow.sklearn.log_model(clf, "model", serialization_format="cloudpickle")

            print(f"{model_name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

            # Keep track of the best performing model based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf

    # 5. Save the best model to the 'models' folder so our API can use it
    joblib.dump(best_model, 'models/best_model.pkl')
    print("\nSuccess! Best model saved to models/best_model.pkl")

if __name__ == "__main__":
    train_models('data/raw_data.csv')