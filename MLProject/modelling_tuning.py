import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

def main():
    # Load data yang sudah diproses
    data_path = "netflixmovie_preprocessing.csv"
    df = pd.read_csv(data_path)
    
    # Preprocessing sederhana (Encoding teks ke angka)
    target_column = 'total_title' 
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup Hyperparameter Tuning
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)

    # 4. Manual Logging
    # mlflow.set_experiment("Eksperimen_Tuning_Netflix")
    
    with mlflow.start_run(run_name="Manual_Logging_Tuning", nested=true):
        print("Sedang melakukan tuning... mohon tunggu.")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Hitung metrik secara manual
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # LOGGING MANUAL
        mlflow.log_params(grid_search.best_params_) # Catat parameter terbaik
        mlflow.log_metric("mse", mse)               # Catat MSE
        mlflow.log_metric("r2_score", r2)           # Catat R2
        mlflow.sklearn.log_model(best_model, "model_tuned") # Simpan model

        print(f"Tuning selesai! R2 Score: {r2:.4f}")

if __name__ == "__main__":

    main()
