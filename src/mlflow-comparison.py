import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def process_and_compare():

    data = pd.read_csv('data/Rainfall.csv')
    data = data.dropna()
    data['rainfall'] = data['rainfall'].replace({'yes': 1, 'no': 0})
    print(data.head())

    # Basic preprocessing
    X = data.drop('rainfall', axis=1)
    y = data['rainfall']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    input_example = X_train.iloc[:1]

    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(n_estimators=100)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Log parameters, metrics, and model
            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("mean_squared_error", mse)
            mlflow.sklearn.log_model(model, "model", input_example=input_example)

            print(f"Model: {model_name} logged with MSE: {mse}")

if __name__ == '__main__':
    process_and_compare()