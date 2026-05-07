import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

from src.data_loader import load_data
from src.preprocessing import create_preprocessor

from src.app_config import (
    TARGET_COLUMN,
    MODEL_PATH,
    RANDOM_STATE
)

# ==================================================
# LOAD DATA
# ==================================================
df = load_data("data/Sample-Superstore.csv")

# ==================================================
# FEATURES & TARGET
# ==================================================
X = df.drop(TARGET_COLUMN, axis=1)

y = df[TARGET_COLUMN]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE
)

# ==================================================
# PREPROCESSOR
# ==================================================
preprocessor, numerical_cols = create_preprocessor(X)

# ==================================================
# PIPELINE
# ==================================================
pipeline = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('model', XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE
    ))
])

# ==================================================
# PARAM GRID
# ==================================================
param_grid = {
    'model__n_estimators': [50, 100],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [3, 5]
}

if len(numerical_cols) > 0:

    param_grid['preprocessor__num__pca__n_components'] = [0.95]

# ==================================================
# GRID SEARCH
# ==================================================
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error'
)

# ==================================================
# TRAIN MODEL
# ==================================================
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ==================================================
# SAVE MODEL
# ==================================================
joblib.dump(best_model, MODEL_PATH)

print("Model Saved Successfully")

# ==================================================
# EVALUATION
# ==================================================
datasets = {
    'Train': (X_train, y_train),
    'Validation': (X_val, y_val),
    'Test': (X_test, y_test)
}

metrics = []

for name, (X_data, y_data) in datasets.items():

    predictions = best_model.predict(X_data)

    mse = mean_squared_error(y_data, predictions)

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(y_data, predictions)

    r2 = r2_score(y_data, predictions)

    metrics.append([name, mse, rmse, mae, r2])

metrics_df = pd.DataFrame(
    metrics,
    columns=['Dataset', 'MSE', 'RMSE', 'MAE', 'R2']
)

print(metrics_df)

metrics_df.to_csv(
    'outputs/metrics.csv',
    index=False
)