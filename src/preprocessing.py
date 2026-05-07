from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

def create_preprocessor(X):

    categorical_cols = X.select_dtypes(include=['object']).columns

    numerical_cols = X.select_dtypes(include=['number']).columns

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    return preprocessor, numerical_cols