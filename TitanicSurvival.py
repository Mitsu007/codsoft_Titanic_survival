import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from typing import Dict

# Load the dataset
file_path = r'C:\Users\ACER\Desktop\Suhas\Internship\Mitu internship\Titanic-Dataset.csv'
data = pd.read_csv(file_path)

# Drop unnecessary columns
data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Features and labels
X = data.drop(columns=['Survived'])
y = data['Survived']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Model pipeline with XGBoost
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

def predict_survival(input_data: Dict[str, object]):
    """
    Predict survival based on input features.

    :param input_data: Dictionary of input features.
    """
    input_df = pd.DataFrame([input_data])
    prediction = best_model.predict(input_df)[0]
    print("\nPrediction: The passenger " + ("survived." if prediction == 1 else "did not survive."))

# Example usage
user_input = {
    'Pclass': 2,
    'Sex': 'female',
    'Age': 28.0,
    'SibSp': 0,
    'Parch': 1,
    'Fare': 24.0,
    'Embarked': 'C'
}

predict_survival(user_input)
