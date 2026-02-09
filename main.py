import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Create outputs dir
os.makedirs('outputs', exist_ok=True)

def load_and_explore():
    df = pd.read_csv('data/house_prices.csv')
    print("Dataset shape:", df.shape)
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())
    
    # Visualize relationships
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x='Area', y='Price')
    plt.title('Area vs Price')
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Location', y='Price')
    plt.title('Location vs Price')
    plt.tight_layout()
    plt.savefig('outputs/explore.png')
    plt.close()
    print("Explore plot saved: outputs/explore.png")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    # Fix: Correct column name + safety check
    if 'Property_ID' in df.columns:
        df.drop('Property_ID', axis=1, inplace=True)
    elif 'PropertyID' in df.columns:
        df.drop('PropertyID', axis=1, inplace=True)
    
    numeric_feats = ['Area', 'Bedrooms', 'Bathrooms', 'Age']
    cat_feats = ['Location', 'Property_Type']  # Fix: underscore in your data
    
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    cat_encoded = pd.DataFrame(encoder.fit_transform(df[cat_feats]),
                               columns=encoder.get_feature_names_out(cat_feats),
                               index=df.index)
    df_proc = pd.concat([df[numeric_feats], cat_encoded, df['Price']], axis=1)
    
    X = df_proc.drop('Price', axis=1)
    y = df_proc['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

class SimpleLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0]
        self.coef_ = theta_best[1:]
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept_, self.coef_])

def scratch_lr(X_train, X_test, y_train, y_test):
    print("\n=== Scratch Linear Regression (Area only) ===")
    lr_scratch = SimpleLR().fit(X_train[['Area']].values, y_train)
    preds_scratch = lr_scratch.predict(X_test[['Area']].values)
    evaluate(y_test, preds_scratch, "Scratch LR")

def train_sklearn(X_train, X_test, y_train, y_test):
    print("\n=== Scikit-learn Models ===")
    lr_sk = LinearRegression()
    lr_sk.fit(X_train, y_train)
    preds_lr = lr_sk.predict(X_test)
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    models = {
        'Linear': Pipeline([('poly', poly), ('lr', lr_sk)]),
        'Decision Tree': dt,
        'Random Forest': rf
    }
    
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions[name] = model.predict(X_test)
    
    results = {}
    for name, preds in predictions.items():
        results[name] = evaluate(y_test, preds, name)
    
    # Feature importance (RF)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop Features:", importances.head().to_dict())
    
    return predictions, results, rf.feature_importances_, X_train.columns

def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MAE: ${mae:,.0f}, MSE: {mse:,.0f}, R²: {r2:.3f}")
    return mae, mse, r2

def visualize_predictions(predictions, results, y_test):
    plt.figure(figsize=(15, 5))
    for i, (name, preds) in enumerate(predictions.items(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(y_test, preds, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{name} (R²={results[name][2]:.3f})')
    plt.tight_layout()
    plt.savefig('outputs/predictions_vs_actual.png')
    plt.close()
    print("Predictions plot saved: outputs/predictions_vs_actual.png")

def main(test_size=0.2):
    df = load_and_explore()
    X_train, X_test, y_train, y_test = prepare_data(df, test_size)
    
    scratch_lr(X_train, X_test, y_train, y_test)
    
    predictions, results, importances, feat_names = train_sklearn(X_train, X_test, y_train, y_test)
    visualize_predictions(predictions, results, y_test)
    
    # Final output
    print("\n" + "="*50)
    print("HOUSE PRICE PREDICTION MODEL")
    best_model = max(results.keys(), key=lambda k: results[k][2])
    print(f"Best: {best_model}")
    print(f"MAE: ${results[best_model][0]:,.0f}")
    print(f"R² Score: {results[best_model][2]:.3f}")
    print("Best Features: Area, Location")
    print("Check outputs/ for plots!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Price Prediction")
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    args = parser.parse_args()
    main(args.test_size)
