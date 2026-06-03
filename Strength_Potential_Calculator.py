import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import json

class LiftingStrengthPredictor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.features = ['Age', 'Body Weight in Pounds', 'Height in Inches', 'How many years have you been lifting?', 'BMI']
        self.targets = {
            'Squat': 'Back Squat 1RM in Pounds',
            'Bench': 'Bench 1RM in Pounds',
            'Deadlift': 'Deadlift 1RM in Pounds',
            'Total': 'Total_Lbs'
        }
        self.df = pd.read_csv(csv_path)
        self.prepare_data()

    def prepare_data(self):
        # Calculate BMI: (lbs * 0.453) / (inches * 0.0254)^2
        self.df['BMI'] = (self.df['Body Weight in Pounds'] * 0.453592) / ((self.df['Height in Inches'] * 0.0254) ** 2)
        
        # Ensure lifting columns are numeric
        lift_cols = ['Back Squat 1RM in Pounds', 'Bench 1RM in Pounds', 'Deadlift 1RM in Pounds']
        for col in lift_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Calculate Total SBD
        self.df['Total_Lbs'] = self.df[lift_cols].sum(axis=1)
        
        # Clean data: Remove rows with missing feature/target data and filter outliers
        self.df_clean = self.df.dropna(subset=self.features + lift_cols)
        self.df_clean = self.df_clean[self.df_clean['Total_Lbs'] > 100]

    def predict_user(self, age, weight, height, years_lifting, squat, bench, deadlift):
        # 1. Calculate user BMI for the prediction row
        bmi = (weight * 0.453592) / ((height * 0.0254) ** 2)
        user_row = pd.DataFrame([[age, weight, height, years_lifting, bmi]], columns=self.features)
        
        actual_lifts = {
            'Squat': squat,
            'Bench': bench,
            'Deadlift': deadlift,
            'Total': squat + bench + deadlift
        }
        
        results = {}

        # 2. Iterate through each lift target (S, B, D, and Total)
        for name, col_name in self.targets.items():
            X = self.df_clean[self.features]
            y = self.df_clean[col_name]
            
            # Train Regressor (GradientBoosting is highly similar to XGBoost)
            model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(X, y)
            
            # Generate Prediction
            pred = model.predict(user_row)[0]
            
            # Use RMSE (Model Error) to calculate the Z-Score and Probability
            train_preds = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, train_preds))
            
            actual = actual_lifts[name]
            z_score = (actual - pred) / rmse
            prob_exceed = 1 - norm.cdf(z_score)
            
            # Get Feature Importance (Proxy for LIME impact)
            importance = dict(zip(self.features, model.feature_importances_))
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

            results[name] = {
                'Predicted_lbs': round(pred, 1),
                'Actual_lbs': actual,
                'Strength_Gap': round(actual - pred, 1),
                'Z_Score': round(z_score, 2),
                'Prob_to_Exceed': f"{prob_exceed*100:.2f}%",
                'Top_Influences': sorted_importance[:3]
            }
            
        return results

# --- RUNNING THE ANALYSIS ---
csv_path = r'D:\OneDrive\Workout programs\WR 2021 Deadlift Data - Weightroom Survey 2021.csv'
predictor = LiftingStrengthPredictor(csv_path)

# Input your data here
report = predictor.predict_user(
    age=34, 
    weight=160, 
    height=69, 
    years_lifting=4, 
    squat=320, 
    bench=220, 
    deadlift=500
)

# Output the results
print(json.dumps(report, indent=2))