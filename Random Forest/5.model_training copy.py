import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class AviationSurvivalPredictor:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """Initialize with data and configuration"""
        self.data = pd.read_csv(data_path)
        self.features = ['year', 'aboard', 'ground', 'fatalities_ratio', 'ground_to_aboard_ratio']
        self.target = 'survival_rate'
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.results = {}

        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)

    def analyze_target_variable(self):
        """Analyze the target variable distribution and statistics"""
        target_data = self.data[self.target]

        stats = {
            'count': len(target_data),
            'mean': target_data.mean(),
            'median': target_data.median(),
            'std': target_data.std(),
            'min': target_data.min(),
            'max': target_data.max(),
            'quartiles': target_data.quantile([0.25, 0.5, 0.75]).to_dict(),
            'zero_survival_count': len(target_data[target_data == 0]),
            'full_survival_count': len(target_data[target_data == 1]),
            'partial_survival_count': len(target_data[(target_data > 0) & (target_data < 1)])
        }

        # Calculate percentage distributions
        total_count = stats['count']
        stats['zero_survival_percent'] = (stats['zero_survival_count'] / total_count) * 100
        stats['full_survival_percent'] = (stats['full_survival_count'] / total_count) * 100
        stats['partial_survival_percent'] = (stats['partial_survival_count'] / total_count) * 100

        # Create distribution plot
        plt.figure(figsize=(12, 6))
        sns.histplot(data=target_data, kde=True, bins=50)
        plt.title('Distribution of Survival Rates')
        plt.xlabel('Survival Rate')
        plt.ylabel('Frequency')

        # Add reference lines
        plt.axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
        plt.axvline(stats['median'], color='g', linestyle='--', label=f"Median: {stats['median']:.3f}")

        plt.legend()
        plt.tight_layout()
        plt.show()

        return stats

    def prepare_data(self):
        """Prepare and split the data"""
        X = self.data[self.features]
        y = self.data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return self

    def train_random_forest(self):
        """Train and optimize Random Forest model"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=self.random_state),
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring='r2'
        )

        print("Training Random Forest model...")
        grid_search.fit(self.X_train, self.y_train)

        self.models['rf'] = grid_search.best_estimator_
        self.results['rf'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

        return self

    def evaluate_model(self):
        """Evaluate model performance"""
        if 'rf' not in self.models:
            raise ValueError("Model has not been trained yet. Call train_random_forest() first.")

        model = self.models['rf']
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'r2': r2_score(self.y_test, y_pred),
            'cv_scores': cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
        }

        # Additional evaluation metrics
        predictions = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': y_pred,
            'Absolute_Error': abs(self.y_test - y_pred)
        })

        metrics['max_error'] = predictions['Absolute_Error'].max()
        metrics['prediction_std'] = predictions['Predicted'].std()
        metrics['predictions_within_10_percent'] = (
                (predictions['Absolute_Error'] <= 0.1).mean() * 100
        )

        self.results['rf'].update(metrics)
        return self

    def print_target_analysis(self, stats):
        """Print detailed target variable analysis"""
        print("\n" + "=" * 50)
        print(" Target Variable Analysis ".center(50, "="))
        print("=" * 50 + "\n")

        print("Dataset Statistics:")
        print(f"  Total Records:        {stats['count']}")
        print(f"  Mean Survival Rate:   {stats['mean']:.6f}")
        print(f"  Median Survival Rate: {stats['median']:.6f}")
        print(f"  Standard Deviation:   {stats['std']:.6f}")
        print(f"  Minimum Value:        {stats['min']:.6f}")
        print(f"  Maximum Value:        {stats['max']:.6f}")

        print("\nSurvival Distribution:")
        print(f"  Zero Survival:        {stats['zero_survival_count']} ({stats['zero_survival_percent']:.2f}%)")
        print(f"  Partial Survival:     {stats['partial_survival_count']} ({stats['partial_survival_percent']:.2f}%)")
        print(f"  Full Survival:        {stats['full_survival_count']} ({stats['full_survival_percent']:.2f}%)")

        print("\nQuartile Distribution:")
        print(f"  25th Percentile:      {stats['quartiles'][0.25]:.6f}")
        print(f"  50th Percentile:      {stats['quartiles'][0.5]:.6f}")
        print(f"  75th Percentile:      {stats['quartiles'][0.75]:.6f}")

        print("\nData Range:")
        print(f"  Range:                {stats['max'] - stats['min']:.6f}")
        print(f"  Interquartile Range:  {stats['quartiles'][0.75] - stats['quartiles'][0.25]:.6f}")

    def print_summary(self):
        """Print comprehensive model performance summary"""
        print("\n" + "=" * 50)
        print(" Model Performance Summary ".center(50, "="))
        print("=" * 50 + "\n")

        print("Best Model Parameters:")
        for param, value in self.results['rf']['best_params'].items():
            print(f"  {param:20}: {value}")

        print("\nCross-validation Results:")
        print(f"  Best R² Score:        {self.results['rf']['best_score']:.6f}")
        print(f"  Mean CV R² Score:     {self.results['rf']['cv_scores'].mean():.6f}")
        print(f"  CV R² Score Std:      {self.results['rf']['cv_scores'].std():.6f}")

        print("\nTest Set Metrics:")
        print(f"  Mean Absolute Error:  {self.results['rf']['mae']:.6f}")
        print(f"  Root Mean Sq Error:   {self.results['rf']['rmse']:.6f}")
        print(f"  R² Score:            {self.results['rf']['r2']:.6f}")
        print(f"  Maximum Error:        {self.results['rf']['max_error']:.6f}")
        print(f"  Prediction Std Dev:   {self.results['rf']['prediction_std']:.6f}")
        print(f"  Within 10% Accuracy:  {self.results['rf']['predictions_within_10_percent']:.2f}%")

    def save_model(self, filepath='best_model.pkl'):
        """Save the trained model"""
        joblib.dump(self.models['rf'], filepath)
        print(f"\nModel saved successfully to: {filepath}")


# Run the analysis
if __name__ == "__main__":
    try:
        predictor = AviationSurvivalPredictor('Enhanced_Airplane_Crashes.csv')

        # Analyze target variable
        target_stats = predictor.analyze_target_variable()
        predictor.print_target_analysis(target_stats)

        # Train and evaluate model
        predictor.prepare_data() \
            .train_random_forest() \
            .evaluate_model()

        # Print results and save model
        predictor.print_summary()
        predictor.save_model()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()