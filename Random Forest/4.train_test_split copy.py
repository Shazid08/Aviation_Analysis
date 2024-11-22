from sklearn.model_selection import train_test_split
import pandas as pd

# Load the enhanced dataset
data = pd.read_csv('Enhanced_Airplane_Crashes.csv')

# Select features and the target variable
features = ['year', 'aboard', 'ground', 'fatalities_ratio', 'ground_to_aboard_ratio']
target = 'survival_rate'

# Create the feature matrix (X) and target vector (y)
X = data[features]
y = data[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and testing sets
print(f"Training data shape (X_train): {X_train.shape}")
print(f"Testing data shape (X_test): {X_test.shape}")
print(f"Training target shape (y_train): {y_train.shape}")
print(f"Testing target shape (y_test): {y_test.shape}")
