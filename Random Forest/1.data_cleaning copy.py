import pandas as pd

# Load the dataset
data = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Clean 'Time': fill NaNs with 'Unknown', standardize the format for available times
data['Time'] = data['Time'].fillna('Unknown')

# Fill missing values in categorical fields with 'Unknown'
categorical_cols = ['Location', 'Operator', 'Flight #', 'Route', 'Type', 'Registration', 'cn/In', 'Summary']
for col in categorical_cols:
    data[col] = data[col].fillna('Unknown')

# Fill missing numerical values with 0
numerical_cols = ['Aboard', 'Fatalities', 'Ground']
for col in numerical_cols:
    data[col] = data[col].fillna(0)

# Feature Engineering: Add 'Year' column from 'Date'
data['Year'] = data['Date'].dt.year

# Calculate Survival Rate = (Aboard - Fatalities) / Aboard (when Aboard > 0)
data['Survival Rate'] = data.apply(lambda x: (x['Aboard'] - x['Fatalities']) / x['Aboard']
                                   if x['Aboard'] > 0 else 0, axis=1)

# Standardize column names for consistency
data.columns = [col.strip().replace(' ', '_').lower() for col in data.columns]

# Save the cleansed dataset
data.to_csv('Cleansed_Airplane_Crashes.csv', index=False)

print("Data cleansing complete. Cleansed data saved to 'Cleansed_Airplane_Crashes.csv'.")
