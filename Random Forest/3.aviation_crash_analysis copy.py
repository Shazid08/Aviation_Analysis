import pandas as pd

# Load the cleansed dataset
data = pd.read_csv('cleansed_airplane_crashes.csv')

# Feature Engineering
data['fatalities_ratio'] = data.apply(lambda x: x['fatalities'] / x['aboard'] if x['aboard'] > 0 else 0, axis=1)
data['ground_to_aboard_ratio'] = data.apply(lambda x: x['ground'] / x['aboard'] if x['aboard'] > 0 else 0, axis=1)

def classify_severity(fatalities):
    if fatalities == 0:
        return 'No Fatalities'
    elif fatalities <= 10:
        return 'Low'
    elif fatalities <= 50:
        return 'Medium'
    else:
        return 'High'

data['incident_severity'] = data['fatalities'].apply(classify_severity)

# Save the enhanced dataset
data.to_csv('Enhanced_Airplane_Crashes.csv', index=False)
print("Enhanced dataset created and saved as 'Enhanced_Airplane_Crashes.csv'.")
