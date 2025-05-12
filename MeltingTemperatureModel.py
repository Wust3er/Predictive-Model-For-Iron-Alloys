import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = "Alloys(Comp&MeltTemp).csv"
data = pd.read_csv(file_path)

# Replace NaN values with zeros
data = data.fillna(0)

# Extract relevant features: Iron min and max compositions for other elements
composition_features = ["Iron (Fe)Fe min"] + [col for col in data.columns if "max" in col]
X = data[composition_features]  # Input features

y = data["Melting Onset (Solidus) (F)"]  # Target variable

# Standardize the data (SVR performs better with scaled data)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Create and train the SVR model
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred_scaled = svr_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # Transform predictions back to original scale

mae = mean_absolute_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel(), y_pred)
mse = mean_squared_error(scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel(), y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")



# Example alloy composition
new_data = {
    "Iron (Fe)Fe min": 80.0,
    "Carbon (C)C max": 12.6,
    "Silicon (Si)Si max": 2.5,
    "Manganese (Mn)Mn max": 0.8,
    "Phosphorus (P)P max": 0.1,
    "Sulfur (S)S max": 0.14,
    "Chromium (Cr)Cr max": 0.6,
    "Molybdenum (Mo)Mo max": 0.3,
    "Nickel (Ni)Ni max": 1.2,
    "Copper (Cu)Cu max": 0.3,
    "Vanadium (V)V max": 0.1,
    "Aluminum (Al)Al max": 0.02,
    "Magnesium (Mg)Mg max": 0.01,
    "Selenium (Se)Se max": 0.01,
    "Tin (Sn)Sn max": 0.01,
    "Arsenic (As)As max": 0.01
}

# Convert to DataFrame
example_input = pd.DataFrame([new_data], columns=composition_features)

# Check for missing values in example_input
#print("Any NaN in example_input before normalization:", example_input.isnull().values.any())

# Ensure total composition sums to 100
if example_input.sum(axis=1).iloc[0] != 100:
    print("Normalizing composition to sum to 100...")
    example_input = example_input.div(example_input.sum(axis=1), axis=0) * 100

# Replace any remaining NaN values
example_input = example_input.fillna(0)

# Verify the total composition again
#print("Total composition of new input:", example_input.sum(axis=1).iloc[0])

# Scale the input
example_input_scaled = scaler_X.transform(example_input)

# Debugging checks after scaling
#print("Any NaN in scaled example input:", np.isnan(example_input_scaled).any())
#print("Any infinite values in scaled example input:", np.isinf(example_input_scaled).any())

# Assert no NaN or infinite values
assert not np.isnan(example_input_scaled).any(), "Scaled input contains NaN values!"
assert not np.isinf(example_input_scaled).any(), "Scaled input contains infinite values!"

# Predict the melting onset temperature
predicted_temp_scaled = svr_model.predict(example_input_scaled)
predicted_temp = scaler_y.inverse_transform(predicted_temp_scaled.reshape(-1, 1)).ravel()

print(f"Predicted Melting Onset Temperature: {predicted_temp[0]} F")