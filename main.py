import pickle
import json
import numpy as np

# Load the trained model
with open('model/home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load column names
with open('model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Function to predict price
def predict_price(location, sqft, bath, bhk):
    # Convert both the location and columns to lowercase for case-insensitive matching
    location = location.lower()
    
    # Check if location exists in the columns
    if location not in data_columns:
        print(f"Error: The location '{location}' is not available in the model.")
        return None
    
    # Get the index of the location column
    loc_index = data_columns.index(location)
    
    x = np.zeros(len(data_columns))  # Create an array of zeros for features
    
    # Assign values to respective positions in the feature array
    x[0] = sqft  # Area (sqft)
    x[1] = bath   # Number of bathrooms
    x[2] = bhk    # Number of bedrooms
    
    # Set the location column to 1 if the location is specified
    x[loc_index] = 1

    # Predict and return the price
    return model.predict([x])[0]

# Take user inputs
print("Welcome to the House Price Prediction!")
location = input("Enter the location: ")  # User input for location
sqft = float(input("Enter the area in sqft: "))  # User input for area
bath = int(input("Enter number of bathrooms: "))  # User input for bathrooms
bhk = int(input("Enter number of bedrooms: "))  # User input for bedrooms

# Predict the price based on user input
predicted_price = predict_price(location, sqft, bath, bhk)

# Print the predicted price
if predicted_price is not None:
    print(f"Predicted House Price: ₹{predicted_price:,.2f}")


