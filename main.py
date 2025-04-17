import pickle

# Load the trained model
with open('model/home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)
   

# Take user input
area = float(input("Enter area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Create input array (adjust this based on your model's expected input)
input_features = [[area, bedrooms, bathrooms]]

# Predict
predicted_price = model.predict(input_features)

print(f"Estimated House Price: ₹{predicted_price[0]:,.2f}")

    return model.predict([x])[0]

# Example usage
predicted_price = predict_price(1000, 2, 2, "1st Phase JP Nagar")
print(f"Predicted Price: ₹{predicted_price:.2f}")
