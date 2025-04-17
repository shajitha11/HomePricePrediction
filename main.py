import pickle

# Load the trained model
with open('model/home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Function to predict price based on user input
def predict_price(area, bedrooms, bathrooms):
    input_features = [[area, bedrooms, bathrooms]]
    predicted_price = model.predict(input_features)
    return predicted_price[0]

# Take user input
area = float(input("Enter area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Predict price
predicted_price = predict_price(area, bedrooms, bathrooms)
print(f"Estimated House Price: ₹{predicted_price:,.2f}")

