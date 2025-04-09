import pickle
import json
import numpy as np

# Load model
with open("model/home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load columns
with open("model/columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

# Sample input — replace with your own values
def predict_price(area, bhk, bath, location):
    x = np.zeros(len(data_columns))
    x[0] = area
    x[1] = bath
    x[2] = bhk
    try:
        loc_index = data_columns.index(location.lower())
        x[loc_index] = 1
    except:
        pass

    return model.predict([x])[0]

# Example usage
predicted_price = predict_price(1000, 2, 2, "1st Phase JP Nagar")
print(f"Predicted Price: ₹{predicted_price:.2f}")
