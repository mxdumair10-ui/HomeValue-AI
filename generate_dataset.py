"""Optional: recreate the illustrative synthetic teaching dataset."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 600
locations = np.array(["Chennai", "Coimbatore", "Madurai", "Tirunelveli", "Tenkasi"])
rates = {"Chennai": 6500, "Coimbatore": 4800, "Madurai": 3800,
         "Tirunelveli": 3000, "Tenkasi": 2400}
location = rng.choice(locations, n, p=[.28, .22, .20, .18, .12])
area = rng.integers(550, 3001, n)
bedrooms = np.clip(np.round(area / 520 + rng.normal(0, .7, n)), 1, 6).astype(int)
bathrooms = np.clip(bedrooms + rng.integers(-1, 2, n), 1, 5).astype(int)
age = rng.integers(0, 31, n)
price = (area * np.array([rates[x] for x in location])
         + bedrooms * 180000 + bathrooms * 90000 - age * 28000
         + rng.normal(0, 350000, n)).clip(900000)
pd.DataFrame({
    "location": location, "area_sqft": area, "bedrooms": bedrooms,
    "bathrooms": bathrooms, "property_age": age,
    "price_inr": np.round(price / 10000) * 10000
}).to_csv("house_data.csv", index=False)
print("Created house_data.csv")
