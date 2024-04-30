# write for me a code that shows how to use liner regression to predict the price of a house
# given the size of the house in square feet. The data is in the file house_price.csv
# and the first column is the size of the house and the second column is the price of the house.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
#url = 'https://raw.githubusercontent.com/ywchiu/riii/master/data/house-prices.csv'
#data = pd.read_csv(url)
data = pd.read_csv('/Users/alexparsee/Downloads/house-prices.csv')

print(data.head())


X = data['SqFt'].values
y = data['Price'].values

# Reshape the data
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# Create the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict the price of a house with size 2000 square feet

print(model.predict([[2000]]))


# Plot the data
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.show()


