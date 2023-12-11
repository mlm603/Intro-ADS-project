import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api
import matplotlib.pyplot as plt



# ----- Process the Data -----



# Load the data
data = pd.read_csv(r"C:\Users\cb5176\Documents\Classes\DSPY\Project\Intro-ADS-project\cleaned_datasets\fullData.csv")

# Subset to interesting columns
keepVars = [
    'bctcb2020', 'shape_area', 'modzcta', 'boroname',
    'numGrocery', 'numRestaurant', 'numSubway',
    'numCrime', 'numViolent',
    'No Internet Access (Percentage of Households)'
]
data = data[keepVars]
data = data.rename({'No Internet Access (Percentage of Households)':'percNoInternet'}, axis='columns')

# Attach price per zip
priceData = pd.read_csv(r"C:\Users\cb5176\Documents\Classes\DSPY\Project\Intro-ADS-project\cleaned_datasets\price_per_zip.csv")
data = data.merge(
    priceData,
    on='modzcta'
)

# Normalize count by area
data = data.assign(groc_per_area = (data['numGrocery'] / data['shape_area']))
data = data.assign(rest_per_area = (data['numRestaurant'] / data['shape_area']))
data = data.assign(subw_per_area = (data['numSubway'] / data['shape_area']))
data = data.assign(crim_per_area = (data['numCrime'] / data['shape_area']))
data = data.assign(viol_per_area = (data['numViolent'] / data['shape_area']))

# Final dataset
keepVars = [
    'sale_price',
    'groc_per_area', 'rest_per_area', 'subw_per_area',
    'crim_per_area', 'viol_per_area',
    'percNoInternet',
    'bctcb2020', 'boroname', 'modzcta'
]
data = data[keepVars]



# ----- Model -----



# Check correlation between predictors
data[['groc_per_area','rest_per_area','subw_per_area','viol_per_area','percNoInternet']].corr()

# Create a multilevel model to predict price per square foot
# Since housing prices are often log-normal, do a log transform on y
model = smf.mixedlm(
    "np.log(sale_price) ~ subw_per_area + viol_per_area + percNoInternet", 
    data, 
    groups=data['boroname']
)
model = model.fit()

# Plot the residuals
fig, ax = plt.subplots(figsize=(10,10))
plt.scatter(
    x=range(0, data.shape[0]),
    y=model.resid,
    c="#00000020"
)
ax.axline((0, 0), slope=0, color='#000000')
ax.set(xlabel = 'Index', ylabel='Residual')
plt.show()

# Q-Q plot
fig, ax = plt.subplots(figsize=(10,10))
statsmodels.api.qqplot(model.resid, line='s', ax=ax)
ax.set(xlabel = 'Theoretical Quantiles', ylabel='Empirical Quantiles')
ax.set_title("Q-Q Plot")
plt.show()

# Calculate the relative change in price per sqft
# Moving from the lowest to highest by predictor
"""
Intercept        13.566     0.141  96.271  0.000  13.290   13.842
subw_per_area    78.985    12.445   6.347  0.000  54.593  103.377
viol_per_area    -0.194     0.035  -5.543  0.000  -0.263   -0.126
percNoInternet    0.126     0.021   5.997  0.000   0.085    0.168
Group Var         0.099     0.335                                                                                                                        
"""
# Subway
# Decrease by 240%
print(100 * (np.exp(78.985 * (0.015498 - 0.000000)) - 1))

# Violence
# Decrease by 60%
print(100 * (np.exp(-0.194 * (4.773469 - 0.000000)) - 1))

# Percent No Internet
# Increase by 3%
print(100 * (np.exp(0.126 * (0.323600 - 0.058300)) - 1))

