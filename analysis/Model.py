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
    'price_per_sqft',
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
    "np.log(price_per_sqft) ~ subw_per_area + viol_per_area + percNoInternet", 
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
Intercept          6.277     0.221   28.418  0.000     5.844    6.710
subw_per_area   -110.649     8.724  -12.683  0.000  -127.749  -93.549
viol_per_area      0.306     0.024   12.650  0.000     0.259    0.354
percNoInternet     0.604     0.015   39.505  0.000     0.574    0.634
Group Var          0.244     1.201                                                              
"""
# Subway
# Decrease by 82%
100 * (np.exp(-110.649 * (0.015498 - 0.000000)) - 1) 

# Violence
# Increase by 330%
100 * (np.exp(0.306 * (4.773469 - 0.000000)) - 1) 

# Percent No Internet
# Increase by 17%
100 * (np.exp(0.604 * (0.323600 - 0.058300)) - 1) 

