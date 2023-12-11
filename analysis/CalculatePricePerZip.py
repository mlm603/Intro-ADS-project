import pandas as pd
import numpy as np
import geopandas as gpd
import regex as re



# Read data function
# Standardizes the column names
def readSaleData(path):
    df = pd.read_excel(path, skiprows=4)
    colnames = df.columns
    colnames = [re.sub(r'\n', '', name) for name in colnames]
    colnames = [re.sub(r'\s', '_', name) for name in colnames]
    colnames = [name.lower() for name in colnames]
    df.columns = colnames
    return df

# Rolling house filter
# Filters unrealistic entries
# Subsets to homes
def saleFilter(x):
    x = x[
        (x['zip_code'] > 0) &
        (x['sale_price'] >= 1e4) &
        (x['sale_price'] <= 2e8)
    ]
    doKeep = []
    for v in ['A', 'B', 'C', 'D']:
        codePtrn = re.compile(r'(^' + v + '.$)')
        doKeep.append([codePtrn.search(y) != None for y in x['building_class_at_time_of_sale']])
    codePtrn = re.compile(r'(^R[0-46-9]$)')
    doKeep.append([codePtrn.search(y) != None for y in x['building_class_at_time_of_sale']])
    doKeep = np.stack(doKeep, axis=1)
    doKeep = [sum(v) > 0 for v in doKeep]
    return x.loc[doKeep]



# ----- Housing Price by Area -----



# List the files
fileList = [
    'rollingsales_manhattan.xlsx',
    'rollingsales_queens.xlsx',
    'rollingsales_statenisland.xlsx',
    'rollingsales_brooklyn.xlsx',
    'rollingsales_bronx.xlsx'
]

# Load the data and process
fPath = '../data_cleaning/RollingSales'
saleData = [readSaleData(fPath + '/' + p) for p in fileList]
saleData = [saleFilter(x) for x in saleData]

# Attach modified zip code area
# Zip code data: https://data.cityofnewyork.us/Health/Modified-Zip-Code-Tabulation-Areas-MODZCTA-/pri4-ifjk
# Zip code data is already in WGS84, so no conversion needed
zipData = gpd.read_file(r"../data_cleaning/zipcode/geo_export_fc721dfd-087f-41ed-b841-bec5c0e62515.shp")

# Merge the sales data with the zip code shapes
# The modified zip code areas contain multiple internet locations
# Assign each internet area to its modified zip code 
zipData['modzcta'] = zipData['modzcta'].astype(int)
zipData = zipData.drop(index=177)
zipTab = []
for i in range(0, len(zipData['zcta'])):
    zipLab = zipData.iloc[i, 1]
    zipZCTA = zipData.iloc[i, 2]
    x = (zipLab + ', ' + zipZCTA)
    x = x.split(', ')
    x = [int(y) for y in x]
    zipTab.append(x)

# Assign by first match
for k in range(0, len(saleData)):
    dat = saleData[k]
    dat['zip_code'] = dat['zip_code'].astype(int)
    inds = []
    for i in range(0, len(dat['zip_code'])):
        x = dat.iloc[i, 10]
        for j in range(0, len(zipTab)):
            zipSet = zipTab[j]
            if x in zipSet:
                inds.append(j)
                break
            if j == (len(zipTab) - 1):
                inds.append(pd.NA)
    modzcta = zipData['modzcta'].iloc[inds].reset_index(drop=True)
    saleData[k] = saleData[k].assign(modzcta = modzcta)

# Calculate average price per square foot in each zip code
result = []
for i in range(0, len(saleData)):
    data = saleData[i]
    data = data.groupby(data['modzcta'], as_index=False).agg(
        sale_price = ('sale_price', 'median')
    )
    result.append(data)
result = pd.concat(result)

# Write the result
result.to_csv("price_per_zip.csv", index=False)
