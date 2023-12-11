import pandas as pd
import geopandas as gpd
import geopy.distance
from multiprocessing import Pool



# ----- Process Data Sources -----



"""
Analyzing factors that influence desirability at the level of census blocks. 
Each factor of desirability comes from a different source dataset. Map the 
factors to the census blocks.

This code chunk replicates Meghan Maloy's desirability_factors_analysis.ipynb
notebook.
"""



# Load the block data
blocks = gpd.read_file(r"2020_census_blocks/geo_export_0cf61a4c-2518-43af-b17b-0ae104636fd0.shp")

# Attach zip code to block data
# Drop blocks without a zipcode
zipData = pd.read_csv(r"zcta_to_block.csv", engine='pyarrow')
blocks['geoid'] = blocks['geoid'].astype('int64')
blocks = blocks.merge(
    zipData,
    left_on="geoid",
    right_on="GEOID_TABBLOCK_20",
    how="inner"
)
blocks = blocks.loc[~pd.isna(blocks['GEOID_ZCTA5_20'])]
del zipData

# Attach internet data by zip code
internet = pd.read_csv(r"..cleaned_datasets/internet.csv")
internet = internet.drop('geometry', axis=1)
blocks = blocks.merge(
    internet,
    left_on='GEOID_ZCTA5_20',
    right_on='modzcta',
    how='left'
)
blocks = blocks.loc[~pd.isna(blocks['No Internet Access (Percentage of Households)'])]


# Load the source data
fPath = '../cleaned_datasets'
pathList = [
    'crime_data',
    'grocery',
    'restaurant',
    'subway_stations'
]
sourceData = [pd.read_csv(fPath + '/' + p + '.csv') for p in pathList]

# Convert source data to geospatial objects
# Match crs with the block data
for i in range(0, len(sourceData)):
    dat = sourceData[i]
    sourceData[i] = gpd.GeoDataFrame(
        dat, 
        geometry=gpd.points_from_xy(dat["longitude"], dat["latitude"]), 
        crs=blocks.crs
    )
    del dat

# Spatial join source data with blocks
sourceData = [gpd.sjoin(d, blocks, predicate='within', how='left') for d in sourceData]



# ----- Aggregate Source Data -----



"""
The unit of analysis is census block 'bctcb2020', so the sources now need to be
aggregated down to block level characteristics.

Crime: Assume that crime prevalence in a specific block is all that matters.
    Aggregation is done by counting crimes in your block.
Grocery, Restaurants, and Subways: Assume that distance to matters. Count all 
    locations within a 15-minue radius around a block.
"""



# Aggregate crime data
# Count total crimes and total violence crimes
crimeData = sourceData[0].copy()
crimeData = crimeData.groupby(crimeData['bctcb2020'], as_index=False).agg(
   numCrime=('id', 'count'),
   numViolent=('is_violent_offense', 'sum'),
   boroname=('boroname','first'),
   geometry=('geometry','first')
)
sourceData[0] = crimeData
del crimeData

# Aggregate grocery data
# Count total grocery stores
groceryData = sourceData[1].copy()
groceryData = groceryData.groupby(groceryData['bctcb2020'], as_index=False).agg(
   numGrocery=('County', 'count'),
   boroname=('boroname','first'),
   geometry=('geometry','first')
)
sourceData[1] = groceryData
del groceryData

# Aggregate restaurant data
# Count total restaurants
restaurantData = sourceData[2].copy()
restaurantData = restaurantData.groupby(restaurantData['bctcb2020'], as_index=False).agg(
   numRestaurant=('camis', 'count'),
   boroname=('boroname','first'),
   geometry=('geometry','first')
)
sourceData[2] = restaurantData
del restaurantData

# Aggregate subway
# Subway stations have multiple entrances
# Subset to one entrance per station
# This might make the count slightly less accurate when a subway station is
# on the edge of the radius, but protects against bias from subway stations
# that have more entrances than others
# Count total stations
subwayData = sourceData[3].copy()
subwayData = subwayData.drop_duplicates('station_name')
subwayData = subwayData.groupby(subwayData['bctcb2020'], as_index=False).agg(
   numSubway=('station_name', 'count'),
   boroname=('boroname','first'),
   geometry=('geometry','first')
)
sourceData[3] = subwayData
del subwayData



# Aggregate the other data by counting in a distance radius to the block
# Assume a radius of 1610 / sqrt(2) meters, which gives 4 mph walking speed for 15 minutes
# Penalized towards Manhattan (L1) distance 
# This is the CDC lower estimate on average adult walking speed
# 15 minutes is considered a typical radius in urban design
#
# This computation will take forever to run serially
# Write a block-parallel implementation
def aggBlock(blockInds):
    
    '''
    WARNING: This function accesses 'blocks' and 'sourceData' as globals 
    from inside the function environment. This is bad practice, but I'm 
    doing it to make the parallel computation convenient. Don't do this in 
    real life.
    '''
    
    # Add a tryCatch to the geodistance
    # Drop errors from the count
    def safeDist(p1, p2):
        try:
            d = geopy.distance.geodesic(p1, p2).m
        except:
            d = 9999
        return d
    
    # Specify the targets to count
    targetList = [
        'numCrime',
        'numViolent',
        'numGrocery',
        'numRestaurant',
        'numSubway'
    ]
    
    aggResult = []
    for i in range(blockInds[0], blockInds[1]):
        
        # Get a point corresponding to the centroid of the block
        row = blocks.iloc[i]
        blockPT = (row.geometry.centroid.y, row.geometry.centroid.x)
        
        # Aggregate the source datasets
        # Subset to the cases in the same borough as the block
        # This makes the computation a little more efficient by reducing N
        countList = []
        for d in sourceData:
            sub = d.loc[d['boroname'] == row['boroname']]
            delta = [safeDist(blockPT, (pt.centroid.y, pt.centroid.x)) for pt in sub['geometry']]
            inRange = [d <= 1138 for d in delta]
            for k in range(0, len(targetList)):
                if targetList[k] in sub.columns:
                    countList.append(sum(sub[targetList[k]].loc[inRange]))
            
        # Record as a dataframe row
        aggResult.append(
            pd.DataFrame([countList], columns=targetList)
        )
        
    return pd.concat(aggResult)



# ----- Main -----



# Parameterize the process
result = []
cores = 10
indexList = []
for i in range(0, cores):
    v = round(blocks.shape[0] / cores)
    indexList.append([i * v, min([(i + 1) * v, blocks.shape[0]])])
    
# Perform the computation in parallel on 10 cores
# Write the result
if __name__ == '__main__':
    with Pool(cores) as pool:
        result = pool.map(aggBlock, indexList)
    out = pd.concat(result)
    out = pd.concat([blocks.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    out.to_csv("fullData.csv", index=False)
