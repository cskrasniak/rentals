### This is an exploration of the airbnb listings for 30 US markets over the last year
### from http://insideairbnb.com/get-the-data/ 
import numpy as np
import pandas as pd
import os
from selenium import webdriver
import selenium
from webdriver_manager.chrome import ChromeDriverManager
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

download_dir = r'/Users/ckrasnia/Downloads'
final_dir = r'/Users/ckrasnia/Documents/application_materials/rental_data'
# navigate to the website
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get(r"http://insideairbnb.com/get-the-data")


# most of the data is hidden, so open all of those paths so it can be seen

show_additional_data = driver.find_elements(by=webdriver.common.by.By.CLASS_NAME,value="showArchivedData")
time.sleep(5)  # allow time to connect to the website
for i in range(len(show_additional_data)):
    # scroll down to the next element
    driver.execute_script("arguments[0].scrollIntoView();", show_additional_data[i])
    time.sleep(.5)  # give some time for the browser to update
    show_additional_data = driver.find_elements(by=webdriver.common.by.By.CLASS_NAME,value="showArchivedData")
    show_additional_data[i].click()  # expand to show the data
    time.sleep(.5)  # give some time for the browser to update

    # because it expands the page, need to refind the data
    show_additional_data = driver.find_elements(by=webdriver.common.by.By.CLASS_NAME,value="showArchivedData")

# now that all the data is exposed, we are going to download each listings.csv file
target_data = driver.find_elements(by=webdriver.common.by.By.LINK_TEXT,value='listings.csv.gz')
driver.execute_script("arguments[0].scrollIntoView(false);", target_data[0])
time.sleep(10)  # first scroll starts all the way at the bottom so need longer to scroll
for i in range(len(target_data)):
    # get the target_data again after scrolls
    target_data = driver.find_elements(by=webdriver.common.by.By.LINK_TEXT,value='listings.csv.gz')
    # scroll down to the next target
    driver.execute_script("arguments[0].scrollIntoView(false);", target_data[i])
    time.sleep(.5)  # time for the browser to update
    # refind target data
    target_data = driver.find_elements(by=webdriver.common.by.By.LINK_TEXT,value='listings.csv.gz')
    target_data[i].click()  # downloads the data
    time.sleep(2)  # time to download

## Now I have the data downloaded, I'll take only the US data, concatenate it to a single df and 
## save it
os.chdir(download_dir)
downloads = os.listdir()

# initilize list
data_list = []
for dl in tqdm(downloads):
    if (dl.startswith('listings') & dl.endswith('.csv.gz')) | \
       (dl.startswith('listings') & dl.endswith('.csv')):
        # I only want US listings, so I need to find where these are from. It will also be useful 
        # to have a column about the location. The issue is that there is no column for location,
        # so to assign one I'll find the place where a majority of the hosts live, and assume that
        # is the location of the whole dataset
        temp_data = pd.read_csv(dl)
        us, cnts = np.unique(temp_data['host_location'].dropna(),return_counts=True)
        location = us[np.argmax(cnts)]
        temp_data['location'] = location
        if 'United States' in location:
            data_list.append(temp_data)

# concatenate and save this data
data = pd.concat(data_list)
if not os.path.exists(final_dir):
    os.mkdir(final_dir)
data.to_csv(os.path.join(final_dir,'raw_US_listings.csv'))
        
# Load data if already saved 
data = pd.read_csv(os.path.join(final_dir,'raw_US_listings.csv'))

# now I have the data, and I want to clean it up a bit, first I'll drop some columns I wont need
# dropping some useless info, some with duplicate data, and some that have very low variance, ie >90% the same value
data['neighbourhood_group_cleansed'][data['neighbourhood_group_cleansed'].isna()] = data['neighbourhood_cleansed'][data['neighbourhood_group_cleansed'].isna()]
drop_columns = ['listing_url','host_url','host_thumbnail_url','host_picture_url',
                'host_neighbourhood','host_total_listings_count', 'host_verifications', 
                'neighbourhood_cleansed','calendar_updated']
data=data.drop(drop_columns,axis=1)

## OK so the next goal is to make a regression model trying to predict the total income from a 
# rental given its characteristics. So to do this I will do a few things, I will turn some 
# currently non-numeric values into numeric, and I will normalize all the data into the range 0-1 
# so we can easily compare the weights of the model

# first convert some values to numeric
# getting the bathroom count
column = data['bathrooms_text']
# tmporarily change the nans to an outrageous number to more easily handle them
column.iloc[column.isna()] = ['1000'] 

def get_bathrooms(list):
    """
    extractsr the numbers from a list of strings, specialized to the bathroom_text column
    Input : list of strings
    output : float of number of bathrooms
    """
    for item in list:
        if any([i.isnumeric() for i in item]):
            return float(item)
        elif 'half' in item.lower():
            return .5

data['bathrooms'] = column.str.split().apply(get_bathrooms)
data['bathrooms'][data['bathrooms'] == 1000] = np.nan # set the nans back to nan

# there are a few nan prices which are useless as the price is key, we'll drop those
data = data[~data['price'].isna()]
# remove the dollar signs and commas from prices and convert to float, there are a few entries that
# already have these stripped though, so we'll have to exclude those, this is a bit messy
# but it works
data['price'][~data['price'].str.isnumeric()] = data['price'].str[1:-3].str.replace(",","") \
    [data['price'].str[1:-3].str.replace(",","").str.isnumeric()].astype(float)

data['price'] = data['price'].astype(float)

# there are a couple boolean columns coded as "t" and "f", replace those with 1, 0
boolean_mapper = {'t' : 1, 'f' : 0}
bools = ['instant_bookable','has_availability', 'host_has_profile_pic',
         'host_identity_verified', 'host_is_superhost']

for column in bools:
    data[column] = data[column].map(boolean_mapper, na_action="ignore")

rt_mapper = {'Entire home/apt' : 3, 'Private room' : 2, 'Hotel room' : 1}
data['room_type'] = data['room_type'].map(rt_mapper)
# the nan room types are shared, so assign these to 0
data['room_type'][data['room_type'].isna()] = 0

# Ok things are starting to look better, but because of the way I grabbed the data, there are many
# duplicates and triplicates of listings in the set. So now I need to make some decisions about how
# to solve this. For this first step of just making a simple regression model, I think I'll just 
# use the numeric columns anyway, so I'll just take the averages of those replicates for each 
# listing to start. If I end up needing more data to train on though, maybe I'll treat them each as
# seperate data points, if something has changed in the listing this would make sense anyways.


use_columns = ['host_is_superhost', 'host_listings_count', 'host_has_profile_pic',
               'host_identity_verified','accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
               'minimum_nights', 'maximum_nights','availability_30', 'number_of_reviews',
               'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
               'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
               'review_scores_location', 'review_scores_value', 'instant_bookable',
               'calculated_host_listings_count','reviews_per_month', 'id', 'has_availability', 
               'room_type']

numeric_data = data[use_columns]
numeric_data = numeric_data.groupby('id').mean()

# look to see how often things changed, this should be easy in the columns that are boolean
for col in bools:
    print(col)
    print(np.unique(numeric_data[col],return_counts=True))

# ok there's some interesting stuff here, it looks like there are a reasonable number of times that 
# a listing switched between values of being instant_bookable and from having a superhost or not,
# should remember this for the future as we could do some interesting experiments to see how these
# changes affected the income for that listing
# Another interesting tidbit from here is that many of these columns have the same number (764) of 
# nans, thats pretty small and if the same listings have a bunch of nans they won't be very helpful
# anyway so we can probably drop those

# look more at where there are nans
for col in numeric_data.columns:
    print('fraction nans in {} : {:1.3f}%'.format(col,np.sum(numeric_data[col].isna())/len(numeric_data)*100))
# Ok we have a few things here, some are easy to fix, others not so much. For those with no host 
# information we can easily drop that .3% of data and be fine. The first problem is in the number 
# of bedrooms whih is ~10% nan. I think this might be from listings that are a single private room
# so the host doesn't bother to put a bedroom count on there, this should be pretty easy to fix. 
# the number of beds we also might be able to infer on, but that is only 3% of data so wouldn't be
# too bad throwing that away. The bigger issue is how to treat not having any reviews. Seems like
# ~25% of listings have no reviews, so dealing with that could be tricky. I definitely want to 
# include review data as it is probably pretty influential, might just try setting NaN to a review
# score of 0 and see how that goes.

# start by giving the listings with no reviews a review of 0
reviews = ['number_of_reviews','number_of_reviews_ltm', 'review_scores_value', 'reviews_per_month',
           'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location']
for review in reviews:
    numeric_data[review][numeric_data[review].isna()] = 0     

for col in numeric_data.columns:
    print('fraction nans in {} : {:1.3f}%'.format(col,np.sum(numeric_data[col].isna())/len(numeric_data)*100))

# lets see how to deal with the 10% of listings with no bedroom count
print(numeric_data[numeric_data['bedrooms'].isna()].describe())

# looks like these are really all over the place unfortunately... I think the most reasonable thing
# to do here is to calculate the mean number of bedrooms there are per the number of beds listed. 
# because only 3% of the data has no beds

# need to take the reciprical of what I actually want to avoid the devide by zero issue for 0 beds
bedrooms_per_beds = 1 / (numeric_data[~numeric_data['bedrooms'].isna()]['beds'] \
    / numeric_data[~numeric_data['bedrooms'].isna()]['bedrooms']).mean()

numeric_data['bedrooms'][numeric_data['bedrooms'].isna()] = \
    numeric_data['beds'][numeric_data['bedrooms'].isna()] * bedrooms_per_beds

# check nans again
for col in numeric_data.columns:
    print('fraction nans in {} : {:1.3f}%'.format(col,np.sum(numeric_data[col].isna())/len(numeric_data)*100))

# ok now we look pretty good, our colun with the most nans is beds with ~3%, so lets just drop the
# rest of the nans
for col in numeric_data.columns:
    numeric_data = numeric_data[numeric_data[col].notna()]

# we still have 243k listings so that is pretty good
print('total number of listings: {}'.format(len(numeric_data)))

# now lets start looking at the data to get some future steps
corr_mat = np.corrcoef(np.array(numeric_data).T)
fig,ax = plt.subplots()
ax.set_xticks(np.arange(len(numeric_data.keys())))
ax.set_yticks(np.arange(len(numeric_data.keys())))
# ax.set_xticklabels(numeric_data.keys())
ax.set_yticklabels(numeric_data.keys())
ax.imshow(corr_mat)

# one glaring thing is that all the reviews are almost perfectly correlated, so I'm just going to
# take the mean of them and group them all into 'reviews'
reviews = ['review_scores_value', 
           'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
           'review_scores_communication', 'review_scores_location']
numeric_data['reviews'] = numeric_data[reviews].mean(axis=1)
for review in reviews:
    numeric_data.drop(review,axis=1, inplace=True)

# lets look again at how correlated things are
corr_mat = np.corrcoef(np.array(numeric_data).T)
fig,ax = plt.subplots()
ax.set_xticks(np.arange(len(numeric_data.keys())))
ax.set_yticks(np.arange(len(numeric_data.keys())))
# ax.set_xticklabels(numeric_data.keys())
ax.set_yticklabels(numeric_data.keys())
ax.imshow(corr_mat)

# This looks better now, still some relatively high correlations between beds, bedrooms, bathrooms,
# and accomodates, but the highest is .83 so still a fair bit of independent information

# finally lets calculate our prediction variable which I'll call monthly income, this will 
# definitely be an imperfect measure of monthly income, but it will essentially be the number of
# non available days over the next 30 days times the price per day.

numeric_data['monthly_income'] = numeric_data['price'] * np.abs(numeric_data['availability_30'] - 30)

# first just try a simple linear regression without much more pre-processing of features
features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
            'accommodates', 'bathrooms', 'bedrooms', 'beds',
            'maximum_nights',  'number_of_reviews', 'price',
            'number_of_reviews_ltm', 'instant_bookable',
            'calculated_host_listings_count', 'reviews_per_month',
            'room_type', 'reviews']

regression = LinearRegression()
kfold = KFold(5, shuffle=True)
cv_scores = []
coefs = []
for train_idx,test_idx in kfold.split(numeric_data[features]):
    regression.fit(numeric_data[features].iloc[train_idx],numeric_data['monthly_income'].iloc[train_idx])
    r2=regression.score(numeric_data[features].iloc[test_idx],numeric_data['monthly_income'].iloc[test_idx])
    cv_scores.append(r2)
    coefs.append(regression.coef_)
# mean r2 of .55 with linear regression and a paired down set of regressors, pretty good starting
# point but lets see if we can do better in other files.

# first I'm going to explore a few of the wordy columns to see if theres any promising information
# there
