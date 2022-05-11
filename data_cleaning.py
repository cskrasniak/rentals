import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from preprocess_text import preprocess_text

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

def cut_to_len(array,length):
    a_len=len(array)
    if a_len > length:
        return array[:length]
    else:
        return np.concatenate([array,np.zeros(length-a_len).astype(int)])


def clean_data_xgb(data):

    # now I have the data, and I want to clean it up a bit, first I'll drop some columns I wont need
    # dropping some useless info, some with duplicate data, and some that have very low variance, ie >90% the same value
    data['neighbourhood_group_cleansed'][data['neighbourhood_group_cleansed'].isna()] = \
        data['neighbourhood_cleansed'][data['neighbourhood_group_cleansed'].isna()]

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
    # I think I want to have this one-hot instead, so going to add that here
    data['room_type_entire'] = data['room_type'].map({3:1, 2:0, 1:0, 0:0})
    data['room_type_private'] = data['room_type'].map({2:1, 3:0, 1:0, 0:0})
    data['room_type_hotel'] = data['room_type'].map({1:1, 3:0, 2:0, 0:0})
    data['room_type_none'] = data['room_type'].map({0:1, 3:0, 2:0, 1:0})
    # Ok things are starting to look better, but because of the way I grabbed the data, there are many
    # duplicates and triplicates of listings in the set. So now I need to make some decisions about how
    # to solve this. For this first step of just making a simple regression model, I think I'll just 
    # use the numeric columns anyway, so I'll just take the averages of those replicates for each 
    # listing to start. If I end up needing more data to train on though, maybe I'll treat them each as
    # seperate data points, if something has changed in the listing this would make sense anyways.
    print(len(data))


    use_columns = ['host_is_superhost', 'host_listings_count', 'host_has_profile_pic',
                'host_identity_verified','accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
                'minimum_nights', 'maximum_nights','availability_30', 'number_of_reviews',
                'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'instant_bookable',
                'calculated_host_listings_count','reviews_per_month', 'has_availability', 
                'room_type_none', 'room_type_hotel', 'room_type_private', 'room_type_entire', 'id',
                'description','name'] 
    numeric_columns = ['host_is_superhost', 'host_listings_count', 'host_has_profile_pic',
                'host_identity_verified','accommodates', 'bathrooms', 'bedrooms', 'beds', 'price',
                'minimum_nights', 'maximum_nights','availability_30', 'number_of_reviews',
                'number_of_reviews_ltm', 'review_scores_rating', 'review_scores_accuracy',
                'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
                'review_scores_location', 'review_scores_value', 'instant_bookable',
                'calculated_host_listings_count','reviews_per_month', 'has_availability', 
                'room_type_none', 'room_type_hotel', 'room_type_private', 'room_type_entire']

    data = data[use_columns]

    # for xgboost, if I do use that, nans are allowed, so I'll keep them in reviews for now

    # start by giving the listings with no reviews a review of 0
    # reviews = ['number_of_reviews','number_of_reviews_ltm', 'review_scores_value', 'reviews_per_month',
    #         'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    #         'review_scores_communication', 'review_scores_location', 'review_scores_rating']
    # for review in reviews:
    #     data[review][data[review].isna()] = 0     

    # I think the most reasonable thing
    # to do here is to calculate the mean number of bedrooms there are per the number of beds listed. 
    # because only 3% of the data has no beds

    # need to take the reciprical of what I actually want to avoid the devide by zero issue for 0 beds
    bedrooms_per_beds = 1 / (data[~data['bedrooms'].isna()]['beds'] \
        / data[~data['bedrooms'].isna()]['bedrooms']).mean()

    data['bedrooms'][data['bedrooms'].isna()] = \
        data['beds'][data['bedrooms'].isna()] * bedrooms_per_beds

    # converting columns I know should be numeric but are objects to a numeric dtype
    numeric_data = data[numeric_columns]
    numeric_data = (numeric_data.drop(numeric_columns, axis=1)
            .join(numeric_data[numeric_columns].apply(pd.to_numeric, errors='coerce')))
    non_numeric = data.keys()[[i not in numeric_data.keys() for i in data.keys()]]
    data = numeric_data.join(data[non_numeric])

    # select columns with <=3% nans, those with more, (~20%) are reviews and it is meaningful for 
    # them to not be a number, the others are just missing data or I have already imputed their 
    # value so we can throw those out
    for col in data.columns:
        if np.sum(data[col].isna())/len(data)*100 <= 3:
            data = data[data[col].notna()]

    # we still have ~480k listings so that is pretty good


    # one glaring thing is that all the reviews are almost perfectly correlated, so I'm just going to
    # take the mean of them and group them all into 'reviews'
    reviews = ['review_scores_value', 'review_scores_rating',
            'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
            'review_scores_communication', 'review_scores_location']
    data['reviews'] = np.nanmean(data[reviews],axis=1)
    for review in reviews:
        data.drop(review,axis=1, inplace=True)

    # This looks better now, still some relatively high correlations between beds, bedrooms, bathrooms,
    # and accomodates, but the highest is .83 so still a fair bit of independent information

    # finally lets calculate our prediction variable which I'll call monthly income, this will 
    # definitely be an imperfect measure of monthly income, but it will essentially be the number of
    # non available days over the next 30 days times the price per day.

    data['monthly_income'] = data['price'] * np.abs(data['availability_30'] - 30)

    # clean the string values for the name and description
    data['descript_words'], descript_dict = preprocess_text(data['description'],return_dict=True)
    data['name'], name_dict = preprocess_text(data['name'], return_dict=True)

    # pad the name to length 34
    data['name'] = data['name'].apply(cut_to_len,length=34)
    # split these up so that each one is an individual feature
    name_array = np.vstack(data['name'])
    for i in range(34):
        col_name = 'name_{}'.format(i)
        data[col_name] = name_array[:,i]
    
    # pad the description words to len 110
    data['descript_words'] = data['descript_words'].apply(cut_to_len,length=110)
    descript_array = np.vstack(data['descript_words'])
    for i in range(34):
        col_name = 'descript_words_{}'.format(i)
        data[col_name] = descript_array[:,i]
    return data
    # # first just try a simple linear regression without much more pre-processing of features
    # features = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    #             'accommodates', 'bathrooms', 'bedrooms', 'beds',
    #             'maximum_nights',  'number_of_reviews', 'price',
    #             'number_of_reviews_ltm', 'instant_bookable',
    #             'calculated_host_listings_count', 'reviews_per_month',
    #             'room_type_none', 'room_type_hotel', 'room_type_private', 'room_type_entire',
    #             'reviews']


    # # let me just standardize the features and see if that helps at all
    # from sklearn.preprocessing import MinMaxScaler

    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(data[features])