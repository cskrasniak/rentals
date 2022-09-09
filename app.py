# from fileinput import filename
import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBRegressor
import datetime
import dill
import altair as alt

## functions for transformer
from sklearn.base import BaseEstimator, TransformerMixin
import requests
import haversine as hs
from ediblepickle import checkpoint
import os
from urllib.parse import quote
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# from sklearn.pipeline import FeatureUnion
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import PolynomialFeatures

class BathroomParser(BaseEstimator, TransformerMixin):
    
    @staticmethod
    def get_bathrooms(string):
        """
        extractsr the numbers from a list of strings, specialized to the bathroom_text column
        Input : list of strings
        output : float of number of bathrooms
        """
        try:
            if np.isnan(string):
                return np.nan
        except TypeError:
            pass
        multiplier = 1 # if bathrooms are shared, I cut the count in half
        count = 0
        if string.strip().lower() == 'private':
            return 1
        if string.strip().lower() == 'shared':
            return .5
        for item in string.split():
            
            if item.isnumeric():
                count += float(item)
                
            if 'half' in item.lower():
                count += .5
                
            # if bathrooms are shared, I cut the count in half
            if 'shared' in item.lower():
                multiplier = .5
            
        return count * multiplier
            
    def get_feature_names_out(self, input_features):
        return self.names
    
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = list(X.keys())
        return self
    
    def transform(self, X):
        out = pd.DataFrame()
        for col in X:
            out[col] = X[col].apply(BathroomParser.get_bathrooms)
        return out
    
    
class BoolMapper(BaseEstimator, TransformerMixin):
    
    def get_feature_names_out(self, input_features):
        return self.names
        
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = list(X.keys())
        return self
    
    def transform(self, X):
        # only change the values that are not numeric and not nan
        out = pd.DataFrame()
        for name in X:
            boolean_mapper = {'t': 1., 'f': 0., 1.: 1., 0.: 0.}
            col=X[name].map(boolean_mapper, na_action="ignore")
            out[name]=col
        return out


class OHEfromDF(BaseEstimator, TransformerMixin):
    """
    Transformer class that formats pandas columns with string values into a valid input for the
    DictVectorizer function to create a one-hot encoding.
    
    params : vals (None or list)
                The values to use for one-hot encoding, they can either be fed in at intitialization or 
                fit by taking all the values that are more common than len(X)/val_cutoff
           : val_cutoff (float) the number >1 to use to select the vals in the fit method as described above.
                val_cutoff=1 means select all values, larger number provides a stricter cutoff
    """
    def __init__(self,vals=None,val_cutoff=100):
        self.vals = vals
        self.val_cutoff=val_cutoff
    
    def get_feature_names_out(self, input_features):
        return self.names
    
    def fit(self, X, y=None):
        # can either provide value, or, can fit them based on the most common entries
        if self.vals == None:
            vals = list(X.value_counts()[X.value_counts()>len(X)/self.val_cutoff].index)
            self.vals = [val[0] for val in vals]
        self.names = []
        for col in X:
            self.names.extend([col+'_'+val for val in self.vals])
        return self
    
    def transform(self, X):
        # only change the values that are not numeric and not nan
        mapper = {val:i for i,val in enumerate(self.vals)}
        new_X = pd.DataFrame()
        for col in X:
            col_names = [col+'_'+val for val in self.vals]
            temp = X[col].map(mapper, na_action="ignore")
            for i,name in enumerate(col_names):
                new_X[name] = (temp == i).astype(float)
        return new_X


class FilterOutFakes(BaseEstimator, TransformerMixin):
    """
    transformer to filter our the properties that have 0 availability over the next year, they probably
    aren't going to be useful to train on, also drops any rows where the availability_90 (outcome variable) 
    is nan
    
    Input : pd.DataFrame with the column 'availability_365'
    
    param cutoff: the integer value from 0-365 used to exclude listings. the excludion works as:
                    availability_365 > cutoff
    """
    
    def __init__(self, max = 365,min=0):
        self.max = max
        self.min = min
        
    def get_feature_names_out(self, input_features):
        return self.names
        
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = list(X.keys())
        return self
    
    def transform(self, X):
        vals=pd.to_numeric(X['availability_365'],errors='coerce')
        X = X[(vals > self.min) & (vals <= self.max)]
        X = X[X['availability_90']!=90]
        X = X[X['availability_90']!=0]
        return X.dropna(subset=['availability_90'])


cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

if not 'temp_cache' in locals():
    temp_cache = {}
@checkpoint(key=lambda args, kwargs: quote(args[0]) + '.pkl', work_dir=cache_dir)
def get_lat_long(address):
    # first check if we already loaded it this session, if so it will be much faster to grab it from here than
    # load it from memory or get it from the api
    latlong = temp_cache.get(address)
    if latlong:
        return latlong
    params = { 'format'        :'json', 
               'addressdetails': 1, 
               'q'             : address}
    headers = { 'user-agent'   : 'TDI' }
    response = requests.get('http://nominatim.openstreetmap.org/search', 
                        params=params, headers=headers)
    try:
        lat = response.json()[0]['lat']
        long = response.json()[0]['lon']
    except IndexError:
        print(f'No location on openstreetmap found for {address}, returning nan')
        temp_cache[address] = (np.nan, np.nan)
        return (np.nan, np.nan)
    
    temp_cache[address] = (float(lat),float(long))
    return (float(lat),float(long))


class DistFromCenter(BaseEstimator, TransformerMixin):
    """
    transformer to get the distance from the neighborhood center of the listing. If there is no neighborhood,
    it uses the city center. This takes a long time, but I have to iterate through them individually to keep 
    from pinging the API too many times at once and to check if I can use the neighbourhood or have to use 
    the city
    
    """
      
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = ['dist_from_center']
        return self
    
    def get_feature_names_out(self, input_features):
        return self.names
    
    def transform(self, X):
        loc1 = [(x,y) for x,y in zip(X['latitude'],X['longitude'])]
        dist = np.ones(len(X))*np.nan
        
        # this is required to keep the cache from creating a new directory when there is something like Cambridge/Boston
        X['neighbourhood'] = X['neighbourhood'].str.replace('/',' ')
        for i, l1 in enumerate(tqdm(loc1)):
            l2 = (np.nan,np.nan)
            # first get neighborhood lat,long
            if type(X['neighbourhood'].iloc[i]) == str:
                l2 = get_lat_long(X['neighbourhood'].iloc[i])
            
            #if l2 is nan or if the neighbourhood is nan, use the city instead
            if (X['neighbourhood'].isna().iloc[i]) | any(map(np.isnan,l2)):
                if X['location'].isna().iloc[i]:
                    # if both location and neighborhood are nan, return something really far away, so that the distance
                    # will be really far and return nan in the end
                    l2 = get_lat_long('Beijing, China')
                else:
                    l2 = get_lat_long(X['location'].iloc[i])
                
            dist[i] = hs.haversine(l1,l2)
            # if the distance is outrageous, set it to nan
            if dist[i] >= 1000:
                dist[i] = np.nan
        X['dist_from_center'] = dist.reshape(-1,1)        
        return X
    
    
class HostLength(BaseEstimator, TransformerMixin):
    """
    calculate the length of time the person has been a host, take the diff og host_since and date_scraped
    
    """
      
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = ['host_length_days']
        return self
    
    def get_feature_names_out(self, input_features):
        return self.names
    
    def transform(self, X):
        #calculate time delta since airbnb was founded
        X['host_length_days'] = pd.DataFrame((pd.to_datetime(X['calendar_last_scraped'],errors='coerce') - pd.to_datetime(X['host_since'],errors='coerce')).dt.days)
        return X
    

class FloatCaster(BaseEstimator, TransformerMixin):
    """
    
    """
      
    def get_feature_names_out(self, input_features):
        return self.names
        
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        self.names = list(X.keys())
        return self
    
    
    def transform(self, X):
        out = pd.DataFrame()
        for name in X:
            col = X[name]
            # if its already 
            numerics = 0
            # heuristic to get columns that are mostly numeric and just have a few non-numeric columns
            for i in range(int(np.ceil(len(X)/10))):
                if (type(col.iloc[i]) == int) | (type(col.iloc[i]) == float):
                    numerics+=1
            if numerics >= len(X)/10-(len(X)/10/10):
                out[name] = pd.to_numeric(col,errors='coerce')
            else:
            # deals with columns that are strings but really should be numbers like things with $ or %
                out[name] = pd.to_numeric(col.str.replace(r'[^0-9\.]+',''),errors='coerce')
        return out
    

class BedandBedroomImputer(BaseEstimator, TransformerMixin):
    """
    A transformer to impute the number of bedrooms/beds from the number of people it accommodates,
    fit method calculates the mean ratio of accomodates to the column in question (beds or bedrooms)
    for non-null values, and fills in the estimated number of beds/bedrooms based on this mean ratio
    Input should be the whole dataframe, with an accommodates, beds, and bedrooms column
    """
      
    def fit(self, X, y=None):
        bed_mask = X['beds'].notna()
        self.bed_ratio_ = (X['beds'][bed_mask] / X['accommodates'][bed_mask]).mean()
        bedroom_mask = X['bedrooms'].notna()
        self.bedroom_ratio_ = (X['beds'][bedroom_mask] / X['accommodates'][bedroom_mask]).mean()
        return self
    
    def transform(self, X):
        # fill in where accommodates is not nan
        bed_mask = X['beds'].isna()
        X['beds'][bed_mask] = X['accommodates'][bed_mask] * self.bed_ratio_
        bedroom_mask = X['bedrooms'].isna()
        X['bedrooms'][bedroom_mask] = X['accommodates'][bedroom_mask] * self.bedroom_ratio_
        return X
    
#### Functions to go along with text preprocessing
lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')
sw.extend(["'d", "'ll", "'re", "'s", "'ve", 'could', 'doe', 'ha', 'might', 'must', "n't", 'need', 'sha', 'wa', 'wo', 'would'])
def tokenize_lemma(text):
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]

def remove_md(string):
    """
    Removes angled brackets (< & >) and the markdown formatting inside them from strings. Should 
    remove anything between two angled brackets, anything before a close bracket if there is only a
    close bracket, and anything after an open bracket if there is only an open bracket
    
    Input : str
    output : str
    """

    # if both an open and close angle brackets are in the string
    if ('<' in string) and ('>' in string):
        open_brac = string.find('<')
        close_brac = string.find('>')
        # if there is something inside the brackets, drop that including the brackets
        if open_brac < close_brac:
            string = string.replace(string[open_brac : close_brac + 1]," ")
        # if there isn't anything in between them, drop everything before and after
        else:
            string = string.replace(string[:close_brac+1]," ")
            string = string.replace(string[open_brac:]," ")
        # use recursion to fix any instances where there are multiple opens and closes
        return remove_md(string)
    elif ('<' in string):
        open_brac = string.find('<')
        string = string.replace(string[open_brac:]," ")
        return remove_md(string)
    elif ('>' in string):
        close_brac = string.find('>')
        string = string.replace(string[:close_brac+1]," ")
        return remove_md(string)
    else:
        return string
    
class TextCleaner(BaseEstimator, TransformerMixin):
    """

    """
      
    def fit(self, X, y=None):

        return self
    
    def transform(self, X):
        new_df = pd.DataFrame()
        for col in ['description','name','amenities']:
            column = X[col]
            column[column.isna()] = ''
            #look at the original distribution of how many words there are per description
            column = column.apply(remove_md)
            #also going to want to remove some punctuation
            column = column.str.replace("[():!.,?/]"," ")
            X[col] = column
        return X
    
class TextConcater(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        new_df = X.copy()
        for col in ['amenities']:
            column = X[col]  # each value looks like its a list but is actually a str containing a list
            column[column.isna()] = '['']'
            column = column.apply(lambda x: ' '.join(eval(x)) if isinstance(eval(x),list) else '')
            new_df[col] = column
        return new_df

####################################
### start streamlit stuff###########
####################################


st.set_page_config(layout="wide")

# @st.experimental_memo(suppress_st_warning=True)
def main():
    
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.DataFrame(columns=['location', 'property_type', 'host_response_time', 'name',
                                 'description', 'amenities', 'price', 'accommodates', 'bathrooms_text',
                                 'bedrooms', 'beds', 'minimum_nights', 'latitude','longitude',
                                 'maximum_maximum_nights', 'host_acceptance_rate', 'neighbourhood',
                                 'host_listings_count', 'review_scores_rating', ' availability_90',
                                 'review_scores_value', 'review_scores_location', 'listing_num',
                                 'review_scores_communication', 'host_is_superhost', 'prediction',
                                 'calendar_last_scraped', 'host_since', 'availability_365',
                                 'host_response_rate', 'host_has_profile_pic',
                                 'host_identity_verified','instant_bookable','reviews_per_month',
                                 'review_scores_checkin','review_scores_cleanliness',
                                 'review_scores_accuracy'])
        st.session_state['model'] = XGBRegressor()
        st.session_state['model'].load_model('xgb_model.json')
        with open('preprocess_pipe.pkl', 'rb') as f:
            st.session_state['transformer'] = dill.load(f)
        st.session_state['num'] = 0
        
    text_column, col1, col2,col3, pred_column = st.columns([2.5, 1, 1, 1, 1])
    
    all_medians = pd.read_pickle('defaults.pkl')
    city_options = all_medians.index.unique()
    

    with text_column:
        text_column.subheader('Text like data input')
        city = st.selectbox(label='Listing Location',
                            options=city_options,
                            help='Select the location of your listing from the currently available\
                                set',
                            index=15,
                            key='location'
                            )
        meds = all_medians.loc[city]
        neighbourhood_options = meds['neighbourhood']
        property_options = meds['property_type']
        rt_options = meds['host_response_time']
        property_type = st.selectbox(label='Property type',
                                     options=property_options,
                                     help='Select the property type of your listing from the \
                                         available set',
                                     index=5,
                                     key='property_type'
                                     )
        response_time = st.selectbox(label='Response time',
                                     options=rt_options,
                                     help='Select how quickly you normally respond to booking\
                                         requests',
                                     index=0,
                                     key='host_response_time'
                                     )

        title = st.text_input(label='Listing Title', value="Your title here", max_chars=60)
        description = st.text_area(label='Listing Description', value="Your description here",
                                   height=200, max_chars=1000)
        amenities = st.text_area(label='Amenities', value="Iron, washer, dryer, etc.",
                                 height=50, max_chars=600,
                                 help='Enter your amenities separated by a comma')
    with col1:
        col1.subheader('Basic listing information')
        price = st.number_input(label='Price per night',
                                min_value=1.0,
                                max_value=10000.0,
                                value=meds['price'],
                                format='%f',
                                key='price',
                                help='Insert your nightly price here',
                                step=1.
                                )
        accomodates = st.number_input(label='Accommodates',
                                      min_value=1.0,
                                      max_value=200.0,
                                      value=meds['accommodates'],
                                      format='%f',
                                      key='accommodates',
                                      help='Insert the total number of people that can stay at\
                                          this listing',
                                      step=1.
                                      )
        bedrooms = st.number_input(label='Bedrooms',
                                   min_value=0.0,
                                   max_value=100.0,
                                   value=float(round(meds['bedrooms'])),
                                   format='%f',
                                   key='bedrooms',
                                   help='Insert the number of bedrooms',
                                   step=1.
                                   )
        beds = st.number_input(label='Beds',
                               min_value=0.0,
                               max_value=100.0,
                               value=float(round(meds['beds'])),
                               format='%f',
                               key='beds',
                               help='Insert the number of beds',
                               step=1.
                               )
        bathrooms = st.number_input(label='Bathrooms',
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=meds['bathrooms_text'],
                                    format='%f',
                                    key='bathrooms',
                                    help='Insert the number of bathrooms (.5 for half)',
                                    step=0.5
                                    )

        minimum_nights = st.number_input(label='Minimum nights',
                                         min_value=1.0,
                                         max_value=2000.0,
                                         value=float(round(meds['minimum_minimum_nights'])),
                                         format='%f',
                                         key='minimum_minimum_nights',
                                         help='Insert the minimum number of nights you allow for\
                                             bookings',
                                         step=1.
                                         )
        maximum_nights = st.number_input(label='Maximum nights',
                                         min_value=2.0,
                                         max_value=2000.0,
                                         value=float(round(meds['maximum_maximum_nights'])),
                                         format='%f',
                                         key='maximum_maximum_nights',
                                         help='Insert the total number of nights you allow',
                                         step=1.
                                         )

    with col2:
        col2.subheader('Host and review information')
        host_acceptance_rate = st.number_input(label='Acceptance rate',
                                               min_value=0.0,
                                               max_value=100.0,
                                               value=float(round(meds['host_acceptance_rate'])),
                                               format='%f',
                                               key='host_acceptance_rate',
                                               help='Insert the acceptance rate for this listing,\
                                                   as a percent',
                                               step=1.
                                               )
        host_listings_count = st.number_input(label='Listings count',
                                              min_value=1.0,
                                              max_value=10000.0,
                                              value=float(round(meds['host_listings_count'])),
                                              format='%f',
                                              key='host_listings_count',
                                              help='Insert your total number of listings',
                                              step=1.
                                              )

        review_scores_rating = st.number_input(label='Overall review score',
                                               min_value=0.0,
                                               max_value=5.0,
                                               value=float(round(meds['review_scores_rating'])),
                                               format='%f',
                                               key='review_scores_rating',
                                               help='Insert your overall rating from reviews',
                                               step=.5
                                               )
        review_scores_value = st.number_input(label='Value rating',
                                              min_value=0.0,
                                              max_value=5.0,
                                              value=float(round(meds['review_scores_value'])),
                                              format='%f',
                                              key='review_scores_value',
                                              help='Insert your rating for value',
                                              step=.5
                                              )
        review_scores_location = st.number_input(label='Location rating',
                                                 min_value=0.0,
                                                 max_value=5.0,
                                                 value=float(round(meds[
                                                     'review_scores_location'])),
                                                 format='%f',
                                                 key='review_scores_location',
                                                 help='Insert your rating for location',
                                                 step=.5
                                                 )
        review_scores_communication = st.number_input(label='Communication rating',
                                                      min_value=0.0,
                                                      max_value=5.0,
                                                      value=float(round(meds[
                                                          'review_scores_communication'])),
                                                      format='%f',
                                                      key='review_scores_communication',
                                                      help='Insert your rating for comunication',
                                                      step=.5
                                                      )
        review_scores_checkin = st.number_input(label='Checkin rating',
                                                      min_value=0.0,
                                                      max_value=5.0,
                                                      value=float(round(meds[
                                                          'review_scores_checkin'])),
                                                      format='%f',
                                                      key='review_scores_checkin',
                                                      help='Insert your rating for checkin',
                                                      step=.5
                                                      )
        review_scores_cleanliness = st.number_input(label='Cleanliness rating',
                                                      min_value=0.0,
                                                      max_value=5.0,
                                                      value=float(round(meds[
                                                          'review_scores_cleanliness'])),
                                                      format='%f',
                                                      key='review_scores_cleanliness',
                                                      help='Insert your rating for cleanliness',
                                                      step=.5
                                                      )
        review_scores_accuracy = st.number_input(label='Accuracy rating',
                                                      min_value=0.0,
                                                      max_value=5.0,
                                                      value=float(round(meds[
                                                          'review_scores_accuracy'])),
                                                      format='%f',
                                                      key='review_scores_accuracy',
                                                      help='Insert your rating for accuracy',
                                                      step=.5
                                                      )
        reviews_per_month = st.number_input(label='Review rate per month',
                                                      min_value=0.0,
                                                      max_value=31.,
                                                      value=float(round(meds[
                                                          'reviews_per_month'])),
                                                      format='%f',
                                                      key='reviews_per_month',
                                                      help='Insert how many reviews you get per month',
                                                      step=1.
                                                      )
        host_is_superhost = st.number_input(label='Superhost',
                                            min_value=0.0,
                                            max_value=1.0,
                                            value=float(round(meds['host_is_superhost'])),
                                            format='%f',
                                            key='host_is_superhost',
                                            help="1 if you are a superhost, 0 if you aren't",
                                            step=1.
                                            )
        with col3:
            latitude = st.number_input(label='Latitude',
                                            min_value=-180.0,
                                            max_value=180.0,
                                            value=float(round(meds['latitude'])),
                                            format='%.5f',
                                            key='latitude',
                                            help="decimal latitude",
                                            step=1.
                                            )
            longitude = st.number_input(label='Longitude',
                                            min_value=-180.0,
                                            max_value=180.0,
                                            value=float(round(meds['longitude'])),
                                            format='%.5f',
                                            key='longitude',
                                            help="decimal longitude",
                                            step=1.
                                            )
            neighbourhood = st.selectbox(label='Neighbourhood',
                                     options=neighbourhood_options,
                                     help='Select the neighbourhood of your listing from the \
                                         available set, or leave empty if it isn\'t there',
                                     index=0,
                                     key='neighbourhood'
                                     )
            
            host_since = st.date_input(label='Host since',
                                       value=datetime.date.today(),
                                       max_value= datetime.date.today(),
                                       key='host_since',
                                       help='Day of your first listing',
                                       )
            host_response_rate = st.number_input(label='Host response rate',
                                       value=99,
                                       min_value=0,
                                       max_value= 100,
                                       key='host_response_rate',
                                       help='Your response rate as a percent',
                                       )
        
        with pred_column:
            pred_column.subheader('Get the prediction for your listing')
            button = st.button(label='Get prediction')

            for col1 in all_medians:
                if col1 not in st.session_state['data'].keys():
                    st.session_state['data'][col1] = meds[col1]
            
            if button:
                st.session_state['num'] += 1
                st.session_state['data'] = st.session_state['data'].append({'location': city,
                         'property_type': property_type,
                         'host_response_time': response_time,
                         'name': title,
                         'description': description,
                         'amenities': str([a.strip() for a in amenities.split(',')]),
                         'price': '$'+str(price),
                         'accommodates': accomodates,
                         'bathrooms_text': str(bathrooms),
                         'bedrooms': bedrooms,
                         'beds': beds,
                         'minimum_nights': str(minimum_nights),
                         'maximum_maximum_nights': str(maximum_nights),
                         'host_acceptance_rate': str(host_acceptance_rate),
                         'host_listings_count': str(host_listings_count),
                         'review_scores_rating': str(review_scores_rating),
                         'review_scores_value': review_scores_value,
                         'review_scores_location': review_scores_location,
                         'review_scores_communication': review_scores_communication,
                         'host_is_superhost': host_is_superhost,
                         'prediction': np.nan,
                         'listing_num': st.session_state['num'],
                         'latitude': latitude,
                         'longitude': longitude,
                         'neighbourhood': neighbourhood,
                         'availability_90': 45,
                         'calendar_last_scraped': datetime.date.today(),
                         'host_since': host_since,
                         'availability_365': 330,
                         'host_response_rate': str(host_response_rate),
                         'host_has_profile_pic': 't',
                         'host_identity_verified': 't',
                         'instant_bookable': 't',
                         'reviews_per_month': reviews_per_month,
                         'review_scores_checkin': review_scores_checkin,
                         'review_scores_cleanliness': review_scores_cleanliness,
                         'review_scores_accuracy': review_scores_accuracy
                         }, ignore_index=True)
                # st.write(st.session_state['data'])
                
                prediction = (st.session_state['model']
                              .predict(st.session_state['transformer']
                                       .transform(st.session_state['data'].iloc[-1:])[:,:-1]
                                       )
                             )[0]

                
                st.session_state['data']['prediction'].iloc[-1] = prediction*price/3
                st.write(f'Your predicted monthly income is: $*{round(prediction*price/3)}*!')
                st.download_button(label='Download best listing',
                                   data=(st.session_state['data']
                                         [st.session_state['data']['prediction']
                                               ==st.session_state['data']['prediction'].max()]
                                         .to_csv().encode('utf-8')),
                                   file_name='optimal_listing.csv')
            
            # st.markdown(f"With this listing, we predict you'll earn *{prediction*price/3}*!")
            if button:
                with text_column:
                    single = alt.selection_single()
                    chart = alt.Chart(st.session_state.data).mark_point(
                        point=alt.OverlayMarkDef(color="red")).encode(
                        x=alt.X('listing_num', axis=alt.Axis(title='Listing attempt'),
                                scale=alt.Scale(domain=(0, st.session_state['data']['listing_num'].max()+1))),
                        y=alt.X('prediction', axis=alt.Axis(format='$', title='predicted monthly income')),
                        color=alt.condition(single, 'Origin:N', alt.value('lightgray')),
                        tooltip='prediction'
                    ).add_selection(single)
                    st.write(chart)
                # st.write(single.selection)




if __name__ == '__main__':
    main()
