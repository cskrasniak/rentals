### This is an exploration of the airbnb listings for 30 US markets over the last year
### from http://insideairbnb.com/get-the-data/ 
import numpy as np
import pandas as pd
import os
from selenium import webdriver, webdrive_manager
# import selenium
# import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
from webdriver.common.by import By
import time
from tqdm import tqdm

download_dir = r'/Users/ckrasnia/Downloads'
final_dir = r'/Users/ckrasnia/Documents/application_materials/rental_data'
# navigate to the website
driver = webdriver.Chrome(webdrive_manager.chrome.ChromeDriverManager().install())
driver.get(r"http://insideairbnb.com/get-the-data")


# most of the data is hidden, so open all of those paths so it can be seen

show_additional_data = driver.find_elements(by=By.CLASS_NAME,value="showArchivedData")
time.sleep(5)  # allow time to connect to the website
for i in range(len(show_additional_data)):
    # scroll down to the next element
    driver.execute_script("arguments[0].scrollIntoView();", show_additional_data[i])
    time.sleep(.5)  # give some time for the browser to update
    show_additional_data = driver.find_elements(by=By.CLASS_NAME,value="showArchivedData")
    show_additional_data[i].click()  # expand to show the data
    time.sleep(.5)  # give some time for the browser to update

    # because it expands the page, need to refind the data
    show_additional_data = driver.find_elements(by=By.CLASS_NAME,value="showArchivedData")

# now that all the data is exposed, we are going to download each listings.csv file
target_data = driver.find_elements(by=By.LINK_TEXT,value='listings.csv.gz')
driver.execute_script("arguments[0].scrollIntoView(false);", target_data[0])
time.sleep(10)  # first scroll starts all the way at the bottom so need longer to scroll
for i in range(len(target_data)):
    # get the target_data again after scrolls
    target_data = driver.find_elements(by=By.LINK_TEXT,value='listings.csv.gz')
    # scroll down to the next target
    driver.execute_script("arguments[0].scrollIntoView(false);", target_data[i])
    time.sleep(.5)  # time for the browser to update
    # refind target data
    target_data = driver.find_elements(by=By.LINK_TEXT,value='listings.csv.gz')
    target_data[i].click()  # downloads the data
    time.sleep(2)  # time to download

## Now I have the data downloaded, I'll take only the US data, concatenate it to a single df and 
## save it
os.chdir(download_dir)
downloads = os.listdir()
# make sure the directory we are moving them to exists
if ~os.path.exists(final_dir()):
    os.mkdir(final_dir)
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
if ~os.path.exists(final_dir):
    os.mkdir(final_dir)
data.to_csv(os.path.join(final_dir,'raw_US_listings.csv'))
        
# now I have the data, and I want to clean it up a bit, first I'll drop some columns I wont need
