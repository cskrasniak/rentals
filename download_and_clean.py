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

basedir = '~/Desktop/airbnb_data'
os.chdir(basedir)
files = os.listdir(basedir)
data_list = []
listings = pd.concat([pd.read_csv(file) for file in files])
pd.read