# -*- coding: utf-8 -*-

'''
CREATES AND INVENTORY OF THE RAW DATA
CREATES A SAMPLE DATA SET TO TEST PROGRAMS WITH AND TO STORE ON GITHUB

To Run, set your working directory to the project/code directory
'''

# import modules
import os
import csv
from shutil import copyfile
from numpy import random


# declare constants
CREATE_SAMPLE = 1   # set to 1 if you want to create a sample directory
NUM_SAMPLES = 500
SAMPLE_PATH = 'data_sample'


# set new working directory to inventory
raw_data_path = '../raw_data/'  #'../raw_data/'
os.chdir(raw_data_path)  # changes the working directory
cwd = os.getcwd()        # stores full path of working directory



# |~~~~~~~~~~~~~~~~~~~~~~~~| 
# |  CREATE LIST OF FILES  |
# |~~~~~~~~~~~~~~~~~~~~~~~~|
 
# creates an empty list
list_of_files = []
file_count = 0
 
# creates a list of file names to skip
skips = ['.DS_Store', 'raw_data_inventory.csv', \
            'email_text.csv', 'raw_data_sample.csv ', 'test3.txt']

# create a list of files using os.walk()
for (dirpath, dirname, files) in os.walk(cwd):
    dirname[:] = [d for d in dirname if d != 'data_sample']
    for filename in files:
        if filename not in skips:
            thefile = os.path.join(dirpath, filename)
            if not thefile.endswith('.txt'): # not all files end with .txt, this fixes those files
                new_name = thefile + '.txt'
                os.rename(thefile, new_name)
                thefile = new_name
            list_of_files.append(thefile)
            file_count += 1
    
# remove non-project related portion of the path (i.e., change to relative path instead of absolute)
# this will also allow the files to be accessed from the 'code' subdirectory of the project directory

# |~~~~~~~~~~~~~~~~~~~~~~| 
# |  WRITE FILES TO CSV  |
# |~~~~~~~~~~~~~~~~~~~~~~|


def absolute_to_relative_path(s):
    ii = s.find('/raw_data')
    s = '..' + s[ii:]
    return s

def create_rand_sample(lof):
    sample_files = []
    random.seed(123)
    rand_sel = random.choice(a=len(lof), size=NUM_SAMPLES, replace=False)
    for ii in range(len(lof)):
        if ii in rand_sel:
            sample_files.append(lof[ii])    
    return sample_files

def save_sample_files(sl):
    for name in sl:
        filename = name[name.rfind('/')+1:]
        copyname = os.path.join(SAMPLE_PATH, filename)
        copyfile(name, copyname)    



# create csv file of email inventory
with open('raw_data_inventory.csv', 'wb') as f:
    wr = csv.writer(f)
    for item in list_of_files:
        item = absolute_to_relative_path(item)
        wr.writerow([item]) # put item in list so that it will be written as a whole and not parsed
 
 
# create sample set to test code and to store on github
if CREATE_SAMPLE == 1:
    sample = create_rand_sample(list_of_files)
    save_sample_files(sample)
    # write sample file list to csv
    with open('raw_data_sample.csv', 'wb') as f:
        wr = csv.writer(f)
        for item in sample:
            item = absolute_to_relative_path(item)
            wr.writerow([item])
