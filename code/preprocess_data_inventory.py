# -*- coding: utf-8 -*-

'''
CREATES AND INVENTORY OF THE FULL DATA SET
CREATES A SAMPLE DATA SET TO TEST PROGRAMS WITH AND TO STORE ON GITHUB

To Run, set your working directory to the project/code directory
'''

# import modules
import os
import csv
from shutil import copyfile

# create sample switch. Set to 1 if you want to create a sample directory
CREATE_SAMPLE = 0
#SAMPLE_PATH =  # insert path

# set new working directory
prepro_data_path = '../preprocessed_data/'
os.chdir(prepro_data_path)  # changes the working directory
cwd = os.getcwd()           # stores full path of working directory in 'cwd'
 
# creates an empty list
list_of_files = []
sample_files = []
file_count = 0
 
# create a list of files using os.walk()
for (dirpath, dirname, files) in os.walk(cwd):
    for filename in files:
        if filename.find('.DS_Store') == -1:    # skip this annoying hidden file in OSX folders
            thefile = os.path.join(dirpath, filename)
            list_of_files.append(thefile)
            if (file_count % 100 == 0) and (CREATE_SAMPLE == 1):
                name_of_copy = os.path.join(SAMPLE_PATH, filename)                
                copyfile(thefile, name_of_copy)
                sample_files.append(name_of_copy)
            file_count += 1

# ~~> If you are running this code on the raw text files (many of which do not have file extensions), 
# ~~> you might want to run the following lines to make it explicit that the files are .txt files
# ~~> after running, need to run the above again to update the inventory with the new names
#for name in list_of_files:    
#    os.rename(name, name + '.txt')
    
# remove non-project related portion of the path (i.e., change to relative path instead of absolute)
# this will also allow the files to be accessed from the 'code' subdirectory of the project directory

def absolute_to_relative_path(lst):
    for ii in xrange(len(lst)):
        jj = lst[ii].find('/preprocessed_data')
        lst[ii] = '..' + lst[ii][jj:]
    return lst

# clean up the path
list_of_files = absolute_to_relative_path(list_of_files)
if CREATE_SAMPLE == 1:
    sample_files = absolute_to_relative_path(sample_files)

# create csv file of email inventory
with open('preprocessed_email_inventory.csv', 'wb') as f:
    wr = csv.writer(f)
    for item in list_of_files:
        wr.writerow([item]) # put item in list so that it will be written as a whole and not parsed
 
# create sample set to test processing code and to store on github
with open('sample_data_inventory.csv', 'wb') as f:
    wr = csv.writer(f)
    for item in sample_files:
        wr.writerow([item])
            
         
 
