#!/usr/bin/env python
# Code for dumping the result dictionaries into a csv file
# and hopefully successfully recovering them
# Taken from https://stackoverflow.com/questions/8685809/python-writing-a-dictionary-to-a-csv-file-with-one-line-for-every-key-value

# Output
import csv
with open('l3Results_2.csv', 'wb') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in clf.cv_results_.items():
       writer.writerow([key, value])


# Input (untested)
with open('dict.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    mydict = dict(reader)
