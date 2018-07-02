# predict_foreclosure
Data cleaning/ ML project to predict foreclosures in Fannie Mae mortgages.

## Data and Directory Structure
Once downloaded, the project expects to have all Python files in a "code" subdirectory and all data in a "data" subdirectory on the same level.  The data can be downloaded here:  http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html. 

You will need all of the performance and acquisition data.

## Requirements

Python version 2.4.14
Pandas version 0.23.1
Sklearn version 0.19.1

## Running the code

To run the reduction script reduce_data.py, first import it

`import reduce_data`

then run the script

`reduce_data.main()`

It will then produce a file named reduced_data.csv that will be a csv text file.

## Opening the Jupyter notebook

To use the Jupyter notebook, 
