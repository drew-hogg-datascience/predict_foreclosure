import pandas as pd
from setup_data_reduction import *
import glob as glob
import re

CHUNKSIZE = 100000 # processing 100,000 rows at a time

#def process_data()

def read_data(file_list, line_count):

        if re.search(r'Acquisition', file_list[0]):
            data_type = 'Acquisition'
        if re.search(r'Performance', file_list[0]):
            data_type = 'Performance'

        first_pass = True
        for file_name in file_list:
            print 'Reading File %s' %file_name
            reader = pd.read_table(file_name, sep="|", header=None, names=HEADERS[data_type], chunksize = CHUNKSIZE, low_memory=False)

            for rows in reader:
                line_count += rows['id'].count()

                data_in = rows[SELECT[data_type]] #Only keep data we select

                if (data_type == 'Acquisition'):
                    year = re.findall("_([0-9]*)", file_name)[0]
                    quarter = re.findall("Q([0-9]*)", file_name)[0]
                    data_in['acquisition_date'] = pd.to_numeric(year)+pd.to_numeric(quarter)/4.

                if (data_type == 'Performance'):
                    if data_in.empty:
                        break
                    data_in = data_in[data_in['foreclosure_date'].notnull()]
                    data_in['foreclosure_date'] = data_in['foreclosure_date'].astype('str')
                    data_in = convert_dates(data_in, 'foreclosure_date')

                if (first_pass):
                    data = data_in
                    first_pass = False
                else:
                    data = pd.concat([data, data_in])

        return data, line_count

def convert_dates(data, column):

    out = data[column].str.split('/', expand=True)
    if (out.shape[1] == 2):
        data[column] = pd.to_numeric(out[1])+pd.to_numeric(out[0])/12.
    if (out.shape[1] == 3):
        data[column] = pd.to_numeric(out[2])+pd.to_numeric(out[0])/12.

    return data

def map_first_time_homebuyer(data):

    data['first_time_homebuyer'].fillna('U', inplace=True)
    data['first_time_homebuyer'].astype(basestring)
    data['first_time_homebuyer'] = data['first_time_homebuyer'].map({'Y': 1, 'N': 0, 'U':2})
    data['first_time_homebuyer'] = data['first_time_homebuyer'].astype(int)

    return data

def map_property_state(data):

    data['property_state'].replace('NaN',0)
    data['property_state'] = data['property_state'].map(STATES_TO_VAL)
    data['property_state'] = pd.to_numeric(data['property_state'])

    return data

#def loan_age(data):

def index_dataframe(data):

    data['id'] = data['id'].astype(int)
    data.sort_values(by='id')
    data.set_index('id')
    data.drop_duplicates(subset='id', keep='first')

    return data

def main():

    count = 0 # Row counter

    ### Read in acquisition data and setup dataframe ###
    acq_files = glob.glob(DATA_DIR+'Acquisition*.txt')

    data, count = read_data(acq_files, count)

    ###   Filter data fields   ###
    data = data[FACTORS['In']]

    ###   Convert first time home buyer  ###
    ###   field into numeric value       ###
    data = map_first_time_homebuyer(data)

    ###   Filter and map property state field   ###
    data = map_property_state(data)

    ###   Convert origination date field into   ###
    ###   float value with decimal for month    ###
    data = convert_dates(data, 'origination_date')
    data['loan_age'] = data['acquisition_date']-data['origination_date']

    ###   Set index to id field and tidy up a bit   ###
    data = index_dataframe(data)

    ### Add data for eventual foreclosure
    prf_files = glob.glob(DATA_DIR+'Performance*.txt')

    foreclosed, count = read_data(prf_files, count)

    #print data.head()

    foreclosed = index_dataframe(foreclosed)

    ###   Add a True foreclosure flag on   ###
    ###  rows in foreclosed dataframe      ###
    foreclosed['foreclosure'] = 1.0

    print 'Joining dataframes'
    data = data.join(foreclosed.set_index('id'), on='id')
    print data['foreclosure'].shape
    data['foreclosure'] = data['foreclosure'].fillna(0)
    print data['foreclosure'].shape

    print 'Saving to %s' %(DATA_DIR+DATA_FILE)
    print data.head()
    data.to_csv(DATA_DIR+DATA_FILE)

    print count
    print 'Finished reduction.  Total rows processed = %s' %count
