import pandas as pd
from datetime import datetime, timedelta


def main():
    df = pd.read_excel(r'data.xlsx')
    process_time_patients(df)


def process_time_patients(df):
    t = datetime(2015,1,7)
    t_end = datetime(2015, 1, 8)
    t_end = datetime(2016,1,1)
    t_delta = timedelta(minutes=10)
    col_transfer_into_icu = df['Transfer into ICU']
    index_list = []
    while t < t_end:
        indices = df.loc[(col_transfer_into_icu >= t) & (col_transfer_into_icu < t+t_delta)]
        index_list.append(indices)




if __name__ == '__main__':
    main()