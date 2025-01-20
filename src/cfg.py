import pandas as pd

class CFG:
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    train_df.date = pd.DatetimeIndex(train_df.date)
    test_df.date = pd.DatetimeIndex(test_df.date)
    
    years_train = train_df.date.dt.year.unique()
    years_test = test_df.date.dt.year.unique()
    years = pd.concat([train_df, test_df]).date.dt.year.unique()
    
    validation_year = 2018
    
    countries = train_df.country.unique()
    stores = train_df.store.unique()
    products = train_df['product'].unique()
    
    alpha3 = {
        'Finland': 'FIN',
        'Canada': 'CAN',
        'Italy': 'ITA',
        'Kenya': 'KEN',
        'Norway': 'NOR',
        'Singapore': 'SGP',
    } # to get per capita GDP
    fft_filter_width = 8
    
    countries_21 = {
        'Finland': 'FI',
        'Canada': 'CA',
        'Italy': 'IT',
        'Kenya': 'KE',
        'Norway': 'NO',
        'Singapore': 'SG',
    } # to get holidays
    holiday_response_len = 10
    
    sincoscol = ['sin t', 'cos t', 'sin t/2', 'cos t/2']
    sincoscol2 = ['sin2t', 'cos 2t', *sincoscol]