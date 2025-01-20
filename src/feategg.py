import  numpy as np
import pandas as pd
from src.cfg import CFG
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Feategg:
    def __init__(self, df: pd.DataFrame):
        self.df = self.feature_eng(df)
        self.df = self._add_gdp(self.df)
        self.train_df = self.df[self.df['test'] == 0]
        self.test_df = self.df[self.df['test'] == 1]
   

    def feature_eng(self, df: pd.DataFrame):
        # abt = pd.concat([train_df, test_df])
        abt = df.copy()
        abt['year'] = abt.date.dt.year
        abt['month'] = abt.date.dt.month
        abt['weekday'] = abt.date.dt.weekday
        abt['dayofyear'] = abt.date.dt.dayofyear
        abt['daynum'] = (abt.date - abt.date.iloc[0]).dt.days
        abt['weeknum'] = abt.daynum // 7


        dayisinyear = (abt.groupby('year').id.count() / len(CFG.countries) / len(CFG.stores) / len(CFG.products)).rename('dayisinyear').astype(int).to_frame()
        abt = abt.merge(dayisinyear, on='year', how='left')
        abt['partofyear'] = (abt['dayofyear'] - 1) / abt['dayisinyear'] # sinusoidal
        abt['partof2year'] = abt['dayofyear'] + abt['year'] % 2 # sinusoidal
        abt['sin 4t'] = np.sin(8 * np.pi * abt['partofyear'])  
        abt['cos 4t'] = np.cos(8 * np.pi * abt['partofyear'])
        abt['sin 3t'] = np.sin(6 * np.pi * abt['partofyear'])
        abt['cos 3t'] = np.cos(6 * np.pi * abt['partofyear'])
        abt['sin 2t'] = np.sin(4 * np.pi * abt['partofyear'])
        abt['cos 2t'] = np.cos(4 * np.pi * abt['partofyear'])
        abt['sin t'] = np.sin(2 * np.pi * abt['partofyear'])
        abt['cos t'] = np.cos(2 * np.pi * abt['partofyear']) # partofyear takes half a year to complete
        abt['sin t/2'] = np.sin(np.pi * abt['partof2year']) # partof2year takes a year to complete
        abt['cos t/2'] = np.cos(np.pi * abt['partof2year'])
        abt.drop(['partofyear', 'partof2year', 'dayisinyear'], axis=1, inplace=True)

        abt = self._add_gdp(abt)

        abt_no_ken_can = abt[~abt.country.isin(['Kenya', 'Canada'])] # keya and canada contains NaN values
        store_df = abt_no_ken_can.groupby(by='store').num_sold.mean().rename('store_factor').to_frame()
        abt = abt.drop('store_factor', axis=1, errors='ignore').join(store_df, on='store', how='left')

        return abt

    def _add_gdp(self, df):
        abt = df.copy()
        gdp_df = pd.read_csv("data/gdp_per_capita.csv")
        gdp_df.index = CFG.countries
        gdp_df.columns = CFG.years

        abt['gdp_factor'] = None
        for year in CFG.years:
            for country in CFG.countries:
                abt.loc[(abt.year == year) & (abt.country == country), 'gdp_factor'] = gdp_df.loc[country, year]
        
        return abt