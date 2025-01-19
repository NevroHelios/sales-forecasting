import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px


class EDA:
    def __init__(self):
        self.df = self._load_data()
        
    def _load_data(self):
        train, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/test.csv")
        train['test'], test['test'] = 0, 1
        train.date = pd.DatetimeIndex(train.date)
        test.date = pd.DatetimeIndex(test.date)
        df = pd.concat([train, test])
        return df
    
    def plot_numSold_date(self):
        self.df[self.df['test'] == 0].groupby('date')['num_sold'].sum().plot()   
        st.plotly_chart(plt.gcf(), use_container_width=True)
        
    def plot_sellTrend_country(self):
        train_df = self.df[self.df['test'] == 0]
        yearly_sales = train_df.groupby([train_df['date'].dt.year, 'country'])['num_sold'].sum().reset_index()

        fig = px.line(
            yearly_sales,
            x='date',
            y='num_sold',
            color='country',
            line_group='country',
            title='Sales Trends by Country (Year-wise)',
            labels={'year': 'Year', 'num_sold': 'Number of Products Sold'},
            color_discrete_sequence=px.colors.diverging.Armyrose
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_sellTrend_store(self):
        train_df = self.df[self.df['test'] == 0]
        yearly_sales = train_df.groupby([train_df['date'].dt.year, 'store'])['num_sold'].sum().reset_index()

        fig = px.line(
            yearly_sales,
            x='date',
            y='num_sold',
            color='store',
            line_group='store',
            title='Sales Trends by Store (Year-wise)',
            labels={'year': 'Year', 'num_sold': 'Number of Products Sold'},
            color_discrete_sequence=px.colors.diverging.Armyrose
        )
        st.plotly_chart(fig, use_container_width=True)
        
    def plot_sellTrend_product(self):
        train_df = self.df[self.df['test'] == 0]
        yearly_sales = train_df.groupby([train_df['date'].dt.year, 'product'])['num_sold'].sum().reset_index()

        fig = px.line(
            yearly_sales,
            x='date',
            y='num_sold',
            color='product',
            line_group='product',
            title='Sales Trends by Product (Year-wise)',
            labels={'year': 'Year', 'num_sold': 'Number of Products Sold'},
            color_discrete_sequence=px.colors.diverging.Armyrose
        )
        st.plotly_chart(fig, use_container_width=True)
        
# HOME Page
def home_page():
    st.title("📈 Sticker Sales Forecasting App")
    st.subheader("Playground Series - Season 5, Episode 1")
    
    st.markdown(
        """
        Welcome to the **Sticker Sales Forecasting** app, inspired by Kaggle's latest challenge:  
        **Forecasting Sticker Sales** 🏷️📊  
        
        ### About This App
        - **Objective**: Predict the number of stickers sold across different countries 🏙️🌍  
        - **Data Features**:  
            - **Date**: Time series data for each sale 📅  
            - **Country**: The location where the stickers were sold 🌎  
            - **Store**: The store type or category 🏬  
            - **Product**: Sticker types 🎨  
            - **Num_sold**: Number of stickers sold 🧾  
            - **Test**: Flag to differentiate between training and test sets ⚙️  
        
        ### What You’ll See
        - **Interactive Visualizations** 🖼️: Analyze sales trends over time  
        - **Model Insights** 🤖: Explore the sinusoidal pattern in sales and how it improves prediction  
        - **Real-time Forecasting** 🚀: Get predictions for future sales data  

        #### Why Stickers?  
        "At Kaggle, we take stickers seriously!"™️ – This app is perfect for honing machine learning skills using an interesting dataset.  

        Click on the menu to explore more and unleash the power of data! 🌟  
        """, 
        unsafe_allow_html=True
    )
