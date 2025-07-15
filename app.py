import streamlit as st
import pandas as pd
import numpy as np
from src.helper import *
st.set_page_config(page_title="Sticker Sales Forecasting | Rank 120", page_icon="🏷️", layout="wide")
from src.eda_page import eda_page
from src.results_page import results_page


with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/15898288?s=200&v=4", width=100)
    st.title("🏷️ Sales Forecasting")
    st.markdown("**Kaggle Rank: 120** 🏆")
    st.header("Navigation")
    options = st.radio("Select a page", ["🏠 Home", "🔍 EDA", "🏆 Results"])
    
    
if options == "🏠 Home":
    home_page()
elif options == "🔍 EDA":
    eda_page()
elif options == "🏆 Results":
    results_page()


# df = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
# st.line_chart(df)

