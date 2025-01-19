import streamlit as st
import pandas as pd
import numpy as np
from helper import *
st.set_page_config(page_title="My Streamlit App", page_icon=":shark:", layout="wide")
from eda_page import eda_page


with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/15898288?s=200&v=4", width=100)
    st.title("Streamlit App")
    st.header("Menu")
    options = st.radio("Select an option", ["Home", "EDA", "Results"])
    
    
if options == "Home":
    home_page()
elif options == "EDA":
    eda_page()
elif options == "Results":
    st.write("Contact me at [email](mailto:contact@streamlit.io)")


# df = pd.DataFrame(np.random.randn(10, 2), columns=["a", "b"])
# st.line_chart(df)

