import streamlit as st
from src.helper import EDA

def eda_page():
    st.title("🔍 Exploratory Data Analysis")
    st.markdown(
        """
        In this section, we dive deep into the **sticker sales data** to uncover patterns and trends.  
        Let's explore how sales vary across **time**, **countries**, **stores**, and **products**.
        """
    )
    
    eda = EDA()
    
    # Sales Over Time and Country Analysis Group
    with st.expander("📊 Time and Geographic Analysis", expanded=True):
        st.subheader("1. Sales Trends Over Time 📅")
        st.markdown(
            """
            **Objective**: Analyze the sinusoidal patterns in sales over time.  
            - The number of stickers sold shows clear seasonal trends.  
            - Peaks and troughs correspond to festive seasons and low-sales periods.  
            """
        )
        eda.plot_numSold_date()
            
    
    # Store and Product Analysis Group
    with st.expander("🏪 Store and Product Analysis", expanded=True):
        st.subheader("2. Country-wise Sales Trends 🌍")
        st.markdown(
            """
            **Objective**: Visualize how sticker sales differ across countries.  
            - **Top-performing countries** have higher sales during certain periods.  
            - Use this insight to adjust marketing strategies for underperforming regions.  
            """
        )
        eda.plot_sellTrend_country()
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("3. Store-wise Sticker Sales 🏬")
            st.markdown(
                """
                **Objective**: Evaluate how different store types influence sticker sales.  
                - Certain store categories sell more stickers due to higher foot traffic.  
                """
            )
            eda.plot_sellTrend_store()
            
        with col4:
            st.subheader("4. Product-wise Sticker Sales 🎨")
            st.markdown(
                """
                **Objective**: Analyze which products are the best sellers.  
                - **Top sticker types** can be targeted for promotions.  
                - Seasonal demand for specific stickers is evident from sales spikes.  
                """
            )
            eda.plot_sellTrend_product()
       
    with st.expander("📈 Sinusoidal Sales Analysis", expanded=True): 
        eda.plot_sinusoidal_sells()
    
    with st.expander("📊 Fourier Analysis", expanded=True):
        eda.fourier_analysis()