import os
import pandas as pd
import numpy as np
import streamlit as st

from prophet import Prophet
from bokeh.plotting import figure, show, ColumnDataSource, output_file

import pydaisi as pyd


def plot_prophet(df: pd.DataFrame):
    '''
    Plot a fitted Prophet model

    This function takes a data frame that contains a fit prophet model
    and returns a Bokeh plot of the results.

    :param pd.DataFrame df: The data to plot, generated from fit_prophet()

    :return: Bokeh Visualization of Prophet Results
    '''
    source = ColumnDataSource(df)

    p = figure(x_axis_type="datetime", plot_width=800, plot_height=350)
    p.line('ds', 'yhat', source=source)

    return p


def fit_prophet(df: pd.DataFrame=None, x_var: str=None, y_var: str=None, periods: int=365):
    '''
    Fit a Prophet model to the given dataset

    This function takes a data frame and, given the specified variables
    and number of components, fits a prophet model to the data. If the input data
    frame is not specified, Daisi creation data is used by default

    :param pd.DataFrame df: The data with which to generate Prinicpal Components
    :param str x_var: The date variable to use in the data frame
    :param str y_var: The response variable to use in the data frame
    :param int periods: The number of periods to predict

    :return: DataFrame of Prophet Results
    '''
    if type(df) == str and os.path.isfile(df):
        df = pd.read_csv(df)

    if df is None:
        daisi_platform_growth = pyd.Daisi("erichare/Daisi Platform Growth", base_url="https://dev3.daisi.io")
        df = daisi_platform_growth.get_growth(instance="dev3").value
        x_var = "date"
        y_var = "total"

    m = Prophet()

    df.dropna(axis=0, how='any', inplace=True)

    df["ds"] = df[x_var]
    df['ds'] = df['ds'].dt.tz_localize(None)
    df["y"] = df[y_var]

    m.fit(df)

    future = m.make_future_dataframe(periods=periods)

    res = m.predict(future)

    return res


if __name__ == "__main__":
    st.set_page_config(layout = "wide")
    st.title("Generalized Prophet Forecasting")

    st.write("This Daisi allows for a generalized specification of a Prophet forecasting model. Upload your data to get started!")
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # daisi_platform_growth = pyd.Daisi("erichare/Daisi Platform Growth")
        # df = daisi_platform_growth.get_growth().value
        df = pd.read_csv("growth.csv")
        df['date'] = pd.to_datetime(df['date'])
        x_var = "date"
        y_var = "total"

    numeric_vars = list(df.select_dtypes([np.number]).columns)
    date_vars = list(df.select_dtypes(include=['datetime64', 'datetime64[ns]', 'datetime64[ns, UTC]']))

    with st.sidebar:
        x_var = st.selectbox("Choose Date Variable", date_vars)
        y_var = st.selectbox("Choose Response Variable", numeric_vars)
        periods = st.number_input("Choose Forecast Periods", min_value=1, max_value=1000, step=1, value=365)

    prophet_data = fit_prophet(df, x_var=x_var, y_var=y_var, periods=periods)
    p = plot_prophet(prophet_data)

    st.bokeh_chart(p, use_container_width=True)
