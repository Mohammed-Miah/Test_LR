import streamlit as st
from datetime import date
import yfinance as yf

import prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start = "2015-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stocks = ("AAPL", "GOOG", "MSFT")

selected_stock = st.selectbox("Select Dataset", stocks)

n_years = st.slider("Years of Predictio", 1, 4)

period = n_years * 365
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, today)

    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load Data..")
data = load_data(selected_stock)
data_load_state.text("Loading Data... Done!")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y = data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data["Date"], y = data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)
    
plot_raw_data()

#forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': "ds", "Close": "y"})

m = prophet.Prophet()
m.fit(df_train)


future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(forecast.head())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

#components
fig2 = m.plot_components(forecast)
st.write(fig2)

