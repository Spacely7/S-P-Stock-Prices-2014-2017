import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="S&P 500 Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("S&P 500 Stock Price Analysis (2014-2017)")
st.markdown("""
This app analyzes the stock prices of S&P 500 companies from 2014 to 2017.
* **Data source:** S&P 500 stock price dataset
* **Features:** Price trends, volume analysis, comparative analysis, and technical indicators
""")


# Function to load data
@st.cache_data
def load_data():
    # This is a placeholder. You would replace this with your actual data loading code
    df = pd.read_csv('S&P 500 Stock Prices 2014-2017.csv')

    # Creating sample data based on the provided format
    # In a real app, replace this with reading your actual dataset

    # Convert date to datetime


    df['date'] = pd.to_datetime(df['date'])

    return df


# Function to calculate technical indicators
def calculate_technical_indicators(df):
    # Calculate daily returns
    df['daily_return'] = df.groupby('symbol')['close'].pct_change() * 100

    # Calculate 20-day moving average (if enough data points)
    df['MA20'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

    # Calculate 50-day moving average (if enough data points)
    df['MA50'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())

    # Calculate RSI (14-day) - simplified version
    delta = df.groupby('symbol')['close'].diff()
    gain = (delta.where(delta > 0, 0)).groupby(df['symbol'])
    loss = (-delta.where(delta < 0, 0)).groupby(df['symbol'])

    avg_gain = gain.transform(lambda x: x.rolling(window=14, min_periods=1).mean())
    avg_loss = loss.transform(lambda x: x.rolling(window=14, min_periods=1).mean())

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


# Load data
df = load_data()

# Create sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a page:",
                           ["Overview", "Stock Price Analysis", "Volume Analysis", "Technical Indicators",
                            "Comparative Analysis", "Data Explorer"])

# Overview page
if options == "Overview":
    st.header("Overview of S&P 500 Stocks")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dataset Information")
        st.write(f"Number of stocks: {df['symbol'].nunique()}")
        st.write(f"Date range: {df['date'].min().strftime('%B %d, %Y')} to {df['date'].max().strftime('%B %d, %Y')}")
        st.write(f"Total records: {len(df)}")

    with col2:
        st.subheader("Sample Data")
        st.dataframe(df.head())

    # Market overview visualization
    st.subheader("Market Overview")

    # Get unique dates for the slider
    dates = sorted(df['date'].dt.strftime('%Y-%m-%d').unique())

    if len(dates) > 1:
        selected_date = st.select_slider(
            "Select date to view market snapshot:",
            options=dates,
            value=dates[0]
        )

        # Filter data for selected date
        daily_data = df[df['date'].dt.strftime('%Y-%m-%d') == selected_date]

        # Create a treemap of market cap (using close price * volume as a proxy)
        daily_data['market_activity'] = daily_data['close'] * daily_data['volume']

        fig = px.treemap(daily_data,
                         path=['symbol'],
                         values='market_activity',
                         color='close',
                         color_continuous_scale='RdBu',
                         title=f'Market Activity on {selected_date}')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("More dates needed for the interactive date slider.")

# Stock Price Analysis page
elif options == "Stock Price Analysis":
    st.header("Stock Price Analysis")

    # Select stocks for analysis
    selected_stocks = st.multiselect(
        "Select stocks to analyze:",
        options=sorted(df['symbol'].unique()),
        default=sorted(df['symbol'].unique())[:3]  # Default to first 3 stocks
    )

    if not selected_stocks:
        st.warning("Please select at least one stock to analyze.")
    else:
        # Filter data for selected stocks
        filtered_df = df[df['symbol'].isin(selected_stocks)]

        # Plot historical prices
        st.subheader("Historical Price Trends")

        # Price visualization
        fig = px.line(filtered_df,
                      x='date',
                      y='close',
                      color='symbol',
                      title='Closing Price Over Time',
                      labels={'close': 'Closing Price ($)', 'date': 'Date'})

        st.plotly_chart(fig, use_container_width=True)

        # Candlestick chart for a single selected stock
        st.subheader("Candlestick Chart")

        selected_stock_candlestick = st.selectbox(
            "Select a stock for candlestick chart:",
            options=selected_stocks
        )

        # Filter data for the selected stock
        stock_data = filtered_df[filtered_df['symbol'] == selected_stock_candlestick]

        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data['date'],
            open=stock_data['open'],
            high=stock_data['high'],
            low=stock_data['low'],
            close=stock_data['close'],
            name=selected_stock_candlestick
        )])

        fig.update_layout(
            title=f'{selected_stock_candlestick} Price Movement',
            xaxis_title='Date',
            yaxis_title='Price ($)'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Stock statistics
        st.subheader("Stock Statistics")

        # Calculate statistics for each selected stock
        stats_data = []
        for symbol in selected_stocks:
            stock_data = filtered_df[filtered_df['symbol'] == symbol]
            stats_data.append({
                'Symbol': symbol,
                'Avg Price': stock_data['close'].mean(),
                'Min Price': stock_data['low'].min(),
                'Max Price': stock_data['high'].max(),
                'Price Range': stock_data['high'].max() - stock_data['low'].min(),
                'Volatility': stock_data['close'].pct_change().std() * 100  # Approximation
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.style.highlight_max(axis=0), use_container_width=True)

# Volume Analysis page
elif options == "Volume Analysis":
    st.header("Trading Volume Analysis")

    # Select stocks for volume analysis
    vol_selected_stocks = st.multiselect(
        "Select stocks to analyze:",
        options=sorted(df['symbol'].unique()),
        default=sorted(df['symbol'].unique())[:3]  # Default to first 3 stocks
    )

    if not vol_selected_stocks:
        st.warning("Please select at least one stock to analyze.")
    else:
        # Filter data for selected stocks
        vol_filtered_df = df[df['symbol'].isin(vol_selected_stocks)]

        # Plot historical volumes
        st.subheader("Historical Trading Volumes")

        # Volume visualization
        fig = px.bar(vol_filtered_df,
                     x='date',
                     y='volume',
                     color='symbol',
                     title='Trading Volume Over Time',
                     labels={'volume': 'Volume', 'date': 'Date'})

        st.plotly_chart(fig, use_container_width=True)

        # Price-Volume relationship
        st.subheader("Price-Volume Relationship")

        selected_stock_pv = st.selectbox(
            "Select a stock for price-volume analysis:",
            options=vol_selected_stocks
        )

        # Filter data for the selected stock
        pv_stock_data = vol_filtered_df[vol_filtered_df['symbol'] == selected_stock_pv]

        # Create subplot with price and volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f'{selected_stock_pv} Price', f'{selected_stock_pv} Volume'))

        # Add price trace
        fig.add_trace(
            go.Scatter(x=pv_stock_data['date'], y=pv_stock_data['close'], name='Close Price'),
            row=1, col=1
        )

        # Add volume trace
        fig.add_trace(
            go.Bar(x=pv_stock_data['date'], y=pv_stock_data['volume'], name='Volume'),
            row=2, col=1
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Top trading days
        st.subheader("Top Trading Days by Volume")

        # Calculate top volume days for each stock
        top_days = []
        for symbol in vol_selected_stocks:
            stock_data = vol_filtered_df[vol_filtered_df['symbol'] == symbol]
            top_day = stock_data.loc[stock_data['volume'].idxmax()]
            top_days.append({
                'Symbol': symbol,
                'Date': top_day['date'].strftime('%Y-%m-%d'),
                'Volume': top_day['volume'],
                'Close Price': top_day['close'],
                'Daily Change': ((top_day['close'] - top_day['open']) / top_day['open']) * 100
            })

        top_days_df = pd.DataFrame(top_days)
        st.dataframe(top_days_df, use_container_width=True)

# Technical Indicators page
elif options == "Technical Indicators":
    st.header("Technical Indicators")

    # Calculate technical indicators
    tech_df = calculate_technical_indicators(df)

    # Select a stock for technical analysis
    tech_selected_stock = st.selectbox(
        "Select a stock for technical analysis:",
        options=sorted(df['symbol'].unique())
    )

    # Filter data for the selected stock
    tech_stock_data = tech_df[tech_df['symbol'] == tech_selected_stock]

    # Moving Averages
    st.subheader("Moving Averages")

    # Create subplot with price and MAs
    fig = go.Figure()

    # Add price trace
    fig.add_trace(
        go.Scatter(x=tech_stock_data['date'], y=tech_stock_data['close'], name='Close Price')
    )

    # Add MA traces
    fig.add_trace(
        go.Scatter(x=tech_stock_data['date'], y=tech_stock_data['MA20'], name='20-day MA')
    )

    fig.add_trace(
        go.Scatter(x=tech_stock_data['date'], y=tech_stock_data['MA50'], name='50-day MA')
    )

    fig.update_layout(
        title=f'{tech_selected_stock} Price with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("Relative Strength Index (RSI)")

    # Create RSI plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=tech_stock_data['date'], y=tech_stock_data['RSI'], name='RSI (14)')
    )

    # Add horizontal lines at 70 and 30
    fig.add_shape(
        type="line", line=dict(dash='dash', width=2, color="red"),
        y0=70, y1=70, x0=tech_stock_data['date'].min(), x1=tech_stock_data['date'].max()
    )

    fig.add_shape(
        type="line", line=dict(dash='dash', width=2, color="green"),
        y0=30, y1=30, x0=tech_stock_data['date'].min(), x1=tech_stock_data['date'].max()
    )

    fig.update_layout(
        title=f'{tech_selected_stock} RSI (14)',
        xaxis_title='Date',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Daily Returns
    st.subheader("Daily Returns")

    # Create daily returns plot
    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=tech_stock_data['date'], y=tech_stock_data['daily_return'], name='Daily Return (%)')
    )

    fig.update_layout(
        title=f'{tech_selected_stock} Daily Returns',
        xaxis_title='Date',
        yaxis_title='Return (%)'
    )

    st.plotly_chart(fig, use_container_width=True)

# Comparative Analysis page
elif options == "Comparative Analysis":
    st.header("Comparative Stock Analysis")

    # Select stocks for comparison
    comp_selected_stocks = st.multiselect(
        "Select stocks to compare:",
        options=sorted(df['symbol'].unique()),
        default=sorted(df['symbol'].unique())[:4]  # Default to first 4 stocks
    )

    if len(comp_selected_stocks) < 2:
        st.warning("Please select at least two stocks to compare.")
    else:
        # Filter data for selected stocks
        comp_filtered_df = df[df['symbol'].isin(comp_selected_stocks)]

        # Normalize prices for comparison
        st.subheader("Normalized Price Comparison")

        # Create normalized price dataframe
        norm_df = pd.DataFrame()

        for symbol in comp_selected_stocks:
            stock_data = comp_filtered_df[comp_filtered_df['symbol'] == symbol]
            if not stock_data.empty:
                first_price = stock_data['close'].iloc[0]
                norm_df[symbol] = stock_data['close'] / first_price * 100
                norm_df['date'] = stock_data['date']

        # Plot normalized prices
        fig = px.line(norm_df,
                      x='date',
                      y=comp_selected_stocks,
                      title='Normalized Price Performance (First day = 100)',
                      labels={'value': 'Normalized Price', 'date': 'Date', 'variable': 'Stock'})

        st.plotly_chart(fig, use_container_width=True)

        # Correlation analysis
        st.subheader("Price Correlation Analysis")

        # Create correlation matrix
        corr_df = pd.pivot_table(
            comp_filtered_df,
            values='close',
            index='date',
            columns='symbol'
        )

        correlation = corr_df.corr()

        # Plot correlation heatmap
        fig = px.imshow(
            correlation,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Price Correlation Matrix'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics comparison
        st.subheader("Performance Statistics")

        # Calculate statistics for each stock
        perf_stats = []
        for symbol in comp_selected_stocks:
            stock_data = comp_filtered_df[comp_filtered_df['symbol'] == symbol]
            if not stock_data.empty:
                # Calculate returns if possible
                if len(stock_data) > 1:
                    first_price = stock_data['close'].iloc[0]
                    last_price = stock_data['close'].iloc[-1]
                    total_return = (last_price - first_price) / first_price * 100

                    # Calculate volatility (standard deviation of daily returns)
                    daily_returns = stock_data['close'].pct_change().dropna()
                    volatility = daily_returns.std() * 100

                    # Calculate max drawdown
                    rolling_max = stock_data['close'].cummax()
                    drawdown = (stock_data['close'] - rolling_max) / rolling_max * 100
                    max_drawdown = drawdown.min()

                    perf_stats.append({
                        'Symbol': symbol,
                        'Total Return (%)': total_return,
                        'Volatility (%)': volatility,
                        'Max Drawdown (%)': max_drawdown,
                        'Avg Volume': stock_data['volume'].mean()
                    })

        if perf_stats:
            perf_df = pd.DataFrame(perf_stats)
            st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen')
                         .highlight_min(axis=0, color='lightcoral'),
                         use_container_width=True)
        else:
            st.info("Not enough data to calculate performance statistics.")

# Data Explorer page
elif options == "Data Explorer":
    st.header("Data Explorer")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Filters
        st.subheader("Filters")

        # Stock selection
        explorer_selected_stocks = st.multiselect(
            "Select stocks:",
            options=sorted(df['symbol'].unique()),
            default=None
        )

        # Date range selection
        date_min = df['date'].min().date()
        date_max = df['date'].max().date()

        start_date, end_date = st.date_input(
            "Select date range:",
            [date_min, date_max],
            min_value=date_min,
            max_value=date_max
        )

        # Price range selection
        price_min = float(df['close'].min())
        price_max = float(df['close'].max())

        min_price, max_price = st.slider(
            "Price range:",
            min_value=price_min,
            max_value=price_max,
            value=(price_min, price_max)
        )

        # Apply filters
        filtered_explorer_df = df.copy()

        if explorer_selected_stocks:
            filtered_explorer_df = filtered_explorer_df[filtered_explorer_df['symbol'].isin(explorer_selected_stocks)]

        filtered_explorer_df = filtered_explorer_df[
            (filtered_explorer_df['date'].dt.date >= start_date) &
            (filtered_explorer_df['date'].dt.date <= end_date) &
            (filtered_explorer_df['close'] >= min_price) &
            (filtered_explorer_df['close'] <= max_price)
            ]

    with col2:
        # Show filtered data
        st.subheader("Filtered Data")

        if len(filtered_explorer_df) > 0:
            st.dataframe(filtered_explorer_df, use_container_width=True)

            # Download button
            csv = filtered_explorer_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data",
                data=csv,
                file_name="filtered_stock_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data matches your filter criteria.")

    # Statistics on filtered data
    if len(filtered_explorer_df) > 0:
        st.subheader("Summary Statistics")

        # Calculate statistics
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        summary_stats = filtered_explorer_df[numeric_cols].describe()

        st.dataframe(summary_stats, use_container_width=True)

        # Visualization options
        st.subheader("Visualization")

        viz_type = st.selectbox(
            "Select visualization type:",
            options=["Price Distribution", "Volume Distribution", "Scatter Plot"]
        )

        if viz_type == "Price Distribution":
            fig = px.histogram(
                filtered_explorer_df,
                x="close",
                color="symbol" if len(explorer_selected_stocks) <= 10 else None,
                nbins=50,
                title="Closing Price Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Volume Distribution":
            fig = px.histogram(
                filtered_explorer_df,
                x="volume",
                color="symbol" if len(explorer_selected_stocks) <= 10 else None,
                nbins=50,
                title="Trading Volume Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Scatter Plot":
            x_axis = st.selectbox("X-axis:", options=numeric_cols, index=3)  # Default to 'close'
            y_axis = st.selectbox("Y-axis:", options=numeric_cols, index=4)  # Default to 'volume'

            fig = px.scatter(
                filtered_explorer_df,
                x=x_axis,
                y=y_axis,
                color="symbol" if len(explorer_selected_stocks) <= 10 else None,
                title=f"{y_axis.capitalize()} vs {x_axis.capitalize()}",
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
            
# Predictive Modeling Page
elif options == "Predictive Modeling":
    st.header("Stock Price Prediction Using Linear Regression")

    # Select stock for prediction
    pred_stock = st.selectbox(
        "Select a stock to predict its closing price:",
        options=sorted(df['symbol'].unique())
    )

    # Filter data
    stock_df = df[df['symbol'] == pred_stock].copy()

    # Sort by date and reset index
    stock_df.sort_values('date', inplace=True)
    stock_df.reset_index(drop=True, inplace=True)

    # Feature Engineering: Convert date to ordinal
    stock_df['date_ordinal'] = stock_df['date'].map(datetime.toordinal)

    # Features and target
    X = stock_df[['date_ordinal']]
    y = stock_df['close']

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Linear Regression
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Evaluation Metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**Train R-Squared:** {train_r2:.4f}")
    st.write(f"**Test R-Squared:** {test_r2:.4f}")
    st.write(f"**MAPE:** {mape:.4f}")

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Close Prices")
    pred_df = pd.DataFrame({
        'Date': stock_df.iloc[y_test.index]['date'],
        'Actual': y_test.values,
        'Predicted': y_pred
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'], mode='lines+markers', name='Predicted'))

    fig.update_layout(
        title=f'{pred_stock} - Actual vs Predicted Close Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


# Add footer
st.markdown("---")
st.markdown("S&P 500 Stock Analysis App | Created with Streamlit")
