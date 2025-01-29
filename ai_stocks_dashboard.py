import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

# Set page config
st.set_page_config(page_title="AI Stocks Dashboard", layout="wide")

# Extended AI stocks dictionary
AI_STOCKS = {
    'Core AI Infrastructure': [
        'NVDA', 'AMD', 'INTC', 'TSM', 'MU', 'AVGO', 'QCOM', 'MRVL', 'ARM', 'WOLF'
    ],
    'Cloud & Data': [
        'MSFT', 'AMZN', 'GOOGL', 'SNOW', 'NET', 'DDOG', 'MDB', 'CFLT', 'DBX', 'BOX',
        'ESTC', 'API', 'SUMO', 'NEWR', 'SPLK'
    ],
    'AI Development & Tools': [
        'PLTR', 'AI', 'PATH', 'SNPS', 'CDNS', 'ANSS', 'TTWO', 'ATVI', 'U', 'GTLB',
        'TDOC', 'DKNG', 'RBLX'
    ],
    'Enterprise AI Apps': [
        'ADBE', 'ADSK', 'NOW', 'HUBS', 'CRM', 'WDAY', 'TEAM', 'ZM', 'OKTA', 'DOCU',
        'COUP', 'TWLO', 'TTD', 'PINS', 'SNAP'
    ],
    'AI-Enabled Tech': [
        'META', 'AAPL', 'ABNB', 'UBER', 'SPOT', 'NFLX', 'DIS', 'ROKU', 'SHOP', 'ETSY',
        'DASH', 'LYFT', 'PTON'
    ],
    'Semiconductor Ecosystem': [
        'ASML', 'AMAT', 'LRCX', 'KLAC', 'TER', 'ACLS', 'ONTO', 'CCMP', 'COHU', 'MKSI'
    ],
    'AI Healthcare': [
        'VEEV', 'TDOC', 'RXRX', 'DNA', 'BEAM', 'CRSP', 'EDIT', 'NVTA', 'PACB', 'TWST',
        'SDGR', 'EXAS', '23ME'
    ],
    'AI Cybersecurity': [
        'CRWD', 'PANW', 'ZS', 'FTNT', 'NET', 'OKTA', 'S', 'RPD', 'TENB', 'CYBR',
        'SAIL', 'VRNS'
    ],
    'AI Financial Tech': [
        'SQ', 'PYPL', 'COIN', 'HOOD', 'UPST', 'AFRM', 'SOFI', 'NU', 'INTU', 'BILL',
        'ADYEY', 'WISE.L'
    ],
    'International AI': [
        'BABA', 'BIDU', '9984.T', 'SAP', '005930.KS', 'TCEHY', '700.HK', 'BEKE',
        'SE', 'GRAB'
    ]
}

def fetch_stock_data(ticker, period='1y'):
    """Fetch stock data for given ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if not hist.empty:
            return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
    return None

def plot_aggregate_performance(data, selected_stocks, weights=None):
    """Plot aggregate performance of selected stocks"""
    if not data or len(selected_stocks) == 0:
        return None
        
    if weights is None:
        weights = np.ones(len(selected_stocks)) / len(selected_stocks)
    
    try:
        # Find common dates across all stocks
        common_dates = pd.DatetimeIndex(sorted(set.intersection(
            *[set(data[stock].index) for stock in selected_stocks]
        )))
        
        # Create DataFrame with normalized prices for all stocks
        normalized_prices = pd.DataFrame(index=common_dates)
        for stock, weight in zip(selected_stocks, weights):
            stock_data = data[stock].loc[common_dates]
            normalized_prices[stock] = stock_data['Close'] / stock_data['Close'].iloc[0] * 100 * weight
        
        # Calculate portfolio value
        portfolio_value = normalized_prices.sum(axis=1)
        
        fig = go.Figure()
        
        # Add portfolio line
        fig.add_trace(go.Scatter(
            x=common_dates,
            y=portfolio_value,
            name='Portfolio',
            line=dict(color='rgb(0,100,80)', width=3)
        ))
        
        # Add individual stock lines if 5 or fewer stocks
        if len(selected_stocks) <= 5:
            for stock in selected_stocks:
                fig.add_trace(go.Scatter(
                    x=common_dates,
                    y=normalized_prices[stock],
                    name=stock,
                    line=dict(dash='dot'),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title="Aggregate Portfolio Performance (Normalized to 100)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating performance plot: {str(e)}")
        return None

def calculate_portfolio_metrics(data, weights=None):
    """Calculate portfolio metrics including Sharpe ratio, beta, etc."""
    try:
        if weights is None:
            weights = np.ones(len(data)) / len(data)
            
        # Find common dates across all stocks
        common_dates = pd.DatetimeIndex(sorted(set.intersection(
            *[set(data[stock].index) for stock in data.keys()]
        )))
        
        # Create DataFrame of closing prices
        prices = pd.DataFrame({
            ticker: data[ticker]['Close']
            for ticker in data.keys()
        }, index=common_dates)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            raise ValueError("Insufficient data for analysis")
            
        # Convert weights to numpy array if needed
        weights = np.array(weights)
        
        # Calculate portfolio metrics
        avg_returns = returns.mean()
        port_return = np.sum(avg_returns * weights) * 252
        cov_matrix = returns.cov()
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        
        # Sharpe Ratio (assuming risk-free rate of 2%)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol != 0 else 0
        
        # Calculate beta using market data
        try:
            spy_data = yf.download('SPY', start=returns.index[0], end=returns.index[-1])
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Align SPY returns with portfolio dates
            common_spy_dates = returns.index.intersection(spy_returns.index)
            if len(common_spy_dates) > 0:
                portfolio_returns = returns.loc[common_spy_dates].dot(weights)
                spy_returns = spy_returns[common_spy_dates]
                beta = stats.linregress(spy_returns, portfolio_returns)[0]
            else:
                beta = 0
        except Exception as e:
            print(f"Error calculating beta: {str(e)}")
            beta = 0
            
        # Calculate drawdown
        portfolio_returns = returns.dot(weights)
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = portfolio_cumulative.expanding().max()
        drawdowns = portfolio_cumulative/rolling_max - 1
        max_drawdown = drawdowns.min() * 100
        
        return {
            'Expected Annual Return': port_return * 100,
            'Annual Volatility': port_vol * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Beta': beta,
            'Maximum Drawdown': max_drawdown
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            'Expected Annual Return': 0.0,
            'Annual Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Beta': 0.0,
            'Maximum Drawdown': 0.0
        }

def plot_correlation_matrix(data):
    """Create correlation matrix heatmap"""
    if not data or len(data) < 2:
        return None
        
    try:
        # Find common dates across all stocks
        common_dates = pd.DatetimeIndex(sorted(set.intersection(
            *[set(data[stock].index) for stock in data.keys()]
        )))
        
        if len(common_dates) < 2:
            return None
            
        # Create DataFrame of closing prices
        prices_dict = {}
        for ticker in data.keys():
            try:
                prices_dict[ticker] = data[ticker].loc[common_dates, 'Close']
            except KeyError:
                continue
                
        if not prices_dict:
            return None
            
        prices = pd.DataFrame(prices_dict)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return None
            
        # Calculate correlation matrix
        corr = returns.corr()
        
        # Create annotation text
        annotations = []
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                annotations.append(dict(
                    x=corr.columns[j],
                    y=corr.index[i],
                    text=f"{corr.iloc[i, j]:.2f}",
                    font=dict(color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black'),
                    showarrow=False
                ))
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        # Update layout with better formatting
        fig.update_layout(
            title={
                'text': 'Stock Correlation Matrix',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=600,
            xaxis_tickangle=-45,
            yaxis_tickangle=0,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            annotations=annotations
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating correlation matrix: {str(e)}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for a stock"""
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def plot_technical_analysis(data, ticker):
    """Create technical analysis chart for a stock"""
    if ticker not in data or data[ticker] is None or data[ticker].empty:
        return None
        
    try:
        df = calculate_technical_indicators(data[ticker])
        
        fig = make_subplots(rows=3, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.6, 0.2, 0.2])
        
        # Price and indicators
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash'),
            opacity=0.5
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            opacity=0.5,
            fill='tonexty'
        ), row=1, col=1)
        
        # Volume
        colors = ['red' if row['Open'] - row['Close'] >= 0 
                 else 'green' for index, row in df.iterrows()]
        
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=3, col=1)
        
        # Add RSI threshold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{ticker} Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=800,
            showlegend=True
        )
        
        # Update yaxis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
        
    except Exception as e:
        print(f"Error creating technical analysis chart: {str(e)}")
        return None

# Sidebar
st.sidebar.title("AI Stocks Dashboard")
selected_category = st.sidebar.selectbox(
    "Select Category",
    list(AI_STOCKS.keys()),
    help="Choose a category of AI-related stocks"
)

selected_stocks = st.sidebar.multiselect(
    "Select Stocks to Analyze",
    AI_STOCKS[selected_category],
    default=AI_STOCKS[selected_category][:2],
    help="Choose up to 5 stocks for detailed analysis"
)

time_period = st.sidebar.selectbox(
    "Select Time Period",
    ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
    index=3,
    help="Choose the time period for analysis"
)

# Main content
st.title("AI Stocks Analysis Dashboard")

if selected_stocks:
    # Fetch data for selected stocks
    data = {}
    with st.spinner('Fetching stock data...'):
        for ticker in selected_stocks:
            stock_data = fetch_stock_data(ticker, time_period)
            if stock_data is not None:
                data[ticker] = stock_data
    
    if not data:
        st.error("No data available for selected stocks.")
        st.stop()
    
    # Portfolio Analysis Section
    st.header("Portfolio Analysis")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        perf_plot = plot_aggregate_performance(data, selected_stocks)
        if perf_plot:
            st.plotly_chart(perf_plot, use_container_width=True)
        else:
            st.warning("Unable to generate performance plot.")
    
    with col2:
        st.subheader("Portfolio Metrics")
        metrics = calculate_portfolio_metrics(data)
        
        # Display metrics with appropriate formatting and explanations
        cols = st.columns(2)
        
        with cols[0]:
            st.metric(
                "Expected Annual Return",
                f"{metrics['Expected Annual Return']:.1f}%",
                help="Annualized expected return based on historical data"
            )
            st.metric(
                "Annual Volatility",
                f"{metrics['Annual Volatility']:.1f}%",
                help="Annualized volatility as a measure of risk"
            )
            st.metric(
                "Maximum Drawdown",
                f"{metrics['Maximum Drawdown']:.1f}%",
                help="Largest peak-to-trough decline"
            )
        
        with cols[1]:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['Sharpe Ratio']:.2f}",
                help="Risk-adjusted return (higher is better)"
            )
            st.metric(
                "Beta",
                f"{metrics['Beta']:.2f}",
                help="Market sensitivity (1 = moves with market)"
            )
        
        # Add interpretation
        st.markdown("---")
        st.subheader("Analysis Summary")
        
        # Performance Analysis
        if metrics['Expected Annual Return'] > 15:
            st.success("🚀 Strong return potential")
        elif metrics['Expected Annual Return'] < 0:
            st.error("📉 Negative expected returns")
        else:
            st.info("📊 Moderate return potential")
        
        # Risk Analysis
        risk_level = (
            "low" if metrics['Annual Volatility'] < 15 
            else "moderate" if metrics['Annual Volatility'] < 25 
            else "high"
        )
        st.write(f"Risk Level: {risk_level.title()} (Volatility: {metrics['Annual Volatility']:.1f}%)")
        
        # Market Sensitivity
        if abs(metrics['Beta'] - 1) < 0.2:
            st.info("📊 Portfolio moves in line with market")
        elif metrics['Beta'] > 1.2:
            st.warning("📈 Higher market sensitivity than average")
        else:
            st.success("🛡️ Lower market sensitivity than average")
    
    # Correlation Analysis
    st.header("Correlation Analysis")
    corr_matrix = plot_correlation_matrix(data)
    if corr_matrix:
        st.plotly_chart(corr_matrix, use_container_width=True)
        
        # Add correlation interpretation
        if len(selected_stocks) > 1:
            st.markdown("#### Correlation Insights")
            high_corr_pairs = []
            low_corr_pairs = []
            
            # Calculate correlations between pairs
            returns_data = {ticker: data[ticker]['Close'].pct_change() 
                          for ticker in selected_stocks}
            returns_df = pd.DataFrame(returns_data).dropna()
            corr_matrix = returns_df.corr()
            
            # Find highly correlated and uncorrelated pairs
            for i in range(len(selected_stocks)):
                for j in range(i+1, len(selected_stocks)):
                    corr = corr_matrix.iloc[i, j]
                    pair = (selected_stocks[i], selected_stocks[j])
                    if abs(corr) > 0.7:
                        high_corr_pairs.append((pair, corr))
                    elif abs(corr) < 0.3:
                        low_corr_pairs.append((pair, corr))
            
            if high_corr_pairs:
                st.warning("Highly correlated pairs (potential diversification opportunity):")
                for pair, corr in high_corr_pairs:
                    st.write(f"• {pair[0]} & {pair[1]}: {corr:.2f}")
            
            if low_corr_pairs:
                st.success("Low correlation pairs (good diversification):")
                for pair, corr in low_corr_pairs:
                    st.write(f"• {pair[0]} & {pair[1]}: {corr:.2f}")
    else:
        st.warning("Unable to generate correlation matrix.")
    
    # Individual Stock Analysis
    st.header("Individual Stock Analysis")
    for ticker in selected_stocks:
        st.subheader(f"{ticker} Analysis")
        
        # Display current stats
        if ticker in data:
            df = data[ticker]
            current_price = df['Close'].iloc[-1]
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
            
            cols = st.columns(4)
            cols[0].metric(
                "Current Price",
                f"${current_price:.2f}",
                f"{price_change:+.2f}%"
            )
            cols[1].metric(
                "Volume",
                f"{df['Volume'].iloc[-1]:,.0f}"
            )
            cols[2].metric(
                "52w High",
                f"${df['High'].max():.2f}"
            )
            cols[3].metric(
                "52w Low",
                f"${df['Low'].min():.2f}"
            )
            
            # Technical Analysis Chart
            tech_chart = plot_technical_analysis(data, ticker)
            if tech_chart:
                st.plotly_chart(tech_chart, use_container_width=True)
            else:
                st.warning(f"Unable to generate technical analysis for {ticker}")

# Add footer with disclaimer
st.markdown("---")
st.markdown("""
    *Disclaimer: This dashboard is for informational purposes only. 
    Not financial advice. Data provided by Yahoo Finance.*
""")