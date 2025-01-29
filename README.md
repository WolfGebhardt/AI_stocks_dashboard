# AI Stocks Dashboard

An interactive dashboard for analyzing AI-related stocks, built with Streamlit and Python. This tool provides comprehensive analysis of AI company stocks across different sectors, including portfolio analytics, correlation analysis, and technical indicators.

## Features

### 1. Portfolio Analysis
- Aggregate performance tracking
- Risk-adjusted return metrics
- Portfolio correlation analysis
- Interactive visualizations
- Key metrics:
  - Expected annual return
  - Annual volatility
  - Sharpe ratio
  - Beta
  - Maximum drawdown

### 2. Stock Categories
- Core AI Infrastructure
- Cloud & Data
- AI Development & Tools
- Enterprise AI Apps
- AI-Enabled Tech
- Semiconductor Ecosystem
- AI Healthcare
- AI Cybersecurity
- AI Financial Tech
- International AI

### 3. Technical Analysis
- Interactive price charts
- Technical indicators:
  - Simple Moving Averages (20 & 50 day)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
- Volume analysis
- Price statistics

### 4. Correlation Analysis
- Interactive correlation heatmap
- Pair-wise correlation insights
- Diversification opportunities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-stocks-dashboard.git
cd ai-stocks-dashboard
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Unix or MacOS
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a desktop shortcut (optional):
```bash
python create_shortcut.py
```

## Usage

1. Start the dashboard:
```bash
streamlit run ai_stocks_dashboard.py
```

2. Select stocks to analyze:
   - Choose a category from the sidebar
   - Select up to 5 stocks for detailed analysis
   - Choose the time period for analysis

3. Analyze the results:
   - View portfolio performance
   - Check correlation between stocks
   - Analyze technical indicators
   - Review risk metrics

## Requirements

- Python 3.8+
- Dependencies (included in requirements.txt):
  - streamlit
  - yfinance
  - pandas
  - plotly
  - numpy
  - scipy

## Project Structure

```
ai-stocks-dashboard/
│
├── ai_stocks_dashboard.py    # Main dashboard application
├── create_shortcut.py        # Script to create desktop shortcut
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── venv/                     # Virtual environment (created during installation)
```

## Troubleshooting

### Common Issues

1. Data Loading Errors
   - Ensure you have a stable internet connection
   - Verify the stock symbols are correct
   - Try refreshing the page

2. Display Issues
   - Make sure all dependencies are correctly installed
   - Clear your browser cache
   - Try using a different browser

3. Performance Issues
   - Reduce the number of selected stocks
   - Choose a shorter time period
   - Restart the application

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## Disclaimer

This dashboard is for informational purposes only. Not financial advice. Data provided by Yahoo Finance.

## License

MIT License - see LICENSE file for details.

## Support

For support, please:
1. Check the [issues](https://github.com/yourusername/ai-stocks-dashboard/issues) page
2. Create a new issue if your problem isn't already listed
3. Provide detailed information about your problem

## Future Improvements

Planned features:
- Portfolio optimization suggestions
- Export functionality for analysis results
- Additional technical indicators
- Backtesting capabilities
- Machine learning-based predictions

Screenshots
![image](https://github.com/user-attachments/assets/d4775a22-5877-404d-a86a-de172a495107)
![image](https://github.com/user-attachments/assets/f87fa348-6544-4171-a54a-9195143ea82d)
![image](https://github.com/user-attachments/assets/717e8e63-8cc9-4c23-bf25-c8f7e7054b89)

