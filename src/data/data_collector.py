"""
Autonomous Gold Trading AI - Data Collection Module
Handles multi-source data collection, real-time streaming, and data validation
"""

import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Tuple
import requests
from alpha_vantage.timeseries import TimeSeries
from pymongo import MongoClient
import yaml

class MultiSourceDataCollector:
    """
    Comprehensive data collector for Gold (XAU/USD) from multiple sources
    with real-time streaming capabilities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the data collector with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.symbol = self.config['data']['symbol']
        self.timeframes = self.config['data']['timeframes']
        self.primary_timeframe = self.config['data']['primary_timeframe']
        self.lookback_bars = self.config['data']['lookback_bars']
        
        # Initialize data sources
        self._init_data_sources()
        
        # Database connection
        self.db_client = MongoClient(
            self.config['database']['host'],
            self.config['database']['port']
        )
        self.db = self.db_client[self.config['database']['name']]
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _init_data_sources(self):
        """Initialize all data source connections"""
        try:
            # Yahoo Finance (primary for historical data)
            self.yf_ticker = yf.Ticker("GC=F")  # Gold futures
            
            # Alpha Vantage for additional data
            if hasattr(self.config, 'alpha_vantage_key'):
                self.av = TimeSeries(key=self.config['alpha_vantage_key'])
            
            # CCXT exchanges for real-time data
            self.exchanges = {
                'binance': ccxt.binance({
                    'apiKey': self.config.get('binance_api_key', ''),
                    'secret': self.config.get('binance_secret', ''),
                    'sandbox': False,
                }),
                'oanda': ccxt.oanda({
                    'apiKey': self.config.get('oanda_api_key', ''),
                    'secret': self.config.get('oanda_secret', ''),
                    'sandbox': False,
                })
            }
            
            self.logger.info("Data sources initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}")
            
    def collect_historical_data(self, 
                              days: int = 365,
                              timeframe: str = "5m") -> pd.DataFrame:
        """
        Collect comprehensive historical data for Gold
        
        Args:
            days: Number of days to collect
            timeframe: Timeframe for data collection
            
        Returns:
            DataFrame with OHLCV data and metadata
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Primary data from Yahoo Finance
            data = self.yf_ticker.history(
                start=start_date,
                end=end_date,
                interval=self._convert_timeframe_yf(timeframe),
                auto_adjust=True,
                prepost=True
            )
            
            if data.empty:
                self.logger.warning("No data received from Yahoo Finance")
                return pd.DataFrame()
            
            # Clean and standardize data
            data = self._clean_ohlcv_data(data)
            
            # Add metadata
            data['source'] = 'yfinance'
            data['symbol'] = self.symbol
            data['timeframe'] = timeframe
            data['collected_at'] = datetime.now()
            
            # Store in database
            self._store_data(data, 'historical')
            
            self.logger.info(f"Collected {len(data)} bars of {timeframe} data")
            return data
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {e}")
            return pd.DataFrame()
    
    def collect_realtime_data(self, callback=None) -> Dict:
        """
        Collect real-time Gold price data
        
        Args:
            callback: Optional callback function for real-time updates
            
        Returns:
            Dictionary with current market data
        """
        try:
            # Get current data from multiple sources
            current_data = {}
            
            # Yahoo Finance current price
            yf_data = self.yf_ticker.history(period="1d", interval="1m").tail(1)
            if not yf_data.empty:
                current_data['yfinance'] = {
                    'price': float(yf_data['Close'].iloc[-1]),
                    'volume': float(yf_data['Volume'].iloc[-1]),
                    'timestamp': yf_data.index[-1].isoformat(),
                    'high': float(yf_data['High'].iloc[-1]),
                    'low': float(yf_data['Low'].iloc[-1]),
                    'open': float(yf_data['Open'].iloc[-1])
                }
            
            # Try to get data from exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    if exchange_name == 'oanda':
                        # OANDA XAU/USD
                        ticker = exchange.fetch_ticker('XAU/USD')
                        current_data[exchange_name] = {
                            'price': ticker['last'],
                            'bid': ticker['bid'],
                            'ask': ticker['ask'],
                            'timestamp': ticker['timestamp'],
                            'volume': ticker['baseVolume'] if ticker['baseVolume'] else 0
                        }
                except Exception as e:
                    self.logger.warning(f"Could not fetch from {exchange_name}: {e}")
            
            # Calculate consensus data
            if current_data:
                consensus = self._calculate_consensus_price(current_data)
                current_data['consensus'] = consensus
                
                # Store real-time data
                self._store_realtime_data(current_data)
                
                # Call callback if provided
                if callback:
                    callback(current_data)
                    
            return current_data
            
        except Exception as e:
            self.logger.error(f"Error collecting real-time data: {e}")
            return {}
    
    async def stream_realtime_data(self, callback=None):
        """
        Stream real-time data using WebSocket connections
        
        Args:
            callback: Callback function for each data update
        """
        try:
            # WebSocket streaming (implement based on available sources)
            while True:
                # Collect current data
                data = self.collect_realtime_data(callback)
                
                # Wait before next collection
                await asyncio.sleep(1)  # 1-second updates
                
        except Exception as e:
            self.logger.error(f"Error in streaming data: {e}")
    
    def get_market_sessions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market session information to the data
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            DataFrame with session information added
        """
        try:
            # Convert to UTC if not already
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            
            # Define session times (UTC)
            sessions = []
            for timestamp in data.index:
                hour = timestamp.hour
                
                if 22 <= hour or hour < 8:  # Asian session
                    session = 'Asian'
                elif 8 <= hour < 16:  # European session
                    session = 'European'
                else:  # US session
                    session = 'US'
                    
                sessions.append(session)
            
            data['session'] = sessions
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding session data: {e}")
            return data
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        Validate data quality and return quality metrics
        
        Args:
            data: OHLCV DataFrame to validate
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                'total_bars': len(data),
                'missing_values': data.isnull().sum().sum(),
                'duplicate_timestamps': data.index.duplicated().sum(),
                'zero_volume_bars': (data['Volume'] == 0).sum() if 'Volume' in data.columns else 0,
                'price_anomalies': 0,
                'data_gaps': 0
            }
            
            # Check for price anomalies (e.g., prices outside reasonable range)
            if not data.empty and 'Close' in data.columns:
                median_price = data['Close'].median()
                price_std = data['Close'].std()
                
                # Anomalies are prices more than 5 std devs from median
                anomaly_threshold = 5 * price_std
                quality_metrics['price_anomalies'] = (
                    (abs(data['Close'] - median_price) > anomaly_threshold).sum()
                )
            
            # Check for data gaps (missing time periods)
            if len(data) > 1:
                expected_interval = pd.infer_freq(data.index)
                if expected_interval:
                    expected_length = len(pd.date_range(
                        start=data.index[0],
                        end=data.index[-1],
                        freq=expected_interval
                    ))
                    quality_metrics['data_gaps'] = expected_length - len(data)
            
            # Calculate quality score (0-100)
            total_issues = (
                quality_metrics['missing_values'] +
                quality_metrics['duplicate_timestamps'] +
                quality_metrics['price_anomalies'] +
                quality_metrics['data_gaps']
            )
            
            quality_metrics['quality_score'] = max(0, 100 - (total_issues / len(data) * 100))
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {'quality_score': 0}
    
    def _clean_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize OHLCV data"""
        try:
            # Remove any rows with all NaN values
            data = data.dropna(how='all')
            
            # Ensure proper column names
            column_mapping = {
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            # Forward fill any missing OHLC values (but not volume)
            ohlc_columns = ['open', 'high', 'low', 'close']
            for col in ohlc_columns:
                if col in data.columns:
                    data[col] = data[col].fillna(method='ffill')
            
            # Fill missing volume with 0
            if 'volume' in data.columns:
                data['volume'] = data['volume'].fillna(0)
            
            # Remove any remaining rows with NaN in OHLC
            data = data.dropna(subset=ohlc_columns)
            
            # Ensure positive prices
            for col in ohlc_columns:
                if col in data.columns:
                    data[col] = data[col].abs()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            return data
    
    def _convert_timeframe_yf(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance format"""
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return mapping.get(timeframe, '5m')
    
    def _calculate_consensus_price(self, data_sources: Dict) -> Dict:
        """Calculate consensus price from multiple sources"""
        try:
            prices = []
            volumes = []
            
            for source, data in data_sources.items():
                if source != 'consensus' and 'price' in data:
                    prices.append(data['price'])
                    volumes.append(data.get('volume', 0))
            
            if prices:
                # Volume-weighted average price if volumes available
                if sum(volumes) > 0:
                    consensus_price = np.average(prices, weights=volumes)
                else:
                    consensus_price = np.mean(prices)
                
                return {
                    'price': consensus_price,
                    'sources_count': len(prices),
                    'price_std': np.std(prices),
                    'timestamp': datetime.now().isoformat()
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error calculating consensus: {e}")
            return {}
    
    def _store_data(self, data: pd.DataFrame, data_type: str):
        """Store data in MongoDB"""
        try:
            collection_name = self.config['database']['collections']['raw_data']
            collection = self.db[collection_name]
            
            # Convert DataFrame to records
            records = data.reset_index().to_dict('records')
            
            # Add metadata
            for record in records:
                record['data_type'] = data_type
                record['stored_at'] = datetime.now()
            
            # Insert records
            if records:
                collection.insert_many(records)
                self.logger.info(f"Stored {len(records)} {data_type} records")
                
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
    
    def _store_realtime_data(self, data: Dict):
        """Store real-time data in MongoDB"""
        try:
            collection_name = 'realtime_data'
            collection = self.db[collection_name]
            
            data['stored_at'] = datetime.now()
            collection.insert_one(data)
            
        except Exception as e:
            self.logger.error(f"Error storing real-time data: {e}")
    
    def get_latest_data(self, timeframe: str = None, bars: int = 100) -> pd.DataFrame:
        """
        Get latest data from database
        
        Args:
            timeframe: Timeframe to retrieve
            bars: Number of bars to retrieve
            
        Returns:
            DataFrame with latest data
        """
        try:
            if timeframe is None:
                timeframe = self.primary_timeframe
                
            collection_name = self.config['database']['collections']['raw_data']
            collection = self.db[collection_name]
            
            # Query latest data
            cursor = collection.find({
                'timeframe': timeframe,
                'data_type': 'historical'
            }).sort('timestamp', -1).limit(bars)
            
            # Convert to DataFrame
            data = list(cursor)
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    collector = MultiSourceDataCollector()
    
    # Collect historical data
    historical_data = collector.collect_historical_data(days=30, timeframe="5m")
    print(f"Collected {len(historical_data)} historical bars")
    
    # Validate data quality
    quality = collector.validate_data_quality(historical_data)
    print(f"Data quality score: {quality['quality_score']:.2f}")
    
    # Collect real-time data
    realtime_data = collector.collect_realtime_data()
    print(f"Real-time data: {realtime_data}")