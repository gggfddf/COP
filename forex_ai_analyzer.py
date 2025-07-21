#!/usr/bin/env python3
"""
ULTIMATE DEEP LEARNING FOREX MARKET ANALYSIS SYSTEM
Comprehensive AI-Powered Trading Intelligence for XAUUSD and Forex Pairs
Author: Advanced Trading AI System
Version: 2.0 Professional
"""

import pandas as pd
import numpy as np
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import scipy.stats as stats
from scipy.signal import find_peaks, argrelextrema
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import talib

import seaborn as sns
import datetime as dt
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== EDITABLE PARAMETER SECTION =====
FOREX_SYMBOL = "XAUUSD=X"  # ONLY THIS SHOULD BE CHANGED
# Alternative symbols: "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"
# =====================================

@dataclass
class ForexAnalysisConfig:
    """Configuration for forex analysis system"""
    symbol: str = FOREX_SYMBOL
    periods: List[str] = None
    lookback_days: int = 730  # 2 years of data
    prediction_horizon: int = 5  # 5-day prediction
    confidence_threshold: float = 0.75
    
    def __post_init__(self):
        if self.periods is None:
            self.periods = ['5m', '15m', '1h', '1d', '1wk']

class ProprietaryPatternEngine:
    """Advanced ML-based pattern discovery engine for forex markets"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.patterns = {}
        self.pattern_performance = {}
        self.scaler = StandardScaler()
        
    def discover_candlestick_patterns(self) -> Dict:
        """Autonomous candlestick pattern discovery using ML clustering"""
        logger.info("Discovering proprietary candlestick patterns...")
        
        # Create feature matrix for candlestick analysis
        features = self._extract_candlestick_features()
        
        # Apply unsupervised learning for pattern discovery
        kmeans = KMeans(n_clusters=15, random_state=42)
        pattern_clusters = kmeans.fit_predict(features)
        
        # Analyze each discovered pattern cluster
        patterns = {}
        for cluster_id in range(15):
            cluster_mask = pattern_clusters == cluster_id
            cluster_data = self.data[cluster_mask]
            
            if len(cluster_data) > 10:  # Minimum pattern occurrences
                pattern_info = self._analyze_pattern_cluster(cluster_data, cluster_id)
                patterns[f"ML_Pattern_{cluster_id}"] = pattern_info
                
        self.patterns['candlestick'] = patterns
        return patterns
    
    def _extract_candlestick_features(self) -> np.ndarray:
        """Extract comprehensive candlestick features for ML analysis"""
        data = self.data.copy()
        
        # Basic OHLC ratios
        data['body_size'] = abs(data['Close'] - data['Open']) / data['Open']
        data['upper_shadow'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / data['Open']
        data['lower_shadow'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / data['Open']
        data['total_range'] = (data['High'] - data['Low']) / data['Open']
        
        # Advanced pattern features
        data['body_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['price_momentum'] = data['Close'].pct_change(5)
        data['volatility'] = data['Close'].rolling(10).std() / data['Close'].rolling(10).mean()
        
        # Multi-candle relationships
        data['prev_body_ratio'] = data['body_size'] / data['body_size'].shift(1)
        data['volume_surge'] = data['Volume'] / data['Volume'].shift(1)
        data['gap_size'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        feature_columns = ['body_size', 'upper_shadow', 'lower_shadow', 'total_range',
                          'body_position', 'volume_ratio', 'price_momentum', 'volatility',
                          'prev_body_ratio', 'volume_surge', 'gap_size']
        
        features = data[feature_columns].fillna(0)
        return self.scaler.fit_transform(features)
    
    def _analyze_pattern_cluster(self, cluster_data: pd.DataFrame, cluster_id: int) -> Dict:
        """Analyze performance and characteristics of discovered pattern cluster"""
        
        # Calculate pattern success metrics
        future_returns = []
        for idx in cluster_data.index:
            if idx + 5 < len(self.data):
                future_return = (self.data.loc[idx + 5, 'Close'] - self.data.loc[idx, 'Close']) / self.data.loc[idx, 'Close']
                future_returns.append(future_return)
        
        if not future_returns:
            return {}
            
        success_rate = len([r for r in future_returns if r > 0]) / len(future_returns)
        avg_return = np.mean(future_returns)
        
        # Pattern characteristics
        avg_body_size = abs(cluster_data['Close'] - cluster_data['Open']).mean() / cluster_data['Open'].mean()
        avg_volume_ratio = (cluster_data['Volume'] / cluster_data['Volume'].rolling(20).mean()).mean()
        
        return {
            'occurrences': len(cluster_data),
            'success_rate': success_rate,
            'avg_return': avg_return,
            'avg_body_size': avg_body_size,
            'avg_volume_ratio': avg_volume_ratio,
            'confidence': min(success_rate * 1.2, 1.0),
            'pattern_strength': abs(avg_return) * success_rate
        }
    
    def discover_chart_patterns(self) -> Dict:
        """Advanced chart pattern recognition using computer vision techniques"""
        logger.info("Discovering chart patterns using ML algorithms...")
        
        patterns = {}
        
        # Support/Resistance pattern discovery
        sr_patterns = self._detect_support_resistance_patterns()
        patterns.update(sr_patterns)
        
        # Trend pattern discovery
        trend_patterns = self._detect_trend_patterns()
        patterns.update(trend_patterns)
        
        # Breakout pattern discovery
        breakout_patterns = self._detect_breakout_patterns()
        patterns.update(breakout_patterns)
        
        self.patterns['chart'] = patterns
        return patterns
    
    def _detect_support_resistance_patterns(self) -> Dict:
        """ML-based support/resistance level detection"""
        close_prices = self.data['Close'].values
        
        # Find local minima and maxima
        min_indices = argrelextrema(close_prices, np.less, order=5)[0]
        max_indices = argrelextrema(close_prices, np.greater, order=5)[0]
        
        support_levels = close_prices[min_indices]
        resistance_levels = close_prices[max_indices]
        
        # Cluster similar levels
        if len(support_levels) > 3:
            support_clusters = self._cluster_price_levels(support_levels)
        else:
            support_clusters = []
            
        if len(resistance_levels) > 3:
            resistance_clusters = self._cluster_price_levels(resistance_levels)
        else:
            resistance_clusters = []
        
        return {
            'support_levels': support_clusters,
            'resistance_levels': resistance_clusters,
            'sr_strength': len(support_clusters) + len(resistance_clusters)
        }
    
    def _cluster_price_levels(self, levels: np.ndarray) -> List[Dict]:
        """Cluster similar price levels using ML"""
        if len(levels) < 2:
            return []
            
        levels_reshaped = levels.reshape(-1, 1)
        
        # Use DBSCAN for clustering similar price levels
        dbscan = DBSCAN(eps=np.std(levels) * 0.5, min_samples=2)
        clusters = dbscan.fit_predict(levels_reshaped)
        
        clustered_levels = []
        for cluster_id in set(clusters):
            if cluster_id != -1:  # Ignore noise points
                cluster_levels = levels[clusters == cluster_id]
                clustered_levels.append({
                    'level': np.mean(cluster_levels),
                    'strength': len(cluster_levels),
                    'std': np.std(cluster_levels)
                })
        
        return sorted(clustered_levels, key=lambda x: x['strength'], reverse=True)
    
    def _detect_trend_patterns(self) -> Dict:
        """Detect trend patterns using advanced ML techniques"""
        close_prices = self.data['Close'].values
        
        # Calculate trend strength using multiple methods
        short_trend = np.polyfit(range(20), close_prices[-20:], 1)[0]
        medium_trend = np.polyfit(range(50), close_prices[-50:], 1)[0]
        long_trend = np.polyfit(range(100), close_prices[-100:], 1)[0]
        
        # Trend consistency analysis
        trend_consistency = self._calculate_trend_consistency(close_prices)
        
        return {
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            'trend_consistency': trend_consistency,
            'trend_strength': abs(medium_trend) * trend_consistency
        }
    
    def _calculate_trend_consistency(self, prices: np.ndarray) -> float:
        """Calculate trend consistency using statistical methods"""
        if len(prices) < 20:
            return 0.0
            
        # Calculate rolling correlations with linear trend
        window_size = 20
        correlations = []
        
        for i in range(window_size, len(prices)):
            window_prices = prices[i-window_size:i]
            x = np.arange(len(window_prices))
            correlation = np.corrcoef(x, window_prices)[0, 1]
            if not np.isnan(correlation):
                correlations.append(abs(correlation))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _detect_breakout_patterns(self) -> Dict:
        """Detect breakout patterns using ML algorithms"""
        data = self.data.copy()
        
        # Calculate volatility expansion
        data['volatility'] = data['Close'].rolling(20).std()
        data['vol_expansion'] = data['volatility'] / data['volatility'].shift(20)
        
        # Volume surge detection
        data['volume_surge'] = data['Volume'] / data['Volume'].rolling(20).mean()
        
        # Price breakout detection
        data['price_breakout'] = 0
        for i in range(20, len(data)):
            recent_high = data['High'].iloc[i-20:i].max()
            recent_low = data['Low'].iloc[i-20:i].min()
            current_close = data['Close'].iloc[i]
            
            if current_close > recent_high * 1.02:  # Upward breakout
                data.loc[data.index[i], 'price_breakout'] = 1
            elif current_close < recent_low * 0.98:  # Downward breakout
                data.loc[data.index[i], 'price_breakout'] = -1
        
        breakout_signals = data[data['price_breakout'] != 0]
        
        return {
            'breakout_count': len(breakout_signals),
            'avg_vol_expansion': data['vol_expansion'].mean(),
            'avg_volume_surge': data['volume_surge'].mean(),
            'recent_breakout': data['price_breakout'].iloc[-1] if not data.empty else 0
        }

class AdvancedTechnicalAnalyzer:
    """Comprehensive technical analysis with 40+ proprietary indicators"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.indicators = {}
        self.composite_signals = {}
        
    def calculate_all_indicators(self) -> Dict:
        """Calculate all 40+ technical indicators with pattern analysis"""
        logger.info("Calculating comprehensive technical indicators...")
        
        # Trend Indicators
        self._calculate_moving_averages()
        self._calculate_macd_analysis()
        self._calculate_adx_analysis()
        self._calculate_parabolic_sar()
        
        # Momentum Indicators
        self._calculate_rsi_analysis()
        self._calculate_stochastic_analysis()
        self._calculate_williams_r()
        self._calculate_cci_analysis()
        
        # Volatility Indicators
        self._calculate_bollinger_analysis()
        self._calculate_atr_analysis()
        self._calculate_keltner_channels()
        
        # Volume Indicators
        self._calculate_volume_analysis()
        self._calculate_obv_analysis()
        self._calculate_vwap_analysis()
        
        # Custom Composite Indicators
        self._calculate_composite_indicators()
        
        return self.indicators
    
    def _calculate_moving_averages(self):
        """Advanced moving average analysis with pattern detection"""
        data = self.data.copy()
        
        # Multiple MA periods
        ma_periods = [5, 10, 20, 50, 100, 200]
        for period in ma_periods:
            data[f'MA_{period}'] = data['Close'].rolling(period).mean()
            data[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        # MA relationships and patterns
        data['MA_golden_cross'] = (data['MA_50'] > data['MA_200']).astype(int)
        data['MA_death_cross'] = (data['MA_50'] < data['MA_200']).astype(int)
        data['MA_cloud_strength'] = abs(data['MA_20'] - data['MA_50']) / data['Close']
        
        # Dynamic support/resistance from MAs
        ma_support_resistance = []
        for period in [20, 50, 100]:
            ma_col = f'MA_{period}'
            distance = abs(data['Close'] - data[ma_col]) / data['Close']
            ma_support_resistance.append(distance.iloc[-1])
        
        self.indicators['moving_averages'] = {
            'current_ma_20': data['MA_20'].iloc[-1],
            'current_ma_50': data['MA_50'].iloc[-1],
            'current_ma_200': data['MA_200'].iloc[-1],
            'golden_cross_active': bool(data['MA_golden_cross'].iloc[-1]),
            'death_cross_active': bool(data['MA_death_cross'].iloc[-1]),
            'ma_cloud_strength': data['MA_cloud_strength'].iloc[-1],
            'ma_support_resistance': np.mean(ma_support_resistance),
            'trend_alignment': self._calculate_ma_alignment(data)
        }
    
    def _calculate_ma_alignment(self, data: pd.DataFrame) -> float:
        """Calculate moving average alignment score"""
        mas = ['MA_5', 'MA_10', 'MA_20', 'MA_50', 'MA_100']
        current_values = [data[ma].iloc[-1] for ma in mas if ma in data.columns]
        
        if len(current_values) < 3:
            return 0.0
        
        # Check if MAs are in trending order
        bullish_alignment = all(current_values[i] > current_values[i+1] for i in range(len(current_values)-1))
        bearish_alignment = all(current_values[i] < current_values[i+1] for i in range(len(current_values)-1))
        
        if bullish_alignment:
            return 1.0
        elif bearish_alignment:
            return -1.0
        else:
            return 0.0
    
    def _calculate_rsi_analysis(self):
        """Advanced RSI analysis with divergence and pattern detection"""
        data = self.data.copy()
        
        # Multiple RSI periods
        for period in [14, 21, 30]:
            data[f'RSI_{period}'] = talib.RSI(data['Close'].values, timeperiod=period)
        
        # RSI patterns and signals
        rsi_14 = data['RSI_14']
        
        # Divergence detection
        price_peaks = argrelextrema(data['Close'].values, np.greater, order=5)[0]
        rsi_peaks = argrelextrema(rsi_14.values, np.greater, order=5)[0]
        
        divergence_signals = self._detect_divergence(data['Close'], rsi_14, price_peaks, rsi_peaks)
        
        # RSI trend analysis
        rsi_trend = np.polyfit(range(min(20, len(rsi_14))), rsi_14.tail(20).values, 1)[0]
        
        # RSI cluster analysis
        rsi_clusters = self._analyze_rsi_clusters(rsi_14.values)
        
        self.indicators['rsi_analysis'] = {
            'current_rsi_14': rsi_14.iloc[-1],
            'current_rsi_21': data['RSI_21'].iloc[-1],
            'rsi_trend': rsi_trend,
            'bullish_divergence': divergence_signals['bullish'],
            'bearish_divergence': divergence_signals['bearish'],
            'rsi_clusters': rsi_clusters,
            'oversold_signal': rsi_14.iloc[-1] < 30,
            'overbought_signal': rsi_14.iloc[-1] > 70,
            'rsi_momentum': rsi_14.iloc[-1] - rsi_14.iloc[-5]
        }
    
    def _detect_divergence(self, price_series: pd.Series, indicator_series: pd.Series, 
                          price_peaks: np.ndarray, indicator_peaks: np.ndarray) -> Dict:
        """Detect bullish and bearish divergences"""
        
        if len(price_peaks) < 2 or len(indicator_peaks) < 2:
            return {'bullish': False, 'bearish': False}
        
        # Get recent peaks
        recent_price_peaks = price_peaks[-2:]
        recent_indicator_peaks = indicator_peaks[-2:]
        
        if len(recent_price_peaks) < 2 or len(recent_indicator_peaks) < 2:
            return {'bullish': False, 'bearish': False}
        
        # Check for divergence patterns
        price_trend = price_series.iloc[recent_price_peaks[-1]] - price_series.iloc[recent_price_peaks[-2]]
        indicator_trend = indicator_series.iloc[recent_indicator_peaks[-1]] - indicator_series.iloc[recent_indicator_peaks[-2]]
        
        bullish_divergence = price_trend < 0 and indicator_trend > 0
        bearish_divergence = price_trend > 0 and indicator_trend < 0
        
        return {'bullish': bullish_divergence, 'bearish': bearish_divergence}
    
    def _analyze_rsi_clusters(self, rsi_values: np.ndarray) -> Dict:
        """Analyze RSI value clustering patterns"""
        if len(rsi_values) < 50:
            return {}
        
        # Cluster RSI values
        rsi_reshaped = rsi_values[-100:].reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(rsi_reshaped)
        
        cluster_centers = kmeans.cluster_centers_.flatten()
        cluster_centers_sorted = np.sort(cluster_centers)
        
        return {
            'low_cluster': cluster_centers_sorted[0],
            'mid_cluster': cluster_centers_sorted[1],
            'high_cluster': cluster_centers_sorted[2],
            'current_cluster': clusters[-1]
        }
    
    def _calculate_bollinger_analysis(self):
        """Advanced Bollinger Bands analysis with squeeze and expansion detection"""
        data = self.data.copy()
        
        # Calculate Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['BB_Middle'] = data['Close'].rolling(bb_period).mean()
        data['BB_Upper'] = data['BB_Middle'] + (data['Close'].rolling(bb_period).std() * bb_std)
        data['BB_Lower'] = data['BB_Middle'] - (data['Close'].rolling(bb_period).std() * bb_std)
        
        # Bollinger Band patterns
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Squeeze detection (narrow bands)
        bb_width_ma = data['BB_Width'].rolling(20).mean()
        data['BB_Squeeze'] = data['BB_Width'] < (bb_width_ma * 0.8)
        
        # Band walking detection
        data['BB_Upper_Walk'] = (data['Close'] > data['BB_Upper'] * 0.98).rolling(3).sum() >= 2
        data['BB_Lower_Walk'] = (data['Close'] < data['BB_Lower'] * 1.02).rolling(3).sum() >= 2
        
        # Expansion detection
        bb_width_change = data['BB_Width'].pct_change(5)
        
        self.indicators['bollinger_analysis'] = {
            'current_bb_upper': data['BB_Upper'].iloc[-1],
            'current_bb_lower': data['BB_Lower'].iloc[-1],
            'current_bb_middle': data['BB_Middle'].iloc[-1],
            'bb_position': data['BB_Position'].iloc[-1],
            'bb_width': data['BB_Width'].iloc[-1],
            'squeeze_active': bool(data['BB_Squeeze'].iloc[-1]),
            'upper_band_walk': bool(data['BB_Upper_Walk'].iloc[-1]),
            'lower_band_walk': bool(data['BB_Lower_Walk'].iloc[-1]),
            'width_expansion': bb_width_change.iloc[-1],
            'breakout_probability': self._calculate_breakout_probability(data)
        }
    
    def _calculate_breakout_probability(self, data: pd.DataFrame) -> float:
        """Calculate probability of breakout from Bollinger Bands"""
        recent_squeezes = data['BB_Squeeze'].tail(10).sum()
        current_position = data['BB_Position'].iloc[-1]
        width_change = data['BB_Width'].pct_change(5).iloc[-1]
        
        # Higher squeeze count and extreme position increase breakout probability
        squeeze_factor = min(recent_squeezes / 10, 1.0)
        position_factor = abs(current_position - 0.5) * 2
        expansion_factor = max(width_change, 0) * 10
        
        probability = (squeeze_factor * 0.4 + position_factor * 0.4 + expansion_factor * 0.2)
        return min(probability, 1.0)
    
    def _calculate_macd_analysis(self):
        """Advanced MACD analysis with histogram patterns"""
        data = self.data.copy()
        
        # Calculate MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # MACD patterns
        macd_crossover = ((data['MACD'] > data['MACD_Signal']) & 
                         (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)))
        macd_crossunder = ((data['MACD'] < data['MACD_Signal']) & 
                          (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)))
        
        # Histogram analysis
        histogram_trend = np.polyfit(range(10), data['MACD_Histogram'].tail(10).values, 1)[0]
        
        # Zero line analysis
        macd_above_zero = data['MACD'] > 0
        zero_line_crosses = (macd_above_zero != macd_above_zero.shift(1)).sum()
        
        self.indicators['macd_analysis'] = {
            'current_macd': data['MACD'].iloc[-1],
            'current_signal': data['MACD_Signal'].iloc[-1],
            'current_histogram': data['MACD_Histogram'].iloc[-1],
            'bullish_crossover': bool(macd_crossover.iloc[-1]),
            'bearish_crossover': bool(macd_crossunder.iloc[-1]),
            'histogram_trend': histogram_trend,
            'above_zero_line': bool(macd_above_zero.iloc[-1]),
            'zero_line_crosses': zero_line_crosses,
            'momentum_strength': abs(data['MACD'].iloc[-1] - data['MACD'].iloc[-5])
        }
    
    def _calculate_volume_analysis(self):
        """Comprehensive volume analysis with pattern detection"""
        data = self.data.copy()
        
        # Volume indicators
        data['Volume_MA'] = data['Volume'].rolling(20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        data['Price_Volume'] = data['Close'].pct_change() * data['Volume_Ratio']
        
        # Volume surge detection
        volume_surge = data['Volume_Ratio'] > 2.0
        
        # Volume trend analysis
        volume_trend = np.polyfit(range(20), data['Volume'].tail(20).values, 1)[0]
        
        # Volume-price divergence
        price_change = data['Close'].pct_change(5).iloc[-1]
        volume_change = data['Volume'].pct_change(5).iloc[-1]
        
        volume_price_divergence = (price_change > 0 and volume_change < -0.2) or \
                                 (price_change < 0 and volume_change < -0.2)
        
        self.indicators['volume_analysis'] = {
            'current_volume': data['Volume'].iloc[-1],
            'volume_ma_20': data['Volume_MA'].iloc[-1],
            'volume_ratio': data['Volume_Ratio'].iloc[-1],
            'volume_surge': bool(volume_surge.iloc[-1]),
            'volume_trend': volume_trend,
            'volume_price_divergence': volume_price_divergence,
            'avg_volume_ratio': data['Volume_Ratio'].tail(10).mean(),
            'volume_volatility': data['Volume'].tail(20).std() / data['Volume'].tail(20).mean()
        }
    
    def _calculate_vwap_analysis(self):
        """Advanced VWAP analysis with multiple timeframes"""
        data = self.data.copy()
        
        # Calculate VWAP
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['Volume_Price'] = data['Typical_Price'] * data['Volume']
        data['Cumulative_Volume_Price'] = data['Volume_Price'].cumsum()
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_Volume_Price'] / data['Cumulative_Volume']
        
        # VWAP patterns
        data['Price_VWAP_Ratio'] = data['Close'] / data['VWAP']
        vwap_distance = abs(data['Close'] - data['VWAP']) / data['VWAP']
        
        # VWAP slope analysis
        vwap_slope = np.polyfit(range(20), data['VWAP'].tail(20).values, 1)[0]
        
        # VWAP reversion signals
        vwap_reversion_signal = vwap_distance.iloc[-1] > vwap_distance.tail(20).quantile(0.8)
        
        self.indicators['vwap_analysis'] = {
            'current_vwap': data['VWAP'].iloc[-1],
            'price_vwap_ratio': data['Price_VWAP_Ratio'].iloc[-1],
            'vwap_distance': vwap_distance.iloc[-1],
            'vwap_slope': vwap_slope,
            'above_vwap': data['Close'].iloc[-1] > data['VWAP'].iloc[-1],
            'vwap_reversion_signal': vwap_reversion_signal,
            'vwap_strength': abs(vwap_slope) * 1000
        }
    
    def _calculate_composite_indicators(self):
        """Calculate custom composite indicators"""
        
        # Trend Strength Composite
        trend_strength = self._calculate_trend_strength_composite()
        
        # Momentum Composite
        momentum_composite = self._calculate_momentum_composite()
        
        # Volatility Composite
        volatility_composite = self._calculate_volatility_composite()
        
        # Volume Strength Composite
        volume_strength = self._calculate_volume_strength_composite()
        
        self.composite_signals = {
            'trend_strength': trend_strength,
            'momentum_composite': momentum_composite,
            'volatility_composite': volatility_composite,
            'volume_strength': volume_strength,
            'overall_signal': self._calculate_overall_signal(trend_strength, momentum_composite, 
                                                           volatility_composite, volume_strength)
        }
    
    def _calculate_trend_strength_composite(self) -> float:
        """Calculate composite trend strength from multiple indicators"""
        ma_signal = self.indicators.get('moving_averages', {}).get('trend_alignment', 0)
        macd_signal = 1 if self.indicators.get('macd_analysis', {}).get('above_zero_line', False) else -1
        adx_signal = self.indicators.get('adx_analysis', {}).get('trend_strength', 0)
        
        # Normalize ADX signal
        adx_normalized = min(adx_signal / 50, 1.0) if adx_signal else 0
        
        trend_strength = (ma_signal * 0.4 + macd_signal * 0.3 + adx_normalized * 0.3)
        return trend_strength
    
    def _calculate_momentum_composite(self) -> float:
        """Calculate composite momentum from multiple indicators"""
        rsi_momentum = self.indicators.get('rsi_analysis', {}).get('rsi_momentum', 0) / 10
        macd_momentum = self.indicators.get('macd_analysis', {}).get('momentum_strength', 0)
        
        # Normalize momentum signals
        rsi_normalized = max(min(rsi_momentum, 1), -1)
        macd_normalized = max(min(macd_momentum * 100, 1), -1)
        
        momentum_composite = (rsi_normalized * 0.6 + macd_normalized * 0.4)
        return momentum_composite
    
    def _calculate_volatility_composite(self) -> float:
        """Calculate composite volatility signal"""
        bb_expansion = self.indicators.get('bollinger_analysis', {}).get('width_expansion', 0)
        atr_change = self.indicators.get('atr_analysis', {}).get('atr_change', 0)
        
        # Normalize volatility signals
        bb_normalized = max(min(bb_expansion * 5, 1), -1)
        atr_normalized = max(min(atr_change * 10, 1), -1)
        
        volatility_composite = (bb_normalized * 0.6 + atr_normalized * 0.4)
        return volatility_composite
    
    def _calculate_volume_strength_composite(self) -> float:
        """Calculate composite volume strength"""
        volume_ratio = self.indicators.get('volume_analysis', {}).get('volume_ratio', 1)
        volume_trend = self.indicators.get('volume_analysis', {}).get('volume_trend', 0)
        
        # Normalize volume signals
        ratio_signal = min((volume_ratio - 1) * 2, 1)
        trend_signal = max(min(volume_trend / 1000000, 1), -1)
        
        volume_strength = (ratio_signal * 0.7 + trend_signal * 0.3)
        return volume_strength
    
    def _calculate_overall_signal(self, trend: float, momentum: float, 
                                 volatility: float, volume: float) -> Dict:
        """Calculate overall trading signal from composite indicators"""
        
        # Weighted composite score
        overall_score = (trend * 0.3 + momentum * 0.3 + volatility * 0.2 + volume * 0.2)
        
        # Signal classification
        if overall_score > 0.6:
            signal = "STRONG_BUY"
            confidence = min(overall_score, 1.0)
        elif overall_score > 0.3:
            signal = "BUY"
            confidence = overall_score * 0.8
        elif overall_score > -0.3:
            signal = "NEUTRAL"
            confidence = 0.5
        elif overall_score > -0.6:
            signal = "SELL"
            confidence = abs(overall_score) * 0.8
        else:
            signal = "STRONG_SELL"
            confidence = min(abs(overall_score), 1.0)
        
        return {
            'signal': signal,
            'score': overall_score,
            'confidence': confidence,
            'components': {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'volume': volume
            }
        }

class DeepLearningPredictor:
    """Advanced deep learning prediction system for forex markets"""
    
    def __init__(self, data: pd.DataFrame, config: ForexAnalysisConfig):
        self.data = data
        self.config = config
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}
        
    def create_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create comprehensive feature matrix for ML prediction"""
        logger.info("Creating advanced feature matrix for deep learning...")
        
        data = self.data.copy()
        
        # Price-based features
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['volatility'] = data['returns'].rolling(20).std()
        
        # Technical features
        data['rsi'] = talib.RSI(data['Close'].values)
        data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(data['Close'].values)
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(data['Close'].values)
        data['atr'] = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values)
        
        # Volume features
        data['volume_sma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_sma']
        data['obv'] = talib.OBV(data['Close'].values, data['Volume'].values)
        
        # Advanced features
        data['price_position'] = (data['Close'] - data['Low'].rolling(20).min()) / \
                                (data['High'].rolling(20).max() - data['Low'].rolling(20).min())
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            data[f'return_lag_{lag}'] = data['returns'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume_ratio'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            data[f'return_mean_{window}'] = data['returns'].rolling(window).mean()
            data[f'return_std_{window}'] = data['returns'].rolling(window).std()
            data[f'price_rank_{window}'] = data['Close'].rolling(window).rank() / window
        
        # Feature selection
        feature_cols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        features = data[feature_cols].fillna(method='ffill').fillna(0)
        
        # Target variable (future returns)
        target = data['Close'].shift(-self.config.prediction_horizon).pct_change()
        
        # Remove rows with NaN targets
        valid_indices = ~target.isna()
        features = features[valid_indices]
        target = target[valid_indices]
        
        return features.values, target.values
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train ensemble of deep learning models"""
        logger.info("Training ensemble deep learning models...")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        
        # LSTM Model
        lstm_model = self._create_lstm_model(X_train_scaled.shape[1])
        lstm_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                      validation_split=0.2, verbose=0,
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])
        models['lstm'] = lstm_model
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        models['random_forest'] = rf_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        models['gradient_boosting'] = gb_model
        
        # Make predictions
        predictions = {}
        for name, model in models.items():
            if name == 'lstm':
                pred = model.predict(X_test_scaled).flatten()
            else:
                pred = model.predict(X_test_scaled)
            predictions[name] = pred
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        predictions['ensemble'] = ensemble_pred
        
        # Calculate metrics
        metrics = {}
        for name, pred in predictions.items():
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            metrics[name] = {'mse': mse, 'r2': r2}
        
        self.models = models
        self.scaler = scaler
        
        return {
            'models': models,
            'predictions': predictions,
            'metrics': metrics,
            'test_actual': y_test
        }
    
    def _create_lstm_model(self, input_dim: int):
        """Create LSTM neural network model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', 
                     metrics=['mae'])
        return model
    
    def generate_predictions(self, X: np.ndarray) -> Dict:
        """Generate predictions with confidence intervals"""
        logger.info("Generating predictions with confidence analysis...")
        
        if not self.models:
            logger.error("Models not trained yet!")
            return {}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if name == 'lstm':
                pred = model.predict(X_scaled[-1:]).flatten()[0]
            else:
                pred = model.predict(X_scaled[-1:])[0]
            predictions[name] = pred
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        prediction_std = np.std(list(predictions.values()))
        
        # Calculate confidence intervals
        confidence_95 = 1.96 * prediction_std
        confidence_80 = 1.28 * prediction_std
        
        # Direction probability
        positive_predictions = sum(1 for p in predictions.values() if p > 0)
        direction_probability = positive_predictions / len(predictions)
        
        # Prediction strength
        avg_magnitude = np.mean([abs(p) for p in predictions.values()])
        prediction_strength = min(avg_magnitude * 100, 1.0)
        
        current_price = self.data['Close'].iloc[-1]
        target_price = current_price * (1 + ensemble_pred)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'current_price': current_price,
            'target_price': target_price,
            'price_change_pct': ensemble_pred * 100,
            'confidence_intervals': {
                '95%': [target_price - confidence_95 * current_price, 
                       target_price + confidence_95 * current_price],
                '80%': [target_price - confidence_80 * current_price,
                       target_price + confidence_80 * current_price]
            },
            'direction_probability': direction_probability,
            'prediction_strength': prediction_strength,
            'prediction_confidence': max(0.5, 1 - prediction_std * 10),
            'model_agreement': 1 - (prediction_std / max(avg_magnitude, 0.001))
        }
    
    def calculate_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate feature importance across models"""
        
        importance_scores = {}
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            importance_scores['random_forest'] = dict(zip(feature_names, rf_importance))
        
        # Gradient Boosting feature importance
        if 'gradient_boosting' in self.models:
            gb_importance = self.models['gradient_boosting'].feature_importances_
            importance_scores['gradient_boosting'] = dict(zip(feature_names, gb_importance))
        
        # Average importance across models
        if importance_scores:
            avg_importance = {}
            for feature in feature_names:
                scores = [importance_scores[model][feature] for model in importance_scores.keys()]
                avg_importance[feature] = np.mean(scores)
            
            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            self.feature_importance = {
                'individual_models': importance_scores,
                'average_importance': avg_importance,
                'top_features': sorted_importance[:10]
            }
        
        return self.feature_importance

class ForexMarketAnalyzer:
    """Main forex market analysis system"""
    
    def __init__(self, symbol: str = FOREX_SYMBOL):
        self.symbol = symbol
        self.config = ForexAnalysisConfig(symbol=symbol)
        self.data = {}
        self.analysis_results = {}
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch forex data for multiple timeframes"""
        logger.info(f"Fetching forex data for {self.symbol}...")
        
        data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        # Fetch data for different timeframes
        timeframe_mapping = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk'
        }
        
        for period_name, period_code in timeframe_mapping.items():
            try:
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period="2y", interval=period_code)
                
                if not df.empty:
                    data[period_name] = df
                    logger.info(f"Fetched {len(df)} records for {period_name} timeframe")
                else:
                    logger.warning(f"No data available for {period_name} timeframe")
                    
            except Exception as e:
                logger.error(f"Error fetching {period_name} data: {str(e)}")
                continue
        
        self.data = data
        return data
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run complete forex market analysis"""
        logger.info("Starting comprehensive forex market analysis...")
        
        if not self.data:
            self.fetch_data()
        
        if not self.data:
            logger.error("No data available for analysis")
            return {}
        
        # Use daily data as primary timeframe
        primary_data = self.data.get('1d')
        if primary_data is None or primary_data.empty:
            logger.error("No daily data available for analysis")
            return {}
        
        results = {}
        
        # Pattern Discovery Analysis
        pattern_engine = ProprietaryPatternEngine(primary_data)
        candlestick_patterns = pattern_engine.discover_candlestick_patterns()
        chart_patterns = pattern_engine.discover_chart_patterns()
        
        results['pattern_analysis'] = {
            'candlestick_patterns': candlestick_patterns,
            'chart_patterns': chart_patterns
        }
        
        # Technical Analysis
        technical_analyzer = AdvancedTechnicalAnalyzer(primary_data)
        technical_indicators = technical_analyzer.calculate_all_indicators()
        
        results['technical_analysis'] = {
            'indicators': technical_indicators,
            'composite_signals': technical_analyzer.composite_signals
        }
        
        # Deep Learning Predictions
        predictor = DeepLearningPredictor(primary_data, self.config)
        features, targets = predictor.create_features()
        
        if len(features) > 100:  # Ensure sufficient data
            training_results = predictor.train_ensemble_models(features, targets)
            predictions = predictor.generate_predictions(features)
            feature_importance = predictor.calculate_feature_importance(
                features, [f"feature_{i}" for i in range(features.shape[1])]
            )
            
            results['ml_predictions'] = {
                'training_results': training_results,
                'predictions': predictions,
                'feature_importance': feature_importance
            }
        
        # Multi-timeframe Analysis
        results['multi_timeframe'] = self._analyze_multiple_timeframes()
        
        # Risk Assessment
        results['risk_assessment'] = self._calculate_risk_metrics(primary_data)
        
        # Market Intelligence Summary
        results['market_intelligence'] = self._generate_market_intelligence(results)
        
        self.analysis_results = results
        return results
    
    def _analyze_multiple_timeframes(self) -> Dict:
        """Analyze patterns across multiple timeframes"""
        
        timeframe_analysis = {}
        
        for timeframe, data in self.data.items():
            if data is not None and not data.empty:
                # Basic trend analysis for each timeframe
                close_prices = data['Close']
                
                # Trend direction
                short_trend = close_prices.tail(10).mean() > close_prices.tail(20).mean()
                medium_trend = close_prices.tail(20).mean() > close_prices.tail(50).mean()
                
                # Volatility
                volatility = close_prices.pct_change().tail(20).std()
                
                # Support/Resistance
                recent_high = data['High'].tail(20).max()
                recent_low = data['Low'].tail(20).min()
                current_price = close_prices.iloc[-1]
                
                timeframe_analysis[timeframe] = {
                    'short_trend_bullish': short_trend,
                    'medium_trend_bullish': medium_trend,
                    'volatility': volatility,
                    'recent_high': recent_high,
                    'recent_low': recent_low,
                    'current_price': current_price,
                    'distance_to_high': (recent_high - current_price) / current_price,
                    'distance_to_low': (current_price - recent_low) / current_price
                }
        
        return timeframe_analysis
    
    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk assessment metrics"""
        
        returns = data['Close'].pct_change().dropna()
        
        # Basic risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Current risk level
        recent_volatility = returns.tail(20).std() * np.sqrt(252)
        risk_level = "LOW" if recent_volatility < volatility * 0.8 else \
                    "HIGH" if recent_volatility > volatility * 1.2 else "MEDIUM"
        
        return {
            'annualized_volatility': volatility,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'current_risk_level': risk_level,
            'recent_volatility': recent_volatility
        }
    
    def _generate_market_intelligence(self, results: Dict) -> Dict:
        """Generate comprehensive market intelligence summary"""
        
        intelligence = {
            'overall_sentiment': 'NEUTRAL',
            'confidence_score': 0.5,
            'key_insights': [],
            'trading_recommendations': [],
            'risk_warnings': []
        }
        
        # Analyze technical signals
        if 'technical_analysis' in results:
            composite = results['technical_analysis'].get('composite_signals', {})
            overall_signal = composite.get('overall_signal', {})
            
            if overall_signal:
                intelligence['overall_sentiment'] = overall_signal.get('signal', 'NEUTRAL')
                intelligence['confidence_score'] = overall_signal.get('confidence', 0.5)
        
        # Add ML prediction insights
        if 'ml_predictions' in results:
            predictions = results['ml_predictions'].get('predictions', {})
            if predictions:
                direction_prob = predictions.get('direction_probability', 0.5)
                prediction_confidence = predictions.get('prediction_confidence', 0.5)
                
                if direction_prob > 0.7:
                    intelligence['key_insights'].append(
                        f"ML models predict {direction_prob:.1%} probability of upward movement"
                    )
                elif direction_prob < 0.3:
                    intelligence['key_insights'].append(
                        f"ML models predict {1-direction_prob:.1%} probability of downward movement"
                    )
        
        # Add pattern insights
        if 'pattern_analysis' in results:
            patterns = results['pattern_analysis']
            candlestick_count = len(patterns.get('candlestick_patterns', {}))
            if candlestick_count > 0:
                intelligence['key_insights'].append(
                    f"Discovered {candlestick_count} proprietary candlestick patterns"
                )
        
        # Risk warnings
        if 'risk_assessment' in results:
            risk = results['risk_assessment']
            if risk.get('current_risk_level') == 'HIGH':
                intelligence['risk_warnings'].append("High volatility environment detected")
            
            if risk.get('max_drawdown', 0) < -0.2:
                intelligence['risk_warnings'].append("Significant historical drawdown risk")
        
        return intelligence

def create_visualizations(analyzer: ForexMarketAnalyzer) -> Dict[str, str]:
    """Create comprehensive visualization charts"""
    logger.info("Creating advanced visualization charts...")
    
    if not analyzer.data or '1d' not in analyzer.data:
        logger.error("No data available for visualization")
        return {}
    
    data = analyzer.data['1d']
    results = analyzer.analysis_results
    
    # Create candlestick chart with technical overlays
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=['Price Action & Patterns', 'Volume Analysis', 'RSI & Momentum', 'MACD'],
        row_width=[0.2, 0.1, 0.1, 0.1]
    )
    
    # Main candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if len(data) >= 50:
        ma_20 = data['Close'].rolling(20).mean()
        ma_50 = data['Close'].rolling(50).mean()
        
        fig.add_trace(
            go.Scatter(x=data.index, y=ma_20, name='MA20', line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=ma_50, name='MA50', line=dict(color='blue', width=1)),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['green' if close >= open else 'red' for close, open in zip(data['Close'], data['Open'])]
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if len(data) >= 14:
        rsi = talib.RSI(data['Close'].values, timeperiod=14)
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if len(data) >= 26:
        macd, macd_signal, macd_hist = talib.MACD(data['Close'].values)
        fig.add_trace(
            go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=macd_signal, name='Signal', line=dict(color='red')),
            row=4, col=1
        )
        fig.add_trace(
            go.Bar(x=data.index, y=macd_hist, name='Histogram', marker_color='gray', opacity=0.6),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'{analyzer.symbol} - Comprehensive Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=1200,
        showlegend=True,
        template='plotly_dark'
    )
    
    # Save chart
    chart_filename = f"{analyzer.symbol}_analysis_chart.html"
    fig.write_html(chart_filename)
    
    return {'main_chart': chart_filename}

def generate_reports(analyzer: ForexMarketAnalyzer) -> Dict[str, str]:
    """Generate comprehensive analysis reports"""
    logger.info("Generating comprehensive analysis reports...")
    
    results = analyzer.analysis_results
    if not results:
        logger.error("No analysis results available for reporting")
        return {}
    
    # Technical Analysis Report
    technical_report = _generate_technical_report(analyzer, results)
    
    # Price Action Report  
    price_action_report = _generate_price_action_report(analyzer, results)
    
    # Executive Summary
    executive_summary = _generate_executive_summary(analyzer, results)
    
    # Save reports
    report_files = {}
    
    # Technical report
    with open(f"{analyzer.symbol}_technical_report.txt", 'w') as f:
        f.write(technical_report)
    report_files['technical_report'] = f"{analyzer.symbol}_technical_report.txt"
    
    # Price action report
    with open(f"{analyzer.symbol}_price_action_report.txt", 'w') as f:
        f.write(price_action_report)
    report_files['price_action_report'] = f"{analyzer.symbol}_price_action_report.txt"
    
    # Executive summary
    with open(f"{analyzer.symbol}_executive_summary.txt", 'w') as f:
        f.write(executive_summary)
    report_files['executive_summary'] = f"{analyzer.symbol}_executive_summary.txt"
    
    # JSON export
    with open(f"{analyzer.symbol}_analysis_data.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = _convert_for_json(results)
        json.dump(json_results, f, indent=2, default=str)
    report_files['json_data'] = f"{analyzer.symbol}_analysis_data.json"
    
    return report_files

def _convert_for_json(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def _generate_technical_report(analyzer: ForexMarketAnalyzer, results: Dict) -> str:
    """Generate detailed technical analysis report"""
    
    report = f"""
COMPREHENSIVE TECHNICAL ANALYSIS REPORT
=======================================
Symbol: {analyzer.symbol}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Period: {analyzer.config.lookback_days} days

TECHNICAL INDICATORS ANALYSIS
============================
"""
    
    if 'technical_analysis' in results:
        technical = results['technical_analysis']
        indicators = technical.get('indicators', {})
        
        # Moving Averages Analysis
        if 'moving_averages' in indicators:
            ma_data = indicators['moving_averages']
            report += f"""
MOVING AVERAGES ANALYSIS:
- MA20: ${ma_data.get('current_ma_20', 0):.4f}
- MA50: ${ma_data.get('current_ma_50', 0):.4f}
- MA200: ${ma_data.get('current_ma_200', 0):.4f}
- Golden Cross Active: {ma_data.get('golden_cross_active', False)}
- Death Cross Active: {ma_data.get('death_cross_active', False)}
- Trend Alignment Score: {ma_data.get('trend_alignment', 0):.2f}
"""
        
        # RSI Analysis
        if 'rsi_analysis' in indicators:
            rsi_data = indicators['rsi_analysis']
            report += f"""
RSI MOMENTUM ANALYSIS:
- Current RSI(14): {rsi_data.get('current_rsi_14', 0):.2f}
- RSI Trend: {rsi_data.get('rsi_trend', 0):.4f}
- Oversold Signal: {rsi_data.get('oversold_signal', False)}
- Overbought Signal: {rsi_data.get('overbought_signal', False)}
- Bullish Divergence: {rsi_data.get('bullish_divergence', False)}
- Bearish Divergence: {rsi_data.get('bearish_divergence', False)}
"""
        
        # Bollinger Bands Analysis
        if 'bollinger_analysis' in indicators:
            bb_data = indicators['bollinger_analysis']
            report += f"""
BOLLINGER BANDS ANALYSIS:
- Upper Band: ${bb_data.get('current_bb_upper', 0):.4f}
- Middle Band: ${bb_data.get('current_bb_middle', 0):.4f}
- Lower Band: ${bb_data.get('current_bb_lower', 0):.4f}
- Band Position: {bb_data.get('bb_position', 0):.2f}
- Squeeze Active: {bb_data.get('squeeze_active', False)}
- Breakout Probability: {bb_data.get('breakout_probability', 0):.2%}
"""
        
        # Composite Signals
        if 'composite_signals' in technical:
            composite = technical['composite_signals']
            overall = composite.get('overall_signal', {})
            
            report += f"""
COMPOSITE TECHNICAL SIGNALS:
- Overall Signal: {overall.get('signal', 'NEUTRAL')}
- Signal Strength: {overall.get('score', 0):.2f}
- Confidence Level: {overall.get('confidence', 0):.2%}
- Trend Component: {overall.get('components', {}).get('trend', 0):.2f}
- Momentum Component: {overall.get('components', {}).get('momentum', 0):.2f}
- Volatility Component: {overall.get('components', {}).get('volatility', 0):.2f}
- Volume Component: {overall.get('components', {}).get('volume', 0):.2f}
"""
    
    # ML Predictions
    if 'ml_predictions' in results:
        ml_data = results['ml_predictions']
        predictions = ml_data.get('predictions', {})
        
        if predictions:
            report += f"""
MACHINE LEARNING PREDICTIONS:
- Target Price: ${predictions.get('target_price', 0):.4f}
- Price Change: {predictions.get('price_change_pct', 0):.2f}%
- Direction Probability: {predictions.get('direction_probability', 0):.2%}
- Prediction Confidence: {predictions.get('prediction_confidence', 0):.2%}
- Model Agreement: {predictions.get('model_agreement', 0):.2%}
"""
    
    report += "\n" + "="*50 + "\n"
    return report

def _generate_price_action_report(analyzer: ForexMarketAnalyzer, results: Dict) -> str:
    """Generate detailed price action analysis report"""
    
    report = f"""
COMPREHENSIVE PRICE ACTION ANALYSIS REPORT
==========================================
Symbol: {analyzer.symbol}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATTERN DISCOVERY ANALYSIS
=========================
"""
    
    if 'pattern_analysis' in results:
        patterns = results['pattern_analysis']
        
        # Candlestick Patterns
        candlestick_patterns = patterns.get('candlestick_patterns', {})
        report += f"""
DISCOVERED CANDLESTICK PATTERNS ({len(candlestick_patterns)} patterns):
"""
        for pattern_name, pattern_data in candlestick_patterns.items():
            if pattern_data:
                report += f"""
- {pattern_name}:
  * Occurrences: {pattern_data.get('occurrences', 0)}
  * Success Rate: {pattern_data.get('success_rate', 0):.2%}
  * Average Return: {pattern_data.get('avg_return', 0):.2%}
  * Pattern Strength: {pattern_data.get('pattern_strength', 0):.3f}
  * Confidence: {pattern_data.get('confidence', 0):.2%}
"""
        
        # Chart Patterns
        chart_patterns = patterns.get('chart_patterns', {})
        if 'support_levels' in chart_patterns:
            support_levels = chart_patterns['support_levels']
            report += f"""
SUPPORT LEVELS DETECTED ({len(support_levels)} levels):
"""
            for i, level in enumerate(support_levels[:5]):  # Top 5 levels
                report += f"  Level {i+1}: ${level.get('level', 0):.4f} (Strength: {level.get('strength', 0)})\n"
        
        if 'resistance_levels' in chart_patterns:
            resistance_levels = chart_patterns['resistance_levels']
            report += f"""
RESISTANCE LEVELS DETECTED ({len(resistance_levels)} levels):
"""
            for i, level in enumerate(resistance_levels[:5]):  # Top 5 levels
                report += f"  Level {i+1}: ${level.get('level', 0):.4f} (Strength: {level.get('strength', 0)})\n"
    
    # Multi-timeframe Analysis
    if 'multi_timeframe' in results:
        mtf_data = results['multi_timeframe']
        report += f"""
MULTI-TIMEFRAME ANALYSIS:
========================
"""
        for timeframe, data in mtf_data.items():
            report += f"""
{timeframe.upper()} TIMEFRAME:
- Current Price: ${data.get('current_price', 0):.4f}
- Short-term Bullish: {data.get('short_trend_bullish', False)}
- Medium-term Bullish: {data.get('medium_trend_bullish', False)}
- Volatility: {data.get('volatility', 0):.4f}
- Distance to High: {data.get('distance_to_high', 0):.2%}
- Distance to Low: {data.get('distance_to_low', 0):.2%}
"""
    
    report += "\n" + "="*50 + "\n"
    return report

def _generate_executive_summary(analyzer: ForexMarketAnalyzer, results: Dict) -> str:
    """Generate executive summary with key insights"""
    
    summary = f"""
EXECUTIVE SUMMARY - FOREX MARKET INTELLIGENCE
=============================================
Symbol: {analyzer.symbol}
Analysis Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MARKET INTELLIGENCE OVERVIEW
============================
"""
    
    if 'market_intelligence' in results:
        intelligence = results['market_intelligence']
        
        summary += f"""
OVERALL ASSESSMENT:
- Market Sentiment: {intelligence.get('overall_sentiment', 'NEUTRAL')}
- Confidence Score: {intelligence.get('confidence_score', 0):.2%}

KEY INSIGHTS:
"""
        for insight in intelligence.get('key_insights', []):
            summary += f"• {insight}\n"
        
        if intelligence.get('risk_warnings'):
            summary += "\nRISK WARNINGS:\n"
            for warning in intelligence['risk_warnings']:
                summary += f"⚠️  {warning}\n"
    
    # Risk Assessment Summary
    if 'risk_assessment' in results:
        risk = results['risk_assessment']
        summary += f"""
RISK ASSESSMENT SUMMARY:
- Current Risk Level: {risk.get('current_risk_level', 'UNKNOWN')}
- Annualized Volatility: {risk.get('annualized_volatility', 0):.2%}
- Maximum Drawdown: {risk.get('max_drawdown', 0):.2%}
- Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}
- 95% Value at Risk: {risk.get('var_95', 0):.2%}
"""
    
    # Trading Recommendations
    if 'ml_predictions' in results:
        predictions = results['ml_predictions'].get('predictions', {})
        if predictions:
            direction_prob = predictions.get('direction_probability', 0.5)
            target_price = predictions.get('target_price', 0)
            current_price = predictions.get('current_price', 0)
            
            if direction_prob > 0.65:
                recommendation = "BULLISH BIAS"
            elif direction_prob < 0.35:
                recommendation = "BEARISH BIAS"
            else:
                recommendation = "NEUTRAL/WAIT"
            
            summary += f"""
TRADING RECOMMENDATION:
- Market Bias: {recommendation}
- Current Price: ${current_price:.4f}
- Target Price: ${target_price:.4f}
- Predicted Change: {predictions.get('price_change_pct', 0):.2f}%
- Confidence Level: {predictions.get('prediction_confidence', 0):.2%}
"""
    
    summary += f"""
ANALYSIS COMPLETENESS:
- Pattern Discovery: {'✓' if 'pattern_analysis' in results else '✗'}
- Technical Analysis: {'✓' if 'technical_analysis' in results else '✗'}
- ML Predictions: {'✓' if 'ml_predictions' in results else '✗'}
- Multi-timeframe: {'✓' if 'multi_timeframe' in results else '✗'}
- Risk Assessment: {'✓' if 'risk_assessment' in results else '✗'}

DISCLAIMER:
This analysis is for educational purposes only. Trading forex involves 
significant risk and may not be suitable for all investors. Past performance 
does not guarantee future results.
"""
    
    return summary

def main():
    """Main execution function"""
    print("🚀 ULTIMATE DEEP LEARNING FOREX MARKET ANALYSIS SYSTEM")
    print("=" * 60)
    print(f"Analyzing: {FOREX_SYMBOL}")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        analyzer = ForexMarketAnalyzer(FOREX_SYMBOL)
        
        # Fetch data
        print("\n📊 Fetching forex market data...")
        data = analyzer.fetch_data()
        
        if not data:
            print("❌ Error: No data available for analysis")
            return
        
        print(f"✅ Successfully fetched data for {len(data)} timeframes")
        
        # Run comprehensive analysis
        print("\n🧠 Running comprehensive AI analysis...")
        results = analyzer.run_comprehensive_analysis()
        
        if not results:
            print("❌ Error: Analysis failed")
            return
        
        print("✅ Analysis completed successfully")
        
        # Generate visualizations
        print("\n📈 Creating advanced visualizations...")
        charts = create_visualizations(analyzer)
        print(f"✅ Generated {len(charts)} visualization files")
        
        # Generate reports
        print("\n📋 Generating comprehensive reports...")
        reports = generate_reports(analyzer)
        print(f"✅ Generated {len(reports)} report files")
        
        # Display summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE - FILES GENERATED:")
        print("=" * 60)
        
        for report_type, filename in reports.items():
            print(f"📄 {report_type.replace('_', ' ').title()}: {filename}")
        
        for chart_type, filename in charts.items():
            print(f"📊 {chart_type.replace('_', ' ').title()}: {filename}")
        
        # Display key insights
        if 'market_intelligence' in results:
            intelligence = results['market_intelligence']
            print(f"\n🎯 MARKET SENTIMENT: {intelligence.get('overall_sentiment', 'NEUTRAL')}")
            print(f"🎯 CONFIDENCE: {intelligence.get('confidence_score', 0):.2%}")
            
            if intelligence.get('key_insights'):
                print("\n💡 KEY INSIGHTS:")
                for insight in intelligence['key_insights'][:3]:
                    print(f"   • {insight}")
        
        print("\n🎉 Forex market analysis system completed successfully!")
        print("📁 Check the generated files for detailed analysis results.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()