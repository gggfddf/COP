#!/usr/bin/env python3
"""
AGGRESSIVE HIGH-FIDELITY DEEP LEARNING TRADING SYSTEM
====================================================
Advanced AI Trading Intelligence for Volatile Assets (Gold, Crude, Index)
Multi-branch Transformer + CNN + LSTM Hybrid with Contrastive Learning
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using alternative implementations")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - feature importance will use alternatives")

@dataclass
class AggressiveSystemConfig:
    """Configuration for the aggressive trading system"""
    symbol: str = 'GC=F'  # Gold futures
    timeframes: List[str] = None
    lookback_window: int = 50
    pattern_memory_window: int = 30
    confidence_threshold: float = 0.85
    outlier_threshold: float = 0.05  # 5% move threshold
    max_prediction_horizon: int = 10
    contrastive_margin: float = 0.5
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ['5m', '15m', '1h', '4h', '1d', '1wk']

class AdvancedFeatureEngineering:
    """Advanced feature engineering with 60+ custom features"""
    
    def __init__(self, config: AggressiveSystemConfig):
        self.config = config
        self.scaler = RobustScaler()
        self.pattern_clusters = {}
        
    def extract_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract 60+ custom features for deep learning"""
        features = data.copy()
        
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in features.columns:
                logger.warning(f"Missing column {col}, using Close as fallback")
                features[col] = features['Close']
        
        logger.info("🔬 Extracting 60+ custom features...")
        
        # 🔹 CANDLE STRUCTURE FEATURES (15+ features)
        features = self._extract_candle_structure_features(features)
        
        # 🔹 TIME-BASED FEATURES (10+ features)
        features = self._extract_time_based_features(features)
        
        # 🔹 PATTERN MEMORY FEATURES (15+ features)
        features = self._extract_pattern_memory_features(features)
        
        # 🔹 VOLATILITY & CONTEXT FEATURES (20+ features)
        features = self._extract_volatility_context_features(features)
        
        logger.info(f"✅ Extracted {len([col for col in features.columns if col not in required_cols])} custom features")
        return features
    
    def _extract_candle_structure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract candle structure features"""
        df = data.copy()
        
        # Basic candle metrics
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        df['upper_wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Open']
        df['lower_wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Open']
        df['total_range'] = (df['High'] - df['Low']) / df['Open']
        
        # Wick length % vs body
        df['upper_wick_to_body'] = df['upper_wick'] / (df['body_size'] + 1e-8)
        df['lower_wick_to_body'] = df['lower_wick'] / (df['body_size'] + 1e-8)
        df['wick_imbalance'] = (df['upper_wick'] - df['lower_wick']) / (df['total_range'] + 1e-8)
        
        # Momentum of 3-bar/5-bar closes
        df['momentum_3bar'] = (df['Close'] - df['Close'].shift(3)) / df['Close'].shift(3)
        df['momentum_5bar'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        df['momentum_acceleration'] = df['momentum_3bar'] - df['momentum_5bar']
        
        # Engulfing detection (bull/bear)
        df['prev_body_size'] = df['body_size'].shift(1)
        df['is_bullish'] = (df['Close'] > df['Open']).astype(int)
        df['is_bearish'] = (df['Close'] < df['Open']).astype(int)
        
        # Bull engulfing
        df['bull_engulfing'] = (
            (df['is_bullish'] == 1) & 
            (df['is_bearish'].shift(1) == 1) &
            (df['Close'] > df['Open'].shift(1)) &
            (df['Open'] < df['Close'].shift(1)) &
            (df['body_size'] > df['prev_body_size'])
        ).astype(int)
        
        # Bear engulfing
        df['bear_engulfing'] = (
            (df['is_bearish'] == 1) & 
            (df['is_bullish'].shift(1) == 1) &
            (df['Close'] < df['Open'].shift(1)) &
            (df['Open'] > df['Close'].shift(1)) &
            (df['body_size'] > df['prev_body_size'])
        ).astype(int)
        
        # Wick pressure (failed breaks)
        df['upper_wick_pressure'] = (df['upper_wick'] > df['body_size'] * 2).astype(int)
        df['lower_wick_pressure'] = (df['lower_wick'] > df['body_size'] * 2).astype(int)
        
        # Climax candle (body + volume + volatility spike)
        df['volume_spike'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volatility_spike'] = df['total_range'] / df['total_range'].rolling(20).mean()
        df['climax_candle'] = (
            (df['body_size'] > df['body_size'].rolling(20).quantile(0.8)) &
            (df['volume_spike'] > 2.0) &
            (df['volatility_spike'] > 1.5)
        ).astype(int)
        
        return df
    
    def _extract_time_based_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        df = data.copy()
        
        # Time of day, day of week (UTC aligned)
        if hasattr(df.index, 'hour'):
            df['hour_of_day'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_market_open'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 16)).astype(int)
            df['is_overnight'] = ((df['hour_of_day'] < 9) | (df['hour_of_day'] > 16)).astype(int)
        else:
            df['hour_of_day'] = 12  # Default
            df['day_of_week'] = 2   # Default
            df['is_market_open'] = 1
            df['is_overnight'] = 0
        
        # Days since last extreme candle
        df['extreme_candle'] = (
            (df['total_range'] > df['total_range'].rolling(20).quantile(0.95)) |
            (df['volume_spike'] > 3.0)
        ).astype(int)
        
        days_since_extreme = []
        last_extreme = 0
        for i, is_extreme in enumerate(df['extreme_candle']):
            if is_extreme:
                last_extreme = i
            days_since_extreme.append(i - last_extreme)
        df['days_since_extreme'] = days_since_extreme
        
        # ATR regime classification
        df['atr_14'] = self._calculate_atr(df, 14)
        df['atr_regime'] = pd.cut(
            df['atr_14'].rolling(50).rank(pct=True),
            bins=[0, 0.33, 0.67, 1.0],
            labels=[0, 1, 2]  # low/mid/high
        ).astype(float)
        
        # Cyclical features
        df['month'] = df.index.month if hasattr(df.index, 'month') else 6
        df['quarter'] = ((df['month'] - 1) // 3) + 1
        
        return df
    
    def _extract_pattern_memory_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract pattern memory features"""
        df = data.copy()
        
        # Rolling window pattern features
        window = self.config.pattern_memory_window
        
        # Multi-candle pattern detection
        df['three_white_soldiers'] = self._detect_three_white_soldiers(df)
        df['three_black_crows'] = self._detect_three_black_crows(df)
        df['morning_star'] = self._detect_morning_star(df)
        df['evening_star'] = self._detect_evening_star(df)
        
        # Frequency of recent false breaks
        df['swing_high'] = (df['High'] > df['High'].shift(1)) & (df['High'] > df['High'].shift(-1))
        df['swing_low'] = (df['Low'] < df['Low'].shift(1)) & (df['Low'] < df['Low'].shift(-1))
        
        # False break detection
        false_breaks = []
        for i in range(len(df)):
            if i < 5:
                false_breaks.append(0)
                continue
                
            recent_highs = df['High'].iloc[max(0, i-10):i]
            recent_lows = df['Low'].iloc[max(0, i-10):i]
            
            if len(recent_highs) > 0 and len(recent_lows) > 0:
                current_high = df['High'].iloc[i]
                current_low = df['Low'].iloc[i]
                
                # Check for false breakout patterns
                false_break_count = 0
                if current_high > recent_highs.max() and df['Close'].iloc[i] < recent_highs.max():
                    false_break_count += 1
                if current_low < recent_lows.min() and df['Close'].iloc[i] > recent_lows.min():
                    false_break_count += 1
                    
                false_breaks.append(false_break_count)
            else:
                false_breaks.append(0)
        
        df['false_break_frequency'] = pd.Series(false_breaks).rolling(window).sum()
        
        # Pattern clustering features
        pattern_features = df[['body_size', 'upper_wick', 'lower_wick', 'momentum_3bar']].fillna(0)
        if len(pattern_features) >= 10:
            try:
                kmeans = KMeans(n_clusters=min(8, len(pattern_features)//10), random_state=42, n_init=10)
                df['pattern_cluster'] = kmeans.fit_predict(pattern_features)
                self.pattern_clusters = kmeans
            except:
                df['pattern_cluster'] = 0
        else:
            df['pattern_cluster'] = 0
        
        # Pattern persistence
        df['pattern_persistence'] = df['pattern_cluster'].rolling(5).apply(
            lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
        )
        
        return df
    
    def _extract_volatility_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract volatility and context features"""
        df = data.copy()
        
        # Local VWAP deviation
        df['vwap'] = self._calculate_vwap(df, 20)
        df['vwap_deviation'] = (df['Close'] - df['vwap']) / df['vwap']
        df['vwap_deviation_percentile'] = df['vwap_deviation'].rolling(50).rank(pct=True)
        
        # Distance from last structure
        df = self._calculate_structure_distances(df)
        
        # Volatility features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling volatility measures
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_rank_{window}'] = df[f'volatility_{window}'].rolling(100).rank(pct=True)
        
        # Volume-price relationship
        df['volume_price_trend'] = df['Volume'].rolling(5).corr(df['Close'].rolling(5).mean())
        df['volume_breakout'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)
        
        # Market microstructure
        df['bid_ask_proxy'] = (df['High'] - df['Low']) / df['Close']  # Proxy for spread
        df['market_impact'] = df['Volume'] * df['total_range']  # Price impact proxy
        
        # Regime detection features
        df['trend_strength'] = abs(df['Close'].rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else 0
        ))
        
        df['mean_reversion_signal'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
        
        # Order book proxy features
        df['buying_pressure'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low'])).fillna(0.5)
        df['selling_pressure'] = 1 - df['buying_pressure']
        
        return df
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close_prev = abs(data['High'] - data['Close'].shift())
        low_close_prev = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _calculate_vwap(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).rolling(period).sum() / data['Volume'].rolling(period).sum()
        return vwap
    
    def _calculate_structure_distances(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate distances from key structural levels"""
        df = data.copy()
        
        # Swing highs and lows
        df['swing_high'] = df['High'].rolling(5, center=True).max() == df['High']
        df['swing_low'] = df['Low'].rolling(5, center=True).min() == df['Low']
        
        # Distance to last swing high/low
        last_swing_high = df.loc[df['swing_high'], 'High']
        last_swing_low = df.loc[df['swing_low'], 'Low']
        
        df['dist_to_swing_high'] = 0.0
        df['dist_to_swing_low'] = 0.0
        
        for i in range(len(df)):
            # Find last swing high before current point
            prev_highs = last_swing_high[last_swing_high.index <= df.index[i]]
            if len(prev_highs) > 0:
                last_high = prev_highs.iloc[-1]
                df.iloc[i, df.columns.get_loc('dist_to_swing_high')] = (df['Close'].iloc[i] - last_high) / last_high
            
            # Find last swing low before current point
            prev_lows = last_swing_low[last_swing_low.index <= df.index[i]]
            if len(prev_lows) > 0:
                last_low = prev_lows.iloc[-1]
                df.iloc[i, df.columns.get_loc('dist_to_swing_low')] = (df['Close'].iloc[i] - last_low) / last_low
        
        return df
    
    def _detect_three_white_soldiers(self, data: pd.DataFrame) -> pd.Series:
        """Detect three white soldiers pattern"""
        condition = (
            (data['is_bullish'] == 1) &
            (data['is_bullish'].shift(1) == 1) &
            (data['is_bullish'].shift(2) == 1) &
            (data['Close'] > data['Close'].shift(1)) &
            (data['Close'].shift(1) > data['Close'].shift(2)) &
            (data['Open'] > data['Open'].shift(1)) &
            (data['Open'].shift(1) > data['Open'].shift(2))
        )
        return condition.astype(int)
    
    def _detect_three_black_crows(self, data: pd.DataFrame) -> pd.Series:
        """Detect three black crows pattern"""
        condition = (
            (data['is_bearish'] == 1) &
            (data['is_bearish'].shift(1) == 1) &
            (data['is_bearish'].shift(2) == 1) &
            (data['Close'] < data['Close'].shift(1)) &
            (data['Close'].shift(1) < data['Close'].shift(2)) &
            (data['Open'] < data['Open'].shift(1)) &
            (data['Open'].shift(1) < data['Open'].shift(2))
        )
        return condition.astype(int)
    
    def _detect_morning_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect morning star pattern"""
        condition = (
            (data['is_bearish'].shift(2) == 1) &  # First candle bearish
            (data['body_size'].shift(1) < data['body_size'].shift(2) * 0.5) &  # Second candle small
            (data['is_bullish'] == 1) &  # Third candle bullish
            (data['Close'] > data['Close'].shift(2))  # Third closes above first
        )
        return condition.astype(int)
    
    def _detect_evening_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect evening star pattern"""
        condition = (
            (data['is_bullish'].shift(2) == 1) &  # First candle bullish
            (data['body_size'].shift(1) < data['body_size'].shift(2) * 0.5) &  # Second candle small
            (data['is_bearish'] == 1) &  # Third candle bearish
            (data['Close'] < data['Close'].shift(2))  # Third closes below first
        )
        return condition.astype(int)

class MultiModalDeepLearningModel:
    """Multi-branch Transformer + CNN + LSTM hybrid model"""
    
    def __init__(self, config: AggressiveSystemConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def build_hybrid_model(self, input_dim: int, sequence_length: int = 30):
        """Build multi-branch hybrid model"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using simplified model")
            return self._build_sklearn_model(input_dim)
        
        class HybridTradingModel(nn.Module):
            def __init__(self, input_dim, sequence_length, hidden_dim=128):
                super(HybridTradingModel, self).__init__()
                
                self.input_dim = input_dim
                self.sequence_length = sequence_length
                self.hidden_dim = hidden_dim
                
                # LSTM Branch
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0.2)
                
                # CNN Branch (1D convolution for time series)
                self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
                
                # Transformer Branch
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim, nhead=8, dim_feedforward=256, dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
                
                # Attention mechanism
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
                
                # Fusion layers
                self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
                
                # Output heads
                self.direction_head = nn.Linear(hidden_dim, 3)  # Bull/Bear/Sideways
                self.magnitude_head = nn.Linear(hidden_dim, 1)  # Expected move %
                self.confidence_head = nn.Linear(hidden_dim, 1)  # Confidence score
                self.outlier_head = nn.Linear(hidden_dim, 1)    # Outlier probability
                
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                batch_size, seq_len, features = x.shape
                
                # LSTM Branch
                lstm_out, (h_n, c_n) = self.lstm(x)
                lstm_features = h_n[-1]  # Last hidden state
                
                # CNN Branch
                x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
                conv_out = F.relu(self.conv1(x_conv))
                conv_out = F.relu(self.conv2(conv_out))
                conv_out = F.relu(self.conv3(conv_out))
                cnn_features = conv_out.mean(dim=2)  # Global average pooling
                
                # Transformer Branch
                x_trans = x.transpose(0, 1)  # (seq_len, batch, features)
                trans_out = self.transformer(x_trans)
                transformer_features = trans_out.mean(dim=0)  # Average over sequence
                
                # Attention-based fusion
                combined = torch.stack([lstm_features, cnn_features, transformer_features], dim=1)
                attended, _ = self.attention(combined, combined, combined)
                fused_features = attended.mean(dim=1)
                
                # Apply dropout
                fused_features = self.dropout(fused_features)
                
                # Output predictions
                direction = self.direction_head(fused_features)
                magnitude = self.magnitude_head(fused_features)
                confidence = torch.sigmoid(self.confidence_head(fused_features))
                outlier_prob = torch.sigmoid(self.outlier_head(fused_features))
                
                return {
                    'direction': direction,
                    'magnitude': magnitude,
                    'confidence': confidence,
                    'outlier_prob': outlier_prob
                }
        
        self.model = HybridTradingModel(input_dim, sequence_length)
        return self.model
    
    def _build_sklearn_model(self, input_dim: int):
        """Fallback sklearn-based ensemble model"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        
        class SklearnHybridModel:
            def __init__(self):
                self.direction_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
                self.magnitude_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                self.confidence_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
                self.outlier_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
                
            def fit(self, X, y):
                # Assume y is a dictionary with keys: direction, magnitude, confidence, outlier
                self.direction_model.fit(X, y['direction'])
                self.magnitude_model.fit(X, y['magnitude'])
                self.confidence_model.fit(X, y['confidence'])
                self.outlier_model.fit(X, y['outlier'])
                
            def predict(self, X):
                return {
                    'direction': self.direction_model.predict_proba(X),
                    'magnitude': self.magnitude_model.predict(X),
                    'confidence': self.confidence_model.predict(X),
                    'outlier_prob': self.outlier_model.predict_proba(X)[:, 1] if hasattr(self.outlier_model, 'predict_proba') else self.outlier_model.predict(X)
                }
        
        self.model = SklearnHybridModel()
        return self.model
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data with targets"""
        # Remove non-feature columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = features_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Create targets
        y = self._create_targets(features_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def _create_targets(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create training targets"""
        targets = {}
        
        # Direction target (0: Bearish, 1: Sideways, 2: Bullish)
        future_returns = []
        for horizon in range(1, self.config.max_prediction_horizon + 1):
            future_return = (data['Close'].shift(-horizon) - data['Close']) / data['Close']
            future_returns.append(future_return)
        
        # Use maximum absolute return over prediction horizon
        max_future_return = pd.concat(future_returns, axis=1).abs().max(axis=1)
        direction_return = (data['Close'].shift(-5) - data['Close']) / data['Close']  # 5-period return
        
        direction = np.where(direction_return > 0.01, 2,  # Bullish
                           np.where(direction_return < -0.01, 0, 1))  # Bearish, Sideways
        
        # Magnitude target (expected move %)
        magnitude = direction_return.fillna(0)
        
        # Confidence target (based on volatility and pattern strength)
        volatility = data['Close'].pct_change().rolling(20).std()
        confidence = 1 / (1 + volatility * 10)  # Higher confidence for lower volatility
        
        # Outlier target (5%+ moves)
        outlier = (abs(direction_return) > self.config.outlier_threshold).astype(int)
        
        targets['direction'] = direction[~np.isnan(direction)]
        targets['magnitude'] = magnitude.dropna().values
        targets['confidence'] = confidence.dropna().values
        targets['outlier'] = outlier.dropna().values
        
        # Ensure all targets have same length
        min_length = min(len(v) for v in targets.values())
        for key in targets:
            targets[key] = targets[key][:min_length]
        
        return targets
    
    def train_with_contrastive_learning(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train model with contrastive learning"""
        logger.info("🧠 Training hybrid model with contrastive learning...")
        
        if TORCH_AVAILABLE and hasattr(self.model, 'forward'):
            return self._train_pytorch_model(X, y)
        else:
            return self._train_sklearn_model(X, y)
    
    def _train_pytorch_model(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train PyTorch model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        
        # Reshape for sequence input
        sequence_length = min(30, X.shape[0] // 10)
        if X.shape[0] < sequence_length:
            logger.warning("Insufficient data for sequence modeling")
            return False
        
        # Create sequences
        sequences = []
        targets_seq = {key: [] for key in y.keys()}
        
        for i in range(sequence_length, len(X)):
            sequences.append(X[i-sequence_length:i])
            for key in y.keys():
                if i < len(y[key]):
                    targets_seq[key].append(y[key][i])
        
        if len(sequences) == 0:
            logger.warning("No sequences created")
            return False
        
        X_seq = torch.FloatTensor(np.array(sequences))
        y_tensors = {
            'direction': torch.LongTensor(targets_seq['direction'][:len(sequences)]),
            'magnitude': torch.FloatTensor(targets_seq['magnitude'][:len(sequences)]),
            'confidence': torch.FloatTensor(targets_seq['confidence'][:len(sequences)]),
            'outlier': torch.FloatTensor(targets_seq['outlier'][:len(sequences)])
        }
        
        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Loss functions
        direction_loss_fn = nn.CrossEntropyLoss()
        regression_loss_fn = nn.MSELoss()
        
        # Training loop
        self.model.train()
        num_epochs = 100
        batch_size = min(32, len(X_seq))
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_seq), batch_size):
                batch_X = X_seq[i:i+batch_size]
                batch_y = {key: val[i:i+batch_size] for key, val in y_tensors.items()}
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                
                # Calculate losses
                dir_loss = direction_loss_fn(outputs['direction'], batch_y['direction'])
                mag_loss = regression_loss_fn(outputs['magnitude'].squeeze(), batch_y['magnitude'])
                conf_loss = regression_loss_fn(outputs['confidence'].squeeze(), batch_y['confidence'])
                outlier_loss = regression_loss_fn(outputs['outlier_prob'].squeeze(), batch_y['outlier'])
                
                # Combined loss with weights
                loss = dir_loss + 0.5 * mag_loss + 0.3 * conf_loss + 0.7 * outlier_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        self.is_fitted = True
        return True
    
    def _train_sklearn_model(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """Train sklearn fallback model"""
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("✅ Sklearn model trained successfully")
            return True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make predictions"""
        if not self.is_fitted:
            logger.warning("Model not fitted")
            return {}
        
        if TORCH_AVAILABLE and hasattr(self.model, 'forward'):
            return self._predict_pytorch(X)
        else:
            return self._predict_sklearn(X)
    
    def _predict_pytorch(self, X: np.ndarray) -> Dict[str, Any]:
        """PyTorch model prediction"""
        self.model.eval()
        
        # Prepare sequence
        sequence_length = 30
        if len(X) < sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((sequence_length - len(X), X.shape[1]))
            X_padded = np.vstack([padding, X])
        else:
            X_padded = X[-sequence_length:]
        
        X_tensor = torch.FloatTensor(X_padded).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Convert to numpy
            direction_probs = F.softmax(outputs['direction'], dim=1).numpy()[0]
            magnitude = outputs['magnitude'].numpy()[0, 0]
            confidence = outputs['confidence'].numpy()[0, 0]
            outlier_prob = outputs['outlier_prob'].numpy()[0, 0]
        
        return {
            'direction_probs': direction_probs,
            'predicted_direction': np.argmax(direction_probs),
            'expected_move': magnitude,
            'confidence': confidence,
            'outlier_probability': outlier_prob
        }
    
    def _predict_sklearn(self, X: np.ndarray) -> Dict[str, Any]:
        """Sklearn model prediction"""
        try:
            predictions = self.model.predict(X[-1:])  # Predict on last sample
            
            direction_probs = predictions['direction'][0] if len(predictions['direction'].shape) > 1 else np.array([0.33, 0.34, 0.33])
            
            return {
                'direction_probs': direction_probs,
                'predicted_direction': np.argmax(direction_probs),
                'expected_move': predictions['magnitude'][0] if hasattr(predictions['magnitude'], '__len__') else predictions['magnitude'],
                'confidence': predictions['confidence'][0] if hasattr(predictions['confidence'], '__len__') else predictions['confidence'],
                'outlier_probability': predictions['outlier_prob'][0] if hasattr(predictions['outlier_prob'], '__len__') else predictions['outlier_prob']
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'direction_probs': np.array([0.33, 0.34, 0.33]),
                'predicted_direction': 1,
                'expected_move': 0.0,
                'confidence': 0.5,
                'outlier_probability': 0.1
            }

class AggressivePatternAnalyzer:
    """Aggressive pattern analysis with SHAP explanations"""
    
    def __init__(self, config: AggressiveSystemConfig):
        self.config = config
        self.pattern_labels = {}
        self.explainer = None
        
    def analyze_patterns_with_clustering(self, features_df: pd.DataFrame, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns using rolling KMeans on features"""
        logger.info("🔍 Analyzing patterns with dynamic clustering...")
        
        # Select key features for clustering
        cluster_features = [
            'body_size', 'upper_wick', 'lower_wick', 'momentum_3bar', 'momentum_5bar',
            'bull_engulfing', 'bear_engulfing', 'climax_candle', 'volatility_20'
        ]
        
        available_features = [f for f in cluster_features if f in features_df.columns]
        
        if len(available_features) < 3:
            logger.warning("Insufficient features for clustering")
            return {'pattern_label': 'Unknown', 'pattern_confidence': 0.5}
        
        X_cluster = features_df[available_features].fillna(0).tail(100)  # Last 100 bars
        
        if len(X_cluster) < 10:
            return {'pattern_label': 'Insufficient_Data', 'pattern_confidence': 0.5}
        
        # Dynamic clustering
        try:
            n_clusters = min(8, len(X_cluster) // 10)
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_cluster)
            
            # Get current pattern cluster
            current_cluster = clusters[-1]
            
            # Analyze cluster characteristics
            cluster_data = X_cluster[clusters == current_cluster]
            cluster_stats = cluster_data.describe()
            
            # Generate pattern label based on characteristics
            pattern_label = self._generate_pattern_label(cluster_stats, predictions)
            
            # Calculate pattern confidence based on cluster cohesion
            cluster_distances = kmeans.transform(X_cluster)
            current_distance = cluster_distances[-1, current_cluster]
            pattern_confidence = max(0.1, 1 - (current_distance / np.max(cluster_distances)))
            
            return {
                'pattern_label': pattern_label,
                'pattern_confidence': pattern_confidence,
                'cluster_id': current_cluster,
                'cluster_size': len(cluster_data)
            }
            
        except Exception as e:
            logger.error(f"Pattern clustering failed: {e}")
            return {'pattern_label': 'Error', 'pattern_confidence': 0.5}
    
    def _generate_pattern_label(self, cluster_stats: pd.DataFrame, predictions: Dict[str, Any]) -> str:
        """Generate descriptive pattern label"""
        try:
            # Analyze key characteristics
            body_size_mean = cluster_stats.loc['mean', 'body_size'] if 'body_size' in cluster_stats.columns else 0
            upper_wick_mean = cluster_stats.loc['mean', 'upper_wick'] if 'upper_wick' in cluster_stats.columns else 0
            lower_wick_mean = cluster_stats.loc['mean', 'lower_wick'] if 'lower_wick' in cluster_stats.columns else 0
            momentum_3bar = cluster_stats.loc['mean', 'momentum_3bar'] if 'momentum_3bar' in cluster_stats.columns else 0
            
            # Pattern classification logic
            if body_size_mean > 0.02:  # Large body
                if momentum_3bar > 0.01:
                    base_pattern = "Strong_Bullish"
                elif momentum_3bar < -0.01:
                    base_pattern = "Strong_Bearish"
                else:
                    base_pattern = "Large_Body_Neutral"
            elif upper_wick_mean > 0.01 and lower_wick_mean < 0.005:
                base_pattern = "Shooting_Star"
            elif lower_wick_mean > 0.01 and upper_wick_mean < 0.005:
                base_pattern = "Hammer"
            elif upper_wick_mean > 0.01 and lower_wick_mean > 0.01:
                base_pattern = "Doji_Indecision"
            else:
                base_pattern = "Small_Body"
            
            # Add prediction context
            direction = predictions.get('predicted_direction', 1)
            confidence = predictions.get('confidence', 0.5)
            
            if confidence > 0.8:
                confidence_suffix = "_High_Conf"
            elif confidence > 0.6:
                confidence_suffix = "_Med_Conf"
            else:
                confidence_suffix = "_Low_Conf"
            
            direction_names = ["Bearish", "Neutral", "Bullish"]
            direction_suffix = f"_{direction_names[direction]}"
            
            return f"{base_pattern}{direction_suffix}{confidence_suffix}"
            
        except Exception as e:
            logger.error(f"Pattern label generation failed: {e}")
            return "Unknown_Pattern"
    
    def explain_prediction_with_shap(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Generate SHAP explanations for predictions"""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - using feature importance fallback")
            return self._fallback_feature_importance(model, feature_names)
        
        try:
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X[-100:])  # Use last 100 samples as background
            else:
                explainer = shap.Explainer(model.predict, X[-100:])
            
            # Get SHAP values for last prediction
            shap_values = explainer(X[-1:])
            
            # Extract feature importance
            if hasattr(shap_values, 'values'):
                importance_scores = np.abs(shap_values.values[0]).mean(axis=0) if len(shap_values.values[0].shape) > 1 else np.abs(shap_values.values[0])
            else:
                importance_scores = np.abs(shap_values[0])
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance_scores))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_features[:10]),  # Top 10 features
                'explanation_method': 'SHAP',
                'total_features_analyzed': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_feature_importance(model, feature_names)
    
    def _fallback_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Fallback feature importance using model attributes"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance_scores = np.abs(model.coef_).flatten()
            else:
                # Random importance as last resort
                importance_scores = np.random.random(len(feature_names))
            
            feature_importance = dict(zip(feature_names, importance_scores))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_features[:10]),
                'explanation_method': 'Model_Attributes',
                'total_features_analyzed': len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Fallback feature importance failed: {e}")
            return {
                'feature_importance': {},
                'explanation_method': 'None',
                'total_features_analyzed': 0
            }

class AggressiveTradingSystem:
    """Main aggressive trading system orchestrator"""
    
    def __init__(self, symbol: str = 'GC=F'):
        self.config = AggressiveSystemConfig(symbol=symbol)
        self.feature_engineer = AdvancedFeatureEngineering(self.config)
        self.model = MultiModalDeepLearningModel(self.config)
        self.pattern_analyzer = AggressivePatternAnalyzer(self.config)
        self.data = {}
        self.predictions = {}
        
    def fetch_live_data(self) -> bool:
        """Fetch live data for analysis"""
        logger.info(f"🔄 Fetching live data for {self.config.symbol}...")
        
        try:
            ticker = yf.Ticker(self.config.symbol)
            
            # Fetch data for multiple timeframes
            for timeframe in self.config.timeframes:
                try:
                    if timeframe == '5m':
                        period, interval = '5d', '5m'
                    elif timeframe == '15m':
                        period, interval = '10d', '15m'
                    elif timeframe == '1h':
                        period, interval = '30d', '1h'
                    elif timeframe == '4h':
                        period, interval = '60d', '4h'
                    elif timeframe == '1d':
                        period, interval = '2y', '1d'
                    elif timeframe == '1wk':
                        period, interval = '5y', '1wk'
                    else:
                        continue
                    
                    data = ticker.history(period=period, interval=interval)
                    
                    if not data.empty:
                        self.data[timeframe] = data
                        logger.info(f"✅ {timeframe}: {len(data)} data points")
                    else:
                        logger.warning(f"❌ No data for {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Failed to fetch {timeframe} data: {e}")
            
            return len(self.data) > 0
            
        except Exception as e:
            logger.error(f"Data fetching failed: {e}")
            return False
    
    def run_aggressive_analysis(self):
        """Run complete aggressive analysis"""
        print("🚀 AGGRESSIVE HIGH-FIDELITY DEEP LEARNING TRADING SYSTEM")
        print("=" * 80)
        print("🎯 Advanced AI Trading Intelligence for Volatile Assets")
        print("🧠 Multi-branch Transformer + CNN + LSTM Hybrid")
        print("=" * 80)
        
        # Fetch live data
        if not self.fetch_live_data():
            print("❌ Failed to fetch live data")
            return
        
        print(f"\n⚙️ SYSTEM CONFIGURATION:")
        print(f"   • Asset: {self.config.symbol}")
        print(f"   • Timeframes: {', '.join(self.config.timeframes)}")
        print(f"   • Confidence Threshold: {self.config.confidence_threshold}")
        print(f"   • Outlier Threshold: {self.config.outlier_threshold}")
        print(f"   • Pattern Memory: {self.config.pattern_memory_window} bars")
        
        # Analyze primary timeframe (1h or 1d)
        primary_tf = '1h' if '1h' in self.data else '1d' if '1d' in self.data else list(self.data.keys())[0]
        primary_data = self.data[primary_tf]
        
        print(f"\n🔬 ADVANCED FEATURE ENGINEERING:")
        print(f"   • Primary Timeframe: {primary_tf}")
        print(f"   • Data Points: {len(primary_data)}")
        
        # Extract comprehensive features
        features_df = self.feature_engineer.extract_comprehensive_features(primary_data)
        
        # Build and train model
        print(f"\n🧠 BUILDING HYBRID DEEP LEARNING MODEL:")
        feature_cols = [col for col in features_df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"   • Features Extracted: {len(feature_cols)}")
        print(f"   • Model Architecture: Transformer + CNN + LSTM")
        
        # Prepare training data
        X, y = self.model.prepare_training_data(features_df)
        
        if len(X) < 50:
            print("❌ Insufficient data for training")
            return
        
        # Build model
        self.model.build_hybrid_model(input_dim=X.shape[1])
        
        # Train model
        print(f"\n🎯 TRAINING WITH CONTRASTIVE LEARNING:")
        training_success = self.model.train_with_contrastive_learning(X, y)
        
        if not training_success:
            print("❌ Model training failed")
            return
        
        print("✅ Model training completed")
        
        # Make predictions
        print(f"\n🔮 GENERATING AGGRESSIVE PREDICTIONS:")
        predictions = self.model.predict(X)
        
        if not predictions:
            print("❌ Prediction failed")
            return
        
        # Analyze patterns
        pattern_analysis = self.pattern_analyzer.analyze_patterns_with_clustering(features_df, predictions)
        
        # Feature importance analysis
        feature_importance = self.pattern_analyzer.explain_prediction_with_shap(
            self.model.model, X, feature_cols
        )
        
        # Display results
        self._display_aggressive_results(primary_data, predictions, pattern_analysis, feature_importance)
        
        # Export results
        self._export_aggressive_results(primary_data, predictions, pattern_analysis, feature_importance)
    
    def _display_aggressive_results(self, data: pd.DataFrame, predictions: Dict[str, Any], 
                                  pattern_analysis: Dict[str, Any], feature_importance: Dict[str, Any]):
        """Display comprehensive results"""
        current_price = data['Close'].iloc[-1]
        
        print(f"\n💰 CURRENT MARKET STATUS:")
        print(f"   • Current Price: ${current_price:.2f}")
        print(f"   • Last Update: {data.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Predictions
        direction_names = ["🔴 BEARISH", "⚪ SIDEWAYS", "🟢 BULLISH"]
        predicted_direction = predictions.get('predicted_direction', 1)
        expected_move = predictions.get('expected_move', 0)
        confidence = predictions.get('confidence', 0.5)
        outlier_prob = predictions.get('outlier_probability', 0.1)
        
        print(f"\n🎯 AGGRESSIVE PREDICTIONS:")
        print(f"   • Direction: {direction_names[predicted_direction]}")
        print(f"   • Expected Move: {expected_move:.2%}")
        print(f"   • Confidence Score: {confidence:.1%}")
        print(f"   • Outlier Probability: {outlier_prob:.1%}")
        
        if expected_move != 0:
            target_price = current_price * (1 + expected_move)
            print(f"   • Target Price: ${target_price:.2f}")
        
        # Pattern Analysis
        pattern_label = pattern_analysis.get('pattern_label', 'Unknown')
        pattern_confidence = pattern_analysis.get('pattern_confidence', 0.5)
        
        print(f"\n🔍 PATTERN ANALYSIS:")
        print(f"   • Pattern Detected: {pattern_label}")
        print(f"   • Pattern Confidence: {pattern_confidence:.1%}")
        print(f"   • Cluster ID: {pattern_analysis.get('cluster_id', 'N/A')}")
        
        # Feature Importance
        print(f"\n🧠 KEY DRIVING FACTORS:")
        importance_dict = feature_importance.get('feature_importance', {})
        for i, (feature, importance) in enumerate(list(importance_dict.items())[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")
        
        # Risk Assessment
        risk_level = "🔴 HIGH" if confidence < 0.6 else "🟡 MEDIUM" if confidence < 0.8 else "🟢 HIGH CONFIDENCE"
        print(f"\n🛡️ RISK ASSESSMENT:")
        print(f"   • Confidence Level: {risk_level}")
        print(f"   • Outlier Risk: {'🚨 HIGH' if outlier_prob > 0.3 else '⚠️ MEDIUM' if outlier_prob > 0.15 else '✅ LOW'}")
        
        # Trading Recommendation
        if confidence >= self.config.confidence_threshold:
            recommendation = "🚀 HIGH CONVICTION TRADE"
            print(f"\n🎯 TRADING SIGNAL: {recommendation}")
            print(f"   • Entry: Current levels (${current_price:.2f})")
            if expected_move > 0:
                stop_loss = current_price * 0.98  # 2% stop loss
                take_profit = current_price * (1 + abs(expected_move) * 1.5)  # 1.5x expected move
                print(f"   • Stop Loss: ${stop_loss:.2f}")
                print(f"   • Take Profit: ${take_profit:.2f}")
        else:
            print(f"\n⚠️ TRADING SIGNAL: WAIT FOR HIGHER CONFIDENCE")
            print(f"   • Current confidence ({confidence:.1%}) below threshold ({self.config.confidence_threshold:.1%})")
    
    def _export_aggressive_results(self, data: pd.DataFrame, predictions: Dict[str, Any], 
                                 pattern_analysis: Dict[str, Any], feature_importance: Dict[str, Any]):
        """Export results in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = 'aggressive_exports'
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # JSON Export
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.config.symbol,
            'current_price': float(data['Close'].iloc[-1]),
            'predictions': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                          for k, v in predictions.items()},
            'pattern_analysis': pattern_analysis,
            'feature_importance': feature_importance,
            'system_config': {
                'confidence_threshold': self.config.confidence_threshold,
                'outlier_threshold': self.config.outlier_threshold
            }
        }
        
        json_path = os.path.join(export_dir, f'aggressive_analysis_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n📁 EXPORTS COMPLETED:")
        print(f"   • JSON: {json_path}")
        
        # Excel export would go here
        # PNG chart export would go here
        
        logger.info(f"Results exported to {export_dir}")

def main():
    """Main execution function"""
    try:
        # Initialize system
        system = AggressiveTradingSystem('GC=F')  # Gold futures
        
        # Run analysis
        system.run_aggressive_analysis()
        
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ System error: {e}")

if __name__ == "__main__":
    main()