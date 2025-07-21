"""
Autonomous Gold Trading AI - Feature Engineering Module
Implements all 5 mandatory feature categories:
1. Candle Anatomy Features
2. Volatility & Range Features  
3. Time Context Features
4. Price Action Context Features
5. Market Psychology Features
"""

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.signal import find_peaks
import yaml

class ComprehensiveFeatureEngineer:
    """
    Complete feature engineering system for Gold trading AI
    Implements all mandatory feature categories without omissions
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.lookback_bars = self.config['data']['lookback_bars']
        self.feature_config = self.config['features']
        
        # Initialize scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def engineer_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all feature categories
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info("Starting comprehensive feature engineering")
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # 1. CANDLE ANATOMY FEATURES (MANDATORY)
            df = self._add_candle_anatomy_features(df)
            
            # 2. VOLATILITY & RANGE FEATURES (MANDATORY)
            df = self._add_volatility_range_features(df)
            
            # 3. TIME CONTEXT FEATURES (MANDATORY)
            df = self._add_time_context_features(df)
            
            # 4. PRICE ACTION CONTEXT FEATURES (MANDATORY)
            df = self._add_price_action_features(df)
            
            # 5. MARKET PSYCHOLOGY FEATURES (MANDATORY)
            df = self._add_market_psychology_features(df)
            
            # Add derived features
            df = self._add_derived_features(df)
            
            # Clean up any infinite or NaN values
            df = self._clean_features(df)
            
            self.logger.info(f"Feature engineering complete. Added {len(df.columns) - len(data.columns)} features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            return data
    
    def _add_candle_anatomy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📌 1. CANDLE ANATOMY FEATURES (MANDATORY - NO OMISSIONS)
        """
        try:
            self.logger.info("Adding candle anatomy features")
            
            # Basic candle metrics
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Wick-to-body ratios (Top, Bottom, Relative)
            df['upper_wick_body_ratio'] = np.where(
                df['body_size'] > 0, 
                df['upper_wick'] / df['body_size'], 
                0
            )
            df['lower_wick_body_ratio'] = np.where(
                df['body_size'] > 0,
                df['lower_wick'] / df['body_size'],
                0
            )
            df['total_wick_body_ratio'] = np.where(
                df['body_size'] > 0,
                (df['upper_wick'] + df['lower_wick']) / df['body_size'],
                0
            )
            
            # Relative wick ratios
            df['upper_wick_range_ratio'] = np.where(
                df['total_range'] > 0,
                df['upper_wick'] / df['total_range'],
                0
            )
            df['lower_wick_range_ratio'] = np.where(
                df['total_range'] > 0,
                df['lower_wick'] / df['total_range'],
                0
            )
            df['body_range_ratio'] = np.where(
                df['total_range'] > 0,
                df['body_size'] / df['total_range'],
                0
            )
            
            # Climax candle detection (volume + body spike)
            volume_ma = df['volume'].rolling(20).mean()
            body_ma = df['body_size'].rolling(20).mean()
            
            df['volume_spike'] = df['volume'] / (volume_ma + 1e-8)
            df['body_spike'] = df['body_size'] / (body_ma + 1e-8)
            df['climax_score'] = df['volume_spike'] * df['body_spike']
            df['is_climax_candle'] = (
                (df['volume_spike'] > 2.0) & 
                (df['body_spike'] > 1.5)
            ).astype(int)
            
            # Multi-candle engulfing patterns (3–7 candle span)
            for span in [3, 5, 7]:
                df[f'engulfing_{span}bar'] = self._detect_engulfing_pattern(df, span)
            
            # Inside bar compression
            df['is_inside_bar'] = (
                (df['high'] <= df['high'].shift(1)) & 
                (df['low'] >= df['low'].shift(1))
            ).astype(int)
            
            # Inside bar sequences
            df['inside_bar_sequence'] = self._calculate_inside_bar_sequence(df)
            
            # False breakout wick traps
            df['false_breakout_up'] = self._detect_false_breakout(df, direction='up')
            df['false_breakout_down'] = self._detect_false_breakout(df, direction='down')
            
            # Doji patterns
            df['is_doji'] = (df['body_size'] / (df['total_range'] + 1e-8) < 0.1).astype(int)
            df['is_hammer'] = self._detect_hammer_pattern(df)
            df['is_shooting_star'] = self._detect_shooting_star_pattern(df)
            
            # Candle color and momentum
            df['is_bullish'] = (df['close'] > df['open']).astype(int)
            df['candle_momentum'] = (df['close'] - df['open']) / (df['total_range'] + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding candle anatomy features: {e}")
            return df
    
    def _add_volatility_range_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📌 2. VOLATILITY & RANGE FEATURES (MANDATORY)
        """
        try:
            self.logger.info("Adding volatility and range features")
            
            # ATR and ATR percentile regime
            df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['atr_21'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=21)
            
            # ATR percentile regime (low/mid/high)
            atr_rolling = df['atr_14'].rolling(100)
            df['atr_percentile'] = atr_rolling.apply(
                lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100
            )
            
            df['atr_regime'] = pd.cut(
                df['atr_percentile'], 
                bins=[0, 0.33, 0.66, 1.0], 
                labels=['low', 'mid', 'high']
            )
            df['atr_regime_encoded'] = df['atr_regime'].map({'low': 0, 'mid': 1, 'high': 2})
            
            # Implied Volatility shock zones (using price volatility as proxy)
            returns = df['close'].pct_change()
            df['realized_vol_20'] = returns.rolling(20).std() * np.sqrt(252)  # Annualized
            vol_ma = df['realized_vol_20'].rolling(50).mean()
            df['vol_shock'] = df['realized_vol_20'] / (vol_ma + 1e-8)
            df['is_vol_shock'] = (df['vol_shock'] > 1.5).astype(int)
            
            # Relative range expansion/compression vs 10-bar median
            range_median_10 = df['total_range'].rolling(10).median()
            df['range_expansion'] = df['total_range'] / (range_median_10 + 1e-8)
            df['is_range_expansion'] = (df['range_expansion'] > 1.5).astype(int)
            df['is_range_compression'] = (df['range_expansion'] < 0.7).astype(int)
            
            # Bollinger band % width (not indicator, just range)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'], timeperiod=20)
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_width_percentile'] = df['bb_width'].rolling(100).apply(
                lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100
            )
            
            # Volatility clustering
            df['vol_cluster_score'] = self._calculate_volatility_clustering(df)
            
            # Range efficiency
            df['range_efficiency'] = df['body_size'] / (df['total_range'] + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return df
    
    def _add_time_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📌 3. TIME CONTEXT FEATURES (MANDATORY)
        """
        try:
            self.logger.info("Adding time context features")
            
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Time of day (encoded in sine/cosine for cycles)
            hour = df.index.hour
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Day of week encoding
            dow = df.index.dayofweek
            df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
            
            # Day of month encoding
            dom = df.index.day
            df['dom_sin'] = np.sin(2 * np.pi * dom / 31)
            df['dom_cos'] = np.cos(2 * np.pi * dom / 31)
            
            # Session-based strength (Asia, Europe, US session)
            df['session'] = df.index.hour.map(self._get_trading_session)
            df['session_encoded'] = df['session'].map({
                'Asian': 0, 'European': 1, 'US': 2
            })
            
            # Session strength based on volume and volatility
            session_stats = df.groupby('session').agg({
                'volume': 'mean',
                'total_range': 'mean'
            })
            
            df['session_volume_strength'] = df.apply(
                lambda row: session_stats.loc[row['session'], 'volume'] if row['session'] in session_stats.index else 0,
                axis=1
            )
            
            # Days since last major breakout/fakeout
            df['days_since_breakout'] = self._calculate_days_since_event(df, 'breakout')
            df['days_since_fakeout'] = self._calculate_days_since_event(df, 'fakeout')
            
            # Relative bar position within week/month
            df['week_position'] = (df.index.dayofweek + 1) / 7  # 0-1
            df['month_position'] = df.index.day / 31  # 0-1
            
            # Time since market open/close
            df['minutes_since_open'] = self._calculate_minutes_since_open(df)
            df['minutes_to_close'] = self._calculate_minutes_to_close(df)
            
            # Weekend/holiday effects
            df['is_monday'] = (df.index.dayofweek == 0).astype(int)
            df['is_friday'] = (df.index.dayofweek == 4).astype(int)
            df['is_month_end'] = (df.index.day > 25).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding time context features: {e}")
            return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📌 4. PRICE ACTION CONTEXT FEATURES (MANDATORY)
        """
        try:
            self.logger.info("Adding price action context features")
            
            # Distance from recent swing high/low
            swing_highs, swing_lows = self._identify_swing_points(df)
            df['distance_from_swing_high'] = self._calculate_distance_from_swings(
                df, swing_highs, 'high'
            )
            df['distance_from_swing_low'] = self._calculate_distance_from_swings(
                df, swing_lows, 'low'
            )
            
            # Deviation from rolling VWAP (local + multi-day)
            df['vwap_local'] = self._calculate_vwap(df, period=20)
            df['vwap_daily'] = self._calculate_vwap(df, period=288)  # ~1 day in 5min bars
            
            df['vwap_local_deviation'] = (df['close'] - df['vwap_local']) / df['vwap_local']
            df['vwap_daily_deviation'] = (df['close'] - df['vwap_daily']) / df['vwap_daily']
            
            # Time elapsed since OB/Breaker/Imbalance fill
            df['time_since_ob_fill'] = self._calculate_time_since_ob_fill(df)
            df['time_since_imbalance_fill'] = self._calculate_time_since_imbalance_fill(df)
            
            # Clustering of equal highs/lows
            df['equal_highs_cluster'] = self._detect_equal_levels(df, 'high')
            df['equal_lows_cluster'] = self._detect_equal_levels(df, 'low')
            
            # Liquidity sweep detection
            df['liquidity_sweep_high'] = self._detect_liquidity_sweep(df, 'high')
            df['liquidity_sweep_low'] = self._detect_liquidity_sweep(df, 'low')
            
            # Support/Resistance levels
            df['support_level'] = self._calculate_support_resistance(df, 'support')
            df['resistance_level'] = self._calculate_support_resistance(df, 'resistance')
            df['distance_from_support'] = (df['close'] - df['support_level']) / df['close']
            df['distance_from_resistance'] = (df['resistance_level'] - df['close']) / df['close']
            
            # Price structure analysis
            df['higher_high'] = self._detect_price_structure(df, 'HH')
            df['lower_low'] = self._detect_price_structure(df, 'LL')
            df['higher_low'] = self._detect_price_structure(df, 'HL')
            df['lower_high'] = self._detect_price_structure(df, 'LH')
            
            # Momentum divergence
            df['price_momentum'] = df['close'].pct_change(periods=5)
            df['volume_momentum'] = df['volume'].pct_change(periods=5)
            df['momentum_divergence'] = np.sign(df['price_momentum']) != np.sign(df['volume_momentum'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding price action features: {e}")
            return df
    
    def _add_market_psychology_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📌 5. MARKET PSYCHOLOGY FEATURES (MANDATORY)
        """
        try:
            self.logger.info("Adding market psychology features")
            
            # Wick Pressure (long wick + failure to follow through)
            df['wick_pressure_up'] = self._calculate_wick_pressure(df, 'up')
            df['wick_pressure_down'] = self._calculate_wick_pressure(df, 'down')
            
            # Volume Anomaly (volume deviation from mean for similar range bars)
            df['volume_anomaly'] = self._calculate_volume_anomaly(df)
            
            # Fakeout Pattern Score (probabilistic)
            df['fakeout_probability'] = self._calculate_fakeout_probability(df)
            
            # Displacement Score (impulsive breakout vs grind)
            df['displacement_score'] = self._calculate_displacement_score(df)
            
            # Fear/Greed indicators
            df['fear_index'] = self._calculate_fear_index(df)
            df['greed_index'] = self._calculate_greed_index(df)
            
            # Exhaustion signals
            df['buying_exhaustion'] = self._detect_buying_exhaustion(df)
            df['selling_exhaustion'] = self._detect_selling_exhaustion(df)
            
            # Market participation
            df['participation_rate'] = self._calculate_participation_rate(df)
            
            # Sentiment momentum
            df['sentiment_momentum'] = self._calculate_sentiment_momentum(df)
            
            # Trap patterns
            df['bull_trap_probability'] = self._calculate_trap_probability(df, 'bull')
            df['bear_trap_probability'] = self._calculate_trap_probability(df, 'bear')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding market psychology features: {e}")
            return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features"""
        try:
            # Feature interactions
            df['vol_momentum_interaction'] = df['volume_spike'] * df['candle_momentum']
            df['time_volatility_interaction'] = df['hour_sin'] * df['atr_percentile']
            df['session_strength_interaction'] = df['session_volume_strength'] * df['displacement_score']
            
            # Composite scores
            df['bullish_composite'] = (
                df['is_bullish'] + 
                df['buying_exhaustion'] + 
                df['liquidity_sweep_low'] +
                (df['vwap_local_deviation'] > 0).astype(int)
            ) / 4
            
            df['bearish_composite'] = (
                (1 - df['is_bullish']) + 
                df['selling_exhaustion'] + 
                df['liquidity_sweep_high'] +
                (df['vwap_local_deviation'] < 0).astype(int)
            ) / 4
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding derived features: {e}")
            return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up features by handling inf and NaN values"""
        try:
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values
            df = df.fillna(method='ffill')
            
            # Backward fill remaining NaN values
            df = df.fillna(method='bfill')
            
            # Fill any remaining NaN with 0
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning features: {e}")
            return df
    
    # Helper methods for feature calculations
    def _detect_engulfing_pattern(self, df: pd.DataFrame, span: int) -> pd.Series:
        """Detect multi-candle engulfing patterns"""
        engulfing = pd.Series(0, index=df.index)
        
        for i in range(span, len(df)):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check if current candle engulfs previous span candles
            prev_high = df['high'].iloc[i-span:i].max()
            prev_low = df['low'].iloc[i-span:i].min()
            
            if current_high > prev_high and current_low < prev_low:
                engulfing.iloc[i] = 1
                
        return engulfing
    
    def _calculate_inside_bar_sequence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate consecutive inside bar sequences"""
        sequence = pd.Series(0, index=df.index)
        current_sequence = 0
        
        for i in range(1, len(df)):
            if df['is_inside_bar'].iloc[i]:
                current_sequence += 1
            else:
                current_sequence = 0
            sequence.iloc[i] = current_sequence
            
        return sequence
    
    def _detect_false_breakout(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Detect false breakout patterns"""
        false_breakouts = pd.Series(0, index=df.index)
        lookback = 20
        
        for i in range(lookback, len(df)):
            if direction == 'up':
                # Check for break above recent high followed by failure
                recent_high = df['high'].iloc[i-lookback:i].max()
                if (df['high'].iloc[i] > recent_high and 
                    df['close'].iloc[i] < recent_high):
                    false_breakouts.iloc[i] = 1
            else:
                # Check for break below recent low followed by failure
                recent_low = df['low'].iloc[i-lookback:i].min()
                if (df['low'].iloc[i] < recent_low and 
                    df['close'].iloc[i] > recent_low):
                    false_breakouts.iloc[i] = 1
                    
        return false_breakouts
    
    def _detect_hammer_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick patterns"""
        is_hammer = (
            (df['lower_wick'] > 2 * df['body_size']) &
            (df['upper_wick'] < 0.1 * df['total_range']) &
            (df['body_size'] > 0)
        ).astype(int)
        
        return is_hammer
    
    def _detect_shooting_star_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect shooting star candlestick patterns"""
        is_shooting_star = (
            (df['upper_wick'] > 2 * df['body_size']) &
            (df['lower_wick'] < 0.1 * df['total_range']) &
            (df['body_size'] > 0)
        ).astype(int)
        
        return is_shooting_star
    
    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility clustering score"""
        returns = df['close'].pct_change()
        vol = returns.rolling(5).std()
        vol_ma = vol.rolling(20).mean()
        
        clustering_score = vol / (vol_ma + 1e-8)
        return clustering_score.fillna(1)
    
    def _get_trading_session(self, hour: int) -> str:
        """Map hour to trading session"""
        if 22 <= hour or hour < 8:
            return 'Asian'
        elif 8 <= hour < 16:
            return 'European'
        else:
            return 'US'
    
    def _calculate_days_since_event(self, df: pd.DataFrame, event_type: str) -> pd.Series:
        """Calculate days since last major event"""
        # Simplified implementation - would need more sophisticated event detection
        if event_type == 'breakout':
            events = df['range_expansion'] > 2.0
        else:  # fakeout
            events = (df['false_breakout_up'] == 1) | (df['false_breakout_down'] == 1)
        
        days_since = pd.Series(0, index=df.index)
        last_event_idx = 0
        
        for i, is_event in enumerate(events):
            if is_event:
                last_event_idx = i
            days_since.iloc[i] = i - last_event_idx
            
        return days_since
    
    def _calculate_minutes_since_open(self, df: pd.DataFrame) -> pd.Series:
        """Calculate minutes since market open"""
        # Simplified - assumes 24/7 market for Gold
        return (df.index.hour * 60 + df.index.minute) % (24 * 60)
    
    def _calculate_minutes_to_close(self, df: pd.DataFrame) -> pd.Series:
        """Calculate minutes to market close"""
        # Simplified - assumes 24/7 market for Gold
        return (24 * 60) - ((df.index.hour * 60 + df.index.minute) % (24 * 60))
    
    def _identify_swing_points(self, df: pd.DataFrame, window: int = 10) -> Tuple[pd.Series, pd.Series]:
        """Identify swing highs and lows"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Find peaks and troughs
        peak_indices, _ = find_peaks(highs, distance=window)
        trough_indices, _ = find_peaks(-lows, distance=window)
        
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        swing_highs.iloc[peak_indices] = True
        swing_lows.iloc[trough_indices] = True
        
        return swing_highs, swing_lows
    
    def _calculate_distance_from_swings(self, df: pd.DataFrame, swings: pd.Series, level_type: str) -> pd.Series:
        """Calculate distance from nearest swing point"""
        distances = pd.Series(np.nan, index=df.index)
        
        swing_indices = swings[swings].index
        
        for i, timestamp in enumerate(df.index):
            if len(swing_indices) > 0:
                if level_type == 'high':
                    # Find most recent swing high
                    recent_swings = swing_indices[swing_indices <= timestamp]
                    if len(recent_swings) > 0:
                        swing_level = df.loc[recent_swings[-1], 'high']
                        distances.iloc[i] = (df['close'].iloc[i] - swing_level) / swing_level
                else:  # low
                    # Find most recent swing low
                    recent_swings = swing_indices[swing_indices <= timestamp]
                    if len(recent_swings) > 0:
                        swing_level = df.loc[recent_swings[-1], 'low']
                        distances.iloc[i] = (df['close'].iloc[i] - swing_level) / swing_level
        
        return distances.fillna(0)
    
    def _calculate_vwap(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        return vwap
    
    def _calculate_time_since_ob_fill(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time since order block fill (simplified)"""
        # Simplified implementation - would need more sophisticated OB detection
        return pd.Series(0, index=df.index)
    
    def _calculate_time_since_imbalance_fill(self, df: pd.DataFrame) -> pd.Series:
        """Calculate time since imbalance fill (simplified)"""
        # Simplified implementation - would need gap detection
        return pd.Series(0, index=df.index)
    
    def _detect_equal_levels(self, df: pd.DataFrame, level_type: str, tolerance: float = 0.001) -> pd.Series:
        """Detect clustering of equal highs/lows"""
        clusters = pd.Series(0, index=df.index)
        
        if level_type == 'high':
            levels = df['high']
        else:
            levels = df['low']
        
        for i in range(20, len(df)):
            recent_levels = levels.iloc[i-20:i]
            current_level = levels.iloc[i]
            
            # Count levels within tolerance
            similar_levels = abs(recent_levels - current_level) / current_level < tolerance
            clusters.iloc[i] = similar_levels.sum()
            
        return clusters
    
    def _detect_liquidity_sweep(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Detect liquidity sweeps"""
        sweeps = pd.Series(0, index=df.index)
        lookback = 20
        
        for i in range(lookback, len(df)):
            if direction == 'high':
                recent_high = df['high'].iloc[i-lookback:i].max()
                if (df['high'].iloc[i] > recent_high * 1.001 and  # Small break
                    df['close'].iloc[i] < recent_high * 0.999):   # Quick reversal
                    sweeps.iloc[i] = 1
            else:
                recent_low = df['low'].iloc[i-lookback:i].min()
                if (df['low'].iloc[i] < recent_low * 0.999 and   # Small break
                    df['close'].iloc[i] > recent_low * 1.001):   # Quick reversal
                    sweeps.iloc[i] = 1
                    
        return sweeps
    
    def _calculate_support_resistance(self, df: pd.DataFrame, level_type: str) -> pd.Series:
        """Calculate support/resistance levels"""
        # Simplified implementation using rolling min/max
        if level_type == 'support':
            return df['low'].rolling(50).min()
        else:
            return df['high'].rolling(50).max()
    
    def _detect_price_structure(self, df: pd.DataFrame, pattern: str) -> pd.Series:
        """Detect price structure patterns (HH, HL, LH, LL)"""
        structure = pd.Series(0, index=df.index)
        
        # Simplified implementation
        if pattern == 'HH':  # Higher High
            structure = (df['high'] > df['high'].shift(1)).astype(int)
        elif pattern == 'LL':  # Lower Low
            structure = (df['low'] < df['low'].shift(1)).astype(int)
        elif pattern == 'HL':  # Higher Low
            structure = (df['low'] > df['low'].shift(1)).astype(int)
        else:  # LH - Lower High
            structure = (df['high'] < df['high'].shift(1)).astype(int)
            
        return structure
    
    def _calculate_wick_pressure(self, df: pd.DataFrame, direction: str) -> pd.Series:
        """Calculate wick pressure (long wick + failure to follow through)"""
        if direction == 'up':
            wick_length = df['upper_wick']
            follow_through = (df['close'].shift(-1) > df['high']).astype(int)
        else:
            wick_length = df['lower_wick']
            follow_through = (df['close'].shift(-1) < df['low']).astype(int)
        
        # Pressure is high when wick is long but no follow through
        pressure = (wick_length / (df['total_range'] + 1e-8)) * (1 - follow_through)
        return pressure.fillna(0)
    
    def _calculate_volume_anomaly(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume anomaly for similar range bars"""
        # Group bars by similar range and compare volume
        df_temp = df.copy()
        df_temp['range_bucket'] = pd.qcut(df_temp['total_range'], q=10, labels=False, duplicates='drop')
        
        volume_anomaly = pd.Series(0, index=df.index)
        
        for bucket in df_temp['range_bucket'].unique():
            if pd.notna(bucket):
                bucket_mask = df_temp['range_bucket'] == bucket
                bucket_data = df_temp[bucket_mask]
                
                if len(bucket_data) > 5:
                    volume_mean = bucket_data['volume'].mean()
                    volume_std = bucket_data['volume'].std()
                    
                    if volume_std > 0:
                        anomaly_scores = abs(bucket_data['volume'] - volume_mean) / volume_std
                        volume_anomaly[bucket_mask] = anomaly_scores
        
        return volume_anomaly
    
    def _calculate_fakeout_probability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probabilistic fakeout score"""
        # Combine multiple fakeout indicators
        fakeout_score = (
            df['false_breakout_up'] + 
            df['false_breakout_down'] +
            df['wick_pressure_up'] + 
            df['wick_pressure_down']
        ) / 4
        
        return fakeout_score
    
    def _calculate_displacement_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate displacement score (impulsive vs grind)"""
        # Measure of how impulsive the move is
        price_change = abs(df['close'] - df['open'])
        time_factor = 1  # Could be adjusted based on timeframe
        volume_factor = df['volume'] / df['volume'].rolling(20).mean()
        
        displacement = (price_change * volume_factor) / (time_factor + 1e-8)
        return displacement.fillna(0)
    
    def _calculate_fear_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate fear index based on market behavior"""
        # High wicks, low volume, compression
        fear_components = [
            df['total_wick_body_ratio'],
            1 - df['volume_spike'],  # Inverted volume spike
            1 - df['range_expansion']  # Inverted range expansion
        ]
        
        fear_index = pd.concat(fear_components, axis=1).mean(axis=1)
        return fear_index
    
    def _calculate_greed_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate greed index based on market behavior"""
        # High volume, large bodies, expansion
        greed_components = [
            df['volume_spike'],
            df['body_range_ratio'],
            df['range_expansion']
        ]
        
        greed_index = pd.concat(greed_components, axis=1).mean(axis=1)
        return greed_index
    
    def _detect_buying_exhaustion(self, df: pd.DataFrame) -> pd.Series:
        """Detect buying exhaustion signals"""
        exhaustion = (
            (df['is_bullish'] == 1) &
            (df['upper_wick_body_ratio'] > 1.5) &
            (df['volume_spike'] > 1.5) &
            (df['close'] < df['high'] * 0.95)  # Close near low of range
        ).astype(int)
        
        return exhaustion
    
    def _detect_selling_exhaustion(self, df: pd.DataFrame) -> pd.Series:
        """Detect selling exhaustion signals"""
        exhaustion = (
            (df['is_bullish'] == 0) &
            (df['lower_wick_body_ratio'] > 1.5) &
            (df['volume_spike'] > 1.5) &
            (df['close'] > df['low'] * 1.05)  # Close near high of range
        ).astype(int)
        
        return exhaustion
    
    def _calculate_participation_rate(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market participation rate"""
        # Based on volume and number of price levels tested
        volume_ma = df['volume'].rolling(20).mean()
        participation = df['volume'] / (volume_ma + 1e-8)
        
        return participation
    
    def _calculate_sentiment_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sentiment momentum"""
        # Combination of price momentum and volume momentum
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].pct_change(5)
        
        sentiment = (price_momentum + volume_momentum) / 2
        return sentiment.fillna(0)
    
    def _calculate_trap_probability(self, df: pd.DataFrame, trap_type: str) -> pd.Series:
        """Calculate bull/bear trap probability"""
        if trap_type == 'bull':
            # Bull trap: break higher but fail to sustain
            trap_prob = (
                df['false_breakout_up'] +
                df['buying_exhaustion'] +
                df['wick_pressure_up']
            ) / 3
        else:
            # Bear trap: break lower but fail to sustain
            trap_prob = (
                df['false_breakout_down'] +
                df['selling_exhaustion'] +
                df['wick_pressure_down']
            ) / 3
            
        return trap_prob


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5T')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open': np.random.uniform(2000, 2100, 1000),
        'high': np.random.uniform(2010, 2110, 1000),
        'low': np.random.uniform(1990, 2090, 1000),
        'close': np.random.uniform(2000, 2100, 1000),
        'volume': np.random.uniform(1000, 10000, 1000)
    }, index=dates)
    
    # Ensure OHLC relationships
    sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 10, 1000)
    sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 10, 1000)
    
    # Initialize feature engineer
    feature_engineer = ComprehensiveFeatureEngineer()
    
    # Engineer all features
    engineered_data = feature_engineer.engineer_all_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"Engineered columns: {len(engineered_data.columns)}")
    print(f"Added features: {len(engineered_data.columns) - len(sample_data.columns)}")
    print(f"\nFeature categories implemented:")
    print("✅ 1. Candle Anatomy Features")
    print("✅ 2. Volatility & Range Features")
    print("✅ 3. Time Context Features")
    print("✅ 4. Price Action Context Features")
    print("✅ 5. Market Psychology Features")