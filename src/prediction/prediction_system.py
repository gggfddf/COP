"""
Autonomous Gold Trading AI - Prediction System
Generates comprehensive predictions with annotated charts and all required outputs:
- Direction (UP/DOWN/NEUTRAL)
- Expected Move (% or $)
- Volatility confidence level
- Pattern name (matched)
- Annotated charts with highlights, zones, cones, and patterns
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import mplfinance as mpf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime, timedelta
import yaml
from dataclasses import dataclass, asdict
import seaborn as sns

@dataclass
class PredictionOutput:
    """Structured prediction output"""
    timestamp: str
    direction_label: str  # UP/DOWN/NEUTRAL
    direction_probability: float
    expected_move_pct: float
    expected_move_points: float
    volatility_level: str  # LOW/NORMAL/HIGH
    volatility_confidence: float
    pattern_name: str
    pattern_similarity: float
    confidence_score: float
    prediction_horizon: int  # bars ahead
    
    # Quantile predictions
    quantile_10: float
    quantile_25: float
    quantile_50: float  # median
    quantile_75: float
    quantile_90: float
    
    # Risk metrics
    max_favorable: float
    max_adverse: float
    risk_reward_ratio: float
    
    # Additional context
    market_regime: str
    session: str
    key_levels: Dict[str, float]

class ComprehensivePredictionSystem:
    """
    Complete prediction system that generates all required outputs
    with annotated charts and interpretable results
    """
    
    def __init__(self, 
                 model, 
                 feature_engineer,
                 pattern_engine,
                 config_path: str = "config/config.yaml"):
        """Initialize prediction system"""
        
        self.model = model
        self.feature_engineer = feature_engineer
        self.pattern_engine = pattern_engine
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_config = self.config['output']
        self.performance_targets = self.config['performance']
        
        # Direction mappings
        self.direction_map = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        self.volatility_map = {0: 'LOW', 1: 'NORMAL', 2: 'HIGH'}
        
        # Chart styling
        self.chart_style = {
            'up_color': '#00ff88',
            'down_color': '#ff4444', 
            'neutral_color': '#ffaa00',
            'bg_color': '#1e1e1e',
            'text_color': '#ffffff',
            'grid_color': '#333333'
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def generate_prediction(self, 
                          market_data: pd.DataFrame,
                          current_price: float,
                          lookback_bars: int = 100) -> PredictionOutput:
        """
        Generate comprehensive prediction for current market state
        
        Args:
            market_data: Historical OHLCV data
            current_price: Current market price
            lookback_bars: Number of historical bars to use
            
        Returns:
            Comprehensive prediction output
        """
        try:
            self.logger.info("Generating comprehensive prediction")
            
            # Prepare input data
            input_data = market_data.tail(lookback_bars).copy()
            
            # Engineer features
            engineered_data = self.feature_engineer.engineer_all_features(input_data)
            
            # Prepare model input
            feature_columns = [col for col in engineered_data.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            X = engineered_data[feature_columns].values
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Get model predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model.predict_step(X, 0)
            
            # Extract predictions
            direction_probs = predictions['direction_probs'][0].cpu().numpy()
            direction_pred = predictions['direction_pred'][0].cpu().item()
            magnitude_pred = predictions['magnitude_pred'][0].cpu().numpy()
            volatility_probs = predictions['volatility_probs'][0].cpu().numpy()
            volatility_pred = predictions['volatility_pred'][0].cpu().item()
            quantile_preds = predictions['quantile_preds'][0].cpu().numpy()
            pattern_probs = predictions['pattern_probs'][0].cpu().numpy()
            pattern_pred = predictions['pattern_pred'][0].cpu().item()
            confidence = predictions['confidence'][0].cpu().item()
            embeddings = predictions['embeddings'][0].cpu().numpy()
            
            # Get pattern information
            pattern_name, pattern_similarity = self.pattern_engine.classify_pattern(embeddings)
            
            # Calculate expected moves
            expected_move_pct = magnitude_pred[0] * 100  # Convert to percentage
            expected_move_points = current_price * magnitude_pred[0]
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                quantile_preds, current_price, direction_pred
            )
            
            # Determine market context
            market_context = self._analyze_market_context(engineered_data)
            
            # Create prediction output
            prediction = PredictionOutput(
                timestamp=datetime.now().isoformat(),
                direction_label=self.direction_map[direction_pred],
                direction_probability=float(direction_probs[direction_pred]),
                expected_move_pct=float(expected_move_pct),
                expected_move_points=float(expected_move_points),
                volatility_level=self.volatility_map[volatility_pred],
                volatility_confidence=float(volatility_probs[volatility_pred]),
                pattern_name=pattern_name,
                pattern_similarity=float(pattern_similarity),
                confidence_score=float(confidence),
                prediction_horizon=1,  # 1 bar ahead for primary prediction
                
                # Quantiles
                quantile_10=float(current_price + quantile_preds[0] * current_price),
                quantile_25=float(current_price + quantile_preds[1] * current_price),
                quantile_50=float(current_price + quantile_preds[2] * current_price),
                quantile_75=float(current_price + quantile_preds[3] * current_price),
                quantile_90=float(current_price + quantile_preds[4] * current_price),
                
                # Risk metrics
                max_favorable=risk_metrics['max_favorable'],
                max_adverse=risk_metrics['max_adverse'],
                risk_reward_ratio=risk_metrics['risk_reward_ratio'],
                
                # Context
                market_regime=market_context['regime'],
                session=market_context['session'],
                key_levels=market_context['key_levels']
            )
            
            self.logger.info(f"Prediction generated: {prediction.direction_label} "
                           f"({prediction.direction_probability:.2f} confidence)")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            raise
    
    def create_annotated_chart(self, 
                             market_data: pd.DataFrame,
                             prediction: PredictionOutput,
                             save_path: Optional[str] = None,
                             chart_type: str = 'plotly') -> Any:
        """
        Create annotated chart with all required elements:
        - Candle highlights
        - Prediction zones
        - Probability cones
        - Pattern overlays
        
        Args:
            market_data: OHLCV data
            prediction: Prediction output
            save_path: Path to save chart
            chart_type: 'plotly' or 'matplotlib'
            
        Returns:
            Chart figure
        """
        try:
            if chart_type == 'plotly':
                return self._create_plotly_chart(market_data, prediction, save_path)
            else:
                return self._create_matplotlib_chart(market_data, prediction, save_path)
                
        except Exception as e:
            self.logger.error(f"Error creating chart: {e}")
            return None
    
    def _create_plotly_chart(self, 
                           market_data: pd.DataFrame,
                           prediction: PredictionOutput,
                           save_path: Optional[str] = None) -> go.Figure:
        """Create comprehensive Plotly chart"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Gold Price with Predictions', 'Volume', 'Pattern Confidence'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Main candlestick chart
        candlestick = go.Candlestick(
            x=market_data.index,
            open=market_data['open'],
            high=market_data['high'],
            low=market_data['low'],
            close=market_data['close'],
            name='Gold Price',
            increasing_line_color=self.chart_style['up_color'],
            decreasing_line_color=self.chart_style['down_color']
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add prediction zone
        current_time = market_data.index[-1]
        future_time = current_time + pd.Timedelta(minutes=5)  # Assuming 5-min bars
        current_price = market_data['close'].iloc[-1]
        
        # Prediction zone based on quantiles
        prediction_zone = go.Scatter(
            x=[current_time, future_time, future_time, current_time, current_time],
            y=[
                prediction.quantile_25,
                prediction.quantile_25, 
                prediction.quantile_75,
                prediction.quantile_75,
                prediction.quantile_25
            ],
            fill='toself',
            fillcolor=f'rgba(0, 255, 136, 0.2)' if prediction.direction_label == 'UP' 
                     else f'rgba(255, 68, 68, 0.2)' if prediction.direction_label == 'DOWN'
                     else f'rgba(255, 170, 0, 0.2)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'Prediction Zone ({prediction.direction_label})',
            hovertemplate=f'<b>Prediction Zone</b><br>Direction: {prediction.direction_label}<br>Confidence: {prediction.direction_probability:.2%}<extra></extra>'
        )
        fig.add_trace(prediction_zone, row=1, col=1)
        
        # Add probability cone
        cone_x = [current_time]
        cone_y_upper = [current_price]
        cone_y_lower = [current_price]
        
        for i in range(1, 11):  # 10 future bars
            future_t = current_time + pd.Timedelta(minutes=5*i)
            volatility_factor = prediction.volatility_confidence * i * 0.01
            
            upper_bound = current_price * (1 + prediction.expected_move_pct/100 + volatility_factor)
            lower_bound = current_price * (1 + prediction.expected_move_pct/100 - volatility_factor)
            
            cone_x.append(future_t)
            cone_y_upper.append(upper_bound)
            cone_y_lower.append(lower_bound)
        
        # Upper cone boundary
        fig.add_trace(go.Scatter(
            x=cone_x,
            y=cone_y_upper,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.5)', dash='dot'),
            name='Probability Cone (Upper)',
            showlegend=False
        ), row=1, col=1)
        
        # Lower cone boundary
        fig.add_trace(go.Scatter(
            x=cone_x,
            y=cone_y_lower,
            mode='lines',
            line=dict(color='rgba(100,100,100,0.5)', dash='dot'),
            name='Probability Cone (Lower)',
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.1)',
            showlegend=False
        ), row=1, col=1)
        
        # Add key levels
        for level_name, level_price in prediction.key_levels.items():
            fig.add_hline(
                y=level_price,
                line_dash="dash",
                line_color="yellow",
                annotation_text=f"{level_name}: {level_price:.2f}",
                row=1, col=1
            )
        
        # Add pattern annotation
        fig.add_annotation(
            x=current_time,
            y=current_price * 1.02,
            text=f"<b>{prediction.pattern_name}</b><br>Similarity: {prediction.pattern_similarity:.2%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.chart_style['text_color'],
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor=self.chart_style['text_color'],
            font=dict(color=self.chart_style['text_color'], size=10),
            row=1, col=1
        )
        
        # Volume chart
        volume_colors = [
            self.chart_style['up_color'] if close >= open else self.chart_style['down_color']
            for open, close in zip(market_data['open'], market_data['close'])
        ]
        
        fig.add_trace(go.Bar(
            x=market_data.index,
            y=market_data['volume'],
            marker_color=volume_colors,
            name='Volume',
            opacity=0.7
        ), row=2, col=1)
        
        # Pattern confidence over time (simplified)
        confidence_data = [prediction.confidence_score] * len(market_data)
        fig.add_trace(go.Scatter(
            x=market_data.index,
            y=confidence_data,
            mode='lines',
            line=dict(color='orange', width=2),
            name='Pattern Confidence',
            fill='tozeroy',
            fillcolor='rgba(255,165,0,0.3)'
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>Gold Trading AI Prediction</b><br>"
                     f"Direction: {prediction.direction_label} "
                     f"({prediction.direction_probability:.1%}) | "
                     f"Expected Move: {prediction.expected_move_pct:+.2f}% | "
                     f"Pattern: {prediction.pattern_name}",
                x=0.5,
                font=dict(size=16, color=self.chart_style['text_color'])
            ),
            plot_bgcolor=self.chart_style['bg_color'],
            paper_bgcolor=self.chart_style['bg_color'],
            font=dict(color=self.chart_style['text_color']),
            xaxis3_title="Time",
            yaxis_title="Price",
            yaxis2_title="Volume", 
            yaxis3_title="Confidence",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        # Update axes
        fig.update_xaxes(gridcolor=self.chart_style['grid_color'])
        fig.update_yaxes(gridcolor=self.chart_style['grid_color'])
        
        # Add prediction summary as annotation
        summary_text = (
            f"<b>Prediction Summary</b><br>"
            f"Direction: {prediction.direction_label}<br>"
            f"Confidence: {prediction.direction_probability:.1%}<br>"
            f"Expected Move: {prediction.expected_move_pct:+.2f}%<br>"
            f"Volatility: {prediction.volatility_level}<br>"
            f"Pattern: {prediction.pattern_name}<br>"
            f"Risk/Reward: {prediction.risk_reward_ratio:.2f}<br>"
            f"Session: {prediction.session}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=summary_text,
            showarrow=False,
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor=self.chart_style['text_color'],
            font=dict(color=self.chart_style['text_color'], size=10),
            align="left"
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def _create_matplotlib_chart(self, 
                               market_data: pd.DataFrame,
                               prediction: PredictionOutput,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive matplotlib chart"""
        
        # Set style
        plt.style.use('dark_background')
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                          gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Main price chart with candlesticks
        data_for_plot = market_data.copy()
        data_for_plot.index = pd.to_datetime(data_for_plot.index)
        
        # Create candlestick plot using mplfinance style
        colors = [self.chart_style['up_color'] if close >= open else self.chart_style['down_color']
                 for open, close in zip(data_for_plot['open'], data_for_plot['close'])]
        
        # Plot candlesticks manually
        for i, (idx, row) in enumerate(data_for_plot.iterrows()):
            color = colors[i]
            # Body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            
            ax1.add_patch(plt.Rectangle(
                (i, body_bottom), 0.8, body_height,
                facecolor=color, edgecolor=color, alpha=0.8
            ))
            
            # Wicks
            ax1.plot([i+0.4, i+0.4], [row['low'], row['high']], 
                    color=color, linewidth=1)
        
        # Add prediction zone
        current_price = data_for_plot['close'].iloc[-1]
        prediction_x = len(data_for_plot)
        
        # Prediction zone polygon
        zone_x = [len(data_for_plot)-1, prediction_x+2, prediction_x+2, len(data_for_plot)-1]
        zone_y = [current_price, prediction.quantile_75, prediction.quantile_25, current_price]
        
        zone_color = (self.chart_style['up_color'] if prediction.direction_label == 'UP' 
                     else self.chart_style['down_color'] if prediction.direction_label == 'DOWN'
                     else self.chart_style['neutral_color'])
        
        ax1.add_patch(Polygon(list(zip(zone_x, zone_y)), 
                             facecolor=zone_color, alpha=0.3, 
                             label=f'Prediction Zone ({prediction.direction_label})'))
        
        # Add probability cone
        cone_steps = 10
        for step in range(1, cone_steps+1):
            x_pos = len(data_for_plot) + step
            volatility_factor = prediction.volatility_confidence * step * 0.01
            
            upper_bound = current_price * (1 + prediction.expected_move_pct/100 + volatility_factor)
            lower_bound = current_price * (1 + prediction.expected_move_pct/100 - volatility_factor)
            
            ax1.plot([x_pos, x_pos], [lower_bound, upper_bound], 
                    color='gray', alpha=0.5, linestyle='--')
        
        # Add key levels
        for level_name, level_price in prediction.key_levels.items():
            ax1.axhline(y=level_price, color='yellow', linestyle='--', alpha=0.7,
                       label=f'{level_name}: {level_price:.2f}')
        
        # Chart formatting
        ax1.set_title(f'Gold Trading AI Prediction\n'
                     f'Direction: {prediction.direction_label} '
                     f'({prediction.direction_probability:.1%}) | '
                     f'Expected Move: {prediction.expected_move_pct:+.2f}%',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Volume chart
        ax2.bar(range(len(data_for_plot)), data_for_plot['volume'], 
               color=colors, alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Pattern confidence chart
        confidence_line = [prediction.confidence_score] * len(data_for_plot)
        ax3.plot(range(len(data_for_plot)), confidence_line, 
                color='orange', linewidth=2)
        ax3.fill_between(range(len(data_for_plot)), confidence_line, 
                        alpha=0.3, color='orange')
        ax3.set_ylabel('Confidence', fontsize=12)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add prediction summary text box
        summary_text = (
            f"Prediction Summary\n"
            f"Direction: {prediction.direction_label}\n"
            f"Confidence: {prediction.direction_probability:.1%}\n"
            f"Expected Move: {prediction.expected_move_pct:+.2f}%\n"
            f"Volatility: {prediction.volatility_level}\n"
            f"Pattern: {prediction.pattern_name}\n"
            f"Risk/Reward: {prediction.risk_reward_ratio:.2f}\n"
            f"Session: {prediction.session}"
        )
        
        props = dict(boxstyle='round', facecolor='black', alpha=0.8)
        ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.chart_style['bg_color'])
            self.logger.info(f"Chart saved to {save_path}")
        
        return fig
    
    def export_predictions(self, 
                         predictions: List[PredictionOutput],
                         format: str = 'json',
                         filepath: str = None) -> str:
        """
        Export predictions in specified format
        
        Args:
            predictions: List of predictions
            format: 'json', 'csv', or 'excel'
            filepath: Output file path
            
        Returns:
            Exported data as string or filepath
        """
        try:
            if format == 'json':
                data = [asdict(pred) for pred in predictions]
                json_str = json.dumps(data, indent=2, default=str)
                
                if filepath:
                    with open(filepath, 'w') as f:
                        f.write(json_str)
                    return filepath
                return json_str
                
            elif format == 'csv':
                df = pd.DataFrame([asdict(pred) for pred in predictions])
                
                if filepath:
                    df.to_csv(filepath, index=False)
                    return filepath
                return df.to_csv(index=False)
                
            elif format == 'excel':
                df = pd.DataFrame([asdict(pred) for pred in predictions])
                
                if filepath:
                    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='Predictions', index=False)
                        
                        # Add summary sheet
                        summary_data = {
                            'Total Predictions': len(predictions),
                            'Bullish Predictions': sum(1 for p in predictions if p.direction_label == 'UP'),
                            'Bearish Predictions': sum(1 for p in predictions if p.direction_label == 'DOWN'),
                            'Neutral Predictions': sum(1 for p in predictions if p.direction_label == 'NEUTRAL'),
                            'Average Confidence': np.mean([p.confidence_score for p in predictions]),
                            'High Confidence (>80%)': sum(1 for p in predictions if p.confidence_score > 0.8)
                        }
                        
                        summary_df = pd.DataFrame(list(summary_data.items()), 
                                                columns=['Metric', 'Value'])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    return filepath
                
        except Exception as e:
            self.logger.error(f"Error exporting predictions: {e}")
            return ""
    
    def _calculate_risk_metrics(self, 
                              quantile_preds: np.ndarray,
                              current_price: float,
                              direction_pred: int) -> Dict[str, float]:
        """Calculate risk metrics from quantile predictions"""
        
        # Convert quantile predictions to price levels
        price_quantiles = current_price * (1 + quantile_preds)
        
        if direction_pred == 2:  # UP
            max_favorable = price_quantiles[4] - current_price  # 90th percentile
            max_adverse = current_price - price_quantiles[0]    # 10th percentile
        elif direction_pred == 0:  # DOWN
            max_favorable = current_price - price_quantiles[0]  # 10th percentile
            max_adverse = price_quantiles[4] - current_price    # 90th percentile
        else:  # NEUTRAL
            max_favorable = max(price_quantiles[4] - current_price, 
                              current_price - price_quantiles[0])
            max_adverse = max_favorable * 0.5  # Conservative estimate
        
        risk_reward_ratio = max_favorable / (max_adverse + 1e-8)
        
        return {
            'max_favorable': float(max_favorable),
            'max_adverse': float(max_adverse),
            'risk_reward_ratio': float(risk_reward_ratio)
        }
    
    def _analyze_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market context"""
        
        latest_data = data.iloc[-1]
        
        # Determine market regime
        if 'atr_percentile' in data.columns:
            atr_pct = latest_data['atr_percentile']
            if atr_pct > 0.7:
                regime = 'HIGH_VOLATILITY'
            elif atr_pct < 0.3:
                regime = 'LOW_VOLATILITY'
            else:
                regime = 'NORMAL_VOLATILITY'
        else:
            regime = 'UNKNOWN'
        
        # Determine session
        if 'session' in data.columns:
            session = latest_data['session']
        else:
            session = 'UNKNOWN'
        
        # Key levels (simplified)
        current_price = latest_data['close']
        key_levels = {
            'Current': current_price,
            'Support': current_price * 0.995,  # 0.5% below
            'Resistance': current_price * 1.005  # 0.5% above
        }
        
        if 'support_level' in data.columns:
            key_levels['Support'] = latest_data['support_level']
        
        if 'resistance_level' in data.columns:
            key_levels['Resistance'] = latest_data['resistance_level']
        
        return {
            'regime': regime,
            'session': session,
            'key_levels': key_levels
        }
    
    def generate_batch_predictions(self, 
                                 market_data: pd.DataFrame,
                                 prediction_count: int = 10) -> List[PredictionOutput]:
        """Generate multiple predictions for different time horizons"""
        
        predictions = []
        current_price = market_data['close'].iloc[-1]
        
        for i in range(prediction_count):
            # Simulate different market states by using different lookback windows
            lookback = max(50, 100 - i * 5)
            
            try:
                prediction = self.generate_prediction(
                    market_data, 
                    current_price, 
                    lookback_bars=lookback
                )
                prediction.prediction_horizon = i + 1
                predictions.append(prediction)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate prediction {i+1}: {e}")
        
        return predictions


# Example usage and testing
if __name__ == "__main__":
    # This would normally be imported from the actual modules
    print("🔮 Prediction System Created Successfully!")
    print("✅ Comprehensive prediction output structure")
    print("✅ Annotated chart generation (Plotly & Matplotlib)")
    print("✅ Multiple export formats (JSON, CSV, Excel)")
    print("✅ Risk metrics calculation")
    print("✅ Market context analysis")
    print("✅ Batch prediction capability")
    
    # Create sample prediction output
    sample_prediction = PredictionOutput(
        timestamp=datetime.now().isoformat(),
        direction_label="UP",
        direction_probability=0.85,
        expected_move_pct=1.2,
        expected_move_points=24.5,
        volatility_level="NORMAL",
        volatility_confidence=0.75,
        pattern_name="HighVolume_Asian_Pattern",
        pattern_similarity=0.82,
        confidence_score=0.87,
        prediction_horizon=1,
        quantile_10=2010.5,
        quantile_25=2015.2,
        quantile_50=2020.0,
        quantile_75=2025.8,
        quantile_90=2030.5,
        max_favorable=30.5,
        max_adverse=15.2,
        risk_reward_ratio=2.0,
        market_regime="NORMAL_VOLATILITY",
        session="Asian",
        key_levels={"Support": 2015.0, "Resistance": 2025.0, "Current": 2020.0}
    )
    
    print(f"\n📊 Sample Prediction:")
    print(f"Direction: {sample_prediction.direction_label} "
          f"({sample_prediction.direction_probability:.1%} confidence)")
    print(f"Expected Move: {sample_prediction.expected_move_pct:+.2f}%")
    print(f"Pattern: {sample_prediction.pattern_name}")
    print(f"Risk/Reward: {sample_prediction.risk_reward_ratio:.2f}")