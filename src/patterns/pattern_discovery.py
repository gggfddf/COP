"""
Autonomous Gold Trading AI - Pattern Discovery Module
Implements unsupervised pattern discovery using clustering over LSTM embeddings
Maintains a Pattern Bank of 50-100 discovered classes with meaningful names
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
import pickle
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PatternDiscoveryEngine:
    """
    Unsupervised pattern discovery and classification system
    Discovers, names, and maintains a bank of trading patterns
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pattern discovery engine"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.pattern_config = self.config['patterns']
        self.num_patterns = self.pattern_config['num_patterns']
        self.clustering_algorithm = self.pattern_config['clustering_algorithm']
        self.similarity_threshold = self.pattern_config['similarity_threshold']
        
        # Pattern bank storage
        self.pattern_bank = {}
        self.pattern_names = {}
        self.pattern_characteristics = {}
        self.pattern_performance = {}
        
        # Clustering components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=20)  # Reduce dimensionality for clustering
        self.clusterer = None
        
        # Known pattern templates
        self.known_patterns = self._initialize_known_patterns()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _initialize_known_patterns(self) -> Dict[str, Dict]:
        """Initialize known pattern templates"""
        return {
            "false_break_liquidity_sweep": {
                "description": "Price breaks key level briefly then reverses, sweeping stops",
                "characteristics": {
                    "wick_length": "high",
                    "volume_spike": "high", 
                    "reversal_speed": "fast",
                    "follow_through": "low"
                },
                "bullish_probability": 0.3,
                "success_rate": 0.0
            },
            "three_bar_exhaustion": {
                "description": "Three consecutive bars showing momentum exhaustion",
                "characteristics": {
                    "bar_count": 3,
                    "volume_pattern": "decreasing",
                    "range_pattern": "decreasing",
                    "wick_progression": "increasing"
                },
                "bullish_probability": 0.2,
                "success_rate": 0.0
            },
            "volume_climax_reversal": {
                "description": "High volume climax followed by reversal",
                "characteristics": {
                    "volume_spike": "extreme",
                    "range_size": "large",
                    "reversal_bar": "present",
                    "follow_through": "strong"
                },
                "bullish_probability": 0.4,
                "success_rate": 0.0
            },
            "session_gap_fill": {
                "description": "Gap at session open gets filled during session",
                "characteristics": {
                    "gap_size": "medium",
                    "fill_speed": "gradual",
                    "volume_profile": "normal",
                    "continuation": "likely"
                },
                "bullish_probability": 0.6,
                "success_rate": 0.0
            },
            "liquidity_grab_continuation": {
                "description": "Brief liquidity grab followed by strong continuation",
                "characteristics": {
                    "grab_duration": "short",
                    "continuation_strength": "high",
                    "volume_confirmation": "strong",
                    "follow_through": "sustained"
                },
                "bullish_probability": 0.8,
                "success_rate": 0.0
            }
        }
    
    def discover_patterns(self, 
                         embeddings: np.ndarray, 
                         market_data: pd.DataFrame,
                         labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Discover patterns using unsupervised clustering
        
        Args:
            embeddings: LSTM/Transformer embeddings (n_samples, embedding_dim)
            market_data: Corresponding market data
            labels: Optional ground truth labels for validation
            
        Returns:
            Dictionary with discovered patterns and metadata
        """
        try:
            self.logger.info("Starting pattern discovery process")
            
            # Preprocess embeddings
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            embeddings_pca = self.pca.fit_transform(embeddings_scaled)
            
            # Perform clustering
            if self.clustering_algorithm == 'dbscan':
                cluster_labels = self._dbscan_clustering(embeddings_pca)
            else:
                cluster_labels = self._kmeans_clustering(embeddings_pca)
            
            # Analyze discovered clusters
            patterns = self._analyze_clusters(
                cluster_labels, 
                embeddings, 
                market_data
            )
            
            # Name patterns based on characteristics
            named_patterns = self._name_patterns(patterns, market_data)
            
            # Update pattern bank
            self._update_pattern_bank(named_patterns)
            
            # Evaluate pattern quality
            quality_metrics = self._evaluate_pattern_quality(
                cluster_labels, 
                embeddings_pca, 
                labels
            )
            
            discovery_results = {
                'patterns': named_patterns,
                'cluster_labels': cluster_labels,
                'quality_metrics': quality_metrics,
                'num_patterns_discovered': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
                'pattern_bank_size': len(self.pattern_bank)
            }
            
            self.logger.info(f"Discovered {discovery_results['num_patterns_discovered']} patterns")
            return discovery_results
            
        except Exception as e:
            self.logger.error(f"Error in pattern discovery: {e}")
            return {}
    
    def _dbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering"""
        # Optimize DBSCAN parameters
        eps_values = np.arange(0.1, 2.0, 0.1)
        min_samples_values = range(5, 20)
        
        best_score = -1
        best_params = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                labels = clusterer.fit_predict(embeddings)
                
                # Skip if all points are noise or all in one cluster
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2 or n_clusters > 50:
                    continue
                
                # Calculate silhouette score
                try:
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                except:
                    continue
        
        # Use best parameters
        if best_params:
            self.clusterer = DBSCAN(eps=best_params[0], min_samples=best_params[1])
            labels = self.clusterer.fit_predict(embeddings)
        else:
            # Fallback parameters
            self.clusterer = DBSCAN(eps=0.5, min_samples=10)
            labels = self.clusterer.fit_predict(embeddings)
        
        return labels
    
    def _kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Perform K-means clustering with optimal K selection"""
        # Find optimal number of clusters using elbow method
        max_k = min(self.num_patterns, len(embeddings) // 10)
        inertias = []
        silhouette_scores = []
        
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(embeddings, labels))
        
        # Choose K with best silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Fit final model
        self.clusterer = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = self.clusterer.fit_predict(embeddings)
        
        return labels
    
    def _analyze_clusters(self, 
                         cluster_labels: np.ndarray,
                         embeddings: np.ndarray,
                         market_data: pd.DataFrame) -> Dict[int, Dict]:
        """Analyze characteristics of each discovered cluster"""
        patterns = {}
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get cluster members
            cluster_mask = cluster_labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_data = market_data.iloc[cluster_mask]
            
            # Calculate cluster characteristics
            characteristics = self._calculate_cluster_characteristics(
                cluster_data, 
                cluster_embeddings
            )
            
            patterns[label] = {
                'size': np.sum(cluster_mask),
                'characteristics': characteristics,
                'centroid': np.mean(cluster_embeddings, axis=0),
                'spread': np.std(cluster_embeddings, axis=0),
                'sample_indices': np.where(cluster_mask)[0]
            }
        
        return patterns
    
    def _calculate_cluster_characteristics(self, 
                                         data: pd.DataFrame,
                                         embeddings: np.ndarray) -> Dict[str, float]:
        """Calculate statistical characteristics of a cluster"""
        characteristics = {}
        
        try:
            # Price action characteristics
            if 'body_size' in data.columns:
                characteristics['avg_body_size'] = data['body_size'].mean()
                characteristics['body_size_std'] = data['body_size'].std()
            
            if 'total_range' in data.columns:
                characteristics['avg_range'] = data['total_range'].mean()
                characteristics['range_std'] = data['total_range'].std()
            
            # Wick characteristics
            if 'upper_wick' in data.columns and 'lower_wick' in data.columns:
                characteristics['avg_upper_wick'] = data['upper_wick'].mean()
                characteristics['avg_lower_wick'] = data['lower_wick'].mean()
                characteristics['wick_ratio'] = (
                    data['upper_wick'].mean() / (data['lower_wick'].mean() + 1e-8)
                )
            
            # Volume characteristics
            if 'volume' in data.columns:
                characteristics['avg_volume'] = data['volume'].mean()
                characteristics['volume_std'] = data['volume'].std()
            
            if 'volume_spike' in data.columns:
                characteristics['avg_volume_spike'] = data['volume_spike'].mean()
            
            # Volatility characteristics
            if 'atr_14' in data.columns:
                characteristics['avg_atr'] = data['atr_14'].mean()
            
            if 'realized_vol_20' in data.columns:
                characteristics['avg_volatility'] = data['realized_vol_20'].mean()
            
            # Time characteristics
            if 'hour_sin' in data.columns and 'hour_cos' in data.columns:
                # Convert back to hour for interpretation
                hours = np.arctan2(data['hour_sin'], data['hour_cos']) * 24 / (2 * np.pi)
                hours = (hours + 24) % 24  # Ensure positive
                characteristics['dominant_hour'] = hours.mode().iloc[0] if not hours.empty else 12
            
            # Market psychology characteristics
            if 'fear_index' in data.columns:
                characteristics['avg_fear'] = data['fear_index'].mean()
            
            if 'greed_index' in data.columns:
                characteristics['avg_greed'] = data['greed_index'].mean()
            
            # Pattern-specific characteristics
            if 'false_breakout_up' in data.columns:
                characteristics['false_breakout_frequency'] = (
                    data['false_breakout_up'].sum() + data.get('false_breakout_down', 0).sum()
                ) / len(data)
            
            if 'liquidity_sweep_high' in data.columns:
                characteristics['liquidity_sweep_frequency'] = (
                    data['liquidity_sweep_high'].sum() + data.get('liquidity_sweep_low', 0).sum()
                ) / len(data)
            
            # Embedding characteristics
            characteristics['embedding_centroid_norm'] = np.linalg.norm(embeddings.mean(axis=0))
            characteristics['embedding_spread'] = np.mean(np.std(embeddings, axis=0))
            
        except Exception as e:
            self.logger.warning(f"Error calculating characteristics: {e}")
        
        return characteristics
    
    def _name_patterns(self, 
                      patterns: Dict[int, Dict],
                      market_data: pd.DataFrame) -> Dict[int, Dict]:
        """Assign meaningful names to discovered patterns"""
        named_patterns = {}
        
        for pattern_id, pattern_info in patterns.items():
            characteristics = pattern_info['characteristics']
            
            # Generate name based on characteristics
            name = self._generate_pattern_name(characteristics)
            
            # Calculate pattern performance if we have future data
            performance = self._calculate_pattern_performance(
                pattern_info['sample_indices'], 
                market_data
            )
            
            named_patterns[pattern_id] = {
                **pattern_info,
                'name': name,
                'performance': performance,
                'confidence': self._calculate_pattern_confidence(characteristics),
                'trading_signal': self._determine_trading_signal(characteristics)
            }
        
        return named_patterns
    
    def _generate_pattern_name(self, characteristics: Dict[str, float]) -> str:
        """Generate meaningful name based on pattern characteristics"""
        name_components = []
        
        # Volume component
        if 'avg_volume_spike' in characteristics:
            if characteristics['avg_volume_spike'] > 2.0:
                name_components.append("HighVolume")
            elif characteristics['avg_volume_spike'] < 0.5:
                name_components.append("LowVolume")
        
        # Range component
        if 'avg_range' in characteristics and 'range_std' in characteristics:
            if characteristics['range_std'] / (characteristics['avg_range'] + 1e-8) > 1.0:
                name_components.append("VolatileRange")
            elif characteristics['range_std'] / (characteristics['avg_range'] + 1e-8) < 0.3:
                name_components.append("TightRange")
        
        # Wick component
        if 'wick_ratio' in characteristics:
            if characteristics['wick_ratio'] > 2.0:
                name_components.append("UpperWickDominant")
            elif characteristics['wick_ratio'] < 0.5:
                name_components.append("LowerWickDominant")
            elif 0.8 <= characteristics['wick_ratio'] <= 1.2:
                name_components.append("BalancedWicks")
        
        # Time component
        if 'dominant_hour' in characteristics:
            hour = characteristics['dominant_hour']
            if 22 <= hour or hour < 8:
                name_components.append("Asian")
            elif 8 <= hour < 16:
                name_components.append("European")
            else:
                name_components.append("US")
        
        # Psychology component
        if 'avg_fear' in characteristics and 'avg_greed' in characteristics:
            fear_greed_ratio = characteristics['avg_fear'] / (characteristics['avg_greed'] + 1e-8)
            if fear_greed_ratio > 1.5:
                name_components.append("FearDriven")
            elif fear_greed_ratio < 0.7:
                name_components.append("GreedDriven")
        
        # Special patterns
        if 'false_breakout_frequency' in characteristics:
            if characteristics['false_breakout_frequency'] > 0.3:
                name_components.append("FakeoutProne")
        
        if 'liquidity_sweep_frequency' in characteristics:
            if characteristics['liquidity_sweep_frequency'] > 0.2:
                name_components.append("LiquiditySweep")
        
        # Combine components
        if name_components:
            base_name = "_".join(name_components)
        else:
            base_name = "UnknownPattern"
        
        # Add pattern type suffix
        return f"{base_name}_Pattern"
    
    def _calculate_pattern_performance(self, 
                                     sample_indices: np.ndarray,
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate historical performance of pattern"""
        performance = {
            'success_rate': 0.0,
            'avg_return': 0.0,
            'max_return': 0.0,
            'min_return': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0
        }
        
        try:
            # Calculate forward returns for pattern occurrences
            returns = []
            
            for idx in sample_indices:
                if idx + 5 < len(market_data):  # Look 5 bars ahead
                    current_price = market_data.iloc[idx]['close']
                    future_price = market_data.iloc[idx + 5]['close']
                    ret = (future_price - current_price) / current_price
                    returns.append(ret)
            
            if returns:
                returns = np.array(returns)
                performance['avg_return'] = np.mean(returns)
                performance['max_return'] = np.max(returns)
                performance['min_return'] = np.min(returns)
                performance['win_rate'] = np.mean(returns > 0)
                
                if np.std(returns) > 0:
                    performance['sharpe_ratio'] = np.mean(returns) / np.std(returns)
                
                # Success rate based on positive returns
                performance['success_rate'] = performance['win_rate']
        
        except Exception as e:
            self.logger.warning(f"Error calculating performance: {e}")
        
        return performance
    
    def _calculate_pattern_confidence(self, characteristics: Dict[str, float]) -> float:
        """Calculate confidence score for pattern based on characteristics"""
        confidence_factors = []
        
        # Volume consistency
        if 'avg_volume_spike' in characteristics:
            volume_factor = min(characteristics['avg_volume_spike'] / 2.0, 1.0)
            confidence_factors.append(volume_factor)
        
        # Range consistency
        if 'avg_range' in characteristics and 'range_std' in characteristics:
            range_consistency = 1.0 - min(
                characteristics['range_std'] / (characteristics['avg_range'] + 1e-8), 1.0
            )
            confidence_factors.append(range_consistency)
        
        # Embedding spread (tighter cluster = higher confidence)
        if 'embedding_spread' in characteristics:
            spread_factor = max(0.0, 1.0 - characteristics['embedding_spread'] / 2.0)
            confidence_factors.append(spread_factor)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _determine_trading_signal(self, characteristics: Dict[str, float]) -> str:
        """Determine trading signal based on pattern characteristics"""
        bullish_score = 0
        bearish_score = 0
        
        # Volume analysis
        if 'avg_volume_spike' in characteristics:
            if characteristics['avg_volume_spike'] > 1.5:
                bullish_score += 1
        
        # Wick analysis
        if 'wick_ratio' in characteristics:
            if characteristics['wick_ratio'] < 0.7:  # Lower wick dominant
                bullish_score += 1
            elif characteristics['wick_ratio'] > 1.3:  # Upper wick dominant
                bearish_score += 1
        
        # Psychology analysis
        if 'avg_greed' in characteristics and 'avg_fear' in characteristics:
            if characteristics['avg_greed'] > characteristics['avg_fear']:
                bullish_score += 1
            else:
                bearish_score += 1
        
        # Determine signal
        if bullish_score > bearish_score:
            return "BULLISH"
        elif bearish_score > bullish_score:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _update_pattern_bank(self, patterns: Dict[int, Dict]):
        """Update the pattern bank with new patterns"""
        for pattern_id, pattern_info in patterns.items():
            pattern_name = pattern_info['name']
            
            # Store in pattern bank
            self.pattern_bank[pattern_name] = pattern_info
            
            # Update pattern characteristics
            self.pattern_characteristics[pattern_name] = pattern_info['characteristics']
            
            # Update performance tracking
            self.pattern_performance[pattern_name] = pattern_info['performance']
        
        self.logger.info(f"Pattern bank updated. Total patterns: {len(self.pattern_bank)}")
    
    def _evaluate_pattern_quality(self, 
                                 cluster_labels: np.ndarray,
                                 embeddings: np.ndarray,
                                 true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Evaluate quality of discovered patterns"""
        metrics = {}
        
        try:
            # Silhouette score
            if len(set(cluster_labels)) > 1:
                metrics['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
            
            # Calinski-Harabasz score
            from sklearn.metrics import calinski_harabasz_score
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)
            
            # Davies-Bouldin score
            from sklearn.metrics import davies_bouldin_score
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
            
            # Number of patterns
            n_patterns = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            metrics['num_patterns'] = n_patterns
            
            # Pattern size distribution
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            if -1 in unique_labels:
                # Remove noise points
                noise_idx = np.where(unique_labels == -1)[0][0]
                counts = np.delete(counts, noise_idx)
            
            metrics['avg_pattern_size'] = np.mean(counts)
            metrics['pattern_size_std'] = np.std(counts)
            
            # If true labels available, calculate adjusted rand index
            if true_labels is not None:
                from sklearn.metrics import adjusted_rand_score
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, cluster_labels)
        
        except Exception as e:
            self.logger.warning(f"Error evaluating pattern quality: {e}")
        
        return metrics
    
    def classify_pattern(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Classify a new embedding against discovered patterns
        
        Args:
            embedding: Single embedding vector
            
        Returns:
            Tuple of (pattern_name, similarity_score)
        """
        if not self.pattern_bank:
            return "UNKNOWN", 0.0
        
        best_match = None
        best_similarity = 0.0
        
        # Normalize input embedding
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        for pattern_name, pattern_info in self.pattern_bank.items():
            # Calculate similarity to pattern centroid
            centroid = pattern_info['centroid']
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            
            similarity = np.dot(embedding_norm, centroid_norm)
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = pattern_name
        
        return best_match if best_match else "UNKNOWN", best_similarity
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific pattern"""
        if pattern_name in self.pattern_bank:
            return self.pattern_bank[pattern_name]
        else:
            return {}
    
    def visualize_patterns(self, 
                          embeddings: np.ndarray, 
                          cluster_labels: np.ndarray,
                          save_path: Optional[str] = None) -> None:
        """Visualize discovered patterns using t-SNE or PCA"""
        try:
            from sklearn.manifold import TSNE
            
            # Reduce to 2D for visualization
            if embeddings.shape[1] > 2:
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings)
            else:
                embeddings_2d = embeddings
            
            # Create plot
            plt.figure(figsize=(12, 8))
            
            unique_labels = set(cluster_labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points in black
                    color = 'black'
                    marker = 'x'
                    alpha = 0.5
                else:
                    marker = 'o'
                    alpha = 0.8
                
                mask = cluster_labels == label
                plt.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1],
                    c=[color], 
                    marker=marker, 
                    alpha=alpha,
                    s=50,
                    label=f'Pattern {label}' if label != -1 else 'Noise'
                )
            
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.title('Discovered Trading Patterns')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error visualizing patterns: {e}")
    
    def save_pattern_bank(self, filepath: str):
        """Save pattern bank to file"""
        try:
            pattern_data = {
                'pattern_bank': self.pattern_bank,
                'pattern_names': self.pattern_names,
                'pattern_characteristics': self.pattern_characteristics,
                'pattern_performance': self.pattern_performance,
                'known_patterns': self.known_patterns,
                'config': self.pattern_config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(pattern_data, f)
            
            self.logger.info(f"Pattern bank saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving pattern bank: {e}")
    
    def load_pattern_bank(self, filepath: str):
        """Load pattern bank from file"""
        try:
            with open(filepath, 'rb') as f:
                pattern_data = pickle.load(f)
            
            self.pattern_bank = pattern_data.get('pattern_bank', {})
            self.pattern_names = pattern_data.get('pattern_names', {})
            self.pattern_characteristics = pattern_data.get('pattern_characteristics', {})
            self.pattern_performance = pattern_data.get('pattern_performance', {})
            
            self.logger.info(f"Pattern bank loaded from {filepath}")
            self.logger.info(f"Loaded {len(self.pattern_bank)} patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading pattern bank: {e}")
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary of all patterns in the bank"""
        if not self.pattern_bank:
            return pd.DataFrame()
        
        summary_data = []
        
        for pattern_name, pattern_info in self.pattern_bank.items():
            summary_data.append({
                'Pattern Name': pattern_name,
                'Size': pattern_info.get('size', 0),
                'Confidence': pattern_info.get('confidence', 0.0),
                'Trading Signal': pattern_info.get('trading_signal', 'NEUTRAL'),
                'Success Rate': pattern_info.get('performance', {}).get('success_rate', 0.0),
                'Avg Return': pattern_info.get('performance', {}).get('avg_return', 0.0),
                'Win Rate': pattern_info.get('performance', {}).get('win_rate', 0.0),
                'Sharpe Ratio': pattern_info.get('performance', {}).get('sharpe_ratio', 0.0)
            })
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Initialize pattern discovery engine
    pattern_engine = PatternDiscoveryEngine()
    
    # Create sample embeddings and market data
    np.random.seed(42)
    n_samples = 1000
    embedding_dim = 128
    
    embeddings = np.random.randn(n_samples, embedding_dim)
    
    # Create sample market data
    market_data = pd.DataFrame({
        'close': np.random.uniform(2000, 2100, n_samples),
        'volume': np.random.uniform(1000, 10000, n_samples),
        'body_size': np.random.uniform(1, 20, n_samples),
        'total_range': np.random.uniform(5, 30, n_samples),
        'upper_wick': np.random.uniform(0, 10, n_samples),
        'lower_wick': np.random.uniform(0, 10, n_samples),
        'volume_spike': np.random.uniform(0.5, 3.0, n_samples),
        'atr_14': np.random.uniform(10, 50, n_samples),
        'hour_sin': np.sin(2 * np.pi * np.random.uniform(0, 24, n_samples) / 24),
        'hour_cos': np.cos(2 * np.pi * np.random.uniform(0, 24, n_samples) / 24),
        'fear_index': np.random.uniform(0, 1, n_samples),
        'greed_index': np.random.uniform(0, 1, n_samples),
        'false_breakout_up': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'false_breakout_down': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'liquidity_sweep_high': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'liquidity_sweep_low': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    # Discover patterns
    results = pattern_engine.discover_patterns(embeddings, market_data)
    
    print("🔍 Pattern Discovery Results:")
    print(f"✅ Patterns discovered: {results['num_patterns_discovered']}")
    print(f"✅ Pattern bank size: {results['pattern_bank_size']}")
    print(f"✅ Quality metrics: {results['quality_metrics']}")
    
    # Show pattern summary
    summary = pattern_engine.get_pattern_summary()
    print("\n📊 Pattern Summary:")
    print(summary.to_string(index=False))
    
    # Test pattern classification
    test_embedding = np.random.randn(embedding_dim)
    pattern_name, similarity = pattern_engine.classify_pattern(test_embedding)
    print(f"\n🎯 Test Classification: {pattern_name} (similarity: {similarity:.3f})")