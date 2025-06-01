"""
User clustering analysis based on activity patterns.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def cluster_users_by_activity(df, n_clusters=3):
    """
    Cluster users by their activity patterns (24-hour activity profile).
    
    Args:
        df: DataFrame with messages containing 'username' and 'date' columns
        n_clusters: Number of clusters to create (default: 3)
        
    Returns:
        tuple: (cluster_results dict, cluster_labels, feature_matrix)
    """
    if df.empty:
        return {}, {}, np.array([])
    
    # Extract hour from date
    df = df.copy()
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], utc=True)
    else:
        # If already datetime but timezone-aware, convert to UTC
        if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
            df['date'] = df['date'].dt.tz_convert('UTC')
        elif df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_convert('UTC')
    df['hour'] = df['date'].dt.hour
    
    # Create 24-hour activity vector for each user
    users = df['username'].unique()
    feature_matrix = []
    user_names = []
    
    for user in users:
        user_data = df[df['username'] == user]
        
        # Skip users with very few messages
        if len(user_data) < 5:
            continue
            
        # Create 24-hour activity profile (normalized)
        hourly_counts = user_data['hour'].value_counts().reindex(range(24), fill_value=0)
        total_messages = hourly_counts.sum()
        
        if total_messages > 0:
            hourly_profile = hourly_counts / total_messages  # Normalize to percentages
            feature_matrix.append(hourly_profile.values)
            user_names.append(user)
    
    if len(feature_matrix) < n_clusters:
        return {"error": f"Недостаточно пользователей для кластеризации. Найдено {len(feature_matrix)} активных пользователей, нужно минимум {n_clusters}."}, {}, np.array([])
    
    # Convert to numpy array
    X = np.array(feature_matrix)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create cluster results
    cluster_results = {}
    
    for i in range(n_clusters):
        cluster_users = [user_names[j] for j in range(len(user_names)) if cluster_labels[j] == i]
        if not cluster_users:
            continue
            
        # Calculate cluster characteristics
        cluster_data = df[df['username'].isin(cluster_users)]
        
        # Find peak activity hours for this cluster
        cluster_hourly = cluster_data['hour'].value_counts().sort_index()
        peak_hours = cluster_hourly.nlargest(3).index.tolist()
        
        # Calculate average activity pattern
        cluster_features = X[cluster_labels == i]
        avg_pattern = np.mean(cluster_features, axis=0)
        
        # Determine cluster type based on peak hours
        cluster_type = categorize_cluster_type(peak_hours, avg_pattern)
        
        cluster_results[f"Кластер {i + 1}"] = {
            "type": cluster_type,
            "users": cluster_users,
            "peak_hours": peak_hours,
            "user_count": len(cluster_users),
            "avg_pattern": avg_pattern
        }
    
    return cluster_results, dict(zip(user_names, cluster_labels)), X


def categorize_cluster_type(peak_hours, pattern):
    """
    Categorize cluster type based on peak activity hours.
    
    Args:
        peak_hours: List of peak activity hours
        pattern: Average 24-hour activity pattern
        
    Returns:
        str: Cluster type description
    """
    # Convert peak hours to primary peak
    if not peak_hours:
        return "🤷 Неопределенный тип"
    
    primary_peak = peak_hours[0]
    
    # Define time periods
    if 6 <= primary_peak <= 9:
        return "🌅 Ранние пташки"
    elif 9 <= primary_peak <= 17:
        return "🏢 Дневные работники" 
    elif 18 <= primary_peak <= 22:
        return "🌆 Вечерние активисты"
    elif 22 <= primary_peak <= 23 or 0 <= primary_peak <= 2:
        return "🌙 Полуночники"
    elif 2 <= primary_peak <= 6:
        return "🦉 Ночные совы"
    else:
        return f"⏰ Активны в {primary_peak:02d}:00"


def format_cluster_analysis_text(cluster_results):
    """
    Format cluster analysis results as text.
    
    Args:
        cluster_results: Dictionary with cluster analysis results
        
    Returns:
        str: Formatted text description
    """
    if "error" in cluster_results:
        return cluster_results["error"]
    
    if not cluster_results:
        return "Не удалось выполнить кластеризацию пользователей."
    
    text = "🔍 Кластерный анализ пользователей по времени активности:\n\n"
    
    total_users = sum(cluster["user_count"] for cluster in cluster_results.values())
    
    for cluster_name, cluster_info in cluster_results.items():
        text += f"{cluster_info['type']} ({cluster_name}):\n"
        text += f"👥 Пользователи ({cluster_info['user_count']}):\n"
        
        for user in cluster_info["users"]:
            text += f"  • {user}\n"
        
        text += f"⏰ Пиковые часы: {', '.join([f'{h:02d}:00' for h in cluster_info['peak_hours']])}\n"
        
        percentage = (cluster_info['user_count'] / total_users) * 100
        text += f"📊 Доля от общего числа: {percentage:.1f}%\n\n"
    
    text += f"Всего проанализировано пользователей: {total_users}\n"
    text += "Кластеризация основана на 24-часовых паттернах активности пользователей."
    
    return text


def create_cluster_visualization(cluster_results, cluster_labels, feature_matrix):
    """
    Create visualization for cluster analysis results.
    
    Args:
        cluster_results: Dictionary with cluster analysis results
        cluster_labels: Dictionary mapping user names to cluster labels
        feature_matrix: Numpy array with user activity patterns
        
    Returns:
        str: Path to saved visualization file or None if error
    """
    if "error" in cluster_results or not cluster_results or len(feature_matrix) == 0:
        return None
        
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create heatmap of activity patterns by cluster
        user_names = list(cluster_labels.keys())
        cluster_nums = list(cluster_labels.values())
        
        # Sort users by cluster
        sorted_indices = np.argsort(cluster_nums)
        sorted_users = [user_names[i] for i in sorted_indices]
        sorted_patterns = feature_matrix[sorted_indices]
        sorted_clusters = [cluster_nums[i] for i in sorted_indices]
        
        # Create color mapping for clusters
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        cluster_colors = [colors[c % len(colors)] for c in sorted_clusters]
        
        # Create the main heatmap
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.8, 0.2],
            subplot_titles=("Паттерны активности пользователей по кластерам", "Средние паттерны кластеров"),
            vertical_spacing=0.15
        )
        
        # Main heatmap: individual users
        fig.add_trace(
            go.Heatmap(
                z=sorted_patterns,
                x=list(range(24)),
                y=sorted_users,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Относительная активность", x=1.02)
            ),
            row=1, col=1
        )
        
        # Add cluster average patterns
        cluster_avg_patterns = []
        cluster_names = []
        
        for cluster_name, cluster_info in cluster_results.items():
            cluster_avg_patterns.append(cluster_info['avg_pattern'])
            cluster_names.append(f"{cluster_info['type']}")
        
        fig.add_trace(
            go.Heatmap(
                z=cluster_avg_patterns,
                x=list(range(24)),
                y=cluster_names,
                colorscale='Plasma',
                showscale=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': "🕐 Кластерный анализ активности пользователей",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            width=1200
        )
        
        # Update x-axes
        fig.update_xaxes(
            title_text="Час дня",
            tickmode='linear',
            tick0=0,
            dtick=2,
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            row=1, col=1
        )
        
        fig.update_xaxes(
            title_text="Час дня",
            tickmode='linear',
            tick0=0,
            dtick=2,
            tickvals=list(range(0, 24, 2)),
            ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)],
            row=2, col=1
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Пользователи", row=1, col=1)
        fig.update_yaxes(title_text="Кластеры", row=2, col=1)
        
        # Save the visualization
        filename = "cluster_analysis.png"
        fig.write_image(filename, scale=2)
        
        return filename
        
    except Exception as e:
        logger.error(f"Error creating cluster visualization: {e}")
        return None