"""
Basic statistical analysis functions.
"""

import json
import calendar
import logging
import pandas as pd
import numpy as np
from pandas import Timestamp

from src.core.config import USER_MAPPING

logger = logging.getLogger(__name__)


def merge_users(df):
    """Merge user names according to mapping."""
    df['username'] = df['username'].replace(USER_MAPPING)
    return df


def parse_date_str(x):
    """
    Parse date string with timezone awareness.
    First tries explicit format, then infers UTC if needed.
    """
    dt = pd.to_datetime(x, format="%Y-%m-%dT%H:%M:%S%z", errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(x, infer_datetime_format=True, utc=True, errors='coerce')
    return dt


def load_and_prepare_data(json_file, start_date=None, end_date=None):
    """
    Load message database from JSON file and prepare DataFrame.
    Selects only messages, parses dates, applies user mapping,
    and filters by optional date range.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    messages = data.get("messages", [])
    df = pd.DataFrame(messages)
    df_messages = df[df.get('type') == 'message'].copy()
    if df_messages.empty:
        return df_messages
    df_messages['date'] = df_messages['date'].apply(parse_date_str)
    df_messages['username'] = df_messages['from'].fillna('Unknown')
    df_messages = df_messages[df_messages['username'] != 'Unknown']
    df_messages = merge_users(df_messages)
    if start_date is not None:
        df_messages = df_messages[df_messages['date'] >= pd.to_datetime(start_date, utc=True)]
    if end_date is not None:
        if not isinstance(end_date, Timestamp):
            end_dt = pd.to_datetime(end_date, utc=True)
        else:
            end_dt = end_date if end_date.tzinfo is not None else end_date.tz_localize('UTC')
        df_messages = df_messages[df_messages['date'] <= end_dt]
    return df_messages


def get_global_top_users(df, top_n=7):
    """Determine global top-N users by message count."""
    user_counts = df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'message_count']
    top_users = user_counts.head(top_n)['username'].tolist()
    return top_users, user_counts


def prepare_relative_data(df, top_users, freq='W'):
    """Prepare data for relative stacked-area chart."""
    df_copy = df.copy()
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'], utc=True)
    df_copy['interval'] = df_copy['date'].dt.to_period(freq).apply(lambda r: r.start_time)
    show_other = len(top_users) < df_copy['username'].nunique()
    if show_other:
        df_copy['category'] = df_copy['username'].apply(lambda x: x if x in top_users else 'Другие')
    else:
        df_copy['category'] = df_copy['username']
    grouped = df_copy.groupby(['interval', 'category']).size().reset_index(name='count')
    totals = grouped.groupby('interval')['count'].sum().reset_index(name='total_count')
    merged = pd.merge(grouped, totals, on='interval')
    merged['relative_share'] = merged['count'] / merged['total_count']
    pivot = merged.pivot(index='interval', columns='category', values='relative_share').fillna(0)
    for u in top_users + (['Другие'] if show_other else []):
        if u not in pivot.columns:
            pivot[u] = 0
    pivot = pivot[top_users + (['Другие'] if show_other else [])]
    pivot['sum'] = pivot.sum(axis=1)
    if not (pivot['sum'].round(4) == 1).all():
        logger.warning("Sum of shares in some intervals is not 1.")
    return pivot.drop(columns=['sum'])


def prepare_cumulative_data(df, top_users, total_messages, freq='W'):
    """Prepare data for cumulative stacked-area chart."""
    df_copy = df.copy()
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'], utc=True)
    df_copy['interval'] = df_copy['date'].dt.to_period(freq).apply(lambda r: r.start_time)
    show_other = len(top_users) < df_copy['username'].nunique()
    if show_other:
        df_copy['category'] = df_copy['username'].apply(lambda x: x if x in top_users else 'Другие')
    else:
        df_copy['category'] = df_copy['username']
    grouped = df_copy.groupby(['interval', 'category']).size().reset_index(name='count')
    grouped = grouped.sort_values('interval')
    pivot = grouped.pivot(index='interval', columns='category', values='count').fillna(0)
    pivot = pivot[top_users + (['Другие'] if show_other else [])]
    pivot = pivot.cumsum() / total_messages
    pivot['sum'] = pivot.sum(axis=1)
    if not (pivot['sum'] <= 1).all():
        logger.warning("Sum of shares in some intervals exceeds 1.")
    return pivot.drop(columns=['sum'])


def build_monthly_stats_text(df, month_str):
    """Generate text summary of statistics for specified month."""
    total_msgs = len(df)
    if total_msgs == 0:
        return f"No messages found for {month_str}."
    user_counts = df['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'message_count']
    user_counts['percentage'] = user_counts['message_count'] / total_msgs * 100
    top_user = user_counts.iloc[0]
    df['date_only'] = df['date'].dt.date
    day_counts = df.groupby('date_only').size()
    active_days = day_counts.count()
    avg_msgs = total_msgs / active_days if active_days else 0
    median_msgs = day_counts.median()
    std_msgs = day_counts.std()
    busiest_day = day_counts.idxmax() if not day_counts.empty else "N/A"
    busiest_day_count = day_counts.max() if not day_counts.empty else 0
    least_day = day_counts.idxmin() if not day_counts.empty else "N/A"
    least_day_count = day_counts.min() if not day_counts.empty else 0
    year, month = map(int, month_str.split('-'))
    total_days = calendar.monthrange(year, month)[1]
    active_pct = active_days / total_days * 100 if total_days else 0
    df['hour'] = df['date'].dt.hour
    hour_counts = df['hour'].value_counts().sort_index()
    busiest_hour = hour_counts.idxmax() if not hour_counts.empty else "N/A"
    busiest_hour_count = hour_counts.max() if not hour_counts.empty else 0
    text = f"Statistics for {month_str}:\n" + f"Total messages: {total_msgs}\n\n" + "Messages by user:\n"
    for i, row in user_counts.iterrows():
        if i >= 5: break
        text += f"{i + 1}. {row['username']} — {row['message_count']} messages ({row['percentage']:.2f}%)\n"
    text += (
        f"\nMost active user: {top_user['username']} ({top_user['message_count']} messages)\n"
        f"\nDays in month: {total_days}, active days: {active_days} ({active_pct:.2f}%)\n"
        f"Average messages per active day: {avg_msgs:.2f}\n"
        f"Median messages per active day: {median_msgs:.2f}\n"
        f"Standard deviation: {std_msgs:.2f}\n"
        f"Busiest day: {busiest_day} ({busiest_day_count} messages)\n"
        f"Least active day: {least_day} ({least_day_count} messages)\n"
        f"\nBusiest hour: {busiest_hour:02d}:00 ({busiest_hour_count} messages)\n"
    )
    return text


def calculate_dominance_index(df):
    """
    Calculate dominance index showing how evenly messages are distributed among participants.
    Uses both Gini coefficient and Herfindahl-Hirschman Index (HHI).
    
    Args:
        df: DataFrame with messages containing 'username' column
        
    Returns:
        str: Formatted text with dominance analysis results
    """
    try:
        if df.empty:
            return "Недостаточно данных для анализа индекса доминирования."
        
        # Calculate message counts per user
        user_counts = df['username'].value_counts().reset_index()
        user_counts.columns = ['username', 'message_count']
        
        total_messages = user_counts['message_count'].sum()
        user_counts['share'] = user_counts['message_count'] / total_messages
        
        # Calculate Gini coefficient
        def calculate_gini(shares):
            """Calculate Gini coefficient (0 = perfect equality, 1 = perfect inequality)"""
            # Sort shares in ascending order
            sorted_shares = np.sort(shares)
            n = len(sorted_shares)
            
            # Calculate cumulative shares
            cumulative_shares = np.cumsum(sorted_shares)
            
            # Calculate Gini using the formula
            gini = (n + 1 - 2 * np.sum((n + 1 - np.arange(1, n + 1)) * sorted_shares) / np.sum(sorted_shares)) / n
            
            return gini
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        def calculate_hhi(shares):
            """Calculate HHI (0 = perfect competition, 10000 = monopoly)"""
            # HHI is sum of squared market shares (in percentage)
            hhi = np.sum((shares * 100) ** 2)
            return hhi
        
        # Calculate indices
        shares = user_counts['share'].values
        gini = calculate_gini(shares)
        hhi = calculate_hhi(shares)
        
        # Calculate normalized HHI for easier interpretation
        # Normalized HHI = (HHI - HHI_min) / (HHI_max - HHI_min)
        n_users = len(user_counts)
        hhi_min = 10000 / n_users  # Perfect equality
        hhi_max = 10000  # Monopoly
        if hhi_max > hhi_min:
            hhi_normalized = (hhi - hhi_min) / (hhi_max - hhi_min)
        else:
            hhi_normalized = 0
        
        # Determine dominance level based on indices
        if gini < 0.3:
            dominance_level = "🟢 Низкий"
            dominance_desc = "Сообщения распределены довольно равномерно"
        elif gini < 0.5:
            dominance_level = "🟡 Средний"
            dominance_desc = "Есть заметные различия в активности"
        elif gini < 0.7:
            dominance_level = "🟠 Высокий"
            dominance_desc = "Несколько пользователей доминируют в чате"
        else:
            dominance_level = "🔴 Очень высокий"
            dominance_desc = "Чат сильно зависит от нескольких активных участников"
        
        # Calculate additional statistics
        top_5_share = user_counts.head(5)['share'].sum() * 100
        top_10_share = user_counts.head(10)['share'].sum() * 100
        
        # Find concentration points
        cumulative_share = 0
        users_for_50_percent = 0
        users_for_80_percent = 0
        
        for _, row in user_counts.iterrows():
            cumulative_share += row['share']
            if cumulative_share >= 0.5 and users_for_50_percent == 0:
                users_for_50_percent = _ + 1
            if cumulative_share >= 0.8 and users_for_80_percent == 0:
                users_for_80_percent = _ + 1
                break
        
        # Format results
        result = "📊 Анализ индекса доминирования:\n\n"
        
        result += f"🎯 Уровень доминирования: {dominance_level}\n"
        result += f"{dominance_desc}\n\n"
        
        result += "📈 Индексы концентрации:\n"
        result += f"• Коэффициент Джини: {gini:.3f}\n"
        result += f"  (0 = идеальное равенство, 1 = монополия)\n"
        result += f"• Индекс Херфиндаля-Хиршмана: {hhi:.0f}\n"
        result += f"  (Нормализованный: {hhi_normalized:.3f})\n\n"
        
        result += "👥 Распределение активности:\n"
        result += f"• Всего участников: {n_users}\n"
        result += f"• Топ-5 пользователей: {top_5_share:.1f}% сообщений\n"
        result += f"• Топ-10 пользователей: {top_10_share:.1f}% сообщений\n"
        result += f"• 50% сообщений написали: {users_for_50_percent} человек(а)\n"
        result += f"• 80% сообщений написали: {users_for_80_percent} человек(а)\n\n"
        
        # Top contributors
        result += "🏆 Основные контрибьюторы:\n"
        for i, row in user_counts.head(5).iterrows():
            result += f"{i+1}. {row['username']} — {row['message_count']} сообщений ({row['share']*100:.1f}%)\n"
        
        # Interpretation
        result += "\n💡 Интерпретация:\n"
        if gini < 0.3:
            result += "• Чат имеет здоровое распределение активности\n"
            result += "• Большинство участников вносят существенный вклад\n"
            result += "• Низкий риск зависимости от отдельных пользователей"
        elif gini < 0.5:
            result += "• Активность умеренно сконцентрирована\n"
            result += "• Есть группа основных участников и менее активные\n"
            result += "• Рекомендуется поощрять участие менее активных"
        elif gini < 0.7:
            result += "• Высокая концентрация активности\n"
            result += "• Чат зависит от небольшой группы активных участников\n"
            result += "• Риск снижения активности при уходе ключевых участников"
        else:
            result += "• Критически высокая концентрация\n"
            result += "• Чат фактически держится на нескольких людях\n"
            result += "• Высокий риск \"смерти\" чата при их уходе"
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating dominance index: {e}")
        return f"Ошибка при расчете индекса доминирования: {str(e)}"


def analyze_user_correlation(df, period='D', min_messages=10):
    """
    Analyze correlation of activity between users.
    
    Args:
        df: DataFrame with messages containing 'date' and 'username' columns
        period: Time period for aggregation ('H', 'D', 'W')
        min_messages: Minimum messages for user to be included
        
    Returns:
        str: Formatted text with correlation analysis results
    """
    try:
        if df.empty:
            return "Недостаточно данных для анализа корреляции активности."
        
        # Ensure date column is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)
        
        # Filter users with minimum messages
        user_counts = df['username'].value_counts()
        active_users = user_counts[user_counts >= min_messages].index.tolist()
        
        if len(active_users) < 2:
            return f"Недостаточно активных пользователей для анализа корреляции. Найдено {len(active_users)} пользователей с более чем {min_messages} сообщениями."
        
        df_active = df[df['username'].isin(active_users)]
        
        # Create time series for each user
        df_active['period'] = df_active['date'].dt.to_period(period)
        user_timeseries = df_active.groupby(['period', 'username']).size().unstack(fill_value=0)
        
        # Calculate correlation matrix
        correlation_matrix = user_timeseries.corr()
        
        # Find pairs with high correlation
        high_correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                user1 = correlation_matrix.index[i]
                user2 = correlation_matrix.index[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:  # Threshold for significant correlation
                    high_correlations.append({
                        'user1': user1,
                        'user2': user2,
                        'correlation': corr_value
                    })
        
        # Sort by absolute correlation value
        high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        # Format results
        result = "📊 Анализ корреляции активности между пользователями:\n\n"
        
        period_names = {
            'H': 'час',
            'D': 'день',
            'W': 'неделя',
            'M': 'месяц'
        }
        period_name = period_names.get(period, period)
        
        result += f"📅 Период анализа: {period_name}\n"
        result += f"👥 Активных пользователей: {len(active_users)}\n"
        result += f"📯 Минимум сообщений для анализа: {min_messages}\n\n"
        
        if not high_correlations:
            result += "❌ Не найдено пар пользователей с высокой корреляцией активности (|корреляция| > 0.5)\n\n"
        else:
            result += "🔗 Пары пользователей с высокой корреляцией:\n\n"
            
            for i, pair in enumerate(high_correlations[:10], 1):  # Top 10 pairs
                corr = pair['correlation']
                if corr > 0:
                    emoji = "📈"
                    desc = "положительная"
                else:
                    emoji = "📉"
                    desc = "отрицательная"
                
                result += f"{i}. {emoji} {pair['user1']} ↔️ {pair['user2']}\n"
                result += f"   Корреляция: {corr:.3f} ({desc})\n"
                
                # Interpret correlation
                if corr > 0.8:
                    result += "   💡 Очень сильная связь - возможно, пишут в одно время\n"
                elif corr > 0.6:
                    result += "   💡 Сильная связь - часто активны одновременно\n"
                elif corr > 0.4:
                    result += "   💡 Умеренная связь\n"
                elif corr < -0.6:
                    result += "   💡 Сильная обратная связь - активны в разное время\n"
                elif corr < -0.4:
                    result += "   💡 Умеренная обратная связь\n"
                
                result += "\n"
        
        # Overall statistics
        all_correlations = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                all_correlations.append(correlation_matrix.iloc[i, j])
        
        if all_correlations:
            avg_corr = np.mean(all_correlations)
            median_corr = np.median(all_correlations)
            std_corr = np.std(all_correlations)
            
            result += "📊 Общая статистика корреляций:\n"
            result += f"• Средняя корреляция: {avg_corr:.3f}\n"
            result += f"• Медианная корреляция: {median_corr:.3f}\n"
            result += f"• Стандартное отклонение: {std_corr:.3f}\n\n"
            
            # Interpretation
            result += "💡 Интерпретация:\n"
            if avg_corr > 0.3:
                result += "• В целом пользователи склонны быть активными в одно время\n"
                result += "• Возможно, это связано с общими интересами или временными зонами\n"
            elif avg_corr < -0.1:
                result += "• Пользователи в основном активны в разное время\n"
                result += "• Чат поддерживается разными группами в разные периоды\n"
            else:
                result += "• Активность пользователей слабо связана между собой\n"
                result += "• Каждый имеет свой индивидуальный паттерн активности\n"
        
        result += f"\n📌 Корреляция показывает, насколько синхронизирована активность пользователей.\n"
        result += "Положительная = активны одновременно, отрицательная = в разное время."
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing user correlation: {e}")
        return f"Ошибка при анализе корреляции активности: {str(e)}"