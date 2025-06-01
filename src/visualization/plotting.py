import logging

import fix_pillow  # Fix for Pillow 10+ compatibility
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

def plot_stacked_area(pivot_df, plot_type='relative', save_as_file=False,
                      file_name="stacked_area_chart", file_format="png"):
    """
    Create stacked area chart using Plotly.
    If save_as_file=True, saves chart to file and returns filename;
    otherwise calls fig.show().
    """
    categories = pivot_df.columns.tolist()
    if 'Другие' in categories:
        categories.remove('Другие')
        categories = ['Другие'] + categories

    colors = px.colors.qualitative.Plotly
    color_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}

    fig = go.Figure()
    for category in categories:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[category],
            mode='lines',
            name=category,
            stackgroup='one',
            fill='tonexty',
            line=dict(width=0.5, color=color_map[category]),
            hoverinfo='x+y+name'
        ))

    title_text = (
        "Relative share of messages from top-N users by period"
        if plot_type == 'relative'
        else "Cumulative share of messages from top-N users"
    )

    fig.update_layout(
        title=title_text,
        xaxis_title='Period',
        yaxis_title='Message Share',
        legend_title='Users',
        hovermode='x unified',
        template='plotly_white'
    )

    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=45)

    if save_as_file:
        valid_formats = ["png", "jpeg", "pdf", "svg"]
        if file_format not in valid_formats:
            file_format = "png"
        file_path = f"{file_name}.{file_format}"
        try:
            fig.write_image(file_path, scale=2)
            logger.info(f"Chart saved as {file_path}.")
            return file_path
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return None
    else:
        fig.show()
        return None