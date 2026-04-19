"""Почему такой риск, Plotly waterfall в mantine Card + Alert."""
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import dcc
from dash_iconify import DashIconify

from app.explainer import humanize_feature


_MAX_FEATURES = 10


def render_shap_waterfall():
    return dmc.Card(
        withBorder=True,
        shadow='sm',
        radius='md',
        p='lg',
        children=[
            dmc.Group(
                justify='space-between',
                children=[
                    dmc.Text('Почему такой риск', fw=600, size='lg'),
                    DashIconify(
                        icon='tabler:chart-bar', width=20,
                        color='var(--mantine-color-indigo-6)',
                    ),
                ],
            ),
            dmc.Text(
                'Какие признаки толкают предсказание вверх (красные) и вниз (зелёные).',
                size='sm',
                c='dimmed',
            ),
            dmc.Space(h='md'),
            dmc.Skeleton(
                visible=False,
                id='shap-waterfall-skeleton',
                height=520,
                children=dcc.Graph(
                    id='shap-waterfall',
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '520px', 'width': '100%'},
                ),
            ),
            dmc.Space(h='md'),
            dmc.Alert(
                id='shap-explanation',
                title='Что это значит',
                color='indigo',
                variant='light',
                icon=DashIconify(icon='tabler:bulb', width=18),
                children='Выберите пациента, чтобы увидеть объяснение прогноза.',
            ),
        ],
    )


def build_waterfall_figure(patient):
    order = np.argsort(np.abs(patient.shap_values))[::-1][:_MAX_FEATURES]
    names = [patient.feature_names[i] for i in order][::-1]
    values = [float(patient.shap_values[i]) for i in order][::-1]
    raw_vals = [patient.feature_values[i] for i in order][::-1]

    labels = [
        f'{_truncate(humanize_feature(n), 32)}'
        f"<br><span style='color:#999;font-size:11px'>{_fmt_value(v)}</span>"
        for n, v in zip(names, raw_vals)
    ]

    span = max(abs(v) for v in values) if values else 1.0
    x_pad = max(span * 0.35, 0.15)

    fig = go.Figure(
        go.Waterfall(
            orientation='h',
            measure=['relative'] * len(values),
            y=labels,
            x=values,
            connector={'line': {'color': '#c5cad1', 'dash': 'dot', 'width': 1}},
            increasing={'marker': {'color': '#e03131'}},
            decreasing={'marker': {'color': '#2f9e44'}},
            text=[f'{v:+.3f}' for v in values],
            textposition='outside',
            textfont={'size': 11, 'color': '#444'},
            hovertemplate='<b>%{y}</b><br>SHAP=%{x:+.4f}<extra></extra>',
        )
    )
    fig.update_layout(
        margin={'l': 10, 'r': 20, 't': 10, 'b': 50},
        height=max(360, 42 * len(values) + 80),
        xaxis_title='Вклад в log-odds',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, system-ui, sans-serif', 'size': 12, 'color': '#444'},
        hoverlabel={'bgcolor': 'white', 'bordercolor': '#d0d7de',
                    'font': {'family': 'Inter, system-ui, sans-serif'}},
        bargap=0.35,
    )
    fig.update_xaxes(
        zeroline=True, zerolinecolor='#888',
        gridcolor='rgba(128,128,128,0.15)',
        range=[min(0, min(values)) - x_pad, max(0, max(values)) + x_pad] if values else None,
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor='rgba(128,128,128,0.08)',
        ticklabelposition='outside',
        tickfont={'size': 12},
    )
    return fig


def empty_figure():
    fig = go.Figure()
    fig.add_annotation(
        text='Выберите пациента, чтобы увидеть SHAP-объяснение',
        showarrow=False,
        font={'size': 14, 'color': '#888', 'family': 'Inter, system-ui, sans-serif'},
    )
    fig.update_layout(
        height=420,
        margin={'l': 10, 'r': 10, 't': 20, 'b': 30},
        xaxis={'visible': False},
        yaxis={'visible': False},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _fmt_value(v):
    if isinstance(v, float):
        if np.isnan(v):
            return '—'
        if v.is_integer():
            return str(int(v))
        return f'{v:.2f}'
    return str(v)


def _truncate(text, max_len):
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + '…'
