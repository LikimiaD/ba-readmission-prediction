"""Таблица метрик и распределение рисков, наполнение нижнего аккордеона"""
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import dcc


_MODEL_ORDER = (
    'baseline',
    'random_forest',
    'catboost',
    'lightgbm',
    'xgboost',
)

_MODEL_LABELS = {
    'baseline': 'Logistic (baseline)',
    'random_forest': 'Random Forest',
    'catboost': 'CatBoost',
    'lightgbm': 'LightGBM',
    'xgboost': 'XGBoost',
}


def render_metrics_table(bundle):
    header = [dmc.TableThead(dmc.TableTr([
        dmc.TableTh('Модель'),
        dmc.TableTh('ROC-AUC'),
        dmc.TableTh('PR-AUC'),
        dmc.TableTh('F1 @ порог'),
        dmc.TableTh('Brier'),
    ]))]

    body_rows = []
    for name in _MODEL_ORDER:
        block = bundle.metrics.get(name)
        if not block:
            continue
        split = block.get('test') or block.get('val') or {}
        if not split:
            continue
        is_main = name == 'catboost'
        cells = [
            dmc.TableTd(dmc.Group(gap=6, children=[
                dmc.Text(_MODEL_LABELS.get(name, name), fw=700 if is_main else 400),
                dmc.Badge('основная', size='xs', color='indigo', variant='light') if is_main else None,
            ])),
            dmc.TableTd(_fmt4(split.get('roc_auc'))),
            dmc.TableTd(_fmt4(split.get('pr_auc'))),
            dmc.TableTd(_fmt4(split.get('f1'))),
            dmc.TableTd(_fmt4(split.get('brier'))),
        ]
        body_rows.append(dmc.TableTr(cells, className='metrics-row-main' if is_main else None))

    body = [dmc.TableTbody(body_rows or [dmc.TableTr(dmc.TableTd(
        'Метрики ещё не собраны — запустите `make train`.', colSpan=5,
    ))])]

    return dmc.Stack(
        gap='sm',
        children=[
            dmc.Text(
                'Пять моделей сошлись в узком окне ROC-AUC ≈ 0.656–0.659 на тесте — '
                'виден структурный потолок, диктуемый самим набором признаков.',
                size='sm',
                c='dimmed',
            ),
            dmc.Table(
                withTableBorder=True,
                withRowBorders=True,
                highlightOnHover=True,
                striped=True,
                children=header + body,
            ),
        ],
    )


def render_risk_distribution(bundle):
    proba = bundle.proba_test
    thr = bundle.threshold
    above = int((proba >= thr).sum())
    share = above / max(len(proba), 1)

    y = bundle.y_test.to_numpy()
    pred = (proba >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)

    edges = np.linspace(0.0, 1.0, 41)
    mask_below = proba < thr
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=proba[mask_below],
        xbins={'start': 0.0, 'end': 1.0, 'size': edges[1] - edges[0]},
        name='Ниже порога',
        marker_color='#2f9e44',
        opacity=0.8,
    ))
    fig.add_trace(go.Histogram(
        x=proba[~mask_below],
        xbins={'start': 0.0, 'end': 1.0, 'size': edges[1] - edges[0]},
        name='Выше порога',
        marker_color='#e03131',
        opacity=0.8,
    ))
    fig.add_vline(x=thr, line_width=2, line_dash='dash', line_color='#555',
                  annotation_text=f'thr={thr:.2f}', annotation_position='top')
    fig.update_layout(
        barmode='overlay',
        height=300,
        margin={'l': 10, 'r': 10, 't': 30, 'b': 30},
        xaxis_title='Предсказанная вероятность',
        yaxis_title='Пациентов',
        legend={'orientation': 'h', 'y': -0.25},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, system-ui, sans-serif', 'size': 12},
    )
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.15)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.15)')

    summary = (
        f'Выше порога {thr:.2f}: {above} из {len(proba)} пациентов '
        f'({share * 100:.1f}%). Recall={recall:.2f}, precision={precision:.2f}.'
    )
    return dmc.Stack(
        gap='sm',
        children=[
            dcc.Graph(figure=fig, config={'displayModeBar': False, 'responsive': True}),
            dmc.Text(summary, size='sm', c='dimmed'),
        ],
    )


def _fmt4(v):
    if v is None:
        return '—'
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return '—'


def render_model_quality(bundle):
    return dmc.Stack(
        gap='lg',
        children=[
            render_metrics_table(bundle),
            render_risk_distribution(bundle),
        ],
    )
