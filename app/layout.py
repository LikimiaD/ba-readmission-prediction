"""Компоновка страницы, AppShell + 2-колоночный грид + аккордеон + FAB"""
import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify

from app.components import (
    render_export_buttons,
    render_metrics_table,
    render_patient_data_card,
    render_patient_selector,
    render_risk_card,
    render_risk_distribution,
    render_shap_waterfall,
    render_what_if_sliders,
)
from app.data_loader import ArtifactBundle


def build_layout(bundle: ArtifactBundle):
    return dmc.MantineProvider(
        id='mantine-provider',
        defaultColorScheme='light',
        forceColorScheme='light',
        theme={
            'primaryColor': 'indigo',
            'fontFamily': 'Inter, system-ui, -apple-system, sans-serif',
            'headings': {'fontFamily': 'Inter, system-ui, sans-serif', 'fontWeight': '600'},
            'defaultRadius': 'md',
        },
        children=[
            dcc.Store(id='store-base-proba', data=None),
            dcc.Store(id='store-color-scheme', data='light'),
            dmc.AppShell(
                header={'height': 64},
                padding='md',
                children=[
                    dmc.AppShellHeader(_build_header(bundle)),
                    dmc.AppShellMain(_build_main(bundle)),
                ],
            ),
            render_export_buttons(),
            dmc.NotificationProvider(),
            dmc.Box(id='notifications-container'),
        ],
    )


def build_unavailable_layout(message):
    return dmc.MantineProvider(
        theme={'primaryColor': 'indigo'},
        children=dmc.Container(
            size='sm',
            pt='xl',
            children=dmc.Alert(
                title='Дашборд недоступен',
                color='red',
                variant='light',
                icon=DashIconify(icon='tabler:alert-triangle', width=22),
                children=dmc.Stack(
                    gap='xs',
                    children=[
                        dmc.Text(message),
                        dmc.Text(
                            'Сначала запустите `make all` для генерации артефактов, '
                            'затем перезапустите дашборд командой `make dashboard`.',
                            size='sm',
                        ),
                    ],
                ),
            ),
        ),
    )


def _build_header(bundle):
    catboost_meta = (bundle.metrics.get('catboost') or {})
    version = 'v1.0'
    roc = (catboost_meta.get('test') or catboost_meta.get('val') or {}).get('roc_auc')

    return dmc.Group(
        h='100%',
        px='lg',
        justify='space-between',
        align='center',
        children=[
            DashIconify(
                icon='tabler:stethoscope', width=28,
                color='var(--mantine-color-indigo-6)',
            ),
            dmc.Group(
                gap='xs',
                children=[
                    dmc.Badge(
                        f'CatBoost · {version}',
                        color='indigo',
                        variant='light',
                        leftSection=DashIconify(icon='tabler:brain', width=14),
                    ),
                    dmc.Badge(
                        f'ROC-AUC {roc:.3f}' if isinstance(roc, (int, float)) else 'ROC-AUC —',
                        color='gray',
                        variant='light',
                    ),
                    dmc.Tooltip(
                        label='Переключить светлую/тёмную тему',
                        children=dmc.ActionIcon(
                            id='theme-toggle',
                            variant='subtle',
                            color='gray',
                            size='lg',
                            children=DashIconify(icon='tabler:sun', width=18),
                        ),
                    ),
                    dmc.Tooltip(
                        label='О проекте',
                        children=dmc.ActionIcon(
                            id='info-toggle',
                            variant='subtle',
                            color='gray',
                            size='lg',
                            children=DashIconify(icon='tabler:info-circle', width=18),
                        ),
                    ),
                    _build_info_modal(bundle),
                ],
            ),
        ],
    )


def _build_info_modal(bundle):
    cat = (bundle.metrics.get('catboost') or {})
    test = cat.get('test') or cat.get('val') or {}
    meta = bundle.metrics.get('meta') or {}
    return dmc.Modal(
        id='info-modal',
        title=dmc.Text('О проекте', fw=600),
        size='md',
        centered=True,
        opened=False,
        children=dmc.Stack(
            gap='sm',
            children=[
                _info_row('ROC-AUC (test)', _fmt(test.get('roc_auc'))),
                _info_row('PR-AUC (test)', _fmt(test.get('pr_auc'))),
                _info_row('F1 (test)', _fmt(test.get('f1'))),
                _info_row('Оптимальный порог', _fmt(cat.get('optimal_threshold'))),
                _info_row('Размер test', str(meta.get('test_size', bundle.n_patients))),
                _info_row('Дата обучения', str(meta.get('run_timestamp_utc', '—'))),
            ],
        ),
    )


def _info_row(label, value):
    return dmc.Group(
        justify='space-between',
        children=[
            dmc.Text(label, size='sm', c='dimmed'),
            dmc.Text(value, size='sm', fw=600),
        ],
    )


def _fmt(v):
    if v is None:
        return '—'
    try:
        return f'{float(v):.4f}'
    except (TypeError, ValueError):
        return str(v)


def _build_main(bundle):
    left_col = dmc.Stack(
        gap='md',
        children=[
            render_patient_selector(bundle),
            render_risk_card(),
            render_patient_data_card(),
        ],
    )
    right_col = dmc.Stack(
        gap='md',
        children=[
            render_shap_waterfall(),
            render_what_if_sliders(),
        ],
    )
    return dmc.Container(
        size='xl',
        px='md',
        pt='md',
        children=[
            dmc.Grid(
                gutter='md',
                children=[
                    dmc.GridCol(left_col, span={'base': 12, 'md': 5}),
                    dmc.GridCol(right_col, span={'base': 12, 'md': 7}),
                ],
            ),
            dmc.Space(h='lg'),
            _build_accordion(bundle),
            dmc.Space(h='xl'),
        ],
    )


def _build_accordion(bundle):
    return dmc.Accordion(
        chevronPosition='right',
        variant='separated',
        multiple=True,
        children=[
            dmc.AccordionItem(
                value='quality',
                children=[
                    dmc.AccordionControl(
                        dmc.Group(
                            gap='xs',
                            children=[
                                DashIconify(icon='tabler:target', width=18,
                                            color='var(--mantine-color-indigo-6)'),
                                dmc.Text('Как точна модель', fw=600),
                            ],
                        )
                    ),
                    dmc.AccordionPanel(render_metrics_table(bundle)),
                ],
            ),
            dmc.AccordionItem(
                value='distribution',
                children=[
                    dmc.AccordionControl(
                        dmc.Group(
                            gap='xs',
                            children=[
                                DashIconify(icon='tabler:chart-histogram', width=18,
                                            color='var(--mantine-color-indigo-6)'),
                                dmc.Text('Риск среди всех пациентов', fw=600),
                            ],
                        )
                    ),
                    dmc.AccordionPanel(render_risk_distribution(bundle)),
                ],
            ),
        ],
    )
