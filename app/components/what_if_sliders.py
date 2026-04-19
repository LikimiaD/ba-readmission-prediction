"""Что если изменить, sliders/selects + delta-бейдж + reset"""
import dash_mantine_components as dmc
from dash_iconify import DashIconify


_SLIDERS = [
    {'id': 'whatif-n-inpatient', 'column': 'n_inpatient',
     'label': 'Прошлых стационаров', 'short': 'n_inpatient', 'min': 0, 'max': 10},
    {'id': 'whatif-n-medications', 'column': 'n_medications',
     'label': 'Назначено препаратов', 'short': 'n_medications', 'min': 1, 'max': 50},
    {'id': 'whatif-time-in-hospital', 'column': 'time_in_hospital',
     'label': 'Длительность госпитализации', 'short': 'дни', 'min': 1, 'max': 14},
]

_SELECTS = [
    {'id': 'whatif-a1c', 'column': 'A1Ctest', 'label': 'Тест HbA1c',
     'options': ['no', 'normal', 'high']},
    {'id': 'whatif-diabmed', 'column': 'diabetes_med',
     'label': 'Противодиабетическая терапия', 'options': ['no', 'yes']},
]


def _slider_block(spec):
    label_row = dmc.Group(
        justify='space-between',
        align='flex-end',
        gap='xs',
        children=[
            dmc.Text(spec['label'], size='sm', fw=500),
            dmc.Text(spec['short'], size='xs', c='dimmed'),
        ],
    )
    slider = dmc.Box(
        px='xs',
        pt=28,
        pb=4,
        children=dmc.Slider(
            id=spec['id'],
            min=spec['min'],
            max=spec['max'],
            step=1,
            value=spec['min'],
            marks=[
                {'value': spec['min'], 'label': str(spec['min'])},
                {'value': spec['max'], 'label': str(spec['max'])},
            ],
            labelAlwaysOn=True,
            color='indigo',
            size='md',
        ),
    )
    baseline = dmc.Text(
        id=f"{spec['id']}-baseline",
        size='xs',
        c='dimmed',
        children='Текущее значение пациента: —',
    )
    return dmc.Stack(
        gap='xs',
        children=[label_row, slider, dmc.Space(h='sm'), baseline],
    )


def render_what_if_sliders():
    sliders = [_slider_block(spec) for spec in _SLIDERS]
    selects = [
        dmc.Select(
            id=spec['id'],
            label=spec['label'],
            data=[{'label': o, 'value': o} for o in spec['options']],
            clearable=False,
            size='sm',
        )
        for spec in _SELECTS
    ]
    return dmc.Card(
        withBorder=True,
        shadow='sm',
        radius='md',
        p='lg',
        children=[
            dmc.Group(
                justify='space-between',
                align='center',
                children=[
                    dmc.Text('Что если изменить', fw=600, size='lg'),
                    dmc.Tooltip(
                        label='Меняйте значения ключевых признаков, чтобы увидеть '
                              'как изменится прогноз. SHAP-объяснение не пересчитывается.',
                        multiline=True,
                        w=260,
                        withArrow=True,
                        children=DashIconify(
                            icon='tabler:info-circle', width=20,
                            color='var(--mantine-color-gray-6)',
                        ),
                    ),
                ],
            ),
            dmc.Space(h='md'),
            dmc.SimpleGrid(
                cols={'base': 1, 'md': 3},
                spacing='xl',
                verticalSpacing='xl',
                children=sliders,
            ),
            dmc.Divider(my='md', variant='dashed'),
            dmc.SimpleGrid(
                cols={'base': 1, 'sm': 2},
                spacing='md',
                children=selects,
            ),
            dmc.Space(h='lg'),
            dmc.Paper(
                withBorder=True,
                p='md',
                radius='md',
                bg='var(--mantine-color-gray-0)',
                children=dmc.Group(
                    justify='space-between',
                    align='center',
                    wrap='nowrap',
                    children=[
                        dmc.Stack(
                            gap=2,
                            children=[
                                dmc.Text('Эффект изменений', size='sm', c='dimmed'),
                                dmc.Badge(
                                    id='whatif-delta-badge',
                                    children='—',
                                    size='xl',
                                    color='gray',
                                    variant='light',
                                    radius='sm',
                                    leftSection=DashIconify(icon='tabler:arrows-diff', width=14),
                                ),
                            ],
                        ),
                        dmc.Button(
                            'Сбросить',
                            id='whatif-reset',
                            n_clicks=0,
                            variant='subtle',
                            color='gray',
                            leftSection=DashIconify(icon='tabler:refresh', width=16),
                        ),
                    ],
                ),
            ),
        ],
    )


def slider_specs():
    return list(_SLIDERS)


def dropdown_specs():
    return list(_SELECTS)
