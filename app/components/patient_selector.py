"""Выбор пациента: dmc.Card с searchable Select и двумя shortcut-кнопками"""
import dash_mantine_components as dmc
from dash_iconify import DashIconify


def _select_data(bundle):
    return [
        {'label': f'Пациент #{i}', 'value': str(i)}
        for i in range(bundle.n_patients)
    ]


def render_patient_selector(bundle):
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
                    dmc.Text('Выбор пациента', fw=600, size='lg'),
                    DashIconify(icon='tabler:users', width=20, color='var(--mantine-color-gray-6)'),
                ],
            ),
            dmc.Space(h='sm'),
            dmc.Select(
                id='patient-select',
                placeholder='Начните вводить номер пациента…',
                data=_select_data(bundle),
                searchable=True,
                clearable=False,
                nothingFoundMessage='Ничего не найдено',
                maxDropdownHeight=280,
                leftSection=DashIconify(icon='tabler:id', width=16),
            ),
            dmc.Space(h='md'),
            dmc.Group(
                grow=True,
                children=[
                    dmc.Button(
                        'Пример высокого риска',
                        id='btn-high-risk',
                        color='red',
                        variant='light',
                        leftSection=DashIconify(icon='tabler:alert-triangle', width=16),
                    ),
                    dmc.Button(
                        'Пример низкого риска',
                        id='btn-low-risk',
                        color='green',
                        variant='light',
                        leftSection=DashIconify(icon='tabler:shield-check', width=16),
                    ),
                ],
            ),
        ],
    )
