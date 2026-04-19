"""Карточка `dmc.Card` с восемью ключевыми клиническими полями"""
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from app.ui import PATIENT_CARD_FIELDS, PATIENT_FIELD_LABELS, format_cell_value


def render_patient_data_card():
    return dmc.Card(
        withBorder=True,
        radius='md',
        shadow='sm',
        p='lg',
        children=[
            dmc.Group(
                justify='space-between',
                children=[
                    dmc.Text('Данные пациента', fw=600, size='lg'),
                    DashIconify(
                        icon='tabler:clipboard-text', width=20,
                        color='var(--mantine-color-gray-6)',
                    ),
                ],
            ),
            dmc.Space(h='xs'),
            dmc.Box(id='patient-data-grid', children=_empty_state()),
        ],
    )


def build_patient_data_rows(row):
    cells = []
    for col in PATIENT_CARD_FIELDS:
        if col not in row.index:
            continue
        label = PATIENT_FIELD_LABELS.get(col, col)
        value = format_cell_value(row[col])
        cells.append(
            dmc.Box(
                className='patient-row',
                children=dmc.Stack(
                    gap=2,
                    children=[
                        dmc.Text(label, size='xs', c='dimmed'),
                        dmc.Text(value, size='sm', fw=600),
                    ],
                ),
            )
        )
    return dmc.SimpleGrid(
        cols={'base': 1, 'sm': 2},
        spacing='xs',
        verticalSpacing='xs',
        children=cells,
    )


def _empty_state():
    return dmc.Center(
        style={'height': '120px'},
        children=dmc.Text('—', c='dimmed', size='sm'),
    )
