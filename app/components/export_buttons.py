"""Floating action button с меню экспорта и модалкой iframe"""
import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify


def render_export_buttons():
    menu = dmc.Menu(
        position='top-end',
        shadow='md',
        width=260,
        children=[
            dmc.MenuTarget(
                dmc.ActionIcon(
                    id='fab-export',
                    variant='filled',
                    color='indigo',
                    size='xl',
                    radius='xl',
                    children=DashIconify(icon='tabler:download', width=22),
                ),
            ),
            dmc.MenuDropdown(
                children=[
                    dmc.MenuLabel('Экспорт и интеграция'),
                    dmc.MenuItem(
                        'Скачать предсказания (CSV)',
                        id='menu-export-csv',
                        leftSection=DashIconify(icon='tabler:file-type-csv', width=16),
                    ),
                    dmc.MenuItem(
                        'Скачать предсказания (Excel)',
                        id='menu-export-xlsx',
                        leftSection=DashIconify(icon='tabler:file-spreadsheet', width=16),
                    ),
                    dmc.MenuDivider(),
                    dmc.MenuItem(
                        'Встроить в BI-систему',
                        id='menu-embed-iframe',
                        leftSection=DashIconify(icon='tabler:code', width=16),
                    ),
                    dmc.MenuItem(
                        'Статус системы',
                        id='menu-health',
                        leftSection=DashIconify(icon='tabler:heartbeat', width=16),
                    ),
                ]
            ),
        ],
    )

    modal = dmc.Modal(
        id='iframe-modal',
        title=dmc.Group(
            gap='xs',
            children=[
                DashIconify(icon='tabler:code', width=18),
                dmc.Text('Встраивание в BI-систему', fw=600),
            ],
        ),
        size='lg',
        centered=True,
        opened=False,
        children=[
            dmc.CodeHighlight(
                id='iframe-modal-code',
                code=(
                    '<iframe\n'
                    '    src="http://host:8050"\n'
                    '    width="100%"\n'
                    '    height="900"\n'
                    '    frameborder="0">\n'
                    '</iframe>'
                ),
                language='html',
            ),
        ],
    )

    return dmc.Affix(
        position={'bottom': 24, 'right': 24},
        children=[
            dcc.Download(id='download-csv'),
            dcc.Download(id='download-xlsx'),
            menu,
            modal,
        ],
    )
