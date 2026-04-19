"""Primary-блок: RingProgress + Badge + три строки, summary"""
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from app.ui import risk_tooltip_text


_EMPTY_CHILDREN = dmc.Center(
    style={'height': '200px'},
    children=dmc.Stack(
        gap='xs',
        align='center',
        children=[
            DashIconify(
                icon='tabler:mouse-2', width=48,
                color='var(--mantine-color-gray-5)',
            ),
            dmc.Text(
                'Выберите пациента для просмотра риска реадмиссии',
                c='dimmed',
                size='md',
                ta='center',
            ),
        ],
    ),
)


def render_risk_card():
    return dmc.Card(
        withBorder=True,
        shadow='md',
        radius='md',
        p='xl',
        id='risk-card',
        style={'minHeight': '240px'},
        children=[
            dmc.Stack(
                gap='md',
                children=[
                    dmc.Group(
                        justify='space-between',
                        children=[
                            dmc.Text('Риск реадмиссии', fw=600, size='lg'),
                            DashIconify(
                                icon='tabler:activity-heartbeat', width=20,
                                color='var(--mantine-color-indigo-6)',
                            ),
                        ],
                    ),
                    dmc.Box(id='risk-card-content', children=_EMPTY_CHILDREN),
                ],
            ),
        ],
    )


def build_risk_card_body(
    probability, category, age, n_inpatient, diag_1,
    delta=None, extrapolation_note=None,
):
    pct = probability * 100.0
    ring = dmc.RingProgress(
        sections=[{'value': pct, 'color': category.color}],
        label=dmc.Center(
            dmc.Stack(
                gap=0,
                align='center',
                children=[
                    dmc.Text(f'{pct:.0f}%', fw=700, size='xl',
                             style={'fontSize': '34px', 'lineHeight': 1}),
                    dmc.Text('вероятность', size='xs', c='dimmed'),
                ],
            )
        ),
        size=170,
        thickness=14,
    )

    badge = dmc.Tooltip(
        label=risk_tooltip_text(probability, category),
        multiline=True,
        w=260,
        withArrow=True,
        children=dmc.Badge(
            category.label,
            size='xl',
            color=category.color,
            variant='light',
            radius='sm',
            leftSection=DashIconify(icon='tabler:alert-circle', width=14),
        ),
    )

    summary_rows = [
        _kv_row('Возрастная группа', age),
        _kv_row('Прошлых стационаров', str(n_inpatient)),
        _kv_row('Основной диагноз', diag_1),
    ]
    if delta is not None:
        summary_rows.append(_kv_row(
            'Δ от базового предсказания',
            _delta_text(delta),
            value_color=_delta_color(delta),
        ))
    if extrapolation_note:
        summary_rows.append(
            dmc.Alert(
                extrapolation_note,
                color='red',
                variant='light',
                icon=DashIconify(icon='tabler:alert-triangle', width=16),
                p='xs',
            )
        )

    right = dmc.Stack(gap='xs', children=[badge, *summary_rows])
    return [
        dmc.Group(
            justify='space-around',
            align='center',
            wrap='nowrap',
            children=[ring, right],
        )
    ]


def _kv_row(label, value, value_color=None):
    value_kwargs = {'fw': 600, 'size': 'sm'}
    if value_color:
        value_kwargs['c'] = value_color
    return dmc.Group(
        justify='space-between',
        gap='xs',
        children=[
            dmc.Text(label, size='sm', c='dimmed'),
            dmc.Text(value, **value_kwargs),
        ],
    )


def _delta_text(delta):
    sign = '+' if delta >= 0 else '−'
    return f'{sign}{abs(delta) * 100:.1f} п.п.'


def _delta_color(delta):
    if abs(delta) < 0.005:
        return 'gray'
    return 'red' if delta > 0 else 'green'
