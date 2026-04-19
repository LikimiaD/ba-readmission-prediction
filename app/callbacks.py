"""Dash-коллбэки: выбор пациента, what-if, экспорт, модалки"""
import io
import logging
from datetime import datetime

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback_context, dcc, no_update
from dash_iconify import DashIconify

from app.components.patient_data_card import build_patient_data_rows
from app.components.risk_card import build_risk_card_body
from app.components.shap_waterfall import build_waterfall_figure
from app.components.what_if_sliders import dropdown_specs, slider_specs
from app.data_loader import ArtifactBundle
from app.explainer import (
    format_explanation_ru,
    humanize_feature,
    patient_shap,
    score_modified_patient,
)
from app.ui import RiskCategory

logger = logging.getLogger(__name__)

_RNG = np.random.default_rng(0)
_EXTREME_BUCKET = 0.10


def register_callbacks(app, bundle: ArtifactBundle):
    slider_ids = [spec['id'] for spec in slider_specs()]
    slider_cols = [spec['column'] for spec in slider_specs()]
    dropdown_ids = [spec['id'] for spec in dropdown_specs()]
    dropdown_cols = [spec['column'] for spec in dropdown_specs()]
    slider_baseline_ids = [f'{sid}-baseline' for sid in slider_ids]

    @app.callback(
        Output('patient-select', 'value'),
        Input('btn-high-risk', 'n_clicks'),
        Input('btn-low-risk', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _shortcut_pick(_high_clicks, _low_clicks):
        trigger = callback_context.triggered[0]['prop_id'] if callback_context.triggered else ''
        proba = bundle.proba_test
        n = len(proba)
        k = max(1, int(_EXTREME_BUCKET * n))
        order = np.argsort(proba)
        pool = order[-k:] if 'btn-high-risk' in trigger else order[:k]
        return str(int(_RNG.choice(pool)))

    @app.callback(
        Output('risk-card-content', 'children'),
        Output('shap-waterfall', 'figure'),
        Output('shap-explanation', 'children'),
        Output('patient-data-grid', 'children'),
        Output('store-base-proba', 'data'),
        Output('whatif-delta-badge', 'children'),
        Output('whatif-delta-badge', 'color'),
        *[Output(sid, 'value') for sid in slider_ids],
        *[Output(did, 'value') for did in dropdown_ids],
        *[Output(bid, 'children') for bid in slider_baseline_ids],
        Input('patient-select', 'value'),
    )
    def _on_patient_change(patient_value):
        if patient_value is None or patient_value == '':
            return (no_update,) * (7 + len(slider_ids) + len(dropdown_ids) + len(slider_baseline_ids))

        idx = int(patient_value)
        p = patient_shap(bundle, idx)
        category = RiskCategory.from_probability(p.prediction)
        row = bundle.X_test.iloc[idx]

        card_body = build_risk_card_body(
            probability=p.prediction,
            category=category,
            age=str(row.get('age', '—')),
            n_inpatient=int(row.get('n_inpatient', 0)),
            diag_1=str(row.get('diag_1', '—')),
        )
        figure = build_waterfall_figure(p)
        explanation = format_explanation_ru(p)
        data_rows = build_patient_data_rows(row)

        slider_values = [int(row[col]) for col in slider_cols]
        dropdown_values = [str(row[col]) for col in dropdown_cols]
        baseline_texts = [
            f'Текущее значение пациента: {int(row[col])}'
            for col in slider_cols
        ]

        return (card_body, figure, explanation, data_rows, float(p.prediction),
                '— изменений нет —', 'gray',
                *slider_values, *dropdown_values, *baseline_texts)

    @app.callback(
        Output('risk-card-content', 'children', allow_duplicate=True),
        Output('whatif-delta-badge', 'children', allow_duplicate=True),
        Output('whatif-delta-badge', 'color', allow_duplicate=True),
        *[Input(sid, 'value') for sid in slider_ids],
        *[Input(did, 'value') for did in dropdown_ids],
        State('patient-select', 'value'),
        State('store-base-proba', 'data'),
        prevent_initial_call=True,
    )
    def _on_whatif_change(*args):
        n_slide = len(slider_ids)
        n_drop = len(dropdown_ids)
        slider_values = args[:n_slide]
        dropdown_values = args[n_slide:n_slide + n_drop]
        patient_value = args[n_slide + n_drop]
        base_proba = args[n_slide + n_drop + 1]

        if patient_value is None or patient_value == '' or base_proba is None:
            return no_update, no_update, no_update
        idx = int(patient_value)

        overrides = {}
        extrapolation = None
        for col, val in zip(slider_cols, slider_values):
            if val is None:
                continue
            overrides[col] = int(val)
            col_series = bundle.X_test[col]
            if val > col_series.quantile(0.995) or val < col_series.quantile(0.005):
                extrapolation = (
                    'Экстраполяция за пределы обучающего распределения — '
                    'прогноз ненадёжен.'
                )
        for col, val in zip(dropdown_cols, dropdown_values):
            if val is None:
                continue
            overrides[col] = str(val)

        new_proba = score_modified_patient(bundle, idx, overrides)
        category = RiskCategory.from_probability(new_proba)
        row = bundle.X_test.iloc[idx]
        delta = new_proba - float(base_proba)

        card_body = build_risk_card_body(
            probability=new_proba,
            category=category,
            age=str(row.get('age', '—')),
            n_inpatient=int(row.get('n_inpatient', 0)),
            diag_1=str(row.get('diag_1', '—')),
            delta=delta,
            extrapolation_note=extrapolation,
        )
        badge_text, badge_color = _delta_badge(delta)
        return card_body, badge_text, badge_color

    @app.callback(
        *[Output(sid, 'value', allow_duplicate=True) for sid in slider_ids],
        *[Output(did, 'value', allow_duplicate=True) for did in dropdown_ids],
        Input('whatif-reset', 'n_clicks'),
        State('patient-select', 'value'),
        prevent_initial_call=True,
    )
    def _reset_whatif(_n_clicks, patient_value):
        n_total = len(slider_ids) + len(dropdown_ids)
        if patient_value is None or patient_value == '':
            return (no_update,) * n_total
        row = bundle.X_test.iloc[int(patient_value)]
        slider_vals = [int(row[col]) for col in slider_cols]
        dropdown_vals = [str(row[col]) for col in dropdown_cols]
        return (*slider_vals, *dropdown_vals)

    @app.callback(
        Output('download-csv', 'data'),
        Input('menu-export-csv', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _export_csv(_n_clicks):
        df = _build_export_frame(bundle)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return dcc.send_data_frame(df.to_csv, f'readmission_predictions_{ts}.csv', index=False)

    @app.callback(
        Output('download-xlsx', 'data'),
        Input('menu-export-xlsx', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _export_xlsx(_n_clicks):
        df = _build_export_frame(bundle)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='predictions')
            ws = writer.sheets['predictions']
            for col_idx, col in enumerate(df.columns, start=1):
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(
                    14, min(40, int(df[col].astype(str).map(len).max()) + 2)
                )
        buf.seek(0)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return dcc.send_bytes(buf.getvalue(), f'readmission_predictions_{ts}.xlsx')

    @app.callback(
        Output('iframe-modal', 'opened'),
        Input('menu-embed-iframe', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _open_iframe_modal(_n_clicks):
        return True

    @app.callback(
        Output('info-modal', 'opened'),
        Input('info-toggle', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _open_info_modal(_n_clicks):
        return True

    @app.callback(
        Output('notifications-container', 'children'),
        Input('menu-health', 'n_clicks'),
        prevent_initial_call=True,
    )
    def _show_health(_n_clicks):
        msg = (
            f'Модель CatBoost загружена, пациентов в тесте: {bundle.n_patients}. '
            f'SHAP-кеш активен.'
        )
        return dmc.Notification(
            id='health-notification',
            title='Статус системы',
            message=msg,
            color='green',
            icon=DashIconify(icon='tabler:heartbeat', width=18),
            action='show',
            autoClose=4000,
        )

    app.clientside_callback(
        """
        function(n_clicks, current) {
            if (!n_clicks) { return current; }
            const next = (current === 'dark') ? 'light' : 'dark';
            document.documentElement.setAttribute('data-mantine-color-scheme', next);
            return next;
        }
        """,
        Output('store-color-scheme', 'data'),
        Input('theme-toggle', 'n_clicks'),
        State('store-color-scheme', 'data'),
        prevent_initial_call=True,
    )


def _delta_badge(delta):
    if abs(delta) < 0.005:
        return '≈ без изменений', 'gray'
    sign = '+' if delta > 0 else '−'
    text = f'{sign}{abs(delta) * 100:.1f} п.п.'
    color = 'red' if delta > 0 else 'green'
    return text, color


def _build_export_frame(bundle):
    proba = bundle.proba_test
    thr = bundle.threshold
    cls = (proba >= thr).astype(int)

    shap_abs = np.abs(bundle.shap_values)
    top3_idx = np.argsort(shap_abs, axis=1)[:, ::-1][:, :3]
    feat_names = np.asarray(bundle.feature_names)

    top1, top2, top3 = [], [], []
    for row in top3_idx:
        names = feat_names[row]
        top1.append(humanize_feature(names[0]))
        top2.append(humanize_feature(names[1]) if len(names) > 1 else '')
        top3.append(humanize_feature(names[2]) if len(names) > 2 else '')

    return pd.DataFrame({
        'patient_id': np.arange(len(proba), dtype=int),
        'probability': np.round(proba, 6),
        'class_at_threshold': cls,
        'top1_shap_feature': top1,
        'top2_shap_feature': top2,
        'top3_shap_feature': top3,
    })
