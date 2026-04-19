"""Тонкая обёртка над кешем SHAP и пересчёт what-if"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.catboost_model import predict_proba

from app.data_loader import ArtifactBundle

logger = logging.getLogger(__name__)


_HUMAN_NAMES = {
    'n_inpatient': 'число прошлых стационарных визитов',
    'n_outpatient': 'число прошлых амбулаторных визитов',
    'n_emergency': 'число прошлых визитов в неотложку',
    'n_medications': 'количество назначенных лекарств',
    'n_procedures': 'количество процедур',
    'n_lab_procedures': 'количество лабораторных анализов',
    'time_in_hospital': 'длительность текущей госпитализации',
    'age': 'возрастная группа',
    'diag_1': 'основной диагноз',
    'diag_2': 'второй диагноз',
    'diag_3': 'третий диагноз',
    'A1Ctest': 'тест на HbA1c',
    'glucose_test': 'тест на глюкозу',
    'diabetes_med': 'назначение противодиабетических препаратов',
    'change': 'смена медикации',
    'medical_specialty': 'специальность врача',
    'total_prior_visits': 'суммарные прежние визиты',
    'emergency_ratio': 'доля неотложных визитов',
    'meds_per_day': 'интенсивность медикаментозной терапии',
    'procedures_per_day': 'интенсивность процедур',
    'labs_per_day': 'интенсивность лаб. анализов',
    'had_emergency_visit': 'был ли визит в неотложку',
    'frequent_inpatient': 'флаг частых госпитализаций',
    'n_distinct_diagnoses': 'число различных диагнозов',
    'poorly_controlled_diabetes': 'плохо контролируемый диабет',
}


@dataclass
class PatientShap:
    feature_names: list[str]
    feature_values: list[object]
    shap_values: np.ndarray
    base_value: float
    prediction: float

    @property
    def top_signed(self):
        order = np.argsort(np.abs(self.shap_values))[::-1]
        return [
            (self.feature_names[i], float(self.shap_values[i]), self.feature_values[i])
            for i in order
        ]


def humanize_feature(name):
    return _HUMAN_NAMES.get(name, name)


def patient_shap(bundle: ArtifactBundle, patient_idx: int) -> PatientShap:
    row = bundle.X_test.iloc[patient_idx]
    values = [row[col] for col in bundle.feature_names]
    return PatientShap(
        feature_names=list(bundle.feature_names),
        feature_values=values,
        shap_values=bundle.shap_values[patient_idx].copy(),
        base_value=float(bundle.shap_base_value),
        prediction=float(bundle.proba_test[patient_idx]),
    )


def score_modified_patient(bundle: ArtifactBundle, patient_idx: int, overrides: dict) -> float:
    row = bundle.X_test.iloc[patient_idx].copy()
    for col, val in overrides.items():
        if val is None or val == '':
            continue
        row[col] = val
    frame = pd.DataFrame([row], columns=bundle.X_test.columns)
    proba = predict_proba(bundle.model, frame, bundle.feature_types)
    return float(proba[0])


def format_explanation_ru(patient: PatientShap, top_k: int = 3) -> str:
    top = patient.top_signed[:top_k]
    if not top:
        return 'Недостаточно данных для построения объяснения.'

    first = top[0]
    parts = [
        (
            'Вероятность реадмиссии '
            + ('повышена' if first[1] > 0 else 'понижена')
            + ' преимущественно из-за признака «'
            + humanize_feature(first[0])
            + f'» (значение: {_format_value(first[2])}).'
        )
    ]
    if len(top) > 1:
        second = top[1]
        parts.append(
            'Дополнительный '
            + ('повышающий' if second[1] > 0 else 'понижающий')
            + ' вклад вносит «'
            + humanize_feature(second[0])
            + f'» (значение: {_format_value(second[2])}).'
        )
    if len(top) > 2:
        third = top[2]
        parts.append(
            'Умеренный '
            + ('повышающий' if third[1] > 0 else 'понижающий')
            + ' эффект даёт «'
            + humanize_feature(third[0])
            + f'» (значение: {_format_value(third[2])}).'
        )
    return ' '.join(parts)


def _format_value(value):
    if isinstance(value, float):
        if np.isnan(value):
            return '—'
        if value.is_integer():
            return str(int(value))
        return f'{value:.2f}'
    return str(value)
