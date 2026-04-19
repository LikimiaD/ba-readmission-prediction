"""UI-помощники: риск-тиры, русские подписи колонок, форматтеры ячеек"""
from dataclasses import dataclass


RISK_TIERS = (
    (0.25, 'green', 'Низкий риск'),
    (0.50, 'yellow', 'Средний риск'),
    (0.75, 'orange', 'Высокий риск'),
    (1.01, 'red', 'Очень высокий риск'),
)


@dataclass(frozen=True)
class RiskCategory:
    label: str
    color: str

    @classmethod
    def from_probability(cls, p):
        for cutoff, color, label in RISK_TIERS:
            if p < cutoff:
                return cls(label=label, color=color)
        return cls(label='Очень высокий риск', color='red')


PATIENT_FIELD_LABELS = {
    'age': 'Возрастная группа',
    'time_in_hospital': 'Дней в стационаре',
    'n_lab_procedures': 'Лабораторных анализов',
    'n_procedures': 'Процедур',
    'n_medications': 'Назначено препаратов',
    'n_outpatient': 'Амбулаторных визитов',
    'n_inpatient': 'Прошлых стационаров',
    'n_emergency': 'Визитов в неотложку',
    'medical_specialty': 'Специальность врача',
    'diag_1': 'Основной диагноз',
    'diag_2': 'Второй диагноз',
    'diag_3': 'Третий диагноз',
    'A1Ctest': 'Тест HbA1c',
    'glucose_test': 'Тест на глюкозу',
    'change': 'Смена медикации',
    'diabetes_med': 'Назначен противодиабетический препарат',
}

PATIENT_CARD_FIELDS = (
    'age',
    'time_in_hospital',
    'n_medications',
    'n_lab_procedures',
    'n_inpatient',
    'n_emergency',
    'diag_1',
    'A1Ctest',
)


def format_cell_value(value):
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return '—'
    except Exception:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def risk_tooltip_text(probability, category):
    pct = probability * 100.0
    if category.color == 'green':
        tail = 'профилактика по стандартной программе.'
    elif category.color == 'yellow':
        tail = 'рекомендуется усилить наблюдение.'
    elif category.color == 'orange':
        tail = 'требуется дополнительное обследование.'
    else:
        tail = 'требуется немедленное внимание.'
    return f'Вероятность реадмиссии: {pct:.1f}%. {category.label} — {tail}'
