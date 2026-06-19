# Ozon Customer Churn Prediction

Выпускная квалификационная работа: прогнозирование оттока клиентов маркетплейса на основе поведенческих данных с использованием CatBoost и Яндекс.Метрики.

## Структура проекта

- `generate_realistic_ozon.py` — генератор синтетического датасета
- `appendix_b.py` — SHAP-визуализации
- `appendix_c.py` — расчёт метрик качества
- `appendix_g.py` — кривые обучения и матрицы ошибок
- `requirements.txt` — зависимости

## Запуск

1. Установите зависимости: `pip install -r requirements.txt`
2. Сгенерируйте данные: `python generate_new_data.py` (необходим модуль `generate_realistic_ozon.py`)
3. Запустите нужный скрипт: `python appendix_c.py`