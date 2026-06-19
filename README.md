markdown
# Ozon Customer Churn Prediction

Выпускная квалификационная работа на тему «Анализ и совершенствование интернет-маркетинга на основе технологий искусственного интеллекта (на примере маркетплейса Ozon)».  
Проект содержит полностью воспроизводимый пайплайн прогнозирования оттока клиентов: от генерации синтетического датасета до визуализации SHAP-значений и расчёта экономической эффективности.

## Структура репозитория
.
├── generate_realistic_ozon.py # Модуль генерации синтетических данных Ozon
├── generate_new_data.py # Скрипт для создания итогового CSV
├── appendix_b.py # SHAP-визуализации (Приложение Б)
├── appendix_c.py # Метрики качества и тест МакНемара (Приложение В)
├── appendix_g.py # Кривые обучения и матрицы ошибок (Приложение Г)
├── requirements.txt # Зависимости Python
├── .gitignore
└── README.md

text

## Требования

- **Python 3.10** (строго, так как часть библиотек нестабильна на более новых версиях)
- pip
- Git (для клонирования)

## Инструкция по воспроизведению

### 1. Клонируйте репозиторий

Откройте терминал и выполните:

```bash
git clone https://github.com/your-username/ozon-churn-prediction.git
cd ozon-churn-prediction
2. Создайте виртуальное окружение
bash
# Windows
py -3.10 -m venv venv

# macOS / Linux
python3.10 -m venv venv
Активируйте окружение:

Windows (PowerShell): venv\Scripts\activate

Windows (cmd): venv\Scripts\activate.bat

macOS/Linux: source venv/bin/activate

Если PowerShell запрещает выполнение скриптов, используйте прямой вызов интерпретатора:

bash
# Вместо активации просто подставляйте venv\Scripts\python (Windows) или venv/bin/python (macOS/Linux)
venv\Scripts\python -m pip install -r requirements.txt
3. Установите зависимости
bash
pip install -r requirements.txt
Если отдельные пакеты не устанавливаются, попробуйте без фиксации версий:

bash
pip install pandas numpy scikit-learn catboost shap matplotlib seaborn mlflow statsmodels
4. Сгенерируйте синтетический датасет
Скрипт generate_new_data.py создаёт 100 000 синтетических клиентов, параметры которых откалиброваны по публичной отчётности Ozon (средний чек, частота заказов, доля оттока и др.). Результат сохраняется в ozon_features_full.csv.

bash
python generate_new_data.py
Генерация занимает 2–5 минут. В консоль будет выведено:

text
Генерация реалистичного датасета Ozon (100 000 клиентов)...
Новый датасет сохранён в ozon_features_full.csv
Размер: 100000 записей, 11 столбцов
Доля оттока: 25.36%
Файл ozon_features_full.csv (~10‑15 МБ) не включён в репозиторий, чтобы не раздувать его размер. При необходимости его всегда можно пересоздать.

5. Запустите скрипты приложений
Приложение Б – SHAP-визуализации

bash
python appendix_b.py
Создаёт файлы:

fig_shap_bar_ozon.png – важность признаков

fig_shap_beeswarm_ozon.png – распределение SHAP-значений

fig_shap_waterfall_ozon.png – объяснение для одного клиента

Приложение В – таблицы метрик

bash
python appendix_c.py
В консоль выводятся три таблицы:

метрики кросс-валидации

метрики на тестовой выборке

результаты теста МакНемара

Приложение Г – кривые обучения и матрицы ошибок

bash
python appendix_g.py
Создаёт файлы:

fig_learning_curve_логистическая_регрессия.png

fig_learning_curve_случайный_лес.png

fig_learning_curve_catboost.png

fig_confusion_matrix_логистическая_регрессия.png

fig_confusion_matrix_случайный_лес.png

fig_confusion_matrix_catboost.png

6. (Опционально) GPU-ускорение для CatBoost
Если на вашей машине установлена видеокарта NVIDIA и драйверы CUDA, можно включить GPU‑обучение. Для этого в скриптах appendix_b.py, appendix_c.py и appendix_g.py найдите строку с созданием CatBoostClassifier и раскомментируйте параметр task_type='GPU', devices='0'. Ускорение заметно на больших выборках (100 000 записей), но не обязательно для воспроизведения.

Как процитировать
Если вы используете этот код в своей работе, пожалуйста, укажите ссылку на репозиторий и автора ВКР.