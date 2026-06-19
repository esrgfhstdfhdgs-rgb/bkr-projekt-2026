import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Загрузка реалистичного датасета Ozon
# ------------------------------
print("Загрузка датасета...")
df = pd.read_csv('ozon_features_full.csv')

# Отделяем признаки и целевую переменную
feature_cols_raw = ['Recency', 'Frequency', 'Monetary', 'IntervalCV', 'TrendCheck',
                    'OrdersLast90', 'ReturnRate', 'Category', 'HasPremium']
X_raw = df[feature_cols_raw]
y = df['Churn']

# Очистка: бесконечности и NaN
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
X_raw[numeric_cols] = X_raw[numeric_cols].fillna(0)

# Кодирование категорий
X = pd.get_dummies(X_raw, columns=['Category', 'HasPremium'], drop_first=True)

# Обучение CatBoost
print("Обучение CatBoost...")
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05,
                           cat_features=[], verbose=0, random_seed=42)
model.fit(X, y)

# SHAP-анализ
print("Построение SHAP-визуализаций...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Рисунок Б.1 – Bar-диаграмма
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('Средняя абсолютная важность признаков (Ozon, CatBoost)', fontsize=14)
plt.tight_layout()
plt.savefig('fig_shap_bar_ozon.png', dpi=300)
plt.close()

# Рисунок Б.2 – Beeswarm
shap.summary_plot(shap_values, X, show=False)
plt.title('Распределение SHAP-значений (Ozon, CatBoost)', fontsize=14)
plt.tight_layout()
plt.savefig('fig_shap_beeswarm_ozon.png', dpi=300)
plt.close()

# Рисунок Б.3 – Waterfall для первого клиента
shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                     base_values=explainer.expected_value,
                                     data=X.iloc[0].values,
                                     feature_names=X.columns),
                   show=False)
plt.title('Waterfall-диаграмма для клиента (Ozon, CatBoost)', fontsize=14)
plt.tight_layout()
plt.savefig('fig_shap_waterfall_ozon.png', dpi=300)
plt.close()

print("Готово! Сохранены: fig_shap_bar_ozon.png, fig_shap_beeswarm_ozon.png, fig_shap_waterfall_ozon.png")