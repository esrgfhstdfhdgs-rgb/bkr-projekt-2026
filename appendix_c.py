import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Загрузка датасета
# ------------------------------
print("Загрузка датасета...")
df = pd.read_csv('ozon_features_full.csv')

feature_cols_raw = ['Recency', 'Frequency', 'Monetary', 'IntervalCV', 'TrendCheck',
                    'OrdersLast90', 'ReturnRate', 'Category', 'HasPremium']
X_raw = df[feature_cols_raw]
y = df['Churn']

# Очистка
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
X_raw[numeric_cols] = X_raw[numeric_cols].fillna(0)

X = pd.get_dummies(X_raw, columns=['Category', 'HasPremium'], drop_first=True)

# Разделение
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Модели
models = {
    'Логистическая регрессия': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Случайный лес': RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42),
    'CatBoost': CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
}

# Таблица В.1
print("\nТаблица В.1 — Метрики (кросс-валидация)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            scoring=['precision','recall','f1','roc_auc'])
    print(f"{name}: Prec={np.mean(scores['test_precision']):.3f}, "
          f"Rec={np.mean(scores['test_recall']):.3f}, "
          f"F1={np.mean(scores['test_f1']):.3f}, "
          f"ROC-AUC={np.mean(scores['test_roc_auc']):.3f}")

# Таблица В.2
print("\nТаблица В.2 — Метрики (тестовая выборка)")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred, target_names=['Остался','Ушёл'], digits=3))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

# Таблица В.3
print("\nТаблица В.3 — Тест МакНемара")
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model_lr.fit(X_train, y_train)
model_rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)
model_cb = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
model_cb.fit(X_train, y_train)

y_lr = model_lr.predict(X_test)
y_rf = model_rf.predict(X_test)
y_cb = model_cb.predict(X_test)

def mcnemar_test(y_a, y_b, name_a, name_b):
    b = np.sum((y_a == 1) & (y_b == 0))
    c = np.sum((y_a == 0) & (y_b == 1))
    table = np.array([[0, b], [c, 0]])
    result = mcnemar(table, exact=False, correction=True)
    print(f"{name_a} vs {name_b}: p={result.pvalue:.4f}")

mcnemar_test(y_lr, y_rf, 'Лог. регрессия', 'Случ. лес')
mcnemar_test(y_rf, y_cb, 'Случ. лес', 'CatBoost')