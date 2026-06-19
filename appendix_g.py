import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Загрузка и подготовка данных
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------
# 1. Кривые обучения (ускоренная версия)
# ------------------------------
def plot_learning_curve_fast(name, X_tr, y_tr, filename):
    """
    Быстрая кривая обучения с 5 размерами выборки и облегчёнными параметрами.
    """
    train_sizes = np.linspace(0.2, 1.0, 5)          # всего 5 шагов
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 3 фолда
    test_scores = []
    
    for i, size in enumerate(train_sizes):
        if size < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_tr, y_tr, train_size=size, stratify=y_tr, random_state=42
            )
        else:
            X_sub, y_sub = X_tr, y_tr
        
        # Создаём облегчённые модели
        if name == 'CatBoost':
            est = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1,
                                     verbose=0, random_seed=42)
        elif name == 'Случайный лес':
            est = RandomForestClassifier(n_estimators=100, max_depth=10,
                                         class_weight='balanced', random_state=42)
        else:  # Логистическая регрессия
            est = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
        fold_scores = []
        for tr_idx, val_idx in cv.split(X_sub, y_sub):
            X_fold_tr, X_fold_val = X_sub.iloc[tr_idx], X_sub.iloc[val_idx]
            y_fold_tr, y_fold_val = y_sub.iloc[tr_idx], y_sub.iloc[val_idx]
            est.fit(X_fold_tr, y_fold_tr)
            y_proba = est.predict_proba(X_fold_val)[:, 1]
            fold_scores.append(roc_auc_score(y_fold_val, y_proba))
        test_scores.append(np.mean(fold_scores))
        print(f"  {name}: размер {size:.0%} завершён (AUC={test_scores[-1]:.3f})")
    
    train_sizes_abs = (np.array(train_sizes) * len(X_tr)).astype(int)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes_abs, test_scores, 'o-', color='green', label='Кросс-валидация')
    plt.xlabel('Размер обучающей выборки', fontsize=12)
    plt.ylabel('ROC-AUC', fontsize=12)
    plt.title(f'Кривая обучения ({name})', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

print("Построение кривых обучения (ускоренный режим)...")
for name in ['Логистическая регрессия', 'Случайный лес', 'CatBoost']:
    plot_learning_curve_fast(name, X_train, y_train,
                             f'fig_learning_curve_{name.replace(" ", "_").lower()}.png')

# ------------------------------
# 2. Матрицы ошибок
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Остался', 'Ушёл'],
                yticklabels=['Остался', 'Ушёл'])
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

print("\nПостроение матриц ошибок...")
models = {
    'Логистическая регрессия': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Случайный лес': RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42),
    'CatBoost': CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
}

for name, model in models.items():
    print(f"Обучение {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred,
                          f'Матрица ошибок ({name})',
                          f'fig_confusion_matrix_{name.replace(" ", "_").lower()}.png')

print("\nГотово! Все изображения сохранены.")