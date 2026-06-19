import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_ozon_dataset(n_customers=500, random_state=42):
    rng = np.random.default_rng(random_state)
    customer_ids = np.arange(1, n_customers + 1)
    categories = rng.choice(['Дом','Электроника','Одежда'], size=n_customers, p=[0.4,0.35,0.25])
    has_premium = rng.choice([0,1], size=n_customers, p=[0.8,0.2])
    freq_multiplier = np.where(has_premium==1, rng.normal(1.4,0.2,n_customers), rng.normal(1.0,0.3,n_customers))
    freq_multiplier = np.clip(freq_multiplier, 0.3, 2.5)
    annual_freq = 18.5 * freq_multiplier

    # Правильные вероятности месяцев (сумма = 1.0)
    month_probs = [0.07,0.07,0.08,0.08,0.08,0.08,0.08,0.09,0.09,0.09,0.10,0.10]
    # Убедимся, что сумма 1.0 (может быть 1.01 из-за ошибок округления, исправим)
    total = sum(month_probs)
    month_probs = [p/total for p in month_probs]

    transactions = []
    for i in range(n_customers):
        n_orders = rng.poisson(annual_freq[i])
        if n_orders == 0:
            continue
        months = rng.choice(np.arange(1,13), size=n_orders, p=month_probs)
        days = rng.integers(1,29,size=n_orders)
        dates = [pd.Timestamp(2024,m,d) for m,d in zip(months,days)]
        dates.sort()
        if categories[i]=='Электроника':
            avg_receipt = rng.lognormal(8.4,0.5)
        elif categories[i]=='Одежда':
            avg_receipt = rng.lognormal(7.6,0.6)
        else:
            avg_receipt = rng.lognormal(7.8,0.5)
        amounts = rng.lognormal(np.log(avg_receipt),0.4,size=n_orders)
        amounts = np.clip(amounts,150,50000)
        is_return = rng.choice([0,1],size=n_orders,p=[0.95,0.05])
        amounts = np.where(is_return==1, -amounts, amounts)
        for j in range(n_orders):
            transactions.append({'CustomerID':customer_ids[i],'InvoiceDate':dates[j],
                                 'Amount':round(amounts[j],2),'IsReturn':is_return[j],
                                 'Category':categories[i]})
    df_trans = pd.DataFrame(transactions)
    ref_date = pd.Timestamp('2024-12-31')
    rfm = df_trans[df_trans['Amount']>0].groupby('CustomerID').agg(
        Recency=('InvoiceDate',lambda x: (ref_date-x.max()).days),
        Frequency=('InvoiceDate','count'),
        Monetary=('Amount','sum')).reset_index()
    rfm['Churn'] = (rfm['Recency']>90).astype(int)
    df_cust = pd.DataFrame({'CustomerID':customer_ids,'Category':categories,'HasPremium':has_premium})
    rfm = rfm.merge(df_cust,on='CustomerID',how='left')
    returns = df_trans.groupby('CustomerID')['IsReturn'].mean().reset_index()
    returns.columns = ['CustomerID','ReturnRate']
    rfm = rfm.merge(returns,on='CustomerID',how='left')
    return df_trans, rfm

def engineer_features(df_trans, rfm, ref_date=pd.Timestamp('2024-12-31')):
    df = rfm.copy()
    def cv_intervals(dates):
        if len(dates)<2: return 0.0
        intervals = dates.sort_values().diff().dt.days.dropna()
        return intervals.std()/intervals.mean() if intervals.mean()>0 else 0.0
    intervals_cv = df_trans[df_trans['Amount']>0].groupby('CustomerID')['InvoiceDate'].apply(cv_intervals)
    df['IntervalCV'] = df['CustomerID'].map(intervals_cv)
    last_30 = df_trans[(df_trans['InvoiceDate']>=ref_date-pd.Timedelta(days=30))&(df_trans['Amount']>0)]
    avg_check_30 = last_30.groupby('CustomerID')['Amount'].mean()
    avg_check_all = df['Monetary']/df['Frequency'].replace(0,1)
    df['TrendCheck'] = df['CustomerID'].map(avg_check_30) / avg_check_all.replace(0,1)
    last_90 = df_trans[(df_trans['InvoiceDate']>=ref_date-pd.Timedelta(days=90))&(df_trans['Amount']>0)]
    orders_90 = last_90.groupby('CustomerID').size()
    df['OrdersLast90'] = df['CustomerID'].map(orders_90).fillna(0)
    df['ReturnRate'] = df['ReturnRate'].fillna(0)
    return df

print("Генерация данных...")
df_trans, rfm = generate_ozon_dataset(n_customers=500)
df = engineer_features(df_trans, rfm)

feature_cols = ['Recency','Frequency','Monetary','IntervalCV','TrendCheck','OrdersLast90','ReturnRate']
cat_features = ['Category','HasPremium']
X = pd.get_dummies(df[feature_cols + cat_features], columns=cat_features, drop_first=True)
y = df['Churn']

print("Обучение CatBoost...")
model = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.1,
                           cat_features=[], verbose=0, random_seed=42)
model.fit(X, y)

print("Построение SHAP-визуализаций...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('Важность признаков (тест)', fontsize=14)
plt.tight_layout()
plt.savefig('test_shap_bar.png', dpi=150)
plt.close()

shap.summary_plot(shap_values, X, show=False)
plt.title('Beeswarm (тест)', fontsize=14)
plt.tight_layout()
plt.savefig('test_shap_beeswarm.png', dpi=150)
plt.close()

print("Всё работает! Графики сохранены: test_shap_bar.png, test_shap_beeswarm.png")