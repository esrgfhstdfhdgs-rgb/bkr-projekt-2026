from generate_realistic_ozon import generate_realistic_ozon
import pandas as pd
import numpy as np

# Генерация с инженерией признаков
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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

print("Генерация реалистичного датасета Ozon (100 000 клиентов)...")
df_trans, rfm = generate_realistic_ozon(n_customers=100000)
df = engineer_features(df_trans, rfm)
df.to_csv('ozon_features_full.csv', index=False)
print("Новый датасет сохранён в ozon_features_full.csv")
print(f"Размер: {df.shape[0]} записей, {df.shape[1]} столбцов")
print(f"Доля оттока: {df['Churn'].mean():.2%}")
