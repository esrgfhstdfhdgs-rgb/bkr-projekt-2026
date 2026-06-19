import pandas as pd
import numpy as np

def generate_realistic_ozon(n_customers=100000, random_state=42):
    rng = np.random.default_rng(random_state)
    customer_ids = np.arange(1, n_customers + 1)
    
    # Категории и подписка (без изменений)
    categories = rng.choice(['Дом','Электроника','Одежда'], size=n_customers, p=[0.4,0.35,0.25])
    has_premium = rng.choice([0,1], size=n_customers, p=[0.8,0.2])
    
    # Разная частота заказов для уходящих и лояльных (скрытая переменная)
    is_loyal = rng.choice([0,1], size=n_customers, p=[0.25,0.75])  # 25% склонны к оттоку
    base_freq_loyal = 20.0
    base_freq_nonloyal = 12.0
    annual_freq = np.where(is_loyal == 1,
                           rng.poisson(base_freq_loyal, n_customers),
                           rng.poisson(base_freq_nonloyal, n_customers))
    annual_freq = np.clip(annual_freq, 1, 60)
    
    month_probs = [0.07,0.07,0.08,0.08,0.08,0.08,0.08,0.09,0.09,0.09,0.10,0.10]
    month_probs = [p/sum(month_probs) for p in month_probs]
    
    transactions = []
    for i in range(n_customers):
        n_orders = annual_freq[i]
        months = rng.choice(np.arange(1,13), size=n_orders, p=month_probs)
        days = rng.integers(1,29,size=n_orders)
        dates = [pd.Timestamp(2024,m,d) for m,d in zip(months,days)]
        dates.sort()
        
        # Средний чек (зависит от категории, но с большим шумом)
        if categories[i]=='Электроника':
            avg_receipt = rng.lognormal(8.4,0.8)
        elif categories[i]=='Одежда':
            avg_receipt = rng.lognormal(7.6,0.9)
        else:
            avg_receipt = rng.lognormal(7.8,0.7)
        
        amounts = rng.lognormal(np.log(avg_receipt), 0.6, size=n_orders)
        amounts = np.clip(amounts, 50, 80000)
        
        # Возвраты (5% базово, но для уходящих больше)
        return_rate = 0.05 if is_loyal[i] else 0.10
        is_return = rng.choice([0,1], size=n_orders, p=[1-return_rate, return_rate])
        amounts = np.where(is_return==1, -amounts, amounts)
        
        for j in range(n_orders):
            transactions.append({
                'CustomerID': customer_ids[i],
                'InvoiceDate': dates[j],
                'Amount': round(amounts[j],2),
                'IsReturn': is_return[j],
                'Category': categories[i]
            })
    
    df_trans = pd.DataFrame(transactions)
    ref_date = pd.Timestamp('2024-12-31')
    
    # Агрегация RFM
    rfm = df_trans[df_trans['Amount']>0].groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency=('InvoiceDate','count'),
        Monetary=('Amount','sum')
    ).reset_index()
    
    # Вносим случайный шум в признаки (чтобы они перекрывались)
    rfm['Recency'] = rfm['Recency'] + rng.normal(0, 15, n_customers)
    rfm['Recency'] = rfm['Recency'].clip(1, None)
    rfm['Frequency'] = rfm['Frequency'] + rng.normal(0, 2, n_customers)
    rfm['Frequency'] = rfm['Frequency'].clip(1, None)
    rfm['Monetary'] = rfm['Monetary'] * rng.lognormal(0, 0.2, n_customers)
    
    # Целевая переменная: смесь Recency и скрытой лояльности
    rfm['Churn'] = ((rfm['Recency'] > 90) | (is_loyal == 0)).astype(int)
    # Делаем отток чуть менее детерминированным (у части лояльных тоже может быть большой Recency)
    mask_loyal_high_rec = (is_loyal == 1) & (rfm['Recency'] > 90)
    rfm.loc[mask_loyal_high_rec, 'Churn'] = rng.choice([0,1], size=mask_loyal_high_rec.sum(), p=[0.3,0.7])
    
    # Добавляем остальные атрибуты
    df_cust = pd.DataFrame({
        'CustomerID': customer_ids,
        'Category': categories,
        'HasPremium': has_premium
    })
    rfm = rfm.merge(df_cust, on='CustomerID', how='left')
    
    # Доля возвратов (шумная)
    returns = df_trans.groupby('CustomerID')['IsReturn'].mean().reset_index()
    returns.columns = ['CustomerID','ReturnRate']
    rfm = rfm.merge(returns, on='CustomerID', how='left')
    
    return df_trans, rfm