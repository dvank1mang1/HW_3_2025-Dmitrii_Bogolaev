#!/usr/bin/env python3

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

SEED = 2025
np.random.seed(SEED)

all_data = pd.read_csv('train.csv')
test_ids_df = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

store_data = pd.read_csv('STORE_LOCATION.csv', delimiter=';')

all_data['period_start_dt'] = pd.to_datetime(all_data['period_start_dt'], format='%Y-%m-%d')
all_data.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
all_data = all_data[all_data['store_location_rk'] != 309].copy()

all_data['PROMO1_FLAG'] = all_data['PROMO1_FLAG'].fillna(0)
all_data['AUTORIZATION_FLAG'] = all_data['AUTORIZATION_FLAG'].fillna(1)

all_data = all_data.sort_values(['product_rk', 'store_location_rk', 'period_start_dt'])
for col in ['PRICE_REGULAR', 'PRICE_AFTER_DISC']:
    all_data[col] = all_data.groupby(['product_rk', 'store_location_rk'])[col].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    all_data[col] = all_data[col].fillna(all_data[col].median())

if 'PROMO2_FLAG' in all_data.columns:
    del all_data['PROMO2_FLAG']
if 'NUM_CONSULTANT' in all_data.columns:
    del all_data['NUM_CONSULTANT']

store_data.rename(columns={'STORE_LOCATION_RK': 'store_location_rk'}, inplace=True)

store_data['STORE_OPEN_DTTM'] = pd.to_datetime(store_data['STORE_OPEN_DTTM'], format='%d%b%Y:%H:%M:%S', errors='coerce')
store_data['store_open_year'] = store_data['STORE_OPEN_DTTM'].dt.year
store_data['store_open_month'] = store_data['STORE_OPEN_DTTM'].dt.month

store_features = ['store_location_rk', 
                  'STORE_LOCATION_LVL_RK1', 'STORE_LOCATION_LVL_RK2', 
                  'STORE_LOCATION_LVL_RK3', 'STORE_LOCATION_LVL_RK4',
                  'store_open_year', 'store_open_month']

hash_cols = [col for col in store_data.columns if 'hashing' in col]
for col in hash_cols[:10]:
    store_data[f'{col}_encoded'] = pd.factorize(store_data[col])[0]
    store_features.append(f'{col}_encoded')

store_features_df = store_data[store_features].copy()

all_data = all_data.merge(store_features_df, on='store_location_rk', how='left')

for col in store_features[1:]:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(0)

all_data['month'] = all_data['period_start_dt'].dt.month
all_data['dayofweek'] = all_data['period_start_dt'].dt.dayofweek
all_data['day'] = all_data['period_start_dt'].dt.day
all_data['weekofyear'] = all_data['period_start_dt'].dt.isocalendar().week.astype(int)
all_data['quarter'] = all_data['period_start_dt'].dt.quarter
all_data['year'] = all_data['period_start_dt'].dt.year
all_data['is_weekend'] = (all_data['dayofweek'] >= 5).astype(int)
all_data['is_holiday_season'] = ((all_data['month'] == 11) | (all_data['month'] == 12)).astype(int)
all_data['is_peak_season'] = (all_data['month'] == 3).astype(int)

all_data['store_age_days'] = (all_data['period_start_dt'].dt.year - all_data['store_open_year']) * 365 + \
                              (all_data['period_start_dt'].dt.month - all_data['store_open_month']) * 30
all_data['store_age_days'] = all_data['store_age_days'].fillna(365)

all_data['discount'] = (all_data['PRICE_REGULAR'] - all_data['PRICE_AFTER_DISC']).clip(lower=0)
all_data['discount_pct'] = (all_data['discount'] / all_data['PRICE_REGULAR'].replace(0, np.nan)).fillna(0).clip(0, 1)
all_data['log_price_regular'] = np.log1p(all_data['PRICE_REGULAR'])
all_data['log_price_after_disc'] = np.log1p(all_data['PRICE_AFTER_DISC'])
all_data['has_discount'] = (all_data['discount'] > 0).astype(int)

all_data['promo_x_discount'] = all_data['PROMO1_FLAG'] * all_data['discount_pct']
all_data['promo_x_weekend'] = all_data['PROMO1_FLAG'] * all_data['is_weekend']
all_data['discount_x_holiday'] = all_data['discount_pct'] * all_data['is_holiday_season']

for lag in [7, 14, 21]:
    all_data[f'demand_lag_{lag}'] = all_data.groupby(['product_rk', 'store_location_rk'])['demand'].shift(lag)

for window in [7, 14, 21]:
    all_data[f'demand_roll_mean_{window}'] = all_data.groupby(['product_rk', 'store_location_rk'])['demand'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    all_data[f'demand_roll_std_{window}'] = all_data.groupby(['product_rk', 'store_location_rk'])['demand'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )

for alpha in [0.5, 0.7]:
    all_data[f'demand_ewm_{alpha}'] = all_data.groupby(['product_rk', 'store_location_rk'])['demand'].transform(
        lambda x: x.shift(1).ewm(alpha=alpha, min_periods=1).mean()
    )

all_data['demand_expanding_mean'] = all_data.groupby(['product_rk', 'store_location_rk'])['demand'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean()
)

train_only = all_data[all_data['demand'].notna()].copy()
train_only['month'] = train_only['period_start_dt'].dt.month

product_stats = train_only.groupby('product_rk')['demand'].agg(['mean', 'std', 'median']).reset_index()
product_stats.columns = ['product_rk', 'product_mean', 'product_std', 'product_median']
all_data = all_data.merge(product_stats, on='product_rk', how='left')

store_stats = train_only.groupby('store_location_rk')['demand'].agg(['mean', 'std']).reset_index()
store_stats.columns = ['store_location_rk', 'store_mean', 'store_std']
all_data = all_data.merge(store_stats, on='store_location_rk', how='left')

ps_stats = train_only.groupby(['product_rk', 'store_location_rk'])['demand'].agg(['mean', 'std']).reset_index()
ps_stats.columns = ['product_rk', 'store_location_rk', 'ps_mean', 'ps_std']
all_data = all_data.merge(ps_stats, on=['product_rk', 'store_location_rk'], how='left')

pm_stats = train_only.groupby(['product_rk', 'month'])['demand'].mean().to_dict()
all_data['pm_mean'] = all_data.apply(lambda row: pm_stats.get((row['product_rk'], row['month']), 0), axis=1)

pw_stats = train_only.groupby(['product_rk', 'weekofyear'])['demand'].mean().to_dict()
all_data['pw_mean'] = all_data.apply(lambda row: pw_stats.get((row['product_rk'], row['weekofyear']), 0), axis=1)

train_df = all_data[all_data['demand'].notna()].copy()
test_df = all_data[all_data['demand'].isna()].copy()

feature_cols = [c for c in all_data.columns if c not in ['id', 'period_start_dt', 'demand', 'STORE_OPEN_DTTM', 'STORE_CLOSURE_DTTM']]

for col in feature_cols:
    train_df[col] = train_df[col].fillna(0).replace([np.inf, -np.inf], 0)
    test_df[col] = test_df[col].fillna(0).replace([np.inf, -np.inf], 0)

lgb_full = lgb.Dataset(train_df[feature_cols], label=train_df['demand'])

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.015,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'lambda_l1': 0.05,
    'lambda_l2': 1.5,
    'min_gain_to_split': 0.01,
    'seed': SEED,
    'verbosity': -1
}

model = lgb.train(
    params,
    lgb_full,
    num_boost_round=2000,
    valid_sets=[lgb_full],
    callbacks=[lgb.log_evaluation(500)]
)

test_pred = np.clip(model.predict(test_df[feature_cols]), 0, None)

pred_df = pd.DataFrame({'id': test_df['id'].astype(int), 'predicted': test_pred})
test_ids_set = set(test_ids_df['id'].values)
submission = sample_submission[['id']].merge(pred_df[pred_df['id'].isin(test_ids_set)], on='id', how='left')
submission['predicted'] = submission['predicted'].fillna(train_df['demand'].mean())

submission.to_csv('submission_with_store_features.csv', index=False)
