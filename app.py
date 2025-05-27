import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os
import subprocess

# Setup
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# Load data
df = pd.read_csv(r'C:\Users\steli\Desktop\AI - Time Series\data\PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

# Plot full time series
df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJME Energy Use in MW')
plt.show()

# Outlier analysis: Distribution plot
df['PJME_MW'].plot(kind='hist', bins=500, figsize=(10, 5), title='PJME_MW Distribution')
plt.show()

# Plot potential outliers
df.query('PJME_MW < 19000')['PJME_MW'].plot(style='.', figsize=(15, 5), color=color_pal[5], title='Outliers (<19000 MW)')
plt.show()

# Outlier removal
df = df.query('PJME_MW > 19000').copy()

# Initial train-test split visualization
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend()
plt.show()

# TimeSeriesSplit setup
tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

# Visualize folds
fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
fold = 0
for train_idx, val_idx in tss.split(df):
    train_fold = df.iloc[train_idx]
    val_fold = df.iloc[val_idx]
    train_fold['PJME_MW'].plot(ax=axs[fold], label='Train')
    val_fold['PJME_MW'].plot(ax=axs[fold], label='Validation')
    axs[fold].axvline(val_fold.index.min(), color='black', ls='--')
    axs[fold].set_title(f'Time Series Split Fold {fold}')
    fold += 1
plt.show()

# Feature engineering
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df):
    target_map = df['PJME_MW'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

df = create_features(df)
df = add_lags(df)

# Forecast horizon explanation (from original)
'''
 The forecast horizon is the length of time into the future for which forecasts are to 
 be prepared. These generally vary from short-term forecasting horizons 
 (less than three months) to long-term horizons (more than two years).
'''

# Model training with cross-validation
FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
TARGET = 'PJME_MW'

fold = 0
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=1000,
                           early_stopping_rounds=50, objective='reg:linear',
                           max_depth=3, learning_rate=0.01)
    reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
    y_pred = reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f'Fold {fold} RMSE: {rmse:.2f}')
    scores.append(rmse)
    fold += 1

print(f'\nAverage RMSE across folds: {np.mean(scores):.2f}')
print(f'Fold RMSE scores: {scores}')

# Retrain on full dataset
X_all, y_all = df[FEATURES], df[TARGET]
final_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', n_estimators=500,
                               objective='reg:linear', max_depth=3, learning_rate=0.01)
final_model.fit(X_all, y_all, eval_set=[(X_all, y_all)], verbose=100)

# Future forecasting
future_dates = pd.date_range('2018-08-03', '2019-08-01', freq='1H')
future_df = pd.DataFrame(index=future_dates)
future_df['isFuture'] = True
df['isFuture'] = False
combined_df = pd.concat([df, future_df])
combined_df = create_features(combined_df)
combined_df = add_lags(combined_df)

future_features = combined_df.query('isFuture')[FEATURES]
future_df['pred'] = final_model.predict(future_features)

# Plot future predictions
future_df['pred'].plot(figsize=(15, 5), color=color_pal[4], title='Future Predictions')
plt.show()

# Save model
final_model.save_model('model.json')
subprocess.run('dir', shell=True)

# Load model and verify predictions
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('model.json')
future_df['pred_loaded'] = loaded_model.predict(future_features)
future_df['pred_loaded'].plot(figsize=(15, 5), color=color_pal[1], title='Future Predictions (Loaded Model)')
plt.show()

# Daily error analysis (from app.py)
test = df[df.index >= '2015-01-01'].copy()
X_test, y_test = test[FEATURES], test[TARGET]
test['prediction'] = final_model.predict(X_test)
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
print("\nTop 5 days with highest average error:")
print(test.groupby('date')['error'].mean().sort_values(ascending=False).head())
print("\nTop 5 days with lowest average error:")
print(test.groupby('date')['error'].mean().sort_values(ascending=True).head())

