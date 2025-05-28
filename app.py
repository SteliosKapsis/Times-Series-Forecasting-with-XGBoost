import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import os
import json

from config import DATA_PATH, RESULTS_DIR

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Setup
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

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

def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    # 1) Full time series plot
    df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJME Energy Use in MW')
    plt.savefig(os.path.join(RESULTS_DIR, 'full_timeseries.png'))
    plt.close()

    # 2) Outlier distribution
    df['PJME_MW'].plot(kind='hist', bins=500, figsize=(10, 5), title='PJME_MW Distribution')
    plt.savefig(os.path.join(RESULTS_DIR, 'distribution.png'))
    plt.close()

    # 3) Plot potential outliers
    df.query('PJME_MW < 19000')['PJME_MW'].plot(style='.', figsize=(15, 5), color=color_pal[5],
                                                    title='Outliers (<19000 MW)')
    plt.savefig(os.path.join(RESULTS_DIR, 'outliers.png'))
    plt.close()

    # Outlier removal
    df = df.query('PJME_MW > 19000').copy()

    # 4) Initial train-test split visualization
    train = df.loc[df.index < '2015-01-01']
    test = df.loc[df.index >= '2015-01-01']
    fig, ax = plt.subplots(figsize=(15, 5))
    train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
    test.plot(ax=ax, label='Test Set')
    ax.axvline('2015-01-01', color='black', ls='--')
    ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, 'train_test_split.png'))
    plt.close(fig)

    # 5) TimeSeriesSplit setup and fold visualization
    tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
    df = df.sort_index()
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)
    for fold, (train_idx, val_idx) in enumerate(tss.split(df)):
        train_fold = df.iloc[train_idx]
        val_fold = df.iloc[val_idx]
        train_fold['PJME_MW'].plot(ax=axs[fold], label='Train')
        val_fold['PJME_MW'].plot(ax=axs[fold], label='Validation')
        axs[fold].axvline(val_fold.index.min(), color='black', ls='--')
        axs[fold].set_title(f'Time Series Split Fold {fold}')
        axs[fold].legend()
    fig.savefig(os.path.join(RESULTS_DIR, 'ts_splits.png'))
    plt.close(fig)

    # Feature engineering
    df = create_features(df)
    df = add_lags(df)

    # Model training with cross-validation
    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1', 'lag2', 'lag3']
    TARGET = 'PJME_MW'
    scores = []
    for fold, (train_idx, val_idx) in enumerate(tss.split(df)):
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

    # Save cross-validation scores
    with open(os.path.join(RESULTS_DIR, 'cv_scores.json'), 'w') as f:
        json.dump({'rmse_scores': scores, 'average_rmse': np.mean(scores)}, f, indent=2)

    # Retrain on full dataset
    X_all, y_all = df[FEATURES], df[TARGET]
    final_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                                   n_estimators=500, objective='reg:linear',
                                   max_depth=3, learning_rate=0.01)
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
    plt.savefig(os.path.join(RESULTS_DIR, 'future_predictions.png'))
    plt.close()

    # Save final model
    final_model.save_model(os.path.join(RESULTS_DIR, 'model.json'))

    # Load model and verify predictions
    loaded_model = xgb.XGBRegressor()
    loaded_model.load_model(os.path.join(RESULTS_DIR, 'model.json'))
    future_df['pred_loaded'] = loaded_model.predict(future_features)
    future_df['pred_loaded'].plot(figsize=(15, 5), color=color_pal[1], title='Future Predictions (Loaded)')
    plt.savefig(os.path.join(RESULTS_DIR, 'future_predictions_loaded.png'))
    plt.close()

    # Daily error analysis
    test = df[df.index >= '2015-01-01'].copy()
    X_test, y_test = test[FEATURES], test[TARGET]
    test['prediction'] = final_model.predict(X_test)
    test['error'] = np.abs(test[TARGET] - test['prediction'])
    test['date'] = test.index.date
    daily_errors = test.groupby('date')['error'].mean()
    top5 = daily_errors.sort_values(ascending=False).head()
    bottom5 = daily_errors.sort_values(ascending=True).head()
    print("\nTop 5 days with highest average error:")
    print(top5)
    print("\nTop 5 days with lowest average error:")
    print(bottom5)
    top5.to_csv(os.path.join(RESULTS_DIR, 'top5_errors.csv'), header=['avg_error'])
    bottom5.to_csv(os.path.join(RESULTS_DIR, 'bottom5_errors.csv'), header=['avg_error'])

if __name__ == "__main__":
    main()
