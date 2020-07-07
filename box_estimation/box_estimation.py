import pandas as pd
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def box_estimation():
    # Title
    title = 'icdar'

    # Read bbox info table
    table_path = './icdar_boxinfo.csv'
    table = pd.read_csv(table_path)
    # Split feature, target value
    feature = table[['inf_boxes', 'img_width', 'img_height']]
    target = table['gt_boxes']
    # Split train, test set
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.4, random_state=77)

    # append airlab dataset
    ail_train = pd.read_csv('./al_train.csv')
    X_atrain = ail_train[['inf_boxes', 'img_width', 'img_height']]
    y_atrain = ail_train['gt_boxes']
    X_train = X_train.append(X_atrain, ignore_index=True)
    y_train = y_train.append(y_atrain, ignore_index=True)

    ail_test = pd.read_csv('./al_test.csv')
    X_atest = ail_test[['inf_boxes', 'img_width', 'img_height']]
    y_atest = ail_test['gt_boxes']
    X_test = X_test.append(X_atest, ignore_index=True)
    y_test = y_test.append(y_atest, ignore_index=True)

    real_train_inf_boxes = X_train['inf_boxes']
    real_test_inf_boxes = X_test['inf_boxes']
    # norm
    X_train['inf_boxes'] = np.log1p(X_train['inf_boxes'])
    X_test['inf_boxes'] = np.log1p(X_test['inf_boxes'])
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        # 'max_depth': 6,
        'learning_rate': 0.01,
        'verbose': 0, 
        'early_stopping_round': 50
        }
    n_estimators = 1000

    # Dataset
    train_ds = lgb.Dataset(X_train, label=y_train)
    test_ds = lgb.Dataset(X_test, label=y_test)

    # train
    model = lgb.train(params, train_ds, 1000, test_ds, verbose_eval=100)

    # prediction
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # accuracy
    mse = mean_squared_error(y_test, pred_test)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, pred_test)
    print()
    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('R2 score: ', r2)
    print()

    final_result = pd.concat([y_test.reset_index(drop=True), pd.DataFrame(pred_test)], axis=1)
    final_result.columns = ['label', 'predict']
    final_result['inf_boxes'] = real_test_inf_boxes
    final_result['target_pred'] = final_result['label'] - final_result['predict']
    final_result.to_csv(title + '_result_fit.csv', index=False)

    # Result fit
    plt.figure(figsize=(8,8))
    sns.regplot(x='label', y='predict', data=final_result)
    plt.savefig('./' + title + '_result_fit.png')

    plt.figure(figsize=(18,10))
    plt.plot(final_result.index[:200], final_result['inf_boxes'][:200], label='inferece_boxes', c='orange', alpha=0.5)
    plt.plot(final_result.index[:200], final_result['predict'][:200], label='estim_boxes', c='blue', alpha=0.8)
    plt.plot(final_result.index[:200], final_result['label'][:200], label='real_boxes', c='red', alpha=0.8)
    plt.legend()
    plt.savefig('./' + title + '_boxes_info.png')

    # Train boxes
    total_train_boxes = y_train.sum()
    avg_train_boxes = total_train_boxes / len(y_train)

    # Assume boxes
    total_assume_boxes = avg_train_boxes * len(y_test)
    total_real_boxes = y_test.sum()
    total_estim_boxes = final_result['predict'].sum()

    print('\nTrainset: {}, Testset: {}\n'.format(len(y_train), len(y_test)))

    print("Total - AssumeBox: {}, RealBox: {}, EstimBox: {}".format(
        total_assume_boxes,
        total_real_boxes,
        total_estim_boxes
    ))

    print('Average - AssumeBox: {}, RealBox: {}, EstimBox: {}'.format(
        avg_train_boxes,
        total_real_boxes / len(y_test),
        total_estim_boxes / len(y_test)
    ))

if __name__ == '__main__':
    box_estimation()
