import pandas as pd
import matplotlib.pyplot as plt

def visual():
    log_df = pd.read_csv('./pths/train_log.csv')

    # plt.figure(figsize=(15,8))
    # plt.plot(log_df['epoch'], log_df['loss'], c='blue', label='loss')
    # plt.plot(log_df['epoch'], log_df['recall'], c='red', label='recall')
    # plt.plot(log_df['epoch'], log_df['precision'], c='green', label='precision')
    # plt.xlabel('epoch')
    # plt.legend()
    # plt.savefig('./acc.png')


    train_time = log_df['epoch_time'].sum() / 3600

    tmp = log_df[log_df['recall'] >= 0.50]
    print(tmp)
    print('Total training time: ', train_time)
    print('Max Recall, Precision', log_df['recall'].max(), log_df['precision'].max())
    print('Min Loss', log_df['loss'].min())

def visual_optim():
    optim_df = pd.read_csv('./pths/optimize.csv')
    optim_df['conf_score'] = optim_df['conf_score'].round(7)

    plt.figure(figsize=(10,10))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, conf in enumerate([0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]):
        df = optim_df[optim_df['conf_score'] == conf].reset_index(drop=True)

        plt.plot(df['iou'], df['recall'], label='R-conf: {}'.format(conf), c=colors[i])
        plt.plot(df['iou'], df['precision'], label='P-conf: {}'.format(conf), c=colors[i], linestyle='--')

    plt.xlabel('IoU')
    plt.ylabel('Recall, Precision')
    plt.legend()
    plt.savefig('./score_iou.png')

visual_optim()
    

