"""
Converts Tennessee Eastman Process (TEP) CSV into numpy arrays.
STATUS == 0 means normal operation; anything else is a fault.
Train: first 80% of normal rows. Test: remaining 20% normal + all fault rows.

Usage:
    python3 prepare_tep.py --csv_path /tmp/new_tep_datasets/matlab_data_1year.csv \
                           --out_path dataset/TEP
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [c for c in pd.read_csv(
    '/tmp/new_tep_datasets/matlab_data_1year.csv', index_col=0, nrows=0
).columns if c != 'STATUS']


def main(csv_path, out_path):
    df = pd.read_csv(csv_path, index_col=0)

    normal = df[df['STATUS'] == 0].copy()
    faulty = df[df['STATUS'] != 0].copy()

    split = int(len(normal) * 0.8)
    train_df = normal.iloc[:split]
    test_normal = normal.iloc[split:]
    test_df = pd.concat([test_normal, faulty]).sort_index()

    train_data = train_df[FEATURE_COLS].values.astype(np.float32)
    test_data = test_df[FEATURE_COLS].values.astype(np.float32)
    test_labels = (test_df['STATUS'] != 0).values.astype(np.float32)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    np.save(f'{out_path}/TEP_train.npy', train_data)
    np.save(f'{out_path}/TEP_test.npy', test_data)
    np.save(f'{out_path}/TEP_test_label.npy', test_labels)

    print(f'Train shape : {train_data.shape}')
    print(f'Test shape  : {test_data.shape}')
    print(f'Label shape : {test_labels.shape}')
    print(f'Anomaly rate: {test_labels.mean():.3%}')
    print(f'Saved to    : {out_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/tmp/new_tep_datasets/matlab_data_1year.csv')
    parser.add_argument('--out_path', default='dataset/TEP')
    args = parser.parse_args()
    main(args.csv_path, args.out_path)
