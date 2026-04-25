"""
Converts GECCO 2018 water quality CSV into numpy arrays.
EVENT == True means anomaly.
Train: first 70% of non-anomalous rows. Test: remaining data (preserving order).

Usage:
    python3 prepare_gecco.py --csv_path /tmp/gecco2018.csv --out_path dataset/GECCO
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']


def main(csv_path, out_path):
    df = pd.read_csv(csv_path, index_col=0)
    df['EVENT'] = df['EVENT'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
    # Fill missing sensor readings via forward-fill then backward-fill
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill().bfill()

    normal = df[df['EVENT'] == False]
    split = int(len(normal) * 0.7)
    train_df = normal.iloc[:split]

    # test set: last 30% of the full dataset (in time order)
    test_start = int(len(df) * 0.7)
    test_df = df.iloc[test_start:]

    train_data = train_df[SENSOR_COLS].values.astype(np.float32)
    test_data = test_df[SENSOR_COLS].values.astype(np.float32)
    test_labels = test_df['EVENT'].astype(float).values.astype(np.float32)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    np.save(f'{out_path}/GECCO_train.npy', train_data)
    np.save(f'{out_path}/GECCO_test.npy', test_data)
    np.save(f'{out_path}/GECCO_test_label.npy', test_labels)

    print(f'Train shape : {train_data.shape}')
    print(f'Test shape  : {test_data.shape}')
    print(f'Label shape : {test_labels.shape}')
    print(f'Anomaly rate: {test_labels.mean():.3%}')
    print(f'Saved to    : {out_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default='/tmp/gecco2018.csv')
    parser.add_argument('--out_path', default='dataset/GECCO')
    args = parser.parse_args()
    main(args.csv_path, args.out_path)
