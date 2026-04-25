"""
Converts raw SKAB CSV files into numpy arrays compatible with the
Anomaly Transformer data loader.

Usage:
    python3 prepare_skab.py --skab_path /tmp/SKAB --out_path dataset/SKAB
"""
import argparse
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SENSOR_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
    'Pressure', 'Temperature', 'Thermocouple',
    'Voltage', 'Volume Flow RateRMS'
]


def load_csvs(paths):
    dfs = []
    for p in sorted(paths):
        df = pd.read_csv(p, sep=';', parse_dates=['datetime'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main(skab_path, out_path):
    # Training data: anomaly-free operation (no labels needed)
    train_df = pd.read_csv(f'{skab_path}/data/anomaly-free/anomaly-free.csv', sep=';')
    train_data = train_df[SENSOR_COLS].values.astype(np.float32)

    # Test data: all anomalous experiments concatenated
    test_files = (
        glob.glob(f'{skab_path}/data/valve1/*.csv') +
        glob.glob(f'{skab_path}/data/valve2/*.csv') +
        glob.glob(f'{skab_path}/data/other/*.csv')
    )
    test_df = load_csvs(test_files)
    test_data = test_df[SENSOR_COLS].values.astype(np.float32)
    test_labels = test_df['anomaly'].values.astype(np.float32)

    # Fit scaler on training data only
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data).astype(np.float32)
    test_data = scaler.transform(test_data).astype(np.float32)

    np.save(f'{out_path}/SKAB_train.npy', train_data)
    np.save(f'{out_path}/SKAB_test.npy', test_data)
    np.save(f'{out_path}/SKAB_test_label.npy', test_labels)

    print(f'Train shape : {train_data.shape}')
    print(f'Test shape  : {test_data.shape}')
    print(f'Label shape : {test_labels.shape}')
    print(f'Anomaly rate: {test_labels.mean():.3%}')
    print(f'Saved to    : {out_path}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skab_path', default='/tmp/SKAB')
    parser.add_argument('--out_path', default='dataset/SKAB')
    args = parser.parse_args()
    main(args.skab_path, args.out_path)
