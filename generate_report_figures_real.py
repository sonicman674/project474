"""
generate_report_figures_real.py
================================
Generates report figures from the actual saved model test outputs when available:

  test_outputs/<DATASET>_test_outputs.npz

Those files are produced by running main.py in test mode with the updated solver.py.
This avoids hand-copying Precision/Recall/F1 values into the plotting script.

Run from the project root:
    python generate_report_figures_real.py

All outputs saved to:  report_figures_real/
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'report_figures_real')
os.makedirs(OUT_DIR, exist_ok=True)

def save(name):
    plt.savefig(os.path.join(OUT_DIR, f'{name}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → report_figures_real/{name}.pdf / .png')

# ── Dataset metadata. Metrics are overwritten from test_outputs/*.npz. ─────────
RESULTS = {
    'SKAB':    {'P': 97.89, 'R': 100.00, 'F1': 98.93,
                'train': 9_405,     'test': 37_401,    'anomaly_rate': 0.349,
                'dims': 8,  'domain': 'Industrial pump/valve',
                'sensors': ['Accel1', 'Accel2', 'Current', 'Pressure',
                            'Temperature', 'Thermocouple', 'Voltage', 'Flow Rate'],
                'color': '#2196F3'},
    'TEP':     {'P': 99.79, 'R':  99.84, 'F1': 99.82,
                'train': 72_573,    'test': 102_628,   'anomaly_rate': 0.823,
                'dims': 52, 'domain': 'Chemical plant (52 sensors)',
                'sensors': [f'X{i+1}' for i in range(52)],
                'color': '#4CAF50'},
    'GECCO':   {'P': 97.47, 'R': 100.00, 'F1': 98.72,
                'train': 96_488,    'test':  41_870,   'anomaly_rate': 0.00184,
                'dims': 9,  'domain': 'Drinking water (IoT)',
                'sensors': ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2'],
                'color': '#FF9800'},
    'MIT-BIH': {'P': 86.74, 'R':  99.94, 'F1': 92.88,
                'train': 3_856_518, 'test': 3_900_000, 'anomaly_rate': 0.252,
                'dims': 2,  'domain': 'ECG arrhythmia (2-lead)',
                'sensors': ['Lead MLII', 'Lead V1'],
                'color': '#E91E63'},
}

# ── Training loss from OSAgnosticReAL notebook outputs ────────────────────────
TRAINING = {
    'SKAB': {
        'epochs': list(range(1, 11)),
        'train':  [-22.24, -22.70, -22.77, -22.72, -22.70, -22.68, -22.68, -22.68, -22.67, -22.67],
        'vali':   [874.86, 868.25, 864.72, 862.91, 862.01, 861.55, 861.33, 861.21, 861.16, 861.13],
        'early_stop': None,
    },
    'TEP': {
        'epochs': [1, 2, 3, 4],
        'train':  [-22.22, -21.38, -21.09, -21.09],
        'vali':   [-20.10, -19.76, -19.76, -19.79],
        'early_stop': 4,
    },
    'GECCO': {
        'epochs': [1, 2, 3, 4],
        'train':  [-22.48, -22.19, -22.14, -22.27],
        'vali':   [-18.27, -18.20, -18.17, -18.23],
        'early_stop': 4,
    },
    'MIT-BIH': {
        'epochs': [1, 2, 3, 4],
        'train':  [-38.99, -47.17, -47.73, -47.86],
        'vali':   [-45.92, -47.44, -47.67, -47.77],
        'early_stop': 4,
    },
}

DATASETS = list(RESULTS.keys())


# ─────────────────────────────────────────────────────────────────────────────
def load_test(name):
    key  = 'MITBIH' if name == 'MIT-BIH' else name
    path = os.path.join(BASE_DIR, 'dataset', key)
    data   = np.load(os.path.join(path, f'{key}_test.npy'))
    labels = np.load(os.path.join(path, f'{key}_test_label.npy'))
    return data, labels


def load_model_outputs(name):
    key = 'MITBIH' if name == 'MIT-BIH' else name
    path = os.path.join(BASE_DIR, 'test_outputs', f'{key}_test_outputs.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing real model outputs for {name}: {path}\n'
            'Run main.py in test mode after the solver.py score-export change to create this file.'
        )
    return np.load(path)


def real_results():
    """Return dataset metadata with P/R/F1 and counts read from saved model outputs."""
    results = {}
    for name, meta in RESULTS.items():
        out = load_model_outputs(name)
        gt = out['gt'].astype(int)
        results[name] = dict(meta)
        results[name].update({
            'P': float(out['precision']) * 100,
            'R': float(out['recall']) * 100,
            'F1': float(out['f_score']) * 100,
            'accuracy': float(out['accuracy']) * 100,
            'test': len(gt),
            'anomaly_rate': float(gt.mean()) if len(gt) else 0.0,
        })
    return results


def load_training_log(name):
    key = 'MITBIH' if name == 'MIT-BIH' else name
    path = os.path.join(BASE_DIR, 'training_logs', f'{key}_training_log.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Missing training log for {name}: {path}\n'
            'Rerun main.py in train mode with --training_log_path training_logs.'
        )
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'vali_loss': float(row['vali_loss']),
                'vali_loss2': float(row.get('vali_loss2', 0.0)),
                'early_stop': bool(int(row.get('early_stop', 0))),
            })
    if not rows:
        raise ValueError(f'Training log is empty: {path}')
    return rows


def find_anomaly_window(labels, min_len=150, context=100):
    n = len(labels)
    for i in range(1, n - min_len - context):
        if labels[i] == 1 and labels[i-1] == 0:
            start = max(0, i - context)
            end   = min(n, i + min_len + context)
            if labels[end-1] == 0:
                return start, end
    anom_idx = np.where(labels == 1)[0]
    if len(anom_idx) == 0:
        return 0, min(n, 600)
    centre = anom_idx[len(anom_idx) // 2]
    return max(0, centre - 300), min(n, centre + 300)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  —  Precision / Recall / F1 for the four new domains
# ─────────────────────────────────────────────────────────────────────────────
def fig1_prf1():
    results = real_results()
    metrics = ['P', 'R', 'F1']
    labels  = ['Precision', 'Recall', 'F1-score']
    colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x       = np.arange(len(DATASETS))
    width   = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for i, (m, c, lbl) in enumerate(zip(metrics, colors, labels)):
        vals = [results[d][m] for d in DATASETS]
        bars = ax.bar(x + (i-1)*width, vals, width, label=lbl, color=c, alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 2:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.9,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=7.8)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Score (%)')
    ax.set_title('Anomaly Transformer — Extended Evaluation on New Domains\n'
                 'Precision, Recall, and F1-score from saved model outputs', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.24),
              ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save('fig1_new_domain_prf1')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  —  F1 across all 9 datasets
# ─────────────────────────────────────────────────────────────────────────────
def fig2_f1_all():
    results = real_results()
    original = {
        'SMD':  92.33, 'MSL':  93.59, 'SMAP': 96.69,
        'SWaT': 94.07, 'PSM':  97.89,
    }
    all_ds  = list(original.keys()) + DATASETS
    all_f1  = list(original.values()) + [results[d]['F1'] for d in DATASETS]
    colors  = ['#5b7fbe'] * len(original) + ['#c44e52'] * len(DATASETS)

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    bars = ax.bar(all_ds, all_f1, color=colors, alpha=0.88, width=0.62, zorder=3)
    for bar, v in zip(bars, all_f1):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8.8)

    ax.set_ylabel('F1-score (%)')
    ax.set_title('Anomaly Transformer: F1-score Across All Evaluated Datasets\n'
                 'New-domain scores read from saved model outputs', pad=10)
    ax.set_ylim(0, 116)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.55, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    sep_x = len(original) - 0.5
    ax.axvline(x=sep_x, color='#888', linestyle='--', linewidth=0.9, zorder=2)
    ax.text(sep_x-0.12, 113.5, 'ICLR 2022\nbenchmarks', ha='right', va='top',
            fontsize=8.5, color='#5b7fbe')
    ax.text(sep_x+0.12, 113.5, 'New domain\ndatasets', ha='left', va='top',
            fontsize=8.5, color='#c44e52')
    ax.legend(handles=[
        mpatches.Patch(facecolor='#5b7fbe', alpha=0.88, label='Original benchmarks (ICLR 2022)'),
        mpatches.Patch(facecolor='#c44e52', alpha=0.88, label='New domain datasets (this project)'),
    ], frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.24),
       ncol=2, fontsize=10)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    save('fig2_f1_all_datasets')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  —  Full results table
# ─────────────────────────────────────────────────────────────────────────────
def fig3_table():
    results = real_results()
    new_rows = [
        [
            name,
            r['domain'],
            str(r['dims']),
            f"{r['train']:,}",
            f"{r['test']:,}",
            f"{r['anomaly_rate'] * 100:.2f}%",
            f"{r['P']:.2f}",
            f"{r['R']:.2f}",
            f"{r['F1']:.2f}",
        ]
        for name, r in results.items()
    ]
    headers = ['Dataset', 'Application Domain', 'Dims',
               'Train Size', 'Test Size', 'Anomaly\nRate',
               'Precision\n(%)', 'Recall\n(%)', 'F1\n(%)']
    rows = [
        ['SMD',     'Server monitoring',         '38',  '566,724',   '708,420',  '4.2%',  '89.40', '95.45', '92.33'],
        ['MSL',     'Space (NASA rover)',         '55',   '46,653',    '73,729', '10.5%',  '92.09', '95.15', '93.59'],
        ['SMAP',    'Space (NASA satellite)',     '25',  '108,146',   '427,617', '12.8%',  '94.13', '99.40', '96.69'],
        ['SWaT',    'Water treatment (ICS)',      '51',  '396,000',   '449,919', '12.1%',  '91.55', '96.73', '94.07'],
        ['PSM',     'Server (eBay)',              '25',  '105,984',    '87,841', '27.8%',  '96.91', '98.90', '97.89'],
    ] + new_rows
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.72)
    for j in range(len(headers)):
        tbl[(0,j)].set_facecolor('#1a3a6b')
        tbl[(0,j)].set_text_props(color='white', fontweight='bold', fontsize=8.8)
    for i in range(1, 6):
        for j in range(len(headers)):
            tbl[(i,j)].set_facecolor('#e8f0fb')
        tbl[(i,0)].set_text_props(fontweight='bold')
    for i in range(6, 10):
        for j in range(len(headers)):
            tbl[(i,j)].set_facecolor('#fdf0ef')
        tbl[(i,0)].set_text_props(fontweight='bold')
    for i in range(1, 10):
        tbl[(i,8)].set_text_props(fontweight='bold')
    ax.set_title(
        'Table 1 — Anomaly Transformer: Results Summary\n'
        'Original ICLR 2022 benchmarks (blue) + new-domain metrics read from saved model outputs (coral)',
        pad=8, fontsize=10, loc='left')
    plt.tight_layout()
    save('fig3_results_table')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4  —  Raw sensor data
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_CFG = {
    'SKAB':    {'cols': [0,2,3,4], 'slabels': ['Accelerometer','Current (A)','Pressure (bar)','Temperature (°C)']},
    'TEP':     {'cols': [0,4,8,12],'slabels': ['Feed A (X1)','Reactor Temp (X5)','Separator (X9)','Stripper (X13)']},
    'GECCO':   {'cols': [0,2,3,4], 'slabels': ['Water Temp (Tp)','pH','Redox','Conductivity']},
    'MIT-BIH': {'cols': [0,1],     'slabels': ['Lead MLII (mV)','Lead V1 (mV)']},
}

def shade_anomaly(ax, lbl_slice):
    in_a, a_s = False, 0
    for j in range(1, len(lbl_slice)):
        if lbl_slice[j]==1 and not in_a:  a_s, in_a = j, True
        elif lbl_slice[j]==0 and in_a:
            ax.axvspan(a_s, j, color='#FF5252', alpha=0.18, zorder=0); in_a=False
    if in_a: ax.axvspan(a_s, len(lbl_slice), color='#FF5252', alpha=0.18, zorder=0)


def shade_predicted_anomaly(ax, pred_slice):
    in_a, a_s = False, 0
    for j in range(1, len(pred_slice)):
        if pred_slice[j] == 1 and not in_a:
            a_s, in_a = j, True
        elif pred_slice[j] == 0 and in_a:
            ax.axvspan(a_s, j, color='#c44e52', alpha=0.16, zorder=0)
            in_a = False
    if in_a:
        ax.axvspan(a_s, len(pred_slice), color='#c44e52', alpha=0.16, zorder=0)


def smooth_signal(values, window=11):
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode='same')

def fig4_raw_sensor_data():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.2))
    axes = axes.flatten()
    for ax, (name, res) in zip(axes, RESULTS.items()):
        data, labels = load_test(name)
        s, e = find_anomaly_window(labels, min_len=200, context=150)
        t = np.arange(e-s); lbl_s = labels[s:e]; d_s = data[s:e]
        shade_anomaly(ax, lbl_s)
        cmap_c = plt.cm.tab10.colors
        cfg = SENSOR_CFG[name]
        for k, (col, slbl) in enumerate(zip(cfg['cols'], cfg['slabels'])):
            ax.plot(t, d_s[:,col], lw=0.9, color=cmap_c[k%10], label=slbl, alpha=0.9, zorder=2)
        ax.set_title(f'{name}  —  {res["domain"]}', fontsize=11, pad=6)
        ax.set_xlabel('Time step'); ax.set_ylabel('Normalised value')
        ax.legend(frameon=True, facecolor='white', edgecolor='#dddddd',
                  framealpha=0.92, fontsize=7.1, loc='upper center',
                  bbox_to_anchor=(0.5, -0.20), ncol=2, handlelength=1.8)
        ax.spines[['top','right']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', lw=0.4, alpha=0.5)
        anom_idx = np.where(lbl_s==1)[0]
        if len(anom_idx):
            mid = anom_idx[len(anom_idx)//2]
            ymin, ymax = ax.get_ylim()
            ax.text(mid, ymax*0.92, 'ANOMALY', ha='center', va='top',
                    color='#D32F2F', fontsize=8, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.5))
    fig.legend(handles=[mpatches.Patch(color='#FF5252',alpha=0.4,label='Anomaly period (ground truth)')],
               loc='lower center', ncol=1, frameon=False, fontsize=10, bbox_to_anchor=(0.5,0.02))
    fig.suptitle('Part 1 — Raw Sensor Readings with Ground-Truth Anomaly Windows\n'
                 'Four New Application Domains  (Colab T4 — definitive run)',
                 fontsize=13, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0,0.08,1,0.94], h_pad=3.8, w_pad=2.2)
    save('fig4_raw_sensor_data')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5  —  Class imbalance
# ─────────────────────────────────────────────────────────────────────────────
def fig5_class_imbalance():
    results = real_results()
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        anom = res['anomaly_rate']*100; norm = 100-anom
        if anom < 1.0:
            bars = ax.bar(['Normal','Anomaly'], [norm,anom],
                          color=['#5b7fbe','#c44e52'], alpha=0.88, width=0.5)
            ax.set_ylim(0,115); ax.set_ylabel('Proportion (%)')
            for bar, v in zip(bars, [norm,anom]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                        f'{v:.3f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.spines[['top','right']].set_visible(False)
            ax.yaxis.grid(True, linestyle='--', lw=0.4, alpha=0.5)
        else:
            wedges, texts, ats = ax.pie([norm,anom], labels=['Normal','Anomaly'],
                colors=['#5b7fbe','#c44e52'], autopct='%1.1f%%', startangle=90,
                wedgeprops={'linewidth':1.2,'edgecolor':'white'}, textprops={'fontsize':10})
            for at in ats: at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
        ax.set_title(f'{name}\n{res["domain"]}', fontsize=10, pad=8)
    fig.suptitle('Part 1 — Class Imbalance: Normal vs Anomalous Data Points\n'
                 'Four New Application Domains  (computed from saved model-output labels)',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.legend(handles=[
        mpatches.Patch(facecolor='#5b7fbe',alpha=0.88,label='Normal'),
        mpatches.Patch(facecolor='#c44e52',alpha=0.88,label='Anomaly'),
    ], loc='lower center', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout()
    save('fig5_class_imbalance')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6  —  Confusion matrices
# ─────────────────────────────────────────────────────────────────────────────
def compute_cm_from_outputs(name):
    out = load_model_outputs(name)
    gt = out['gt'].astype(int)
    pred = out['pred'].astype(int)
    TN = int(((gt == 0) & (pred == 0)).sum())
    FP = int(((gt == 0) & (pred == 1)).sum())
    FN = int(((gt == 1) & (pred == 0)).sum())
    TP = int(((gt == 1) & (pred == 1)).sum())
    return np.array([[TN,FP],[FN,TP]])

def fig6_confusion_matrices():
    results = real_results()
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 4.4))
    for ax, (name, res) in zip(axes, results.items()):
        cm    = compute_cm_from_outputs(name)
        total = cm.sum(); cm_n = cm.astype(float)/total
        cmap  = LinearSegmentedColormap.from_list('c',['#ffffff',res['color']],N=256)
        ax.imshow(cm_n, cmap=cmap, vmin=0, vmax=cm_n.max()*1.2)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Pred\nNormal','Pred\nAnomaly'], fontsize=9)
        ax.set_yticklabels(['Actual\nNormal','Actual\nAnomaly'], fontsize=9)
        cell_labels = [['TN','FP'],['FN','TP']]
        for i in range(2):
            for j in range(2):
                bright = cm_n[i,j]/(cm_n.max()*1.2)
                col = 'white' if bright > 0.55 else '#222'
                count = cm[i,j]; pct = 100*count/total
                ax.text(j,i,f'{cell_labels[i][j]}\n{count:,}\n({pct:.1f}%)',
                        ha='center',va='center',fontsize=8.5,fontweight='bold',color=col)
        ax.set_title(name, fontsize=11, pad=6)
        ax.text(0.5,-0.16,f'F1 = {res["F1"]:.2f}%',ha='center',va='top',
                transform=ax.transAxes, fontsize=9, color='#006400', fontweight='bold')
    fig.suptitle('Part 5 — Confusion Matrices: Anomaly Transformer on Four New Domains\n'
                 '(computed directly from saved model predictions)',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0.07,1,0.90], w_pad=2.4)
    save('fig6_confusion_matrices')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7  —  Training convergence
# ─────────────────────────────────────────────────────────────────────────────
def fig7_training_convergence():
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.8))
    for ax, (name, res) in zip(axes, RESULTS.items()):
        log_rows = load_training_log(name)
        epochs = [row['epoch'] for row in log_rows]
        train_loss = [row['train_loss'] for row in log_rows]
        vali_loss = [row['vali_loss'] for row in log_rows]
        early_stop_epoch = next((row['epoch'] for row in log_rows if row['early_stop']), None)
        pcol = res['color']
        if name == 'SKAB':
            ax2 = ax.twinx()
            lv, = ax2.plot(epochs, vali_loss, 'o--', color='#FF9800',
                           lw=1.4, ms=4, label='Validation loss')
            ax2.set_ylabel('Validation loss', color='#FF9800', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='#FF9800', labelsize=7)
            ax2.spines['right'].set_color('#FF9800')
            lt, = ax.plot(epochs, train_loss, 's-', color=pcol,
                          lw=1.6, ms=4, label='Train loss')
            ax.set_ylabel('Train loss', color=pcol, fontsize=9)
            ax.tick_params(axis='y', labelcolor=pcol)
        else:
            lt, = ax.plot(epochs, train_loss, 's-', color=pcol, lw=1.6, ms=5, label='Train loss')
            lv, = ax.plot(epochs, vali_loss,  'o--', color='#FF9800', lw=1.4, ms=5, label='Validation loss')
            ax.set_ylabel('Loss (neg. ELBO)', fontsize=9)
        if early_stop_epoch:
            ax.axvline(x=early_stop_epoch, color='#888', linestyle=':', lw=1.2)
            ax.text(early_stop_epoch-0.03, 0.06, 'early stop',
                    fontsize=7, color='#555', va='bottom', ha='right',
                    rotation=90, transform=ax.get_xaxis_transform())
            ax.set_xlim(min(epochs) - 0.15, max(epochs) + 0.35)
        ax.set_xlabel('Epoch'); ax.set_title(name, fontsize=11)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.spines[['top']].set_visible(False)
        ax.yaxis.grid(True, linestyle='--', lw=0.4, alpha=0.5)
    fig.legend(handles=[
        Line2D([0], [0], color='#555555', marker='s', lw=1.6, label='Train loss'),
        Line2D([0], [0], color='#FF9800', marker='o', lw=1.4, linestyle='--',
               label='Validation loss'),
        Line2D([0], [0], color='#888888', lw=1.2, linestyle=':', label='Early stop'),
    ], loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.035))
    fig.suptitle('Part 4 — Training Loss Convergence\n'
                 'Read directly from training_logs/*.csv generated by solver.py',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0,0.12,1,0.88], w_pad=2.2)
    save('fig7_training_convergence')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8  —  Detection overlay
# ─────────────────────────────────────────────────────────────────────────────
def fig8_detection_overlay():
    fig = plt.figure(figsize=(14.4, 7.8))
    gs  = gridspec.GridSpec(2, 4, hspace=0.12, wspace=0.32,
                            figure=fig, left=0.06, right=0.98, top=0.84, bottom=0.17)
    for col, (name, res) in enumerate(RESULTS.items()):
        ax_sig   = fig.add_subplot(gs[0, col])
        ax_score = fig.add_subplot(gs[1, col], sharex=ax_sig)
        data, labels = load_test(name)
        outputs = load_model_outputs(name)
        score = outputs['score']
        threshold = float(outputs['threshold'])
        pred = outputs['pred'].astype(int)
        gt = outputs['gt'].astype(int)

        n = min(len(data), len(gt), len(score), len(pred))
        data = data[:n]
        gt = gt[:n]
        score = score[:n]
        pred = pred[:n]

        s, e = find_anomaly_window(gt)
        t = np.arange(e-s)
        lbl_s = gt[s:e]
        d_s = data[s:e]
        score_s = score[s:e]
        pred_s = pred[s:e]
        shade_anomaly(ax_sig, lbl_s)
        ax_sig.plot(t, d_s[:,0], color=res['color'], lw=0.9, zorder=2)
        ax_sig.set_ylabel(f'{res["sensors"][0]}\n(norm.)', fontsize=8)
        ax_sig.set_title(name, fontsize=11, pad=4)
        ax_sig.spines[['top','right']].set_visible(False)
        ax_sig.tick_params(labelbottom=False)
        ax_sig.yaxis.grid(True, linestyle='--', lw=0.35, alpha=0.5)
        # The raw model energy is highly skewed, so use a log transform for display.
        eps = max(np.percentile(score[score > 0], 1) if np.any(score > 0) else 1e-12, 1e-12)
        log_score = np.log10(score + eps)
        log_score_s = np.log10(score_s + eps)
        log_threshold = np.log10(threshold + eps)
        lo, hi = np.percentile(log_score, [1, 99.7])
        if hi <= lo:
            lo, hi = float(np.min(log_score)), float(np.max(log_score) + 1e-8)
        score_norm = np.clip((log_score_s - lo) / (hi - lo), 0, 1)
        threshold_norm = np.clip((log_threshold - lo) / (hi - lo), 0, 1)
        score_smooth = smooth_signal(score_norm, window=15)

        shade_predicted_anomaly(ax_score, pred_s)
        ax_score.plot(t, score_norm, color='#9ecae1', linewidth=0.6, alpha=0.42,
                      zorder=2, label='Raw model score')
        ax_score.plot(t, score_smooth, color='#08519c', linewidth=1.5,
                      zorder=3, label='Smoothed model score')
        pred_idx = np.where(pred_s == 1)[0]
        if len(pred_idx) > 0:
            stride = max(1, len(pred_idx) // 120)
            ax_score.scatter(pred_idx[::stride], score_norm[pred_idx][::stride],
                             s=9, color='#c44e52', alpha=0.85, zorder=4,
                             edgecolors='none')
        ax_score.axhline(y=threshold_norm, color='#333', linestyle='--', lw=1.1, zorder=3,
                         label='Threshold')
        ax_score.set_ylabel('Model\nscore', fontsize=8)
        ax_score.set_xlabel('Time step', fontsize=8)
        ax_score.set_ylim(0, 1.15)
        ax_score.spines[['top','right']].set_visible(False)
        ax_score.yaxis.grid(True, linestyle='--', lw=0.35, alpha=0.5)
    fig.legend(handles=[
        Line2D([0], [0], color='#08519c', lw=1.5, label='Smoothed real model score'),
        Line2D([0], [0], color='#333333', lw=1.1, linestyle='--', label='Threshold'),
        mpatches.Patch(facecolor='#c44e52',alpha=0.16,label='Predicted anomaly region'),
        mpatches.Patch(facecolor='#FF5252',alpha=0.35,label='Ground-truth anomaly window'),
    ], loc='lower center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5,0.055))
    fig.suptitle('Part 5 — Real Sensor Readings with Model Detection Overlay  (Colab T4 — definitive run)\n'
                 'Top: raw sensor signal  |  Bottom: log-scaled real model score vs threshold',
                 fontsize=12, fontweight='bold', y=0.965)
    save('fig8_detection_overlay')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9  —  Baseline comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig9_baseline_comparison():
    skab_methods = ['Isolation\nForest','LSTM-AD','OmniAnomaly','THOC','Anomaly\nTransformer']
    skab_f1      = [76.40, 81.22, 88.30, 91.50, 98.93]
    mit_methods  = ['Autoencoder','LSTM-AD','BeatGAN','OmniAnomaly','Anomaly\nTransformer']
    mit_f1       = [72.10, 79.50, 84.30, 88.60, 92.88]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for ax, (methods, f1s, title) in zip(axes, [
        (skab_methods, skab_f1, 'SKAB — Industrial Pump/Valve Fault Detection'),
        (mit_methods,  mit_f1,  'MIT-BIH — ECG Arrhythmia Detection'),
    ]):
        colors = ['#5b7fbe']*(len(methods)-1)+['#c44e52']
        bars = ax.bar(methods, f1s, color=colors, alpha=0.88, width=0.60, zorder=3)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.6,
                    f'{v:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        bars[-1].set_edgecolor('#8B0000'); bars[-1].set_linewidth(1.5)
        ax.set_ylabel('F1-score (%)'); ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylim(50, 108)
        ax.yaxis.grid(True, linestyle='--', lw=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=9)
    fig.suptitle('Part 6 — F1-Score Comparison: Anomaly Transformer vs Baselines  (Colab T4)\n'
                 'SKAB (industrial) and MIT-BIH (biomedical)',
                 fontsize=12, fontweight='bold', y=0.985)
    fig.legend(handles=[
        mpatches.Patch(facecolor='#5b7fbe',alpha=0.88,label='Baseline methods'),
        mpatches.Patch(facecolor='#c44e52',alpha=0.88,label='Anomaly Transformer (ours)'),
    ], loc='lower center', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5,0.02))
    plt.tight_layout(rect=[0,0.10,1,0.90], w_pad=2.4)
    save('fig9_baseline_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10  —  Side-by-side comparison: local run vs Colab T4 run
# ─────────────────────────────────────────────────────────────────────────────
def fig10_local_vs_colab():
    """Highlights the improvement — especially GECCO fixed on Colab."""
    local = {'SKAB':98.87,'TEP':99.82,'GECCO':0.00,'MIT-BIH':92.90}
    colab = {'SKAB':98.93,'TEP':99.82,'GECCO':98.72,'MIT-BIH':92.88}

    x = np.arange(len(DATASETS)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars_l = ax.bar(x - width/2, [local[d] for d in DATASETS], width,
                    label='Local run (CPU)', color='#aec7e8', alpha=0.88,
                    edgecolor='#1f77b4', linewidth=0.8, zorder=3)
    bars_c = ax.bar(x + width/2, [colab[d] for d in DATASETS], width,
                    label='Colab T4 (definitive)', color='#c44e52', alpha=0.88,
                    edgecolor='#8B0000', linewidth=0.8, zorder=3)

    for bar, v in zip(bars_l, [local[d] for d in DATASETS]):
        if v > 2:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)
        else:
            ax.text(bar.get_x()+bar.get_width()/2, 2.5,
                    'FAIL', ha='center', va='bottom', fontsize=8, color='#b22222',
                    fontweight='bold')
    for bar, v in zip(bars_c, [colab[d] for d in DATASETS]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color='#8B0000')

    ax.set_xticks(x); ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_ylabel('F1-score (%)')
    ax.set_title('Local Run vs Colab T4 — F1-Score Comparison\n'
                 'Key difference: GECCO threshold calibration fixed on Colab (GPU precision)',
                 pad=10)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, linestyle='--', lw=0.55, alpha=0.6, zorder=0)
    ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Arrow annotation for GECCO improvement
    gecco_i = DATASETS.index('GECCO')
    ax.annotate('GECCO fixed\non Colab',
                xy=(gecco_i + width/2, colab['GECCO']),
                xytext=(gecco_i - 0.35, 58),
                ha='center', va='center', fontsize=9, color='#006400',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#006400', lw=1.6,
                                shrinkA=4, shrinkB=4))

    plt.tight_layout(rect=[0,0.12,1,0.92])
    save('fig10_local_vs_colab_comparison')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('Generating report figures — OSAgnosticReAL (Colab T4) results')
    print(f'Output folder: report_figures_real/')
    print('=' * 65)

    print('\n[1/10] Figure 1 — P/R/F1 bar chart...')
    fig1_prf1()
    print('\n[2/10] Figure 2 — F1 all datasets...')
    fig2_f1_all()
    print('\n[3/10] Figure 3 — Results table...')
    fig3_table()
    print('\n[4/10] Figure 4 — Raw sensor data...')
    fig4_raw_sensor_data()
    print('\n[5/10] Figure 5 — Class imbalance...')
    fig5_class_imbalance()
    print('\n[6/10] Figure 6 — Confusion matrices...')
    fig6_confusion_matrices()
    print('\n[7/10] Figure 7 — Training convergence...')
    try:
        fig7_training_convergence()
    except FileNotFoundError as exc:
        print(f'  Skipped fig7_training_convergence: {exc}')
        print('  Create training_logs/*.csv by rerunning train mode with the updated solver.py.')
    print('\n[8/9] Figure 8 — Real model detection overlay...')
    try:
        fig8_detection_overlay()
    except FileNotFoundError as exc:
        print(f'  Skipped fig8_detection_overlay: {exc}')
        print('  Create test_outputs/*.npz by rerunning test mode with the updated solver.py.')
    print('\n[9/9] Figure 9 — Baseline comparison skipped.')
    print('  Figure 9 requires cited external baseline results, not model-output files.')

    print('\n' + '=' * 65)
    print('Done. Figures 1, 2, 3, 5, 6, and 8 use saved model outputs.')
    print('Figure 7 uses training_logs/*.csv when those files exist.')
    print('Figure 9 and Figure 10 are intentionally not generated for the report.')
    print('=' * 65)
