"""Data contracts, explicit signal transforms and evaluation shared by this project."""
import hashlib
import importlib.metadata
import json
import platform
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

AGES = np.array([25, 35, 45, 55, 65, 75])


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def environment():
    versions = {}
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'tensorflow-cpu', 'keras', 'torch', 'torchvision']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def read_frame(path):
    df = pd.read_csv(path, dtype={'subject_id': str})
    if df.empty or 'subject_id' not in df or df.subject_id.isna().any():
        raise ValueError('A nonempty CSV with subject_id is required; use prepare_data.py.')
    df.subject_id = df.subject_id.str.strip()
    if df.subject_id.eq('').any() or df.subject_id.duplicated().any():
        raise ValueError('Subject IDs must be nonempty and unique: one waveform per subject/site.')
    return df


def load_signals(path, length):
    df = read_frame(path)
    columns = [c for c in df if c != 'subject_id']
    if not columns or any(re.fullmatch(r's\d+', c) is None for c in columns):
        raise ValueError('Waveform columns must be s0000, s0001, ...; metadata must be excluded.')
    if [int(c[1:]) for c in columns] != list(range(len(columns))):
        raise ValueError('Waveform columns must be contiguous and ordered from s0000.')
    values = df[columns].apply(pd.to_numeric, errors='raise').to_numpy(np.float32)
    x = np.zeros((len(df), length), np.float32)
    for i, row in enumerate(values):
        valid = np.flatnonzero(~np.isnan(row))
        if not len(valid): raise ValueError('Empty waveform.')
        end = valid[-1] + 1
        if end > length: raise ValueError(f'Waveform length {end} exceeds --length {length}; no silent cropping.')
        part = row[:end]
        if not np.isfinite(part).all(): raise ValueError('Internal missing samples or infinite values.')
        if np.ptp(part) <= 1e-8: raise ValueError('Constant waveform.')
        x[i, :end] = part
    return df.subject_id.to_numpy(), x


def load_data(signals, targets, length, classification):
    ids, x = load_signals(signals, length)
    labels = read_frame(targets)
    column = 'age' if classification else 'cfpwv_m_s'
    if set(labels.columns) != {'subject_id', column}: raise ValueError(f'Target columns must be subject_id,{column}.')
    if set(ids) != set(labels.subject_id): raise ValueError('Signal/target subject ID sets differ.')
    y = pd.to_numeric(labels.set_index('subject_id').loc[ids, column], errors='raise').to_numpy(float)
    if not np.isfinite(y).all() or (y <= 0).any(): raise ValueError('Targets must be finite and positive.')
    if classification:
        if not np.isin(y, AGES).all(): raise ValueError('Expected age categories 25,35,45,55,65,75.')
        y = np.searchsorted(AGES, y)
    return ids, x, y


def demo_data(folder, classification, seed=42):
    """Artificial software fixtures, not physiological simulations or PWDB records."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes = np.repeat(np.arange(6), 12)
    rng.shuffle(classes)
    t = np.linspace(0, 1, 512, endpoint=False)
    waves, targets = [], []
    for c in classes:
        z = c / 5 + rng.normal(0, .02)
        wave = np.sin(2*np.pi*(1.1 + .3*z)*t) + (.15 + .25*z)*np.sin(4*np.pi*t + .2)
        wave += rng.normal(0, .015, len(t))
        waves.append((wave-wave.min()) / np.ptp(wave))
        targets.append(AGES[c] if classification else 5 + 5*z + rng.normal(0, .08))
    ids = [f'demo-{i:04d}' for i in range(len(classes))]
    df = pd.DataFrame(waves, columns=[f's{i:04d}' for i in range(512)])
    df.insert(0, 'subject_id', ids)
    sig, lab = folder/'signals.csv', folder/'targets.csv'
    df.to_csv(sig, index=False, float_format='%.7g')
    pd.DataFrame({'subject_id': ids, 'age' if classification else 'cfpwv_m_s': targets}).to_csv(lab, index=False)
    return sig, lab


def split_data(y, seed, folds, classification, holdout=False):
    if len(y) < 20: raise ValueError('At least 20 subjects are required.')
    if classification and (np.bincount(y, minlength=6) < max(5, folds)).any():
        raise ValueError('Each age category needs at least max(5,folds) subjects.')
    indices = np.arange(len(y))
    if holdout:
        outer = [train_test_split(indices, test_size=.2, random_state=seed)]
    else:
        splitter = (StratifiedKFold if classification else KFold)(folds, shuffle=True, random_state=seed)
        outer = splitter.split(indices, y)
    result = []
    for train, test in outer:
        train, val = train_test_split(train, test_size=.2, random_state=seed,
                                      stratify=y[train] if classification else None)
        result.append((train, val, test))
    return result


def fit_scale(x):
    return {'mean': float(np.mean(x, dtype=np.float64)), 'std': max(float(np.std(x, dtype=np.float64)), 1e-8)}


def scale(x, state):
    return ((x-state['mean'])/state['std']).astype(np.float32)


def inverse(y, state):
    return np.asarray(y).reshape(-1)*state['std']+state['mean']


def represent(x, kind, config):
    if kind == 'mlp': return x
    if kind in ['waveform', 'cnn1d']: return x[..., None]
    window = ('tukey', .25) if config['window'] == 'tukey' else config['window']
    _, _, power = signal.spectrogram(x, fs=config['fs'], nperseg=config['nperseg'], window=window,
        noverlap=config['nperseg']//8, detrend='constant', axis=-1)
    spec = (10*np.log10(np.maximum(power, 1e-10))).astype(np.float32)
    if kind in ['spectrogram', 'cnn2d']:
        if kind == 'cnn2d' and min(spec.shape[1:]) < 4: raise ValueError('Spectrogram too small for two pooling layers; increase --length.')
        return spec[..., None]
    size = 32 if kind == 'vgg16' else config['image_size']
    lo = spec.min(axis=(1,2), keepdims=True)
    values = (spec-lo)/np.maximum(spec.max(axis=(1,2), keepdims=True)-lo, 1e-8)
    resized = np.stack([ndimage.zoom(a, (size/a.shape[0], size/a.shape[1]), order=1, prefilter=False) for a in values])
    rgb = np.repeat(resized[..., None], 3, axis=-1).astype(np.float32)
    if kind == 'vgg16': return rgb[..., ::-1]*255-np.array([103.939,116.779,123.68], np.float32)
    return (rgb-np.array([.485,.456,.406], np.float32))/np.array([.229,.224,.225], np.float32)


def metrics(y, prediction, classification):
    if classification:
        return {'accuracy': float(accuracy_score(y,prediction)),
                'macro_f1': float(f1_score(y,prediction,labels=np.arange(6),average='macro',zero_division=0)),
                'weighted_f1': float(f1_score(y,prediction,labels=np.arange(6),average='weighted',zero_division=0))}
    return {'mae_m_s': float(mean_absolute_error(y,prediction)),
            'rmse_m_s': float(np.sqrt(mean_squared_error(y,prediction))), 'r2': float(r2_score(y,prediction)),
            'mape_percent': float(np.mean(np.abs((y-prediction)/y))*100)}


def plot_evaluation(y, prediction, path, classification, demo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    title = 'Artificial-data software check' if demo else 'Held-out evaluation'
    if classification:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        fig, ax = plt.subplots(figsize=(6,5))
        ConfusionMatrixDisplay(confusion_matrix(y,prediction,labels=np.arange(6)), display_labels=AGES).plot(ax=ax,cmap='Blues',colorbar=False)
        ax.set(title=title,xlabel='Predicted age category',ylabel='True age category')
    else:
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        lo,hi = min(y.min(),prediction.min()),max(y.max(),prediction.max())
        axes[0].scatter(y,prediction,s=15,alpha=.6)
        axes[0].plot([lo,hi],[lo,hi],'--',color='gray')
        axes[0].set(xlabel='True cf-PWV (m/s)',ylabel='Predicted cf-PWV (m/s)')
        mean,diff = (y+prediction)/2,y-prediction
        axes[1].scatter(mean,diff,s=15,alpha=.6)
        for line in [diff.mean(),diff.mean()-1.96*diff.std(ddof=1),diff.mean()+1.96*diff.std(ddof=1)]:
            axes[1].axhline(line,linestyle='--',color='gray')
        axes[1].set(xlabel='Mean of true and predicted (m/s)',ylabel='True − predicted (m/s)')
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path,dpi=140)
    plt.close(fig)
