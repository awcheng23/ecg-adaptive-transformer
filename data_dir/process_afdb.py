"""
Process MIT-BIH Atrial Fibrillation Database for rhythm classification.

AFDB contains rhythm annotations (episodes of AFIB, AFL, normal, etc.)
rather than beat-by-beat labels. We segment into windows and label by
the dominant rhythm in each window.
"""

import numpy as np
import wfdb
import scipy as spy
from os.path import join as join_path
from typing import Optional, Tuple, List, Dict


# AFDB record IDs (25 records, each ~10 hours)
AFDB_RECORDS = [
    '04015', '04043', '04048', '04126', '04746',
    '04908', '04936', '05091', '05121', '05261',
    '06426', '06453', '06995', '07162', '07859',
    '07879', '07910', '08215', '08219', '08378',
    '08405', '08434', '08455'
]


def get_record(record_id: str, dt_path: str = "data/afdb/"):
    """Load AFDB record and rhythm annotations."""
    path = join_path(dt_path, record_id)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")  # rhythm annotations
    return record, annotation


def get_bandpass_filter_signal(
    record: wfdb.Record, 
    channel: int = 0,
    lowcut: float = 0.5, 
    highcut: float = 40.0
) -> np.ndarray:
    """
    Apply bandpass filter to remove noise.
    
    AFDB sampled at 250 Hz (vs 360 Hz in MIT-BIH Arrhythmia).
    Use slightly lower highcut (40 Hz) to avoid aliasing.
    """
    ecg_signal = record.p_signal[:, channel]
    fs = record.fs
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = spy.signal.butter(5, [low, high], btype="band")
    return spy.signal.filtfilt(b, a, ecg_signal)


def normalize_signal(
    signal: np.ndarray,
    min_signal: Optional[float] = None,
    max_signal: Optional[float] = None,
) -> np.ndarray:
    """Normalize signal to [0, 1]."""
    min_signal = np.min(signal) if min_signal is None else min_signal
    max_signal = np.max(signal) if max_signal is None else max_signal
    return (signal - min_signal) / (max_signal - min_signal + 1e-8)


def segment_by_rhythm(
    signal: np.ndarray,
    annotation: wfdb.Annotation,
    fs: float,
    window_sec: float = 10.0,
    stride_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Segment signal into fixed windows and label by dominant rhythm.
    
    AFDB rhythm annotations:
    - (AFIB : atrial fibrillation
    - (AFL  : atrial flutter  
    - (N    : normal sinus rhythm
    - (AB   : atrial bigeminy
    - (IVR  : idioventricular rhythm
    - (PREX : pre-excitation (WPW)
    - (SBR  : sinus bradycardia
    - (SVTA : supraventricular tachyarrhythmia
    - (T    : ventricular trigeminy
    - (VFL  : ventricular flutter
    - (VT   : ventricular tachycardia
    - (J    : junctional rhythm
    
    For binary classification:
    - Label 0: Normal rhythm (N, SBR)
    - Label 1: Abnormal rhythm (AFIB, AFL, J)
    - Discard other rhythms (label = -1)
    
    Parameters
    ----------
    signal : np.ndarray
        Preprocessed ECG signal.
    annotation : wfdb.Annotation
        Rhythm annotations from WFDB.
    fs : float
        Sampling rate (250 Hz for AFDB).
    window_sec : float
        Window length in seconds.
    stride_sec : float or None
        Stride for sliding window.
    
    Returns
    -------
    segments : List[np.ndarray]
        Windowed ECG segments.
    labels : List[int]
        Binary labels: 0=normal, 1=abnormal (AFIB/AFL/J), -1=other (discarded later).
    """
    window_samples = int(window_sec * fs)
    stride_samples = int((stride_sec if stride_sec else window_sec) * fs)
    N = len(signal)
    
    ann_samples = np.asarray(annotation.sample)
    ann_symbols = np.asarray(annotation.aux_note, dtype=object)  # rhythm annotations in aux_note
    
    segments = []
    labels = []
    
    # Build rhythm timeline: map each sample to its rhythm label
    rhythm_timeline = np.empty(N, dtype=object)
    rhythm_timeline[:] = None
    
    for i in range(len(ann_samples)):
        start_sample = ann_samples[i]
        end_sample = ann_samples[i + 1] if i + 1 < len(ann_samples) else N
        rhythm = ann_symbols[i]
        rhythm_timeline[start_sample:end_sample] = rhythm
    
    # Sliding window segmentation
    for start in range(0, N - window_samples + 1, stride_samples):
        end = start + window_samples
        seg = signal[start:end]
        
        # Get rhythm annotations in this window
        window_rhythms = rhythm_timeline[start:end]
        window_rhythms = window_rhythms[window_rhythms != None]  # filter None
        
        if len(window_rhythms) == 0:
            continue  # skip windows with no annotations
        
        # Find dominant rhythm (most common)
        unique, counts = np.unique(window_rhythms, return_counts=True)
        dominant_rhythm = unique[np.argmax(counts)]
        
        # Map to binary label
        if dominant_rhythm in ['(N', '(SBR']:
            label = 0  # normal
        elif dominant_rhythm in ['(AFIB', '(AFL', '(J']:
            label = 1  # abnormal: atrial fibrillation, flutter, or junctional
        else:
            label = -1  # other (will discard)
        
        segments.append(seg)
        labels.append(label)
    
    return segments, labels


def downsample_signal(
    signal: np.ndarray, 
    orig_fs: float, 
    target_fs: float = 100.0
) -> np.ndarray:
    """Downsample from orig_fs to target_fs using Fourier resampling."""
    if orig_fs == target_fs:
        return signal
    num_samples = int(len(signal) * target_fs / orig_fs)
    return spy.signal.resample(signal, num_samples)


def get_record_segments(
    record_id: str,
    dt_path: str = "data/afdb/",
    window_sec: float = 10.0,
    stride_sec: Optional[float] = 2.0,
    target_fs: Optional[float] = 100.0,
    channel: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Process one AFDB record into windowed segments.
    
    Parameters
    ----------
    record_id : str
        AFDB record ID (e.g., '04015').
    dt_path : str
        Path to AFDB data directory.
    window_sec : float
        Window duration in seconds.
    stride_sec : float
        Stride for sliding window.
    target_fs : float or None
        Target sampling rate for resampling.
    channel : int
        ECG channel (0 or 1, AFDB has 2 leads).
    
    Returns
    -------
    segments : List[np.ndarray]
        Processed segments.
    labels : List[int]
        Binary rhythm labels (0=normal, 1=abnormal: AFIB/AFL/J).
    """
    record, annotation = get_record(record_id, dt_path)
    
    # Filter and normalize
    signal = get_bandpass_filter_signal(record, channel=channel)
    signal = normalize_signal(signal)
    
    # Segment by rhythm
    segments, labels = segment_by_rhythm(
        signal=signal,
        annotation=annotation,
        fs=record.fs,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    
    # Downsample if requested
    if target_fs is not None and record.fs != target_fs:
        segments = [
            downsample_signal(seg, orig_fs=record.fs, target_fs=target_fs)
            for seg in segments
        ]
    
    return segments, labels


def _get_label_distribution(labels: List[int]) -> Dict[int, List[int]]:
    """Get indices for each label class."""
    unique_labels = set(labels)
    dist: Dict[int, List[int]] = {label: [] for label in unique_labels}
    for i, lab in enumerate(labels):
        dist[lab].append(i)
    return dist


def split_train_val_test(
    labels: List[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]]
]:
    """Split indices per class into train/val/test."""
    train_dist = {}
    val_dist = {}
    test_dist = {}
    
    overall_dist = _get_label_distribution(labels)
    
    for lab, idx_list in overall_dist.items():
        data_length = len(idx_list)
        shuffled = np.array(idx_list)[np.random.permutation(data_length)]
        
        train_end = int(train_size * data_length)
        val_end = int((train_size + val_size) * data_length)
        
        train_dist[lab] = shuffled[:train_end].tolist()
        val_dist[lab] = shuffled[train_end:val_end].tolist()
        test_dist[lab] = shuffled[val_end:].tolist()
    
    return train_dist, val_dist, test_dist


def get_sampled_data(
    segments: List[np.ndarray],
    labels: List[int],
    dist: Dict[int, List[int]],
    num_samples: Optional[int] = None,
    augment: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Sample segments to desired amount with optional class balancing."""
    samples = []
    
    if num_samples is None:
        # Use all data, optionally balance classes
        if augment:
            class_sizes = {lab: len(idx_list) for lab, idx_list in dist.items()}
            max_class_size = max(class_sizes.values())
            
            for lab, idx_list in dist.items():
                length = len(idx_list)
                if length < max_class_size:
                    indices = np.concatenate(
                        [
                            np.random.permutation(length)
                            for _ in range(int(np.ceil(max_class_size / length)))
                        ]
                    )[:max_class_size]
                else:
                    indices = np.random.permutation(length)
                samples.extend(np.array(idx_list)[indices].tolist())
        else:
            for lab, idx_list in dist.items():
                samples.extend(idx_list)
    else:
        for lab, idx_list in dist.items():
            length = len(idx_list)
            if augment:
                indices = np.concatenate(
                    [
                        np.random.permutation(length)
                        for _ in range(int(np.ceil(num_samples / length)))
                    ]
                )[:num_samples]
            else:
                indices = np.random.permutation(length)[:num_samples]
            samples.extend(np.array(idx_list)[indices].tolist())
    
    segments_samp = [segments[i] for i in samples]
    labels_samp = [labels[i] for i in samples]
    
    return segments_samp, labels_samp
