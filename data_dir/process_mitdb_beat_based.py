"""
Beat-based segmentation pipeline for MITDB.

Instead of fixed time windows, this extracts ECG segments centered on individual beats.
Useful for beat-level anomaly detection and exploring temporal locality of arrhythmias.
"""

import numpy as np
import wfdb
import scipy as spy
from os.path import join as join_path
from typing import Optional, Tuple, List, Dict


def get_record(ID: int, dt_path: str = "data/mitdb/"):
    """Obtain a patient record"""
    path = join_path(dt_path, f"{ID}")
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    return record, annotation


def get_bandpass_filter_signal(
    record: wfdb.Record, lowcut: float = 0.5, highcut: float = 45.0
) -> np.ndarray:
    """Apply a bandpass filter to remove noise and get signal."""
    ecg_signal = record.p_signal[:, 0]
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
    """Normalize a signal to the range [0, 1]"""
    min_signal = np.min(signal) if min_signal is None else min_signal
    max_signal = np.max(signal) if max_signal is None else max_signal
    return (signal - min_signal) / (max_signal - min_signal + 1e-8)


def segment_by_beats(
    signal: np.ndarray,
    annotation: wfdb.Annotation,
    fs: float,
    context_sec: float = 2.5,
    target_fs: Optional[float] = 100.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract segments centered on individual beats.

    For each beat annotation, extract a window of `context_sec` seconds
    centered on that beat (or context_sec/2 before and after).

    Label rule (binary):
      - 0: beat is 'N' (normal)
      - 1: beat is NOT 'N' (abnormal)

    Parameters
    ----------
    signal : np.ndarray
        Raw/preprocessed ECG signal.
    annotation : wfdb.Annotation
        Beat annotations from WFDB.
    fs : float
        Sampling rate.
    context_sec : float
        Total window duration in seconds around each beat.
    target_fs : float or None
        If provided, resample segments to target_fs after extraction.

    Returns
    -------
    segments : List[np.ndarray]
        Beat-centered segments of size context_sec * fs.
    labels : List[int]
        Binary anomaly labels: 0 = normal beat, 1 = abnormal beat.
    """
    context_samples = int(context_sec * fs)
    half_context = context_samples // 2

    ann_samples = np.asarray(annotation.sample)
    ann_symbols = np.asarray(annotation.symbol, dtype=object)

    segments = []
    labels = []

    N = len(signal)

    # Extract segment around each beat
    for beat_idx, (beat_sample, beat_symbol) in enumerate(zip(ann_samples, ann_symbols)):
        # Define window around beat
        start = beat_sample - half_context
        end = beat_sample + (context_samples - half_context)

        # Skip if window goes out of bounds
        if start < 0 or end > N:
            continue

        seg = signal[start:end]

        # Label based on beat symbol
        label = 0 if beat_symbol == "N" else 1

        # Optionally downsample to target_fs
        if target_fs is not None and target_fs != fs:
            num_samples = int(len(seg) * target_fs / fs)
            seg = spy.signal.resample(seg, num_samples)

        segments.append(seg)
        labels.append(label)

    return segments, labels


def get_patients_segments_beat_based(
    ID: int,
    dt_path: str = "data/mitdb/",
    context_sec: float = 2.5,
    target_fs: Optional[float] = 100.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Get all beat-centered segments for a patient and their binary labels.

    - Filter: 0.5–45 Hz bandpass.
    - Normalize to [0, 1] over the whole record.
    - Extract context_sec second windows centered on each beat.
    - Label = 1 if beat is not 'N', else 0.
    - Optionally resample to target_fs.

    Parameters
    ----------
    ID : int
        Patient ID from MITDB.
    dt_path : str
        Path to MITDB data directory.
    context_sec : float
        Window duration (seconds) around each beat.
    target_fs : float or None
        Target sampling rate for resampling.

    Returns
    -------
    segments : List[np.ndarray]
        Beat-centered segments.
    labels : List[int]
        Binary labels (0=normal, 1=abnormal).
    """
    record, annotation = get_record(ID=ID, dt_path=dt_path)

    # Filter and normalize
    signal = get_bandpass_filter_signal(record=record)
    signal = normalize_signal(signal=signal)

    # Beat-based segmentation
    segments, labels = segment_by_beats(
        signal=signal,
        annotation=annotation,
        fs=record.fs,
        context_sec=context_sec,
        target_fs=target_fs,
    )

    return segments, labels


def _get_label_distribution(labels: List[int]) -> Dict[int, List[int]]:
    """Get the indices of where each beat type occurs"""
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
    """
    Split indices of each label into train, validation, and test groups.

    Parameters
    ----------
    labels : List[int]
        List of integer labels (0 or 1).
    train_size : float
        Fraction of samples per class for training.
    val_size : float
        Fraction of samples per class for validation.
        Test size will be 1 - train_size - val_size.

    Returns
    -------
    train_dist, val_dist, test_dist : Dict[int, List[int]]
        Dicts mapping label -> list of indices.
    """
    train_dist = {}
    val_dist = {}
    test_dist = {}

    overall_dist = _get_label_distribution(labels)

    for lab, idx_list in overall_dist.items():
        data_length = len(idx_list)

        # Shuffle indices for this label
        shuffled = np.array(idx_list)[np.random.permutation(data_length)]

        # Calculate split boundaries
        train_end = int(train_size * data_length)
        val_end = int((train_size + val_size) * data_length)

        # Assign splits
        train_dist[lab] = shuffled[:train_end].tolist()
        val_dist[lab] = shuffled[train_end:val_end].tolist()
        test_dist[lab] = shuffled[val_end:].tolist()

    return train_dist, val_dist, test_dist


def get_sampled_data(
    beats: List[np.ndarray],
    beat_IDs: List[int],
    dist: Dict[int, List[int]],
    num_samples: int,
    augment: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Sample beat types to desired amount"""

    # Take a sample of the indices
    samples = []
    for id in dist:
        length = len(dist[id])
        if augment:  # up sample minority labels
            indices = np.concatenate(
                [
                    np.random.permutation(length)
                    for _ in range(int(np.ceil(num_samples / length)))
                ]
            )[:num_samples]
        else:
            indices = np.random.permutation(length)[:num_samples]
        samples.extend(np.array(dist[id])[indices].tolist())

    # Keep the data and labels of the sampled indices
    beats_samp = [beats[i] for i in samples]
    beat_IDs_samp = [beat_IDs[i] for i in samples]

    return beats_samp, beat_IDs_samp
