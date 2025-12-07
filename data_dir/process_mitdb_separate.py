import numpy as np
import wfdb
import scipy as spy
from os.path import join as join_path
from typing import Optional, Tuple, List, Dict


def get_record(ID: int, dt_path: str = "data/mitdb/"):
    """Obtain a patient record."""
    path = join_path(dt_path, f"{ID}")
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")
    return record, annotation


def get_bandpass_filter_signal(
    record: wfdb.Record, lowcut: float = 0.5, highcut: float = 45.0
) -> np.ndarray:
    """
    Apply a bandpass filter to remove noise and get signal.
    """
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
    """Normalize a signal to the range [0, 1]."""
    min_signal = np.min(signal) if min_signal is None else min_signal
    max_signal = np.max(signal) if max_signal is None else max_signal
    return (signal - min_signal) / (max_signal - min_signal + 1e-8)


def segment_signal(
    signal: np.ndarray,
    annotation: wfdb.Annotation,
    fs: float,
    window_sec: float = 50.0,
    stride_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Segment signal into fixed-width windows (default 50 seconds) with adjustable stride.

    Label rule (binary):
      - 0: all beats in the window are 'N'
      - 1: at least one beat in the window is NOT 'N'

    Parameters
    ----------
    signal : np.ndarray
        Raw/preprocessed ECG signal.
    annotation : wfdb.Annotation
        Beat annotations from WFDB.
    fs : float
        Sampling rate.
    window_sec : float
        Window length in seconds.
    stride_sec : float or None
        If None: defaults to window_sec (non-overlapping)
        If < window_sec: overlapping windows
        If > window_sec: skip windows

    Returns
    -------
    segments : List[np.ndarray]
        Segmented windows of size window_sec * fs.
    labels : List[int]
        Binary anomaly labels: 0 = all N, 1 = any abnormal beat.
    """

    window_samples = int(window_sec * fs)
    stride_samples = int((stride_sec if stride_sec else window_sec) * fs)
    N = len(signal)

    ann_samples = np.asarray(annotation.sample)
    ann_symbols = np.asarray(annotation.symbol, dtype=object)

    segments: List[np.ndarray] = []
    labels: List[int] = []

    for start in range(0, N - window_samples + 1, stride_samples):
        end = start + window_samples
        seg = signal[start:end]

        # Beat annotations inside this window
        mask = (ann_samples >= start) & (ann_samples < end)
        seg_symbols = ann_symbols[mask]

        # Label = 1 if any beat is not N
        label = 0
        if len(seg_symbols) > 0 and np.any(seg_symbols != "N"):
            label = 1

        segments.append(seg)
        labels.append(label)

    return segments, labels


def downsample_signal(
    signal: np.ndarray, orig_fs: float, target_fs: float = 100.0
) -> np.ndarray:
    """
    Downsample a 1D signal from orig_fs to target_fs using Fourier resampling.
    Used AFTER segmentation so labels based on original annotations stay correct.
    """
    if orig_fs == target_fs:
        return signal
    num_samples = int(len(signal) * target_fs / orig_fs)
    return spy.signal.resample(signal, num_samples)


def get_patients_segments(
    ID: int,
    dt_path: str = "data/mitdb/",
    window_sec: float = 50.0,
    stride_sec: Optional[float] = None,
    target_fs: Optional[float] = 100.0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Get all segments for a patient and their binary labels.

    - Filter: 0.5–45 Hz bandpass.
    - Normalize to [0, 1] over the whole record.
    - Segment into windows of `window_sec` sec with adjustable `stride_sec`.
    - Label = 1 if any beat in the segment is not 'N', else 0.
    - Optionally downsample each segment to `target_fs` (default 100 Hz).
    """
    record, annotation = get_record(ID=ID, dt_path=dt_path)

    # Filter and normalize in original sampling rate
    signal = get_bandpass_filter_signal(record=record)
    signal = normalize_signal(signal=signal)

    # Segment using original fs so annotation.sample stays valid
    segments, labels = segment_signal(
        signal=signal,
        annotation=annotation,
        fs=record.fs,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )

    # Optional downsampling of each segment to target_fs (e.g., 100 Hz)
    if target_fs is not None and record.fs != target_fs:
        segments = [
            downsample_signal(seg, orig_fs=record.fs, target_fs=target_fs)
            for seg in segments
        ]

    return segments, labels


def _get_label_distribution(labels: List[int]) -> Dict[int, List[int]]:
    """Get the indices of where each label occurs."""
    unique_labels = set(labels)
    dist: Dict[int, List[int]] = {label: [] for label in unique_labels}
    for i, lab in enumerate(labels):
        dist[lab].append(i)
    return dist


def split_patients(
    patient_ids: List[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
):
    """
    Split unique patient IDs into train/val/test sets.

    Returns
    -------
    train_pats, val_pats, test_pats : List[int]
        Lists of patient IDs for each split.
    """
    unique_pats = np.array(sorted(set(patient_ids)))
    np.random.shuffle(unique_pats)

    n = len(unique_pats)
    train_end = int(train_size * n)
    val_end = int((train_size + val_size) * n)

    train_pats = unique_pats[:train_end].tolist()
    val_pats = unique_pats[train_end:val_end].tolist()
    test_pats = unique_pats[val_end:].tolist()

    return train_pats, val_pats, test_pats


def split_train_val_test_by_patient(
    labels: List[int],
    patient_ids: List[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
):
    """
    Build label->indices dicts for train/val/test, ensuring NO patient appears
    in more than one split.

    Parameters
    ----------
    labels : List[int]
        Segment-level labels (e.g., 0/1).
    patient_ids : List[int]
        Same length as labels; patient_ids[i] is the patient for labels[i].
    train_size : float
        Fraction of patients for training.
    val_size : float
        Fraction of patients for validation.
        Test fraction is 1 - train_size - val_size.

    Returns
    -------
    train_dist, val_dist, test_dist : Dict[int, List[int]]
        Dicts mapping label -> list of indices in that split.
    """
    assert len(labels) == len(patient_ids), "labels and patient_ids must have same length"

    train_pats, val_pats, test_pats = split_patients(
        patient_ids=patient_ids,
        train_size=train_size,
        val_size=val_size,
    )

    train_pats_set = set(train_pats)
    val_pats_set = set(val_pats)
    test_pats_set = set(test_pats)

    train_dist: Dict[int, List[int]] = {}
    val_dist: Dict[int, List[int]] = {}
    test_dist: Dict[int, List[int]] = {}

    for idx, (lab, pid) in enumerate(zip(labels, patient_ids)):
        if pid in train_pats_set:
            train_dist.setdefault(lab, []).append(idx)
        elif pid in val_pats_set:
            val_dist.setdefault(lab, []).append(idx)
        elif pid in test_pats_set:
            test_dist.setdefault(lab, []).append(idx)

    return train_dist, val_dist, test_dist


def get_sampled_data(
    beats: List[np.ndarray],
    beat_IDs: List[int],
    dist: Dict[int, List[int]],
    num_samples: int,
    augment: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Sample labels to desired amount given label->indices distribution."""

    # Take a sample of the indices
    samples: List[int] = []
    for lab in dist:
        length = len(dist[lab])  # use permutation to ensure each label is visited once
        if length == 0:
            continue
        if augment:  # up sample minority labels
            indices = np.concatenate(
                [
                    np.random.permutation(length)
                    for _ in range(int(np.ceil(num_samples / length)))
                ]
            )[:num_samples]
        else:
            indices = np.random.permutation(length)[: min(num_samples, length)]
        samples.extend(np.array(dist[lab])[indices].tolist())

    # Keep the data and labels of the sampled indices
    beats_samp = [beats[i] for i in samples]
    beat_IDs_samp = [beat_IDs[i] for i in samples]

    return beats_samp, beat_IDs_samp
