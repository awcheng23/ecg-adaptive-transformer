from typing import Optional, Tuple, List, Dict
import numpy as np
import wfdb
import numpy as np
import scipy as spy
from os.path import join as join_path

def get_record(ID: int, dt_path: str = 'data/mitdb/'):

    """Obtain a patient record"""

    path = join_path(dt_path, f'{ID}')
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, 'atr')

    return record, annotation

def get_bandpass_filter_signal(record: wfdb.Record, 
                               lowcut: float = 0.5,
                               highcut: float = 45.0) -> np.ndarray:
    
    """Apply a bandpass filter to remove noise and get signal"""

    ecg_signal = record.p_signal[:,0]
    fs = record.fs
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = spy.signal.butter(5, [low, high], btype='band')

    return spy.signal.filtfilt(b, a, ecg_signal)

def normalize_signal(signal: np.ndarray,
                     min_signal: Optional[float] = None,
                     max_signal: Optional[float] = None) -> np.ndarray:
    
    """Normalize a signal to the range [0, 1]"""

    min_signal = np.min(signal) if type(min_signal) == type(None) else min_signal
    max_signal = np.max(signal) if type(max_signal) == type(None) else max_signal

    return (signal - min_signal) / (max_signal - min_signal)

def segment_signal(signal: np.ndarray,
                   annotation: wfdb.Annotation) -> Tuple[List[np.ndarray], List[int]]:
    
    """Segment signal for fixed time interval of ???"""

    pass

def _get_label_distribution(labels: List[int]) -> Dict[int, List[int]]:
        
    """Get the indices of where each beat type occurs"""

    unique_labels = set(labels)
    dist = {}
    for label in unique_labels:
        dist[label] = []

    for i in range(len(labels)):
        dist[labels[i]].append(i)

    return dist

def split_train_test(labels: List[int],
                     train_size: float = 0.8) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    
    """Split the indices of the each beat type into training and testing groups"""

    train_dist = {}
    test_dist = {}

    overall_dist = _get_label_distribution(labels)
    for id in overall_dist:
        data_length = len(overall_dist[id])
        shuffle_order = np.random.permutation(data_length)
        shuffled = np.array(overall_dist[id])[shuffle_order]

        split = int(train_size * data_length) 
        train_dist[id] = shuffled[:split].tolist()
        test_dist[id] = shuffled[split:].tolist()

    return train_dist, test_dist

def get_sampled_data(beats: List[np.ndarray],
                     beat_IDs: List[int],
                     dist: Dict[int, List[int]],
                     num_samples: int,
                     augment: bool = False) -> Tuple[List[np.ndarray], List[int]]:
    
    """Sample beat types to desired amount"""

    # Take a sample of the indices 
    samples = []
    for id in dist:
        length = len(dist[id]) # use permutation to ensure each label is visited once
        if augment == True: # up sample minority labels
            indices = np.concatenate([np.random.permutation(length) for _ in range(int(np.ceil(num_samples/length)))])[:num_samples] 
        else:
            indices = np.random.permutation(length)[:num_samples]
        samples.extend(np.array(dist[id])[indices].tolist())

    # Keep the data and labels of the sampled indices
    beats_samp = [beats[i] for i in samples]
    beat_IDs_samp = [beat_IDs[i] for i in samples]

    return beats_samp, beat_IDs_samp

def get_patients_segments(ID: int, dt_path: str = 'data/mitdb/') -> Tuple[List[np.ndarray], List[int]]:

    """Get all segments for a patient"""

    record, annotation = get_record(ID=ID, dt_path=dt_path)
    signal = get_bandpass_filter_signal(record=record)
    signal = normalize_signal(signal=signal)
    beats, beat_IDs = segment_signal(signal=signal, annotation=annotation)

    return beats, beat_IDs
