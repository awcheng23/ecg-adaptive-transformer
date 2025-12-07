"""
Beat-based preprocessing for ECG anomaly detection.

Generates train/val/test splits using beat-centered segments instead of fixed time windows.
Saves to separate .npz files for comparison with the original pipeline.
"""

import numpy as np
from data_dir.process_mitdb_beat_based import (
    get_patients_segments_beat_based,
    split_train_val_test,
    get_sampled_data,
)
from data_dir import PATIENT_IDS


def main():
    """
    Process all patients using beat-based segmentation.
    
    Each beat becomes a sample with context_sec seconds of surrounding ECG.
    Labels: 0=normal beat, 1=abnormal beat.
    """
    segments = []
    segment_IDs = []
    data_path = "data/mitdb/"

    print("Processing patients with beat-based segmentation...")
    for patient_num, id in enumerate(PATIENT_IDS):
        print(f"  Patient {patient_num + 1}/{len(PATIENT_IDS)}: ID {id}", flush=True)
        try:
            pat_segments, pat_segment_ID = get_patients_segments_beat_based(
                ID=id,
                dt_path=data_path,
                context_sec=2.5,  # 2.5 second window around each beat
                target_fs=100.0,
            )
            segments.extend(pat_segments)
            segment_IDs.extend(pat_segment_ID)
            
            # Show label breakdown for this patient
            pat_IDs_arr = np.array(pat_segment_ID, dtype=int)
            n_norm = np.sum(pat_IDs_arr == 0)
            n_abnorm = np.sum(pat_IDs_arr == 1)
            print(f"    Extracted {len(pat_segments)} beats: {n_norm} normal, {n_abnorm} abnormal", flush=True)
        except Exception as e:
            print(f"    Warning: Failed to process patient {id}: {e}", flush=True)
            continue

    print(f"\nTotal beat-centered segments extracted: {len(segments)}")

    # Convert to numpy array for analysis
    segment_IDs = np.array(segment_IDs, dtype=int)
    
    # Analyze class distribution
    n_normal = np.sum(segment_IDs == 0)
    n_abnormal = np.sum(segment_IDs == 1)
    print(f"Normal beats: {n_normal} ({100*n_normal/len(segment_IDs):.1f}%)")
    print(f"Abnormal beats: {n_abnormal} ({100*n_abnormal/len(segment_IDs):.1f}%)")
    print(f"Unique labels found: {np.unique(segment_IDs)}")

    print("\nSplitting into train/val/test...")
    train_dist, val_dist, test_dist = split_train_val_test(segment_IDs)

    # Sample to balanced sizes
    segments_train, segment_IDs_train = get_sampled_data(
        segments, segment_IDs, train_dist, num_samples=8680, augment=True
    )
    segments_val, segment_IDs_val = get_sampled_data(
        segments, segment_IDs, val_dist, num_samples=1302
    )
    segments_test, segment_IDs_test = get_sampled_data(
        segments, segment_IDs, test_dist, num_samples=1302
    )

    print(f"Train: {len(segments_train)} samples")
    print(f"Val:   {len(segments_val)} samples")
    print(f"Test:  {len(segments_test)} samples")

    print("\nSaving beat-based datasets...")
    np.savez_compressed(
        "data/db_train_anomaly_beat_based.npz",
        segments=segments_train,
        labels=segment_IDs_train,
    )
    np.savez_compressed(
        "data/db_val_anomaly_beat_based.npz",
        segments=segments_val,
        labels=segment_IDs_val,
    )
    np.savez_compressed(
        "data/db_test_anomaly_beat_based.npz",
        segments=segments_test,
        labels=segment_IDs_test,
    )
    print("Beat-based datasets saved:")
    print("  - data/db_train_anomaly_beat_based.npz")
    print("  - data/db_val_anomaly_beat_based.npz")
    print("  - data/db_test_anomaly_beat_based.npz")


if __name__ == "__main__":
    main()
