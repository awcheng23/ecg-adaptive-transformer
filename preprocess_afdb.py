"""
Preprocess AFDB for binary rhythm classification (Normal vs Abnormal).

Generates train/val/test splits with 30-second windows, no overlap.
Keeps original 250 Hz sampling rate for maximum detail.
Labels: 0=normal rhythm (N, SBR), 1=abnormal (AFIB, AFL, J).
"""

import numpy as np
from data_dir.process_afdb import (
    AFDB_RECORDS,
    get_record_segments,
    split_train_val_test,
    get_sampled_data,
)


def main():
    """
    Process all AFDB records into rhythm-labeled segments.
    
    Window: 30 seconds @ 100Hz = 3000 samples
    Stride: 30 seconds (no overlap)
    Label: dominant rhythm in window (0=normal, 1=abnormal: AFIB/AFL/J)
    """
    segments = []
    segment_labels = []
    data_path = "data/afdb/"
    
    print("Processing AFDB records for rhythm classification...")
    print(f"Total records: {len(AFDB_RECORDS)}\n")
    
    for record_num, record_id in enumerate(AFDB_RECORDS):
        print(f"  Record {record_num + 1}/{len(AFDB_RECORDS)}: {record_id}", flush=True)
        try:
            rec_segments, rec_labels = get_record_segments(
                record_id=record_id,
                dt_path=data_path,
                window_sec=30.0,    # 30-second windows
                stride_sec=30.0,    # 30-second stride (no overlap)
                target_fs=None,     # keep original 250 Hz
                channel=0,          # use first ECG lead
            )
            
            # Filter out "other" rhythms (label=-1)
            valid_indices = [i for i, lab in enumerate(rec_labels) if lab != -1]
            rec_segments = [rec_segments[i] for i in valid_indices]
            rec_labels = [rec_labels[i] for i in valid_indices]
            
            segments.extend(rec_segments)
            segment_labels.extend(rec_labels)
            
            # Show label breakdown for this record
            labels_arr = np.array(rec_labels, dtype=int)
            n_normal = np.sum(labels_arr == 0)
            n_afib = np.sum(labels_arr == 1)
            print(f"    Extracted {len(rec_segments)} windows: {n_normal} normal, {n_afib} AFib", flush=True)
            
        except Exception as e:
            print(f"    Warning: Failed to process record {record_id}: {e}", flush=True)
            continue
    
    print(f"\n{'='*60}")
    print(f"Total segments extracted: {len(segments)}")
    
    # Convert to numpy for analysis
    segment_labels = np.array(segment_labels, dtype=int)
    
    # Analyze class distribution
    n_normal = np.sum(segment_labels == 0)
    n_abnormal = np.sum(segment_labels == 1)
    print(f"Normal rhythm: {n_normal} ({100*n_normal/len(segment_labels):.1f}%)")
    print(f"Abnormal (AFIB/AFL/J): {n_abnormal} ({100*n_abnormal/len(segment_labels):.1f}%)")
    print(f"Unique labels: {np.unique(segment_labels)}")
    
    if len(segments) == 0:
        print("\nError: No segments extracted. Check if AFDB data exists in data/afdb/")
        print("Run: python3 data_dir/download_afdb.py")
        return
    
    print(f"\n{'='*60}")
    print("Splitting into train/val/test (70/15/15)...")
    train_dist, val_dist, test_dist = split_train_val_test(
        segment_labels.tolist(), 
        train_size=0.7, 
        val_size=0.15
    )
    
    # Use all data WITHOUT class balancing (realistic imbalanced distribution)
    segments_train, labels_train = get_sampled_data(
        segments, segment_labels.tolist(), train_dist, 
        num_samples=None,  # use all
        augment=False      # NO balancing - keep real distribution
    )
    
    segments_val, labels_val = get_sampled_data(
        segments, segment_labels.tolist(), val_dist, 
        num_samples=None,
        augment=False      # NO balancing
    )
    
    segments_test, labels_test = get_sampled_data(
        segments, segment_labels.tolist(), test_dist, 
        num_samples=None,
        augment=False      # NO balancing
    )
    
    # Analyze final splits
    train_labels_arr = np.array(labels_train, dtype=int)
    val_labels_arr = np.array(labels_val, dtype=int)
    test_labels_arr = np.array(labels_test, dtype=int)
    
    print(f"\nTrain: {len(segments_train)} samples")
    print(f"  Normal: {np.sum(train_labels_arr == 0)}, Abnormal: {np.sum(train_labels_arr == 1)}")
    print(f"Val:   {len(segments_val)} samples")
    print(f"  Normal: {np.sum(val_labels_arr == 0)}, Abnormal: {np.sum(val_labels_arr == 1)}")
    print(f"Test:  {len(segments_test)} samples")
    print(f"  Normal: {np.sum(test_labels_arr == 0)}, Abnormal: {np.sum(test_labels_arr == 1)}")
    
    print(f"\n{'='*60}")
    print("Saving datasets...")
    np.savez_compressed(
        "data/db_train_afib.npz",
        segments=segments_train,
        labels=labels_train,
    )
    np.savez_compressed(
        "data/db_val_afib.npz",
        segments=segments_val,
        labels=labels_val,
    )
    np.savez_compressed(
        "data/db_test_afib.npz",
        segments=segments_test,
        labels=labels_test,
    )
    print("AFDB datasets saved:")
    print("  - data/db_train_afib.npz")
    print("  - data/db_val_afib.npz")
    print("  - data/db_test_afib.npz")
    print(f"\n{'='*60}")
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
