import numpy as np
from data_dir.process_mitdb import (
    get_patients_segments,
    split_train_val_test_by_patient,
    get_sampled_data,
)
from data_dir import PATIENT_IDS


def main():
    segments = []
    segment_IDs = []
    patient_ids = []   # NEW: track which patient each segment comes from
    data_path = "data/mitdb/"

    for pid in PATIENT_IDS:  # loop over patients
        pat_segments, pat_segment_ID = get_patients_segments(
            ID=pid,
            dt_path=data_path,
            window_sec=50.0,
            stride_sec=2.5,
            target_fs=100.0,
        )

        segments.extend(pat_segments)
        segment_IDs.extend(pat_segment_ID)
        patient_ids.extend([pid] * len(pat_segments))

    print("Completed data load")
    print(f"Total segments: {len(segments)}")

    # Patient-wise split: no patient appears in more than one split
    train_dist, val_dist, test_dist = split_train_val_test_by_patient(
        labels=segment_IDs,
        patient_ids=patient_ids,
        train_size=0.7,
        val_size=0.15,
    )

    segments_train, segment_IDs_train = get_sampled_data(
        segments, segment_IDs, train_dist, num_samples=8680, augment=True
    )
    segments_val, segment_IDs_val = get_sampled_data(
        segments, segment_IDs, val_dist, num_samples=1302
    )
    segments_test, segment_IDs_test = get_sampled_data(
        segments, segment_IDs, test_dist, num_samples=1302
    )

    print("Completed data sampling")
    print(f"Train: {len(segments_train)}, Val: {len(segments_val)}, Test: {len(segments_test)}")

    np.savez_compressed(
        "data/db_train_anomaly.npz",
        segments=segments_train,
        labels=segment_IDs_train,
    )
    np.savez_compressed(
        "data/db_val_anomaly.npz",
        segments=segments_val,
        labels=segment_IDs_val,
    )
    np.savez_compressed(
        "data/db_test_anomaly.npz",
        segments=segments_test,
        labels=segment_IDs_test,
    )

    print("Completed file save")


if __name__ == "__main__":
    main()
