import numpy as np
from data_dir.process_mitdb import (
    get_patients_segments,
    split_train_val_test,
    get_sampled_data,
)
from data_dir import PATIENT_IDS


def main():
    segments = []
    segment_IDs = []
    data_path = "data/mitdb/"

    for id in PATIENT_IDS:  # take out patients for testing
        pat_segments, pat_segment_ID = get_patients_segments(
            ID=id, dt_path=data_path, window_sec=50.0, stride_sec=2.5, target_fs=100.0
        )

        segments.extend(pat_segments)
        segment_IDs.extend(pat_segment_ID)

    print("Completed data load")

    train_dist, val_dist, test_dist = split_train_val_test(segment_IDs)
    segments_train, segment_IDs_train = get_sampled_data(
        segments, segment_IDs, train_dist, num_samples=15577, augment=True
    )
    segments_val, segment_IDs_val = get_sampled_data(
        segments, segment_IDs, val_dist, num_samples=1302
    )
    segments_test, segment_IDs_test = get_sampled_data(
        segments, segment_IDs, test_dist, num_samples=1302
    )
    print("Completed data sampling")

    np.savez_compressed(
        "data/db_train_anomaly.npz", segments=segments_train, labels=segment_IDs_train
    )
    np.savez_compressed(
        "data/db_val_anomaly.npz", segments=segments_val, labels=segment_IDs_val
    )
    np.savez_compressed(
        "data/db_test_anomaly.npz", segments=segments_test, labels=segment_IDs_test
    )
    print("Completed file save")


if __name__ == "__main__":
    main()
