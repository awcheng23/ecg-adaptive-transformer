import os
import wfdb

def download_mitdb(data_dir="data/mitdb"):
    # Make parent data folder if needed
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading MIT-BIH Arrhythmia Database into {data_dir}...")
    wfdb.dl_database(
        'mitdb',
        dl_dir=data_dir
    )
    print("Download complete.")

if __name__ == "__main__":
    download_mitdb()