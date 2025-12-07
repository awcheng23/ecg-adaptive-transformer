import os
import wfdb

def download_afdb(data_dir="data/afdb"):
    """
    Download the MIT-BIH Atrial Fibrillation Database (AFDB)
    from PhysioNet using wfdb.
    """
    # Create parent directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading MIT-BIH Atrial Fibrillation Database into {data_dir}...")
    wfdb.dl_database(
        'afdb',        # <--- database name for AFDB
        dl_dir=data_dir
    )
    print("Download complete.")

if __name__ == "__main__":
    download_afdb()
