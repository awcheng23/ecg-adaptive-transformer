"""
Download MIT-BIH Atrial Fibrillation Database from PhysioNet.
Uses Python's urllib to avoid wget dependency.
"""

import os
import urllib.request
import urllib.error
import ssl
from pathlib import Path


# AFDB record IDs (23 available records)
AFDB_RECORDS = [
    '04015', '04043', '04048', '04126', '04746',
    '04908', '04936', '05091', '05121', '05261',
    '06426', '06453', '06995', '07162', '07859',
    '07879', '07910', '08215', '08219', '08378',
    '08405', '08434', '08455'
]


def download_file(url: str, dest_path: str, ssl_context=None) -> bool:
    """Download a single file from URL to destination path."""
    try:
        # Use urlopen with SSL context, then write to file
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with open(dest_path, 'wb') as f:
                f.write(response.read())
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False  # File doesn't exist
        else:
            print(f"  Error downloading {url}: {e}")
            return False
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def download_afdb(target_dir="data/afdb"):
    """Download AFDB records from PhysioNet."""
    os.makedirs(target_dir, exist_ok=True)
    
    # Create SSL context that doesn't verify certificates (for macOS)
    ssl_context = ssl._create_unverified_context()
    
    base_url = "https://physionet.org/files/afdb/1.0.0/"
    
    print(f"Downloading AFDB to {target_dir}...")
    print("This may take several minutes...\n")
    
    # File extensions to download for each record
    extensions = ['.hea', '.dat', '.atr']
    
    total_files = 0
    downloaded_files = 0
    
    for record_id in AFDB_RECORDS:
        print(f"Downloading record {record_id}...", end=" ", flush=True)
        record_success = True
        
        for ext in extensions:
            filename = f"{record_id}{ext}"
            url = base_url + filename
            dest_path = os.path.join(target_dir, filename)
            
            # Skip if already exists
            if os.path.exists(dest_path):
                continue
            
            total_files += 1
            if download_file(url, dest_path, ssl_context):
                downloaded_files += 1
            else:
                record_success = False
        
        if record_success:
            print("✓")
        else:
            print("✗ (some files missing)")
    
    # Also download RECORDS file
    print("\nDownloading metadata...", end=" ", flush=True)
    for filename in ['RECORDS', 'SHA256SUMS.txt']:
        url = base_url + filename
        dest_path = os.path.join(target_dir, filename)
        if not os.path.exists(dest_path):
            download_file(url, dest_path, ssl_context)
    print("✓")
    
    print(f"\n{'='*60}")
    print(f"Download complete! Files saved to {target_dir}")
    
    # List downloaded files
    files = [f for f in os.listdir(target_dir) if not f.startswith('.')]
    print(f"Downloaded {len(files)} files")
    
    # Count records (*.hea files)
    hea_files = [f for f in files if f.endswith('.hea')]
    print(f"Found {len(hea_files)} ECG records")
    
    if len(hea_files) < len(AFDB_RECORDS):
        print(f"\nWarning: Expected {len(AFDB_RECORDS)} records, found {len(hea_files)}")
        print("Some records may be missing. You can:")
        print("  1. Re-run this script")
        print("  2. Install wget: brew install wget")
        print("  3. Manually download from: https://physionet.org/content/afdb/1.0.0/")


if __name__ == "__main__":
    download_afdb()
