import shutil
from pathlib import Path
import kagglehub

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

path = Path(kagglehub.dataset_download("thoughtvector/customer-support-on-twitter"))
print("Cache folder contents:", list(path.iterdir()))

# find the csv inside (it should be twcs.csv — verify the name from the print above)
for f in path.rglob("*.csv"):
    dest = RAW_DIR / f.name
    if not dest.exists():
        shutil.copy(f, dest)
    print("Copied to:", dest)
