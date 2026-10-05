import io
import os
import zipfile
import requests
import pandas as pd

ZENODO = "https://zenodo.org/records/18686475/files/"
MACH_TS_URL = ZENODO + "MACH_ts.zip?download=1"
DISCHARGE_URL = ZENODO + "discharge_cfs.zip?download=1"

DATA_DIR = "data"
MACH_ZIP = os.path.join(DATA_DIR, "MACH_ts.zip")
DISCHARGE_ZIP = os.path.join(DATA_DIR, "discharge_cfs.zip")
FINAL = os.path.join(DATA_DIR, "leaf_river_2003_2012.csv")

def download(url, path):
    if os.path.exists(path):
        print(f"Already downloaded: {path}")
        return
    print(f"Downloading {url}")
    print("This can take several minutes because the MACH archive is large.")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r{done/total*100:.1f}%", end="")
    print()

def find_member(z, name):
    candidates = [x for x in z.namelist() if x.replace("\\", "/").endswith(name)]
    if not candidates:
        raise FileNotFoundError(f"{name} not found in archive.")
    return candidates[0]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # MACH time-series archive: used for PRCP.
    download(MACH_TS_URL, MACH_ZIP)

    # Raw USGS discharge archive: used for ft^3/s target.
    download(DISCHARGE_URL, DISCHARGE_ZIP)

    site = "02472000"

    with zipfile.ZipFile(MACH_ZIP) as z:
        member = find_member(z, f"basin_{site}_MACH.csv")
        with z.open(member) as f:
            mach = pd.read_csv(f)

    with zipfile.ZipFile(DISCHARGE_ZIP) as z:
        member = find_member(z, f"{site}.csv")
        with z.open(member) as f:
            discharge = pd.read_csv(f)

    # Normalize column names.
    mach.columns = [str(c).strip() for c in mach.columns]
    discharge.columns = [str(c).strip() for c in discharge.columns]

    # Find date/precipitation columns robustly.
    date_m = next((c for c in mach.columns if c.lower() == "date"), None)
    prcp = next((c for c in mach.columns if c.upper() == "PRCP"), None)

    date_d = next(
    (c for c in discharge.columns if c.lower() in {"date", "datetime"}),
    None
)

    flow = next(
        (c for c in discharge.columns
        if c.lower() in {
            "discharge",
            "flow",
            "streamflow",
            "00060",
            "x_00060_00003"
        }),
        None
    )

    if not date_m or not prcp:
        raise ValueError(f"Could not identify Date/PRCP in MACH columns: {list(mach.columns)}")
    if not date_d or not flow:
        raise ValueError(f"Could not identify Date/streamflow in discharge columns: {list(discharge.columns)}")

    mach["Date"] = pd.to_datetime(mach[date_m])
    discharge["Date"] = pd.to_datetime(discharge[date_d])

    out_m = mach[["Date", prcp]].rename(columns={prcp: "Rainfall_mm"})
    out_d = discharge[["Date", flow]].rename(columns={flow: "Streamflow_cfs"})

    out = pd.merge(out_m, out_d, on="Date", how="inner")
    out = out[(out["Date"] >= "2003-01-01") & (out["Date"] <= "2012-12-31")]
    out = out.dropna().sort_values("Date").drop_duplicates("Date")

    if len(out) < 3000:
        raise ValueError(f"Unexpectedly few rows after filtering: {len(out)}")

    out.to_csv(FINAL, index=False)

    print(f"\nCreated: {FINAL}")
    print(f"Rows: {len(out)}")
    print(out.head())
    print(out.tail())

if __name__ == "__main__":
    main()
