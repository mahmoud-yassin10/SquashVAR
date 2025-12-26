import csv, math, sys, os, argparse

parser = argparse.ArgumentParser(description="Compute detection rate from tracks.csv")
parser.add_argument("csv_path", help="Path to tracks.csv")
parser.add_argument("-t", "--thresh", type=float, default=0.30, help="Confidence threshold (default: 0.30)")
args = parser.parse_args()

p = args.csv_path
rows = det = 0

try:
    with open(p, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError("CSV appears empty or malformed.")

        # try common column names
        candidates = {"conf", "confidence", "score"}
        conf_col = next((c for c in rdr.fieldnames if c.strip().lower() in candidates), None)
        if not conf_col:
            raise KeyError(f"No confidence column found. Columns: {rdr.fieldnames}")

        for r in rdr:
            rows += 1
            try:
                c = float(r.get(conf_col, "nan"))
            except (TypeError, ValueError):
                c = float("nan")
            if not math.isnan(c) and c >= args.thresh:
                det += 1

    rate = det / max(rows, 1) * 100
    print(f"{os.path.basename(p)} -> frames: {rows} | detected>={args.thresh:.2f}: {det} | rate={rate:.1f}%")

except FileNotFoundError:
    print(f"File not found: {p}")
except Exception as e:
    print(f"Error: {e}")
