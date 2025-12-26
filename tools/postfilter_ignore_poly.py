import argparse, csv, ast
def inside(pt, poly):
    # ray-casting
    x,y = pt; n=len(poly); c=False
    for i in range(n):
        x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
        if ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1): c=not c
    return c

ap=argparse.ArgumentParser()
ap.add_argument("--tracks", required=True)
ap.add_argument("--out_csv", required=True)
ap.add_argument("--poly", required=True, help="e.g. '[[1150,40],[1270,40],[1270,140],[1150,140]]' (pixel coords)")
a=ap.parse_args()


poly = ast.literal_eval(a.poly)

with open(a.tracks) as fh, open(a.out_csv,"w",newline="") as out:
    rd=csv.DictReader(fh); wr=csv.DictWriter(out, fieldnames=rd.fieldnames); wr.writeheader()
    kept=0; dropped=0
    for r in rd:
        try: x=float(r["x_px"]); y=float(r["y_px"])
        except: wr.writerow(r); continue
        if inside((x,y), poly): dropped+=1
        else: wr.writerow(r); kept+=1
print(f"kept={kept}, dropped_in_poly={dropped}")
