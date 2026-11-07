import joblib, glob, os, re, collections
print("== raw KIT ==")
raw=glob.glob("/data/ljx/datasets/KIT/*/*.npz")
raw_stairs=[p for p in raw if "upstairs" in p or "downstairs" in p]
raw_flat=[p for p in raw if not ("upstairs" in p or "downstairs" in p)]
print("raw total",len(raw),"stairs",len(raw_stairs),"flat",len(raw_flat))

for path in ["data/amass/pkls/amass_isaac_kit_all.pkl","data/amass/pkls/amass_isaac_train_stairs_flat.pkl"]:
    print("==",path)
    d=joblib.load(path); ks=list(d)
    st=[k for k in ks if "upstairs" in k or "downstairs" in k]
    fl=[k for k in ks if not ("upstairs" in k or "downstairs" in k)]
    print("total",len(ks),"stairs",len(st),"flat",len(fl))
    per=collections.defaultdict(lambda:[0,0])
    for k in ks:
        m=re.search(r"KIT_(\d+)",k); sub=m.group(1) if m else "other"
        per[sub][0 if ("upstairs" in k or "downstairs" in k) else 1]+=1
    print("subjects",len(per),"per-subject (stairs,flat):",dict(sorted(per.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)))
    print("flat samples:",fl[:8])

# compare all raw stairs to pkl keys
d=joblib.load("data/amass/pkls/amass_isaac_train_stairs_flat.pkl")
keys=set(d.keys())
missing=[]
for p in raw_stairs:
    rel=os.path.relpath(p,"/data/ljx/datasets")
    norm="0-"+rel[:-4].replace("/","_")
    if norm not in keys: missing.append(norm)
print("raw stairs missing from merged pkl:",len(missing))
if missing: print(missing[:10])
