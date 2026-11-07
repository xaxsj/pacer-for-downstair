import joblib, sys
d=joblib.load(sys.argv[1])
ks=list(d)
stairs=sum(("upstairs" in k or "downstairs" in k) for k in ks)
print("total",len(ks),"stairs",stairs,"flat",len(ks)-stairs)
