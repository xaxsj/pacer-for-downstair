import joblib, os, sys
for path in sys.argv[1:]:
    if not os.path.exists(path):
        print(path, "MISSING")
        continue
    data = joblib.load(path)
    print(path, type(data), len(data))
    if isinstance(data, dict):
        keys = list(data.keys())
        print("  first:", keys[:5])
