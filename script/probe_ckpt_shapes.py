import torch, sys, os
for p in sys.argv[1:]:
    print("==", p)
    try:
        d=torch.load(p, map_location="cpu")
        if isinstance(d, dict):
            print("keys:", list(d.keys())[:8])
            m=d.get("model", d)
            for k,v in m.items():
                if hasattr(v,"shape") and ("actor_mlp.0.weight" in k or "critic_mlp.0.weight" in k or k.endswith("mu.weight")):
                    print(k, tuple(v.shape))
        else:
            print(type(d))
    except Exception as e:
        print("ERR", type(e).__name__, e)
