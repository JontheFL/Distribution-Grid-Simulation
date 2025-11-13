import importlib, traceback, sys

print("Python:", sys.version)

pkgs = ['numpy', 'pandas', 'matplotlib', 'pandapower', 'scipy']

for p in pkgs:
    print('\n==', p)
    try:
        mod = importlib.import_module(p)
        ver = getattr(mod, '__version__', None)
        print('OK', 'version=' + str(ver))
        if p == 'numpy':
            # show build info
            import numpy as np
            print('numpy build info:', np.__config__.show())
    except Exception:
        traceback.print_exc()
