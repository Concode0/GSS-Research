import sys, os

_versor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'versor')
if _versor_path not in sys.path:
    sys.path.insert(0, _versor_path)
