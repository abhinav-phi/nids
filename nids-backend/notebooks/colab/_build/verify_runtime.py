"""Run the inference smoke suite against sources or generated notebooks.

SHAP absence is deliberate: optional explainability must not stop the API.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import pack

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--generated', action='store_true')
args = parser.parse_args()

shap_stub = types.ModuleType('shap')
def unavailable(*args, **kwargs):
    raise ImportError('SHAP deliberately unavailable in fallback regression')
shap_stub.TreeExplainer = unavailable
sys.modules['shap'] = shap_stub

source = (HERE / 'smoke_test.py').read_text(encoding='utf-8')
if not args.generated:
    source = source.replace(
        'nb = json.loads(NB3.read_text(encoding="utf-8"))',
        'nb = {"cells": __import__("pack").parse_file(HERE / "03_Inference_API.py")}')
exec(compile(source, str(HERE / 'smoke_test.py'), 'exec'),
     {'__file__': str(HERE / 'smoke_test.py'), '__name__': '__main__'})
