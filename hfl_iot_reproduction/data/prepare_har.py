"""Deprecated: use prepare_intel_lab.py."""

import runpy

if __name__ == "__main__":
    print("[data_prep] prepare_har.py foi substituído por prepare_intel_lab.py")
    runpy.run_path("/workspace/data/prepare_intel_lab.py", run_name="__main__")
