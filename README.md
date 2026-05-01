# LPHP

Anonymous minimal code release for reproducibility.

This repository provides a minimal runnable implementation of the PH-based topology component used in the paper. The current release keeps only the code path required to run `topo_addon` and removes unrelated training pipelines and auxiliary modules.

## Repository Structure

```text
LPHP/
├─ README.md
├─ demo_topo_addon.py
├─ LPHP/
│  ├─ __init__.py
│  ├─ interfaces.py
│  ├─ loss.py
│  ├─ ph.py
│  └─ utils.py
└─ Fig/
   └─ Methods.png
````

## Environment

A minimal Python environment with PyTorch is sufficient.

```bash
conda create -n lphp python=3.10 -y
conda activate lphp
pip install torch
```

If a CUDA-enabled GPU is available, install the corresponding PyTorch build for your platform.

## Run the Demo

From the repository root, run:

```bash
python demo_topo_addon.py
```

The demo constructs random student and teacher logits, evaluates the current `topo_addon`, and verifies that the backward pass runs successfully.

## Notes

* This is a minimal anonymous release for reproducibility.
* The current repository only includes the PH-related path used by `topo_addon`.
* It is not intended to reproduce the full training framework of the paper.
* `Fig/Methods.png` provides the method overview corresponding to the submitted manuscript.