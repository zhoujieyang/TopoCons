# LPHP

Anonymous minimal code release for reproducibility.

This repository provides a minimal runnable implementation of the topology component used in the submitted paper, including the differentiable PH proxy loss and the conservative topology-consistency injection pathway exposed through `topo_addon`.

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

If a CUDA-enabled GPU is available, install the PyTorch build corresponding to your platform and CUDA version.

## Run the Demo

From the repository root, run:

```bash
python demo_topo_addon.py
```

The demo constructs random student and teacher logits, evaluates `topo_addon`, and verifies that the forward and backward passes run successfully.

## Method Overview

![Method Overview](Fig/Methods.png)

## Notes

* This is a minimal anonymous release for reproducibility.
* It is intended to demonstrate the proposed topology component and its backward behavior.
* It is not intended to reproduce the full UCOD training framework or all experimental results in the paper.
* Full experimental settings, datasets, carrier protocol, and run-specific configurations are described in the paper and appendix.

README 本身现在没有匿名性问题。当前只建议按上面版本稍微增强与论文主张的一致性。
