
# AIR-Mamba

**AIR-Mamba** is a lightweight Selective State-space Model (SSM)-based architecture with real-time domain adaptation capability, specifically designed for robust UAV recognition using IR-UWB radar across complex environments.

> This repository contains the official implementation of our paper:  
> **"Domain-Adaptive UAV Recognition Using IR-UWB Radar and a Lightweight Mamba-Based Network"**  
> Code and full dataset will be released upon publication.

---

## 🌟 Highlights

- Lightweight SSM backbone based on the Mamba architecture
- Supports domain-adaptive training across **urban**, **suburban**, **marine**, and **microwave anechoic chamber** environments
- Specially tailored for **IR-UWB radar UAV Recognition**

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/lsysysu/AIR-Mamba.git
cd AIR-Mamba
````

Create the Conda environment:

```bash
conda env create -n AIRMamba --file requirements.txt
conda activate AIRMamba
```

---

## 📁 Dataset

This project evaluates cross-environment radar-based UAV recognition on two datasets:

### 1. Public Dataset: FOI 77GHz FMCW Radar

We use the 77GHz FMCW radar dataset from the Swedish Defence Research Agency (FOI):

* Title: *Radar measurements on drones, birds and humans with a 77GHz FMCW sensor*
* Link: [FOI Radar Dataset](https://zenodo.org/records/5896641)
* Classes: D1–D6 drone types
* Format: `.npy` files with complex-valued segments (5 × 256)
* Preprocessing: Converted to 256×256 RGB images using real/imag/magnitude

Only center-aligned segments are retained for model training and testing.

### 2. Proprietary Dataset: UWB-Drone4Env (To be released)

We introduce a new dataset **UWB-Drone4Env** (*Ultra-Wideband Radar Dataset for Drone Recognition across Four Environments*), collected using a custom IR-UWB radar platform.

* Radar Type: IR-UWB
* Classes: 6 UAV types
* Environments: Urban, Forest, Ocean, Desert
* Format: 2D radar TR images (256×256)
* Purpose: Evaluate domain generalization and adaptation

The dataset will be made public following paper acceptance.

---

## 🚀 Training and Evaluation

> Training scripts and pretrained weights will be released after acceptance.

* `train.py`: Main training loop with domain adaptation options
* `datasets/`: Preprocessing scripts and dataset format converters
* `models/`: Contains AirMamba and baseline model definitions
* `eval.py`: Evaluate classification and domain adaptation accuracy

---

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{li2024airmamba,
  title={Domain-Adaptive UAV Recognition Using IR-UWB Radar and a Lightweight Mamba-Based Network},
  author={Li, Shengyuan and Dong, Xinyue and Fan, Yiheng and Zhu, Xiangwei and Yuan, Xuelin},
  journal={IEEE Sensors Journal},
  year={2024},
  note={under review}
}
```

---

## 📜 License

This repository is released under the MIT License.
Please adhere to the licenses of any third-party datasets or codebases used.

---

## 🧭 Acknowledgements

* [FOI Radar Dataset](https://zenodo.org/records/5896641)
* [Mamba: Linear-Time Selective SSMs](https://github.com/state-spaces/mamba)

