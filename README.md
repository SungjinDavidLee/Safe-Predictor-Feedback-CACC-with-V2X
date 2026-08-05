# Safe Predictor-Feedback CACC with V2X-Aware Adaptive Spacing

[![Sensors](https://img.shields.io/badge/Sensors-26(15)%2C%204806-2E7D32)](https://www.mdpi.com/1424-8220/26/15/4806)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs26154806-blue)](https://doi.org/10.3390/s26154806)

This repository accompanies the following paper:

> **Jaehyeon Shin, Junhyeok An, and Sungjin Lee**,  
> “Safe Predictor-Feedback CACC with V2X-Aware Adaptive Spacing for Heterogeneous Vehicle Platoons,”  
> *Sensors*, vol. 26, no. 15, Article 4806, 2026.  
> DOI: [10.3390/s26154806](https://doi.org/10.3390/s26154806)

## Overview

Vehicle-to-everything (V2X)-enabled cooperative adaptive cruise control (CACC) can improve traffic throughput and driving safety by sharing motion and control information among platooned vehicles. Practical platoons, however, contain heterogeneous vehicles with different computation, mechanical, actuation, and response delays, while V2X communication latency varies over time. A fixed spacing policy designed under homogeneous and fixed-delay assumptions can therefore become either unsafe during severe braking or unnecessarily conservative during normal operation.

The paper proposes **Safe PF-CACC**, which integrates a physically grounded **Safe Inter-Vehicle Distance (Safe IV Distance)** with predictor-feedback control and adaptive time-headway scheduling. The Safe IV Distance is composed of:

- **Minimum Margin (MM):** maintains a nonzero safety margin in low-speed and standstill conditions.
- **Response-Lag Loss (RLL):** accounts for V2X communication latency and heterogeneous vehicle-response delays.
- **Braking-Performance Limit (BPL):** reflects friction-dependent differences in braking capability between consecutive vehicles.

The resulting safe distance is converted into a dynamic effective time headway. A first-order low-pass filter suppresses abrupt headway and gain changes, and the filtered headway is incorporated into the predictor-feedback gain for risk-adaptive control. The framework retains a common controller structure across vehicles with heterogeneous and time-varying delays.

## Main Contributions

1. A V2X-aware Safe IV Distance that jointly considers communication latency, processing and mechanical delays, actuator-response dynamics, and road-friction-dependent braking limits.
2. Dynamic time-headway and predictor-feedback gain scheduling based on the instantaneous physical risk level.
3. A unified implementation structure for heterogeneous platoons, including the extended predictor-feedback case associated with different vehicle-delay relationships.
4. Quantitative evaluation in passenger-vehicle, emergency-vehicle, and heavy-duty truck platooning scenarios using safety, comfort, spacing-efficiency, and string-stability metrics.

## Reported Results

| Scenario | Main result reported in the paper |
|---|---|
| **S1: Heterogeneous passenger vehicles** | Maximum jerk reduced by **20.6%** and mean spacing reduced by **49.4%** compared with the conservative reference method. |
| **S2: Emergency vehicles** | Collision-free operation, with maximum jerk reduced by **18.6%** and mean spacing reduced by **53.2%**. |
| **S3: Heavy-duty trucks** | Stable jerk and acceleration responses, with mean spacing reduced by **59.6%**. |

The full paper provides the baseline definitions, vehicle parameters, V2X assumptions, stability analysis, and detailed comparison tables.

## Repository Structure

```text
.
├── README.md
├── CACC_Code.ipynb
├── requirements.txt
└── experiments
    ├── scenario1_heterogeneous_passenger.py
    ├── scenario2_emergency_vehicle.py
    └── scenario3_truck_platooning.py
```

- `CACC_Code.ipynb` is a lightweight notebook for running the three experiment scripts.
- `experiments/scenario1_heterogeneous_passenger.py` implements the heterogeneous passenger-vehicle platoon.
- `experiments/scenario2_emergency_vehicle.py` implements the high-speed emergency-vehicle platoon.
- `experiments/scenario3_truck_platooning.py` implements the heavy-duty truck platoon with actuator limits.

The simulation scripts include Safe IV Distance calculation, time-varying V2X latency, low-pass-filtered headway scheduling, predictor-feedback control, performance analysis, static plots, and HTML animations.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/SungjinDavidLee/Safe-Predictor-Feedback-CACC-with-V2X.git
cd Safe-Predictor-Feedback-CACC-with-V2X

python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
```

## Running the Experiments

### Jupyter Notebook

```bash
jupyter lab CACC_Code.ipynb
```

Run the scenario cells individually or execute the full notebook.

### Python Scripts

```bash
python experiments/scenario1_heterogeneous_passenger.py
python experiments/scenario2_emergency_vehicle.py
python experiments/scenario3_truck_platooning.py
```

Each script reports metrics such as maximum jerk, maximum acceleration, minimum time-to-collision, spacing-violation rate, spacing error, mean spacing, and the last-vehicle-to-leader acceleration ratio used for string-stability diagnosis. It also generates acceleration, velocity, spacing, and effective-headway plots.

### Animation Note

The scripts call `FuncAnimation.to_jshtml()` to display platoon motion in Jupyter. HTML animation generation can take additional time and produce large notebook outputs. Comment out the final animation block when only numerical metrics and static plots are required.

## Citation

```bibtex
@article{shin2026safe,
  author  = {Shin, Jaehyeon and An, Junhyeok and Lee, Sungjin},
  title   = {Safe Predictor-Feedback CACC with V2X-Aware Adaptive Spacing for Heterogeneous Vehicle Platoons},
  journal = {Sensors},
  year    = {2026},
  volume  = {26},
  number  = {15},
  pages   = {4806},
  doi     = {10.3390/s26154806}
}
```

## Authors

- **Jaehyeon Shin** — Department of Smart Automotive, Soonchunhyang University
- **Junhyeok An** — Department of Smart Automotive, Soonchunhyang University
- **Sungjin Lee** — Corresponding author, Department of Smart Automotive, Soonchunhyang University

Correspondence: **sungjinlee@sch.ac.kr**

## Disclaimer

This repository contains research-oriented numerical simulation code. It is not a safety-certified controller or a production V2X implementation, and real-vehicle deployment requires independent verification, hardware-in-the-loop testing, communication-stack integration, and compliance with applicable safety standards.
