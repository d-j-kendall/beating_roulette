# beating_roulette

An experiment in whether the physics of a roulette spin — captured from video — can predict the winning pocket. It pairs a computer-vision front end with a small neural network: a YOLOv3 model detects the ball and the wheel's zero marker frame by frame, those detections become a motion state vector (angular position, velocity, acceleration), and a PyTorch network is trained to map that motion to an outcome.

_Status: senior-design experiment (2019) · Python · PyTorch · YOLOv3_

## How it works

1. **Detect** — a [YOLOv3 fork](https://github.com/ultralytics/yolov3) (included as a submodule) locates the ball and the zero pocket in each video frame.
2. **Track** — `pv2sv.py` converts detections to polar coordinates around the wheel center and derives angular velocity and acceleration (`StateVector`).
3. **Assemble** — `generate_training_data.py` turns tracked spins into labeled training rows.
4. **Predict** — `predict/Models.py` trains a small feed-forward network (8 → 30 → 30 → 1) on those state vectors.

## Run it

Clone with the YOLOv3 submodule:

```bash
git clone --recurse-submodules https://github.com/d-j-kendall/beating_roulette.git
```

Requires Python with PyTorch, torchvision, and OpenCV. The `Examples/` notebooks (CIFAR-10, VGG-19, background subtraction) are course exercises kept alongside the project.

## Honest caveats

- This is a proof-of-concept from a university project, not a system that beats a casino. It explores the modeling idea end to end; it doesn't claim a profitable edge.
- Some file paths in the scripts are hardcoded to the original author's machine — adjust them before running.

---
Built by **Dustin Kendall** — production readiness & rescue for AI-built apps.
[dustinkendall.com](https://dustinkendall.com) · [LinkedIn](https://www.linkedin.com/in/dustinjkendall)
