# unitree-g1-humanoid-place

*Place / drop policy for the Unitree G1: set the carried box down on a target.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Framework](https://img.shields.io/badge/framework-mjlab%20%2B%20rsl__rl-lightgrey.svg)](https://github.com/mujocolab/mjlab)

<p align="center">
  <img src="docs/demo.gif" width="720" alt="Unitree G1 setting a carried box down on a target surface">
</p>

<p align="center"><sub>Full walk to pick to carry to place chain in mjlab; this repo trains the last stage (setting the box down).</sub></p>

- Final stage of the loco-manipulation pipeline: sets a carried box down on a target surface without over-shooting or dropping.
- Warm-starts from the pick terminal-state buffer, so training begins from a distribution of realistic already-grasped states.
- Reward geometry uses a broad-above / narrow-below tracking kernel on the box z-axis, encouraging controlled descent rather than a hard cliff.

## Getting Started

### Prerequisites
- Python 3.10 exactly.
- A CUDA-capable GPU (developed on B300 SXM6; scale `NUM_ENVS` down for smaller cards).
- `uv` package manager (`pip install uv`).
- A trained pick checkpoint and its terminal-state buffer (see the pick repo).

### Install
```bash
git clone https://github.com/narayanerm/unitree-g1-humanoid-place.git
cd unitree-g1-humanoid-place
uv pip install -r requirements.txt
```

### Run
Open `01b_place.ipynb` and execute top-to-bottom. Notebook flags at the top:

| Flag | Purpose |
|---|---|
| `REUSE_TRAINING` | Skip training if a checkpoint exists under `outputs/logs/g1_drop/`. |
| `RESUME` | Continue from the latest checkpoint. |
| `WARM_START_FROM_PICK` | Shape-filtered warm-start from a pick checkpoint. |
| `DEBUG_LIGHT` | 64 envs, few iters smoke test. |

Quick viewer over the latest checkpoint:
```bash
python view.py
```

TensorBoard: `tensorboard --logdir outputs/logs/g1_drop`.

## Method

PPO fine-tune of a pick-pretrained actor with a place-specific reward geometry. The critical design choice is the z-axis kernel around the target surface: broad above (pulling the box down toward the target from above) and narrow below (punishing overshoot into the surface). Symmetric hand grip is preserved via a mirror of pick's hand-lost-grip penalty so the box is not released prematurely mid-descent. Reset uses the pick terminal-state buffer, mirroring the same convention as carry so that place and carry can be swapped in a state machine without re-tuning.

## Related repositories

- [`unitree-g1-humanoid-walk`](../unitree-g1-humanoid-walk) - upstream base locomotion.
- [`unitree-g1-humanoid-pick`](../unitree-g1-humanoid-pick) - upstream: emits the terminal buffer that resets place.
- [`unitree-g1-humanoid-carry`](../unitree-g1-humanoid-carry) - sibling stage; carry and place chain together in the full pipeline.
- [`unitree-g1-humanoid-reach`](../unitree-g1-humanoid-reach) - sibling reaching stage.
- [`unitree-g1-humanoid-fsm`](../unitree-g1-humanoid-fsm) - chains place with walk, pick, and carry through a finite state machine.

## Citation

```bibtex
@misc{medeiros2026g1place,
  author = {Ribeiro Medeiros, Narayane},
  title  = {unitree-g1-humanoid-place: place/drop policy for the Unitree G1 in mjlab},
  year   = {2026},
  howpublished = {\url{https://github.com/narayanerm/unitree-g1-humanoid-place}}
}
```

## Author

**Narayane Ribeiro Medeiros** - ITAndroids Humanoid Team, Instituto Tecnologico de Aeronautica.
[GitHub](https://github.com/narayanerm) - narayane.rm@gmail.com

## License

Released under the [MIT License](LICENSE).
