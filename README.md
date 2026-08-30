# TensorFlow in 15 Days

A beginner-friendly TensorFlow / deep learning course, organized day by
day as folders — browse straight through, in order.

No prior deep learning experience assumed, but comfort with Python
(functions, classes, loops) is expected — see
[Python-tutorial](https://github.com/meharuhanzz/Python-tutorial) first
if you need that. This is a TensorFlow/Keras companion to
[Pytorch-tutorial](https://github.com/meharuhanzz/Pytorch-tutorial) —
same 15-day syllabus, same order, same beginner level, different
framework.

## Syllabus

| Day | Folder | Topic |
|---|---|---|
| 1 | [`day01/`](day01/) | Tensors — creation, operations, indexing, GPU tensors |
| 2 | [`day02/`](day02/) | `tf.GradientTape` — automatic differentiation |
| 3 | [`day03/`](day03/) | Building a model with `tf.keras.Model` |
| 4 | [`day04/`](day04/) | Loss functions & optimizers |
| 5 | [`day05/`](day05/) | The training loop |
| 6 | [`day06/`](day06/) | `tf.data.Dataset` — datasets & batching |
| 7 | [`day07/`](day07/) | Real image data — `image_dataset_from_directory` & preprocessing |
| 8 | [`day08/`](day08/) | Convolutional Neural Networks |
| 9 | [`day09/`](day09/) | Training a CNN end-to-end |
| 10 | [`day10/`](day10/) | Overfitting & regularization |
| 11 | [`day11/`](day11/) | Transfer learning |
| 12 | [`day12/`](day12/) | Saving, loading & checkpointing |
| 13 | [`day13/`](day13/) | GPU & mixed-precision training |
| 14 | [`day14/`](day14/) | Evaluation & metrics |
| 15 | [`day15/`](day15/) | Capstone: a full image classifier project |

## How each day is organized

Every `dayNN/` folder has the same shape:

- **`README.md`** — what you're learning today, explained, with examples
- **`main.py`** — runnable code demonstrating today's concepts
- **`exercises.py`** — practice problems (with TODOs) — try these yourself

Some days also include a `make_*.py` script that generates a small
synthetic dataset the day's code needs — run it once before `main.py`
(each day's own README says exactly when).

```bash
git clone https://github.com/meharuhanzz/Tensorflow-tutorial.git
cd Tensorflow-tutorial/day01
python3 main.py
```

## Setup

```bash
pip install tensorflow numpy pillow
```

Every code example runs fine on CPU — GPU is used automatically if
TensorFlow detects one, but nothing here requires it.

## Author

Meharuniza
