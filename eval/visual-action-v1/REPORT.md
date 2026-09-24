# Visual action head experiment — 24 September 2026

The experimental policy makes **one skill decision per fresh RGB observation**. It leaves target pixels, RGB grasp geometry, and IK to the existing vision/motion side. The action head never predicts a pixel, 3D coordinate, or wrist angle. The current Qwen browser policy remains the default because this experiment is weaker on unfamiliar scenes.

## Architecture

- Frozen SigLIP 2 supplies full-scene and selected-target image features. Its existing RGB target selector supplies the highlighted object region.
- The earlier trained MiniLM/GRU action head supplies language and arm-state logits. It is frozen.
- A small learned head receives those logits, image features, current target position, target movement between frames, jaw/arm state, and action history. It adjusts the logits for the **next** `reach`, `lower`, `close`, `lift`, `left`, `right`, `up`, `down`, `open`, or `done` decision. The robot re-observes after every skill.
- After a small lift, an RGB height check and jaw gap report whether the object appears held. A separate small learned retry head sees the new RGB-derived state and visual-head logits; it decides whether to `open`, `reach`, and continue. Its training includes varied simulated slips. The target is tracked by appearance across retries, and failed actions do not count as completed task progress.
- Training labels come from broad simulated skill trajectories. Some training objects are moved after `reach`; the held-out recovery scenes use different seeds. Simulator object poses identify teacher targets and grade outcomes, never select runtime skills or IK targets.

## Evidence

| Check | Frozen earlier head | Visual head |
| --- | ---: | ---: |
| Ordinary held-out step labels | 386/390 | 381/390 |
| Held-out moved-target next action | 0/11 | **9/11** |
| Same moved-target labels with RGB features removed | 0/11 | **0/11** |
| Complete physical tasks on 11 fixed scenes | 8/11 current Qwen policy | **10/11** |
| Induced slips after closing the gripper | — | **2/3** |
| Pre-existing unseen color/shape/environment scenes | 6/10 current Qwen policy | **3/10** |

The 11-scene set includes reaching without pickup, lifting, arm/gripper movements, two pick/carry/release commands, and two targets moved after `reach`. Both place-left/place-right tasks passed. The [moved blue cube](head/280101.jpg) now needed a visual regrasp to lift; the [yellow-block place task](head/280004.jpg) still selected a different object. See [head results](head/results.json) and [baseline results](baseline/results.json).

In three induced slip trials, the model reopened and retried. It [lifted the red cube](slip/280002-lift.jpg) and [completed pick, carry left, and release](slip/280002-drop-left.jpg) in the same scene. In a [second scene](slip/280008-drop-left.jpg), it correctly retried the red cube twice but the RGB grasp geometry missed both times. The two successes share a scene seed, so this is a small controlled test, not a general success rate. See [slip traces](slip/results.json).

On the pre-existing unseen set, three commands stopped after `reach` despite asking for a lift. Three unfamiliar shapes failed at contact despite choosing the correct target; one near command selected the wrong object. Two successes needed an extra `reach`. These are single seeded simulation trials, not real-world rates. See [all ten outcomes](ood/results.json) and their image pairs in [ood/](ood/).

**Decision:** keep this as a separate experiment. It demonstrates a visual check and learned retry after a missed grasp, but unfamiliar-object language and grasp geometry still fail often. Named destination placement is not implemented: that requires a second grounded region for the destination.

## Run

```bash
source scripts/vulkan_env.sh
uv run --frozen python -m so100.train_visual_action
uv run --frozen python -m so100.train_visual_action_recovery
uv run --frozen python -m scripts.eval_visual_action
uv run --frozen python -m scripts.eval_visual_action --slip
uv run --frozen python -m scripts.eval_visual_action --baseline
uv run --frozen python -m scripts.eval_visual_action --ood
uv run --frozen python -m so100.visual_action_demo  # separate browser demo, port 8773
```

The original head was trained on 96 scene seeds and 2,438 labeled decisions. The retry head adds 64 simulated-slip training seeds and 16 held-out seeds. The frozen visual and language encoders are downloaded on first use; only the small heads are trained. Temporary RGB feature caches live under `/tmp/` to avoid repeated image encoding during local retraining.
