# Visual action head experiment — 24 September 2026

The experimental policy makes **one skill decision per fresh RGB observation**. It leaves target pixels, RGB grasp geometry, and IK to the existing vision/motion side. The action head never predicts a pixel, 3D coordinate, or wrist angle. The current Qwen browser policy remains the default because this experiment is weaker on unfamiliar scenes.

## Architecture

- Frozen SigLIP 2 supplies full-scene and selected-target image features. Its existing RGB target selector supplies the highlighted object region.
- The earlier trained MiniLM/GRU action head supplies language and arm-state logits. It is frozen.
- A small learned head receives those logits, image features, current target position, target movement between frames, jaw/arm state, and action history. It adjusts the logits for the **next** `reach`, `lower`, `close`, `lift`, `left`, `right`, `up`, `down`, `open`, or `done` decision. The robot re-observes after every skill.
- Training labels come from broad simulated skill trajectories. Some training objects are moved after `reach`; the held-out recovery scenes use different seeds. Simulator object poses identify teacher targets and grade outcomes, never select runtime skills or IK targets.

## Evidence

| Check | Frozen earlier head | Visual head |
| --- | ---: | ---: |
| Ordinary held-out step labels | 386/390 | 381/390 |
| Held-out moved-target next action | 0/11 | **9/11** |
| Same moved-target labels with RGB features removed | 0/11 | **0/11** |
| Complete physical tasks on 11 fixed scenes | 8/11 current Qwen policy | **9/11** |
| Pre-existing unseen color/shape/environment scenes | 6/10 current Qwen policy | **3/10** |

The 11-scene set includes reaching without pickup, lifting, arm/gripper movements, two pick/carry/release commands, and two targets moved after `reach`. Both place-left/place-right tasks passed; [red cube moved mid-task](head/280102.jpg) was re-approached and lifted. [Blue cube moved mid-task](head/280101.jpg) was re-approached but the physical grasp still failed. The [yellow-block place task](head/280004.jpg) selected a different object. See [head results](head/results.json) and [baseline results](baseline/results.json).

On the pre-existing unseen set, three commands stopped after `reach` despite asking for a lift. Three unfamiliar shapes failed at contact despite choosing the correct target; one near command selected the wrong object. Two successes needed an extra `reach`. These are single seeded simulation trials, not real-world rates. See [all ten outcomes](ood/results.json) and their image pairs in [ood/](ood/).

**Decision:** keep this as a separate experiment. It proves RGB can change the action head's next decision and recover one physical interruption, but the lost open-vocabulary language accuracy and grasp failures rule out replacing the current demo. Named destination placement is not implemented: that requires a second grounded region for the destination.

## Run

```bash
source scripts/vulkan_env.sh
uv run --frozen python -m so100.train_visual_action
uv run --frozen python -m scripts.eval_visual_action
uv run --frozen python -m scripts.eval_visual_action --baseline
uv run --frozen python -m scripts.eval_visual_action --ood
uv run --frozen python -m so100.visual_action_demo  # separate browser demo, port 8773
```

Training uses 96 scene seeds and 2,438 labeled decisions. The frozen visual and language encoders are downloaded on first use; only the small visual correction head is trained. A temporary feature cache is stored under `/tmp/so100-visual-action-features-v2.pt` to avoid repeating image encoding during local retraining.
