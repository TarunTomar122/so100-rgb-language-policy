# RGB-only robustness revision — 23 September 2026

**Complete physical outcomes:** 21/28 on the previously inspected stress scenes, 8/13 on new combinations, and 6/10 on a second new set. The original frozen policy passed 16/28 on the first set. These are single seeded MuJoCo trials, not estimated real-world rates.

No model was trained or fine-tuned for these scenes. The old target and GRU action checkpoints were not changed. The revision uses an empty-table RGB reference to filter foreground, frozen SigLIP 2 to rank object crops against the instruction, and frozen Qwen3-4B to plan from the existing ten skills. RGB grasp geometry and IK still execute the physical motion.

| Scene set | Correct full task | Correct target selected | Full results |
| --- | ---: | ---: | --- |
| Original stress conditions, already inspected | **21/28** (previously 16/28) | 27/28 | [JSON](stress/results.json), [frames](stress/) |
| New shape/color/environment combinations | **8/13** | 10/13 | [JSON](holdout/results.json), [frames](holdout/) |
| Second new set | **6/10** | 9/10 | [JSON](fresh/results.json), [frames](fresh/) |

The same 28 scenes informed the design, so their five additional successes are an engineering regression check, not independent generalization evidence. The first new set was inspected during development. The second set was fixed before its first run; implementation was then simplified and the set rerun without training on its outcomes. The 5/5 core end-to-end check also passes.

## What improved

- New color words no longer need a trained target-head weight for each color: [violet cube](clips/holdout/violet-cube.mp4), [orange cube](stress/orange-cube.jpg), and [pink block](stress/pink-block.jpg) succeed. Crop ranking requires more image processing than the earlier single full-frame encoder pass.
- The planner now gives a full pickup sequence for [cyan cylinder](stress/cyan-cylinder.jpg) and [red ball](stress/red-ball.jpg), where the old small head sometimes stopped after `reach`. The red ball still fails physically, which separates language progress from grasp success.
- Empty-table calibration removes the large teal-table false candidate. The [green-cylinder rollout](clips/stress/teal-table.mp4) lifts the object 58 mm according to the simulator grader. It is hidden from the fixed camera after lifting, so the policy records `lift_unverified=true` and relies on jaw opening for a provisional held-object signal.
- The compressed [yellow-block scene](stress/jpeg-25.jpg) now completes; its earlier RGB grasp point was displaced by about 21 mm.

## Failures still visible

- **Target identity:** [yellow bar](stress/yellow-bar.jpg), [coral block](clips/holdout/coral-block.mp4), [JPEG red egg](holdout/jpeg-red-egg.jpg), and [near lilac block](fresh/near-lilac-block.jpg) select the wrong object. Two similar red objects in [red sphere near cube](holdout/red-sphere-near-cube.jpg) expose the fixed color-cluster proposal limit.
- **Grasp geometry/contact:** [blue tube](clips/stress/blue-tube.mp4), [tall cyan tube](clips/holdout/tall-cyan-tube.mp4), flat puck, spheres, and some wide placements fail despite the intended target being selected. The estimator assumes an approximately 30 mm object and a visible top surface; single-view RGB does not provide metric height or hidden contact geometry.
- **Feedback uncertainty:** The policy can finish a plan after grasping the wrong object or when the target is occluded. The browser's neutral “Plan complete” state is intentionally **not** a success badge. The external grader uses simulator object poses and motion to score the actual result; those poses do not choose runtime actions.

## Methods tried and rejected

A 1.5B local instruction model produced invalid or incomplete action sequences on basic commands. Qwen3-4B passed the five core physical checks and was used without fine-tuning. On representative simulator frames, frozen CLIPSeg often activated on the table instead of the named object, and OWLv2 often boxed the arm or broad scene with low scores. A larger SigLIP 2 checkpoint also ranked saved object crops worse than the smaller checkpoint in an offline comparison. None was inserted as an unverified fallback. [Research notes](../../docs/robustness-research.md) link their primary sources.

## Reproduce and calibration boundary

```bash
source scripts/vulkan_env.sh
uv run --frozen python -m so100.action_demo --check
uv run --frozen python -m scripts.eval_ood --output eval/ood-v2/stress
uv run --frozen python -m scripts.eval_ood --mode holdout --output eval/ood-v2/holdout
uv run --frozen python -m scripts.eval_ood --mode fresh --output eval/ood-v2/fresh
```

The simulator captures the empty-table reference **before** placing objects. For altered camera/table/light test scenes, the evaluation setup temporarily clears the simulated scene, captures the equivalent reference, then restores it before the policy starts. A real setup would require a real empty-table photo at the same camera pose and lighting. No depth render, object pose, or simulator contact ID is passed to the runtime grasp or skill decision. Gripper encoder position and RGB frames are available to the policy; simulator poses are used only for grading and diagnostic fields.
