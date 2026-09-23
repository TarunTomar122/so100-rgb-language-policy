# Frozen-policy stress test — 23 September 2026

**Result: 16/28 exploratory trials passed.** The matched, unmodified scenes passed **24/26**. No model was trained, no checkpoint changed, and no policy code was tuned for these trials.

The policy source and checkpoints were those published in commit `2dbc23c`. Each condition used one fixed random scene seed, so these counts describe these scenes only; they are not estimates of real-world success rates.

## What changed

One of the four movable objects was sometimes given an unseen primitive shape (ball, egg, tube, bar, or capsule), an unseen color, or both. Other trials changed camera pose, table color, light, placement, friction, or added a fifth colored distractor. Image tests applied sensor noise, blur, JPEG compression, or darkening to the RGB frames passed to the policy.

The object bodies kept their original mass and inertia when primitive geometry changed. This is a **shape/contact stress test**, not a physically exact simulation of new real objects. The grader used simulator object positions to check the result; the policy still received RGB, text, and arm state. Each image has three panels: clean initial render, the RGB observation, and final render.

For a lift to pass, the head had to choose `reach → lower → close → lift`, select the intended object, and leave it at least 30 mm higher with the gripper closed. The approach-only case had to stop after `reach`, with the jaw open and object unmoved.

| Condition | Passed / tried |
| --- | ---: |
| Original objects and commands | 2 / 2 |
| New shapes, original colors | 2 / 5 |
| New colors, original shapes | 0 / 4 |
| New colors and shapes | 1 / 3 |
| Environment changes | 5 / 7 |
| RGB noise and compression | 4 / 5 |
| Combined changes | 2 / 2 |
| **Total** | **16 / 28** |

The full trial log is [results.json](results.json). [Same-seed clean controls](controls/results.json) and [known-word diagnostics](known-word/results.json) are separate from the 28-trial total. The latter deliberately uses an old object name for a changed object to isolate wording from grasp behavior; it is not a natural user command.

The measured first grasp points behind the millimetre comparisons are in [failure diagnostics](diagnostics/results.json) and their [clean counterparts](diagnostics/controls/results.json).

## What failed

1. **Unseen color can misground the target.** “Lift the purple cube” selected and lifted the blue cube ([frames](purple-cube.jpg), [video](clips/purple-cube.mp4)). The orange cube and pink block also selected the wrong object. The pink error still had **0.92** target-head confidence, so confidence is not reliable for these unfamiliar inputs.
2. **New nouns can change the action sequence.** “Lift the cyan cylinder” selected the correct object but predicted only `reach` ([frames](cyan-cylinder.jpg), [video](clips/cyan-cylinder.mp4)). In a diagnostic run on the same changed scene, “lift the green cylinder” completed the lift. The red ball and yellow bar also stopped after `reach`; separate diagnostics found grasp failures for those shapes when familiar wording did request a full pickup.
3. **A correct target can still be ungraspable.** The blue tube was selected correctly, but its first RGB grasp point missed the object center by about **16 mm**; the clean same-seed cube missed by about **1 mm**. Retries knocked the tube away ([frames](blue-tube.jpg), [video](clips/blue-tube.mp4)). A widely placed red cube had an accurate initial grasp point but failed at contact/retry, which is a different failure mode.
4. **A saturated tabletop becomes a false object candidate.** One of the four color masks covered **31,449 pixels of teal table**. The requested green cylinder was misgrounded to the blue cube, with **0.996** confidence; the grasp point was about **86 mm** from the intended center, versus about **0.5 mm** on the clean same-seed table ([frames](teal-table.jpg), [video](clips/teal-table.mp4)). Four returned masks did not mean four actual objects.
5. **JPEG compression can shift grasp geometry.** At quality 25, the correct yellow block was selected, but the estimated grasp point was about **21 mm** off and `lower` failed IK. The same seed without compression lifted it with about **0.7 mm** initial grasp error ([frames](jpeg-25.jpg), [video](clips/jpeg-25.mp4)). Gaussian noise, blur, and darkening each passed their single tested scenes; that is limited evidence, not a robustness guarantee.

There were real successes on a [green sphere](green-ball.jpg) ([video](clips/green-ball.mp4)), red ellipsoid, and orange capsule. Moderate camera shifts, dim lighting, and an extra colored distractor also passed their individual trials. One combined case passed while its own clean control failed, so the combined 2/2 should not be read as a robustness win.

Eleven of the twelve stress failures had a passing same-seed clean control. The yellow-bar seed also failed unmodified, so its physical outcome is confounded by scene difficulty. The yellow-bar *instruction* failure remains visible: the changed-scene head stopped after `reach`.

## Reproduce

From the repo root on the documented macOS setup:

```bash
source scripts/vulkan_env.sh
uv run --frozen python -m scripts.eval_ood --output eval/ood-v1
uv run --frozen python -m scripts.eval_ood --mode control --output eval/ood-v1/controls
uv run --frozen python -m scripts.eval_ood --mode known-word --output eval/ood-v1/known-word
```

`scripts/eval_ood.py` fixes the condition list and seeds before evaluation. Pass `--select CASE ... --video` to record selected runs with ffmpeg. The next useful test is more seeds and real object meshes before deciding what data to collect for training.
