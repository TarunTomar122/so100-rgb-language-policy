# Visual action head: outcome-aware stopping and varied objects

The experimental visual action head now learns when to stop from **observed task outcomes**. Its new training scenes include boxes, spheres, cylinders, capsules, and ellipsoids with varied colors and descriptions. A lift is labeled `done` only if the intended simulator object actually rose at least 30 mm; a missed grasp becomes a non-`done` example. The simulator supplies training labels and evaluation grades, never runtime target coordinates or actions.

| Seeded simulation check | Previous head | Current head |
| --- | ---: | ---: |
| Fixed commands, including movements and placement | 10/11 | **10/11** |
| Deliberate slips after grasp | 2/3 | **2/3** |
| Fresh shapes, colors, and descriptions: physical goal | 3/10 | **6/10** |
| Separate object/environment holdout: physical goal | 6/13 | **8/13** |
| Fresh set: exact requested action sequence | — | **4/10** |
| Holdout: exact requested action sequence | — | **5/13** |

The *physical goal* score counts the named object being lifted at least 30 mm and the model stopping; it permits unnecessary intermediate actions. The *exact sequence* score also rejects those extra actions. These are single seeded trials, not a real-world success rate. [Fixed tasks](head/results.json), [slips](slip/results.json), [fresh objects](ood/results.json), and [holdout](holdout/results.json) contain action traces and outcome measurements.

Visual examples: [violet cube lifted](holdout/violet-cube.jpg), [lilac block lifted](ood/lilac-block.jpg), [red capsule left on table](ood/red-capsule.jpg), and [flat puck left on table](holdout/flat-amber-puck.jpg). Each image shows before, observed before, and after. I inspected these rendered outcomes alongside the traces.

The stop model is a small correction to the frozen visual action head's `done` score. Training added 34 varied-object episodes: 19 verified completions, 11 missed grasps, and 14 skipped scenes where RGB grasp estimation could not produce a candidate. It also includes verified arm-movement completions. On 630 held-out step labels, false `done` predictions fell from 67 to 24; this measures next-action labels, while the table above measures physical rollouts.

Remaining failures have different causes. Capsules, ellipsoids, and a low puck often get the right target but miss physical contact. The current RGB grasp estimator assumes a fixed object height; that limits unusual shapes and heights. The coral block, red egg under JPEG noise, and red sphere near another red object expose target-selection or mask failures. A `go near` command for a lilac block stopped at the wrong object. Better stop training cannot repair those perception and grasp errors. The browser status says “Model chose to stop” because runtime success is still uncertain; it does not claim the physical goal was verified.

## Reproduce

```bash
source scripts/vulkan_env.sh
uv run --frozen python -m so100.train_visual_action
uv run --frozen python -m so100.train_visual_action_recovery
uv run --frozen python -m so100.train_done_head
uv run --frozen python -m scripts.eval_visual_action
uv run --frozen python -m scripts.eval_visual_action --slip
uv run --frozen python -m scripts.eval_visual_action --ood
uv run --frozen python -m scripts.eval_visual_action --holdout
uv run --frozen python -m so100.visual_action_demo  # browser on port 8773
```

The first two training commands make temporary feature caches under `/tmp/`. The third command adds varied training scenes; its checkpoint is [done.pt](../../data/visual-action/done.pt). The first two checkpoints and frozen vision/language backbones are unchanged.
