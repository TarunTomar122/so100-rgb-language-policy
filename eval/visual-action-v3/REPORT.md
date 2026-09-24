# Visual action v3: check the object after carrying it

The browser policy now checks RGB again after a movement made while carrying an object. If the object has fallen, it sends that failed outcome to the existing recovery head. It still has a 12-action limit and can stop with the task unfinished.

The reported red-cube command (`lift the red cube then move left then drop it`, seed 290000) initially lifted the cube and dropped it during the left move. The new check detected that loss, retried the pickup and move, then released the cube **28.7 mm left** of its start. It passed the physical goal check. [Watch the 13-second policy recording](../../media/carry-retry-red-left.mp4) or see its [first and final frames](../../media/carry-retry-red-left.jpg).

| Corrected MuJoCo checks | Physical goal | Exact requested sequence |
| --- | ---: | ---: |
| Eleven fixed commands | **10/11** | — |
| Three placement commands, including the reported one | **3/3** | 1/3 |
| Three deliberate slips after grasp | **2/3** | — |
| Fresh objects and scene changes | **5/10** | 3/10 |
| Separate holdout objects and scenes | **7/13** | 5/13 |

[Fixed-command traces](head/results.json), [placement traces](head/placements.json), [slip traces](slip/results.json), [fresh results](ood/results.json), and [holdout results](holdout/results.json) include simulator-graded object motion and before/after frames. A physical-goal pass can include extra actions. These are seeded simulation trials, not a real-world success rate.

## Why older object results changed

The earlier OOD tests changed MuJoCo object shape and size **after model compilation**. The new shape rendered, but collision geometry was intermittently wrong. This version compiles each altered object into the XML before simulation. The [shape-contact check](../../scripts/check_ood_shapes.py) confirms that capsules and ellipsoids settle against the table. The earlier v2 shape scores are historical and are not comparable to these corrected trials.

## Training attempts

I added slipped-carry examples and physical completion labels to candidate stop and recovery training, then compared the resulting checkpoints on the same simulator tasks. Neither improved the physical sets; one stopped late on a valid placement. The previous checkpoints remain deployed. A separate RGB grasp-center model reduced held-out synthetic XY error from 19.0 to 7.2 mm, but its physical holdout fell from 7/13 to 6/13 and it failed the reported red-cube pickup. It was not deployed.

The unresolved failures are mainly wrong object selection, merged same-color masks, and unstable contact on unfamiliar shapes. The action head can also still choose `done` before the requested outcome. Simulator object poses are used for training labels and evaluation, never runtime motion selection.

## Reproduce

```bash
source scripts/vulkan_env.sh
uv run --frozen python scripts/check_ood_shapes.py
uv run --frozen python scripts/eval_visual_action.py --placements
uv run --frozen python scripts/eval_visual_action.py --slip
uv run --frozen python scripts/eval_visual_action.py --ood
uv run --frozen python scripts/eval_visual_action.py --holdout
uv run --frozen python -m so100.visual_action_demo  # http://127.0.0.1:8773/
```
