# How we reached this demo

All results below are from MuJoCo on a fixed side view. They do not measure real-arm performance.

1. **Click a 3D point, then IK.** The side-view point demo (`so100.ik_point_demo`) proved the SO-100 fingertip could reach selected table or above-table coordinates. It did not solve object selection or grasping.
2. **Put boxes on the table.** A scripted top-down grasp worked when a box aligned with the gripper. Rotating a box exposed corner contact and grasp failure: a target centre alone was insufficient.
3. **Estimate grasps from RGB.** The current estimator fits a footprint and angle to a color mask, projects pixels to the known object-top plane, and commands an open approach, close, and lift. This improved rotated box, cylinder, and block trials, but still depends on fixed camera and assumed height.
4. **Learn which object the sentence names.** Frozen SigLIP 2 image/text features plus a small trained target head rank four color masks. The saved `action-target-v4.pt` report was **1,278/1,280** on its held-out simulated prompt/scene pairs. The objects, rendering style, and color-cluster setup remain the same.
5. **Learn the next skill.** MiniLM text tokens, arm height, jaw state, and completed-skill counts feed a GRU action head. MuJoCo demonstrations supply labels. The current v10 head was adapted from v9; its held-out *step-label* score was **5,238/5,502** on generated language/state examples. That score does not measure full physical completion.
6. **Check complete behavior.** On the recorded v10 simulation checks, **8/8** approach-only commands and **16/16** supported mixed commands finished with the expected action sequence and checked physical outcome. These are small, specific held-out suites, not a claim about arbitrary language or objects. The reproducible commands are in the README.

## Failures that shaped the design

- The first reach demo moved a marker to an image patch, yet did not prove a grasp. The current pipeline uses calibrated 3D geometry and MuJoCo contact instead.
- “Go near the blue cube” initially picked it up because the action head generalized it as a pickup. Approach-only training examples were added, and the end-to-end check now verifies the gripper stays open and the object stays on the table.
- The earlier `unsupported` action produced a label for unfamiliar commands. It was removed. If the current model chooses `done` for “dance for me,” nothing moves; that is the model's own output, not a hardcoded rejection.
- Scattered objects and rotated grasps can still cause missed contact. RGB lift checking and limited retarget/retry logic improved the tested scenes, but neither guarantees success on unfamiliar shapes or real hardware.

## Files worth reading

| File | Role |
| --- | --- |
| `so100/action_demo.py` | End-to-end policy and browser server |
| `so100/action_head.py` | Skill head, state features, and executable skills |
| `so100/rgb_target.py`, `so100/encode.py` | RGB candidate masks and frozen vision/text features |
| `so100/rgb_grasp.py`, `so100/vision.py` | RGB grasp geometry and camera projection |
| `so100/executor.py`, `so100/sim.py` | IK, arm control, physics, and rendering |
| `so100/train_action_head.py`, `so100/train_action_target.py` | Latest head training and generated training examples |
| `so100/eval_action_demo.py` | End-to-end check and image evidence |
| `scripts/record_demos.py` | Reproducible MP4 capture from policy execution |

The four included `.pt` files are the current target/action heads and their immediate warm-start checkpoints. The much larger generated datasets and abandoned experiment branches remain outside this focused repo.
