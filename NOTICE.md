# Attribution

`so_arm100.xml`, `assets/*.stl`, and the included Apache 2.0 `LICENSE` originate from [Google DeepMind MuJoCo Menagerie's `trs_so_arm100` model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100), which describes the [SO-ARM100 by The Robot Studio](https://github.com/TheRobotStudio/SO-ARM100). The arm XML was adjusted for this tabletop simulation. `tabletop.xml`, the policy, and the demo are this project's additions.

The pretrained [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) weights are fetched separately from Hugging Face on first run. Their respective model cards provide their licensing and usage terms.
