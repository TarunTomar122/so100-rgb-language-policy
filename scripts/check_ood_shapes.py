"""Check that unseen object shapes have working MuJoCo table contacts."""

import numpy as np

from so100.sim import TABLE_TOP
from so100.train_rgb_target import setup_world


for name, variant, half in (
    ("green_cylinder", None, .015),
    ("yellow_block", None, .015),
    ("red_cube", ("red_cube", "capsule", (.011, .012, 0), .023, None), .023),
    ("red_cube", ("red_cube", "ellipsoid", (.013, .013, .020), .020, None), .020),
):
    world = setup_world(variant)
    world.park_scene()
    world.set_object(name, np.array([0.0, -.30, TABLE_TOP + half + .04]))
    world.step(200)
    actual = float(world.object_xyz(name)[2])
    assert abs(actual - (TABLE_TOP + half)) < .004, (name, actual)
    assert any({world.obj_geom[name], world._table} ==
               {int(world.data.contact[i].geom1), int(world.data.contact[i].geom2)}
               for i in range(world.data.ncon)), name
    print(f"{name} {variant[1] if variant else 'default'}: {actual * 1000:.1f} mm with table contact")
