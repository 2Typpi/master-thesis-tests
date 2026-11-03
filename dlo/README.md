# DER cable

This package adds the adapteddlo_muj Discrete Elastic Rod controller to a free MuJoCo capsule chain. The original C++
solver sources and committed SWIG wrapper are vendored under `cpp/` with their license notice.

## Build

Activate the `mujoco` conda environment, install Eigen if needed, then build the SWIG extension:

```bash
conda install -c conda-forge eigen
bash dlo/cpp/build.sh
```

The build uses the active Python's NumPy and Python headers. To select another interpreter, set `PYTHON=/path/to/python`
and keep its NumPy installed. SWIG is not needed unless the committed wrapper fails to compile and needs regeneration.

## Run the cube scene

```bash
python -m dlo.demo
```

This is a CPU-only simulation because each MuJoCo step needs the Python DER hook between `mj_step1` and `mj_step2`.
MuJoCo Warp/MJX stepping does not invoke that hook. The default scene is a 2 m, 40-segment cable lying beside the cube,
away from the arm's reset pose. The rope XML is generated at `franka_emika_panda/dlo_rope.xml`;
`mjx_single_cube_dlo.xml` is kept separate from the stock-composite scene so its existing keyframes and `arm_load.py`
workflow remain unchanged.

## Run the standalone bending test

```bash
python -m dlo.bend_test
```

The viewer starts with a straight 2 m cable suspended at 1 m. It first sags under gravity, then moves both mocap anchors
inward smoothly to compress the cable and displays the resulting arch. To run headless checks and save a side-view plot:

```bash
python -m dlo.bend_test --check --plot /tmp/dlo-bend.png
```

`--alpha` and `--beta` set the DER bending and twist stiffness; `--compress` chooses the end-span shortening fraction.
By default the viewer then releases the `B_last` weld and simulates six more seconds; choose the other end with
`--drop-anchor start`, or disable release with `--drop 0`. The test reports cable length, endpoint chord, tip height/offset,
vertical sag, lateral deflection, maximum curvature, DER bending energy, twist, anchor error, and maximum speed. The absolute
mapping of these dimensionless stiffness values to a physical EI is not established.

Run the fast package smoke checks after building:

```bash
python -m dlo.check
```

The original adapteddlo_muj values were tested on a 9.29 m rod at a 1.5 ms timestep with both ends welded and gravity disabled;
the cube scene and the standalone gravity/bending test are different setups.
