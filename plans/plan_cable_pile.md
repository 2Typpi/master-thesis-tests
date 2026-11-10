# Plan: Panda + cable-pile environment (Newton)

## Context
Goal: a simulation environment with a Franka Panda and a pile of cables. Pulling one cable should tie a knot, and pulling another should free it from the pile. The arm must reach the pile and be able to contact the cables. The user later wants to train a neural network on it and transfer to a real robot, so the grasp model matters.

Findings from exploration (`example_robot_panda_only.py` is the inspiration only, not to be copied):
- Env: `/home/mwuerstle@corp.exxcellent.de/miniconda3/envs/mujoco/bin/python` (Newton 1.0.0, Warp 1.17 CPU-only). Bare `python` cannot import Newton.
- `SolverVBD` rejects revolute joints, so the Panda cannot live in the cable model. Architecture stays at **two unconnected models**:
  - Panda model: no solver, joints written to `joint_q`, then `newton.eval_fk`.
  - Cable model: stepped by `SolverVBD`, containing kinematic "mirror" bodies of the hand and fingers (`add_body(is_kinematic=True)`, MESH re-added as CONVEX_MESH, mirrors interpolated across substeps).
  - One-way coupling: the arm deflects cables, cables never push back. Do not retry two-way coupling (earlier attempts failed).
- Newton APIs to use: `ModelBuilder.add_rod`, `newton.utils.create_cable_stiffness_from_elastic_moduli`, `newton.utils.create_straight_cable_points_and_quaternions`, `builder.color()`, and `newton.ik` (`IKSolver`, `IKObjectivePosition`, `IKObjectiveRotation`, `IKObjectiveJointLimit`; usage in `example_robot_panda_hydro.py:450-485`).

## Decision on grasping (the user's open question)
Only a **friction/pinch grasp transfers to a real robot**. A "cable sticks to the hand" grasp is a simulation shortcut: a policy trained on it would learn to rely on a perfect, never-slipping grip.
- Default `--grasp friction`: the finger mirrors close on the cable and contact friction holds it. The mirrors are kinematic and stiff, so the grip can be held reliably. Finger width is a real action.
- `--grasp sticky`: debug-only fallback that rigidly attaches the nearest segment to the hand. Use it to check the pull logic independently of the grip.
- Caveat to tell the user: with one-way coupling there is no grip force or compliance modelling, so some sim-to-real gap remains.
- Caveat: CPU-only Warp will probably be too slow for RL training. A GPU or fewer cables will be needed later.

## New file: `example_robot_panda_cable_pile.py`
A single `Example` class, same `newton.examples` structure (`create_parser`, `run`).

1. **Panda**: load via `newton.utils.download_asset("franka_emika_panda")` and `add_urdf(fr3_franka_hand.urdf)`. Place the base so the pile centre is about 0.4-0.5 m in front of it (Panda reach is 0.855 m).
2. **Cables (hybrid pile, `--seed`, `--num-cables 5`)**:
   - Cable length about 0.5 m, 40 segments, radius about 4 mm. Stiffness from `create_cable_stiffness_from_elastic_moduli`. `friction_epsilon` about 0.1.
   - Cable "knot": hand-authored points forming a loose overhand-knot loop, with the crossings separated in z by more than 2 radii. Its far end is pinned (kinematic, like a plugged-in cable) so that pulling the tail tightens the knot.
   - Cable "free": laid on and partly under the pile, one end exposed, no loops, so it slides out when pulled.
   - Remaining cables: dropped with random seeded poses over and around the two authored ones.
   - A settle phase with the arm parked lets the pile come to rest before the robot starts.
3. **Hand contacts**: mirror bodies for `fr3_link7` and both fingers, as in the memory note. Collision-filter the mirrors against each other. Do not filter them against the cables except near a pinned anchor, if needed.
4. **Control (`--pull knot|free`)**: a state machine driven by `newton.ik`:
   - home -> pre-grasp above the target cable's end -> descend -> close fingers -> pull along a straight path -> release.
   - Finger joints are interpolated to a commanded width.
5. **Viewer**: arm via `log_state`. Cables via `log_capsules`, drawing all cable bodies in different colours (the knot cable and the free cable are highlighted). `picking_enabled=False`.
6. **Structure for later RL use**: keep the scene build, `reset(seed)` and the `step()` physics separate from the scripted controller, so a policy can replace the controller.

## Verification
All runs headless: `<env python> example_robot_panda_cable_pile.py --viewer null --num-frames N --pull knot|free` (filter the Warp banner from stdout).
1. Reachability: the IK residual at the pre-grasp and grasp poses for both cables is below 1 cm and the joint limits are respected.
2. Sanity: all state finite, segment lengths within 10%, no cable passes through the ground.
3. Contact check (open item from memory): a control run with the mirror collision pairs filtered must show a different cable pose than the normal run.
4. Knot vs free metric: after the pull, compute end-to-end distance divided by cable length.
   - `--pull free`: the ratio is close to 1 (straight), and the cable has left the pile.
   - `--pull knot`: the ratio is clearly smaller, and the loop's diameter shrinks (knot tightened).
5. Grasp: the cable is still held at the end of the pull with `--grasp friction`, otherwise compare against `--grasp sticky`.
6. Then the user checks it visually in the GL viewer.

## Risks
- A loosely laid knot may not tighten, or may fall apart, during settling. Mitigation: tune the loop geometry and the pinned end; the metric in step 4 detects this.
- 5 cables x 40 segments with cable-cable contact is expensive on CPU. Mitigation: `--broad-phase sap`, fewer segments, or fewer substeps.
- Friction grip may slip. Mitigation: raise `mu`, slow the pull, or use `--grasp sticky` for debugging.

## Out of scope
Two-way robot-cable coupling, whole-arm collisions, and the actual neural network training.
