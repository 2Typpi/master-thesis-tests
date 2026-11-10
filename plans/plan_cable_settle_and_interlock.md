# Plan: settle the cable pile, make the "knot" a multi-cable interlock

## Context
`example_robot_panda_cable_pile.py` works (arm, IK, friction pinch, one-way coupling), but there are two problems:
1. The cables never come to rest.
2. The knot is a single cable with a pinned end, so pulling it only ever tightens or straightens that one cable. Intended: pulling the knot cable pulls out several cables.

## Issue 1: why the cables keep wiggling
Reading `SolverVBD` (Newton 1.0.0, `solvers/vbd/rigid_vbd_kernels.py`, `solver_vbd.py`) gives these causes, most likely first. This is from reading the code, not yet measured:
- **Contact penalty is pinned at 100 N/m.** The shape `ke=1e2` equals the default `rigid_contact_k_start=1e2`. AVBD ramps the contact penalty as `k = min(k + beta*penetration, ke)`, so with `ke=1e2` it never stiffens. Capsules rest on soft springs and sink and bounce. The solver's own pile example uses `ke=1e6`.
- **Contact damping is one-sided.** `kd` only acts while bodies approach (`v_n < 0`). Rebound is not damped, and `kd=10` is far above critical for the soft spring, so it reads as chatter. The pile example uses `kd=0.1`.
- **Friction never fully sticks.** `friction_epsilon=0.1` acts as a viscous damper below 0.1 m/s, so a load causes slow creep. It also has almost no normal load to work with (see the first point).
- **Almost no joint damping.** The solver has no global damping, velocity clamp or sleeping. Stretch damping is 0 and bend damping 1e-3; the pile example uses 1e-4 / 0.5.
- **Stiff joints with few iterations.** The cable joint stiffness is about 4e5 N/m against about 3e2 N/m inertia, with only 5 sweeps. Residual stretch error keeps being pushed along the chain.
- **The "settle" check never worked.** It waits up to `--settle-time`, then goes on regardless, so the arm started while the pile was still moving.

### Fix
In `_build_cables` (`example_robot_panda_cable_pile.py`):
- Cable shape cfg: `ke=1e6`, `kd=0.1`, `mu=5`. Keep `rigid_gap=0.004`.
- `_add_cable`: `stretch_damping=1e-4`, `bend_damping=1e-2` (try 5e-2 if needed).
- `SolverVBD(..., friction_epsilon=1e-2, iterations=10)`.
- Drop the random cables from lower heights (smaller impact energy).
- Hand finger pads keep their own cfg (`HAND_MU`, `HAND_KE`). Re-check the grip, because the contact stiffness changes.
- Make settle meaningful: log max cable speed per second in the driver and require `max_cable_speed() < REST_SPEED` for ~0.3 s before the arm starts. If the cap is hit, print a warning instead of continuing silently.
- If it still jitters, next knobs in order: iterations 10 -> 20, 20 substeps per frame, `set_rigid_history_update` every 2nd substep.

## Issue 2: knot of multiple cables (interlocked loops, no pin)
Decisions: interlocked loops, and remove the pinned end so the whole cluster can leave the pile.

New geometry in place of `knot_cable_points`:
- `hook_cable_points(...)`: a cable whose end has a U-shaped hook (radius ~2 cm), tail running out of the pile.
- Chain of 3 cables: **A** (knot cable, graspable tail) hooks into **B**, and **B** hooks into **C**. B has a hook at each end. Neighbouring hooks lie in perpendicular planes (one horizontal, one vertical) so they thread through each other, with clearance well above 2 cable radii at the start.
- Hooks pull tight against each other under tension and cannot slip apart unless a cable opens. The knot cable is not pinned.
- `_add_cable` no longer takes `pin_first`; `_zero_mass` is only used for the hand mirrors.
- The rest of the pile (`--num-cables`) drops randomly over and around the cluster, as before. The `free` cable stays for `--pull free`.
- The grasp point stays ~3 segments behind A's tail tip.

New metrics (`cable_metrics` and the driver, replacing the single-cable ratio for `--pull knot`):
- Displacement of each cable's centroid before and after the pull, for A, B and C and for the unlinked cables.
- Pass criterion: B and C both move toward the pull direction by a clear margin (> ~5 cm), while unlinked cables move much less.
- The chain stays linked when the pull ends.

## Files
- Modify only `example_robot_panda_cable_pile.py`: `knot_cable_points` -> `hook_cable_points`, `_build_cables`, `_add_cable`, `cable_metrics` / `knot_loop_diameter`, `test_final`, settle logic.
- Verification helper scripts stay in `/tmp` (`drive.py`, `diag*.py`).

## Verification (headless, `<env python> ... --viewer null`)
1. Settle: after the settle phase, max body speed < 0.02 for 0.3 s, and the pile does not drift afterwards (compare centroids 1 s apart). Print the time to rest.
2. `test_final` still passes: finite state, segment lengths within 10%, nothing through the ground.
3. `--pull knot`: centroid displacement of A, B and C as above; cables A-B and B-C still linked (hook endpoints still within hook radius of each other).
4. `--pull free`: free cable still slides out straight.
5. Grip (friction) holds through the pull; the grasped segment stays within ~2 cm of the TCP. `--grasp sticky` as a cross-check.
6. `--filter-mirrors` control run still shows the hand has no effect.
7. Visual check of the settled pile and the pull in the GL viewer.

## Risks
- Hooks may tangle with other pile cables or open while settling. Mitigation: place the cluster on top of the pile and check hook separation in the metrics.
- Stiffer contacts (`ke=1e6`) can make VBD less stable and slower on CPU. Mitigation: the iteration knobs above.
- Pulling through a chain needs a stronger grip. Mitigation: raise `HAND_MU` or slow the pull.
