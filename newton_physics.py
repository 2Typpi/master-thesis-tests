import time
import mujoco
import mujoco.viewer
import warp as wp
import numpy as np
import newton

NUM_ARM_ACTIONS = 7
MAX_DELTA_Q = 0.05
BASKET_RELATIVE_BOUNDS = {
    "min_x": -0.095, "max_x": 0.095,
    "min_y": -0.095, "max_y": 0.095,
    "min_z":  0.005, "max_z": 0.180,
}

def is_object_in_basket(
    d: mujoco.MjData, 
    obj_body_id: int, 
    basket_mocap_id: int
) -> bool:
    """Fast check if object center is inside basket bounds."""
    basket_pos = d.mocap_pos[basket_mocap_id]
    obj_pos = d.xpos[obj_body_id]

    rel_x = obj_pos[0] - basket_pos[0]
    rel_y = obj_pos[1] - basket_pos[1]
    rel_z = obj_pos[2] - basket_pos[2]

    b = BASKET_RELATIVE_BOUNDS
    return (
        b["min_x"] <= rel_x <= b["max_x"] and
        b["min_y"] <= rel_y <= b["max_y"] and
        b["min_z"] <= rel_z <= b["max_z"]
    )


GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 0.04

def step(action: np.ndarray, d: mujoco.MjData):
    """
    Applies an 8-dim action vector [delta_q(7), gripper(1)] to the simulation.
    """
    # copy current positions
    current_ctrl = d.ctrl.copy()

    # Scale and apply changes
    arm_action = np.asarray(action[:NUM_ARM_ACTIONS], dtype=float)
    current_ctrl[:NUM_ARM_ACTIONS] += arm_action * MAX_DELTA_Q

    # action > 0 -> open, if < 0 -> close
    gripper_command = 0.04 if action[7] > 0 else 0.0
    current_ctrl[7] = gripper_command

    d.ctrl[:] = current_ctrl
        

def teleport_in_basket(mjm, d, qpos_adr):
    d.qpos[qpos_adr : qpos_adr + 3] = [0.4, 0.4, 0.03]
    mujoco.mj_forward(mjm, d)

def reset(mjm, d, basket_mocap_id, key_id):
    mujoco.mj_resetDataKeyframe(mjm, d, key_id)
    d.mocap_pos[basket_mocap_id] = [0.4, 0.4, 0]

def build_model():
    num_elements = 40
    span_length = 1.0
    nodes = np.zeros((num_elements + 1, 3))
    for i in range(num_elements + 1):
        alpha = i / num_elements
        nodes[i] = [alpha * span_length, 0.0, 1.0 - 0.5 * np.sin(alpha * np.pi)]

    points = [wp.vec3(*p) for p in nodes]

    # 3. Configure cable sizing and contact properties
    seg_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    mean_seg_length = float(np.mean(seg_lengths))
    cable_radius = 0.4 * mean_seg_length
    # Create model builder with gravity
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))

    # Define contact configuration for self-collision and environment collision
    shape_cfg = newton.ModelBuilder.ShapeConfig(
        ke=3.0e5,  # Contact stiffness
        kd=10.0,   # Contact damping
        gap=0.6 * mean_seg_length
    )

    # Instantiate the Rod primitive
    rod = newton.Rod(points, radius=cable_radius)

    # Add the cable to the model builder with material stiffness parameters
    bodies, joints = builder.add_rod(
        rod=rod,
        cfg=shape_cfg,
        stretch_stiffness=1.0e6,
        stretch_damping=0.0,
        bend_stiffness=50.0,
        bend_damping=0.01,
        twist_stiffness=40.0,
        twist_damping=0.01
    )

    # Finalize the model to compile data structures for the GPU
    return builder.finalize()

def main():
    wp.init()
    wp.set_device("cpu")

    scene_path = "franka_emika_panda/mjx_single_cube_dlo.xml"

    spec = mujoco.MjSpec.from_file(scene_path)

    mjm = spec.compile()

    print("No CUDA device found. Falling back to Standard MuJoCo Interactive Viewer...")

    d = mujoco.MjData(mjm)

    mujoco.mj_forward(mjm, d)

    cube1_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "object1")
    joint_cube1_id = mjm.body_jntadr[cube1_id]
    
    body_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "basket")
    basket_mocap_id = mjm.body_mocapid[body_id]

    print(f"cube: {cube1_id}")
    print(f"basket: {basket_mocap_id}")

    key_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "pickup1")
    mujoco.mj_resetDataKeyframe(mjm, d, key_id)

    # Set Spawn for basket
    d.mocap_pos[basket_mocap_id] = [0.4, 0.4, 0]

    # Run standard MuJoCo with GUI viewer
    with mujoco.viewer.launch_passive(mjm, d, show_right_ui=True) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            step_count = int(round(d.time / mjm.opt.timestep))

            if step_count % 200 == 0:
                print(f"Elapsed time: {d.time:.4f}s | Steps: {step_count}")

            if is_object_in_basket(d=d, obj_body_id=cube1_id, basket_mocap_id=basket_mocap_id):
                print("Cube in Basket!")
                reset(mjm, d, basket_mocap_id, key_id)

            mujoco.mj_step(mjm, d)
            viewer.sync()

            # Maintain real-time pacing
            time_until_next_step = mjm.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()