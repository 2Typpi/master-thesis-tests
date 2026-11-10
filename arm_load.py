import time
import mujoco
import mujoco.viewer
import warp as wp
import numpy as np
from dlo.controller import DloRope
from dlo.rope_xml import RopeConfig, generate_rope_xml
from dlo.sim import step


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

def gripper_object_forces(
    mjm: mujoco.MjModel,
    d: mujoco.MjData,
    obj_body_id: int,
) -> dict[str, float]:
    """Return summed normal contact forces on the two fingers, in newtons."""
    finger_ids = {
        "left": mujoco.mj_name2id(
            mjm, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        ),
        "right": mujoco.mj_name2id(
            mjm, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        ),
    }
    forces = {name: 0.0 for name in finger_ids}
    contact_force = np.zeros(6)

    for contact_id in range(d.ncon):
        contact = d.contact[contact_id]
        if contact.dist > 0:
            continue

        body1 = mjm.geom_bodyid[contact.geom1]
        body2 = mjm.geom_bodyid[contact.geom2]
        finger_name = next(
            (
                name for name, finger_id in finger_ids.items()
                if (body1 == obj_body_id and body2 == finger_id)
                or (body2 == obj_body_id and body1 == finger_id)
            ),
            None,
        )
        if finger_name is None:
            continue

        # mj_contactForce() returns force in the contact frame.  The first
        # component is normal force; abs() removes the body-order sign.
        mujoco.mj_contactForce(mjm, d, contact_id, contact_force)
        forces[finger_name] += abs(float(contact_force[0]))

    return forces


def is_object_grasped(
    mjm: mujoco.MjModel,
    d: mujoco.MjData,
    obj_body_id: int,
    min_normal_force: float = 0.5,
) -> bool:
    """Return true when both fingers apply at least the given force."""
    forces = gripper_object_forces(mjm, d, obj_body_id)
    return all(force >= min_normal_force for force in forces.values())


def grip_until_grasped(
    mjm: mujoco.MjModel,
    d: mujoco.MjData,
    viewer,
    obj_body_id: int,
    min_normal_force: float = 0.5,
    stable_steps: int = 10,
    max_steps: int = 200,
) -> bool:
    """Close the gripper and wait for a stable force on both fingers."""
    stable_count = 0
    for _ in range(max_steps):
        if not viewer.is_running():
            return False

        grip(d)
        mujoco.mj_step(mjm, d)
        viewer.sync()

        forces = gripper_object_forces(mjm, d, obj_body_id)
        print(f"gripper forces [N]: left={forces['left']:.3f}, right={forces['right']:.3f}")
        if all(force >= min_normal_force for force in forces.values()):
            stable_count += 1
            if stable_count >= stable_steps:
                return True
        else:
            stable_count = 0

    return False


GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 0.04


def grip(d: mujoco.MjData) -> None:
    """Command the gripper to close around an object."""
    d.ctrl[NUM_ARM_ACTIONS] = GRIPPER_CLOSED


def release(d: mujoco.MjData) -> None:
    """Command the gripper to open and release an object."""
    d.ctrl[NUM_ARM_ACTIONS] = GRIPPER_OPEN


def step_action(action: np.ndarray, d: mujoco.MjData):
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

def move_to(
    mjm,
    d,
    viewer,
    target_ctrl,
    speed=0.05,
    tolerance=0.01,
    max_steps=1000,
    integral_gain=0.02,
    max_trim=0.2,
):
    """Move to a joint target, compensating static load with control trim."""
    target_ctrl = np.asarray(target_ctrl, dtype=float)
    if target_ctrl.shape != d.ctrl.shape:
        raise ValueError(
            f"target_ctrl must have shape {d.ctrl.shape}, got {target_ctrl.shape}"
        )

    # The first seven actuators transmit to the seven arm joints.  Looking up
    # their qpos addresses avoids assuming that the arm is the first qpos block
    # in the model (the free cube joint is another qpos block).
    arm_actuator_ids = np.arange(NUM_ARM_ACTIONS)
    arm_joint_ids = mjm.actuator_trnid[arm_actuator_ids, 0]
    arm_qpos_adr = mjm.jnt_qposadr[arm_joint_ids]

    # Position actuators settle below their target under gravity/load.  Keep a
    # bounded trim of the desired position to remove that static error.
    trim = np.zeros(NUM_ARM_ACTIONS)

    for _ in range(max_steps):
        if not viewer.is_running():
            return False

        step_start = time.time()

        # Position actuators interpret ctrl as desired joint positions.  Ramp
        # the desired positions instead of changing qpos directly.
        d.ctrl[:NUM_ARM_ACTIONS] += np.clip(
            target_ctrl[:NUM_ARM_ACTIONS] + trim - d.ctrl[:NUM_ARM_ACTIONS],
            -speed,
            speed,
        )
        d.ctrl[NUM_ARM_ACTIONS:] = target_ctrl[NUM_ARM_ACTIONS:]
        mujoco.mj_step(mjm, d)
        viewer.sync()

        qpos_error = target_ctrl[:NUM_ARM_ACTIONS] - d.qpos[arm_qpos_adr]
        error_norm = np.linalg.norm(qpos_error)
        # print(
        #     "errors:",
        #     np.array2string(qpos_error, precision=6),
        #     "max:",
        #     np.max(np.abs(qpos_error)),
        # )
        # print("actuator forces:", d.actuator_force[:NUM_ARM_ACTIONS])


        # print("force limits:", mjm.actuator_forcerange[:NUM_ARM_ACTIONS])
        if error_norm < tolerance:
            print("Target reached!")
            return True

        # Integrate only persistent position error.  This compensates gravity
        # and payload forces without changing qpos directly.
        trim += integral_gain * qpos_error
        trim = np.clip(trim, -max_trim, max_trim)

        # Pace against the simulation timestep.  Do not subtract an extra
        # second here; that made every iteration sleep for roughly one second.
        elapsed = time.time() - step_start
        if elapsed < mjm.opt.timestep:
            time.sleep(mjm.opt.timestep - elapsed)

    return False
        

def teleport_in_basket(mjm, d, qpos_adr):
    d.qpos[qpos_adr : qpos_adr + 3] = [0.4, 0.4, 0.03]
    mujoco.mj_forward(mjm, d)

def reset(mjm, d, basket_mocap_id, key_id):
    mujoco.mj_resetDataKeyframe(mjm, d, key_id)
    d.mocap_pos[basket_mocap_id] = [0.4, 0.4, 0]


def main():
    wp.init()

    # Check if a CUDA-enabled device is available in Warp
    cuda_available = any(d.is_cuda for d in wp.get_devices())

    scene_path = "franka_emika_panda/mjx_single_cube.xml"

    spec = mujoco.MjSpec.from_file(scene_path)
    
    config = RopeConfig(
        segments=40,
        initial_position=(0.6, 0.35, 0.01),
    )
    rope_spec = mujoco.MjSpec.from_string(generate_rope_xml(config))

    mount = spec.worldbody.add_frame(name="rope_mount")
    spec.attach(rope_spec, frame=mount, prefix="")
    mjm = spec.compile()

    if cuda_available:
        print("CUDA detected! Running with MuJoCo Warp (Headless/GPU mode)...")
        import mujoco_warp as mjw

        # Setup Warp structures
        m = mjw.put_model(mjm)
        d = mjw.make_data(mjm, nworld=1)

        # Run Warp simulation steps
        for steps in range(1000):
            mjw.step(m, d)
            if steps % 100 == 0:
                print(f"Warp step {steps} completed.")

    else:
        print("No CUDA device found. Falling back to Standard MuJoCo Interactive Viewer...")

        d = mujoco.MjData(mjm)

        mujoco.mj_forward(mjm, d)
        rope = DloRope(mjm, d, segments=config.segments)

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

        qpos_adr = mjm.jnt_qposadr[joint_cube1_id]

        # Use the arm qpos from the pickup1 keyframe as the first target.  The
        # keyframe qpos and ctrl are intentionally different for this model.
        pickup_target = mjm.key_ctrl[key_id].copy()
        pickup_target[:NUM_ARM_ACTIONS] = mjm.key_qpos[key_id][:NUM_ARM_ACTIONS]

        reached_num = 0

        target_positions = [
            np.array([0.2897, 0.495673, -0.142836, -2.14746,
                -0.0295746, 2.52378, -0.492496, GRIPPER_OPEN]),
            np.array([0.2897, 0.495673, -0.142836, -2.14746,
                    -0.0295746, 2.52378, -0.492496, GRIPPER_CLOSED]),
            np.array([1.04, -0.09620754, -0.14481577, -2.16147839,
                    -0.03115591, 2.51646667, -0.49248524, GRIPPER_CLOSED]),
            np.array([1.04, -0.09620754, -0.14481577, -2.16147839,
                    -0.03115591, 2.51646667, -0.49248524, GRIPPER_OPEN]),
        ]
        grasped = False
        # Run standard MuJoCo with GUI viewer
        with mujoco.viewer.launch_passive(mjm, d, show_right_ui=True) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                step_count = int(round(d.time / mjm.opt.timestep))
                # if step_count <= 200:
                #     mujoco.mj_step(mjm, d)
                #     viewer.sync()
                #     time_until_next_step = mjm.opt.timestep - (time.time() - step_start)
                #     if time_until_next_step > 0:
                #         time.sleep(time_until_next_step)
                #     continue

                if step_count % 200 == 0:
                    print(f"Elapsed time: {d.time:.4f}s | Steps: {step_count}")

                if reached_num < len(target_positions):
                    # The pickup pose is reached with the gripper open.  Once
                    # there, close it and wait until contact force is stable before
                    # moving the object away.
                    if reached_num == 1 and not grasped:
                        grasped = grip_until_grasped(
                            mjm, d, viewer, cube1_id, min_normal_force=1
                        )
                        if not grasped:
                            print("Grasp was not detected")
                            continue
                        print("Object grasped")

                    # reached = move_to(
                    #     mjm, d, viewer, target_ctrl=np.array(target_positions[reached_num])
                    # )
                    # if reached:
                    #     reached_num += 1

                # Keep the gripper command active between movement calls.
                # if reached_num == 1:
                #     grip(d)
                # elif reached_num >= 2:
                #     release(d)

                #step_action(action=target_actions[reached_num], d=d)

                if is_object_in_basket(d=d, obj_body_id=cube1_id, basket_mocap_id=basket_mocap_id):
                    print("Cube in Basket!")
                    reset(mjm, d, basket_mocap_id, key_id)


                step(mjm, d, rope)
                viewer.sync()

                # Maintain real-time pacing
                time_until_next_step = mjm.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()