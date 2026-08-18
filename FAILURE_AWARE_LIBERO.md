# Failure-Aware Visual Recovery for LIBERO

This project augments a pretrained PI0.5 vision-language-action policy with
structured scene memory, explicit failure detection, and closed-loop recovery
for long-horizon manipulation in LIBERO.

## Task

The current experiment targets LIBERO Spatial task 5:

> Pick up the black bowl on the ramekin and place it on the plate.

The nominal policy is `lerobot/pi05_libero_finetuned`. Its 7D actions contain
relative end-effector translation, relative rotation, and one gripper command.

## System

```text
Language instruction + RGB observations + robot state
                         |
                         v
                    PI0.5 policy
                         |
                         v
RGB-D scene memory -> failure detector -> recovery controller
        |                                      |
        +---------- grasp / place state -------+
```

The system contains:

- per-step rollout tracing for actions, rewards, robot state, and scene state;
- a task-specific scene memory with approach, grasp, transport, place,
  verification, completion, and failure stages;
- grasp- and placement-failure detectors with temporal confirmation;
- an object-relative recovery controller that can regrasp and replace the bowl;
- RGB-D backprojection from the fixed agent-view camera;
- persistent target identity and confidence-aware landmark memory;
- shadow-mode comparison between visual estimates and simulator ground truth.

## Perception milestone

Object positions are estimated from agent-view depth and calibrated camera
geometry. MuJoCo instance segmentation is currently used only to provide
oracle object masks. Object body positions are not used by the visual recovery
controller. End-effector position comes from ordinary robot state.

The plate is normally kept as a stable landmark. A relocation is accepted only
when its mask retains at least 80% of the reset visibility and the measured
displacement remains spatially consistent for five frames. This prevents bowl
occlusion from being mistaken for physical plate motion.

## Results

All numbers below are observed evaluation results for LIBERO Spatial task 5.

| Configuration | Success | Notes |
| --- | ---: | --- |
| PI0.5 baseline | 8/10 | One grasp failure and one placement failure |
| Ground-truth scene recovery | 10/10 | Validated recovery logic before perception integration |
| Visual detector, ground-truth recovery targets | 3/3 | Recovery triggered only for the failed rollout |
| RGB-D detector and RGB-D recovery targets | 10/10 | Two failed rollouts recovered; eight nominal rollouts were not modified |

In the final 10-episode RGB-D evaluation:

- episode 2 recovered from a placement failure with 127 overridden actions;
- episode 3 recovered from a grasp failure with 140 overridden actions;
- the remaining eight episodes received zero recovery actions;
- no plate relocalization false positives occurred;
- placement-failure agreement on episode 2 was 97.8%;
- plate-position p95 error on episode 2 was 0.72 cm;
- the final task success rate was 100% (10/10).

These results show selective closed-loop intervention rather than unconditional
scripted control: the learned policy remains in control during nominal
execution, and the recovery controller acts only after a confirmed failure.

## Main files

- `task5_trace.py`: evaluation wrapper and per-step structured trace logging;
- `scene_memory.py`: simulator-state scene memory used for validation;
- `vision_scene_memory.py`: RGB-D scene estimation and task-state tracking;
- `failure_detector.py`: grasp and placement failure detection;
- `placement_recovery.py`: object-relative recovery state machine;
- `analyze_scene_traces.py`: ground-truth trace analysis;
- `analyze_vision_shadow.py`: visual-versus-ground-truth evaluation;
- `inspect_task5_vision.py`: camera, depth, and segmentation inspection;
- `inspect_task5_masks.py`: task-object mask inspection;
- `inspect_task5_backprojection.py`: camera calibration and 3D backprojection.

## Running the full visual recovery evaluation

From the LeRobot environment:

```bash
PLACE_RECOVERY=1 \
VISION_DETECTOR=1 \
VISION_RECOVERY_TARGETS=1 \
TRACE_OUT=./task5_full_vision_recovery_trace_n10 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python task5_trace.py \
  --output_dir=./eval_task5_full_vision_recovery_n10 \
  --job_name=task5_full_vision_recovery_n10 \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[5]' \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --env.max_parallel_tasks=1
```

## Limitations

- Instance masks come from the simulator rather than a learned segmenter.
- Recovery skills and task-state thresholds are currently task-specific.
- Results are from simulation, one task, and a limited number of episodes.
- Ground-truth scene state is retained for analysis, but not for visual recovery
  triggers or recovery target coordinates in the final configuration.
- Larger multi-seed evaluation is required before making a robustness claim.

## Next steps

1. Replace oracle masks with RGB-based object detection or segmentation.
2. Fuse agent-view and wrist-camera observations near grasp and placement.
3. Evaluate multiple seeds and additional LIBERO tasks.
4. Generalize recovery skills behind task-independent interfaces.
5. Connect the skill and scene-memory interfaces to ROS 2 actions.
