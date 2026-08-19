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
- Grounding DINO open-vocabulary detection and SAM mask prediction;
- RGB-D backprojection from agent-view and wrist cameras;
- persistent target identity and confidence-aware landmark memory;
- shadow-mode comparison between visual estimates and simulator ground truth.

## Perception milestone

Object positions are estimated from RGB images, rendered depth, and calibrated
camera geometry. Grounding DINO Tiny detects two black bowls, one plate, and
one ramekin; SAM ViT-B converts the boxes into masks. Masked depth pixels are
backprojected into world coordinates. End-effector position comes from ordinary
robot state.

The memory fuses the fixed agent-view and wrist cameras, associates the two
identical bowls over time, propagates an occluded held target with end-effector
motion, rejects implausible visual innovations, and retains the reset-time
plate and ramekin positions as static anchors. A monotonic task state machine
tracks approach, grasp, transport, place, verification, completion, and
placement failure. After a release near the plate, failure is declared only if
the success predicate remains unconfirmed for a temporal verification window.

On a static reset frame, learned-mask IoU against simulator masks ranged from
0.935 to 0.981. Learned RGB-D object-position error ranged from 0.05 to 0.50 cm.
After model loading, the median perception latency was 117.5 ms per inference
frame on an RTX 4090; inference was run every five control steps.

## Results

All numbers below are observed evaluation results for LIBERO Spatial task 5.

| Configuration | Success | Notes |
| --- | ---: | --- |
| PI0.5 baseline | 9/10 | Episode 2 ended in a placement failure |
| Ground-truth scene recovery | 10/10 | Validated recovery logic before perception integration |
| Oracle-mask RGB-D recovery | 10/10 | Validated 3D estimation and visual closed-loop control |
| Learned-mask RGB-D recovery | 10/10 | Grounding DINO + SAM detection and recovery targets |

In the final 10-episode learned-perception evaluation:

- episode 2 recovered from a placement failure with 116 overridden actions;
- the remaining nine episodes received zero recovery actions;
- the recovery trigger and bowl/plate target coordinates both came from the
  learned visual scene memory;
- no nominal rollout was modified by a false-positive recovery trigger;
- the final task success rate improved from 90% (9/10) to 100% (10/10).

These results show selective closed-loop intervention rather than unconditional
scripted control: the learned policy remains in control during nominal
execution, and the recovery controller acts only after a confirmed failure.

## Main files

- `task5_trace.py`: evaluation wrapper and per-step structured trace logging;
- `scene_memory.py`: simulator-state scene memory used for validation;
- `vision_scene_memory.py`: RGB-D scene estimation and task-state tracking;
- `learned_mask_perception.py`: persistent Grounding DINO and SAM inference;
- `learned_vision_scene_memory.py`: learned-mask multi-view 3D scene memory;
- `failure_detector.py`: grasp and placement failure detection;
- `placement_recovery.py`: object-relative recovery state machine;
- `analyze_scene_traces.py`: ground-truth trace analysis;
- `analyze_vision_shadow.py`: visual-versus-ground-truth evaluation;
- `inspect_task5_vision.py`: camera, depth, and segmentation inspection;
- `inspect_task5_masks.py`: task-object mask inspection;
- `inspect_task5_backprojection.py`: camera calibration and 3D backprojection;
- `inspect_task5_learned_detection.py`: open-vocabulary detector inspection;
- `inspect_task5_learned_masks.py`: learned-mask and oracle-IoU inspection;
- `inspect_task5_learned_backprojection.py`: learned-mask 3D validation;
- `benchmark_task5_learned_perception.py`: learned perception latency benchmark.

## Running the full visual recovery evaluation

From the LeRobot environment:

```bash
PLACE_RECOVERY=1 \
LEARNED_VISION_DETECTOR=1 \
LEARNED_VISION_RECOVERY_TARGETS=1 \
LEARNED_VISION_INTERVAL=5 \
TRACE_OUT=./task5_learned_closed_loop_trace_n10 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
python task5_trace.py \
  --output_dir=./eval_task5_learned_closed_loop_n10 \
  --job_name=task5_learned_closed_loop_n10 \
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

- Recovery skills and task-state thresholds are currently task-specific.
- Results are from simulation, one task, and a limited number of episodes.
- Simulator state and oracle masks are retained for evaluation traces only;
  neither supplies final recovery triggers or recovery target coordinates.
- Larger multi-seed evaluation is required before making a robustness claim.

## Next steps

1. Evaluate multiple seeds and additional LIBERO tasks.
2. Add controlled grasp, perception, and placement perturbations.
3. Generalize recovery skills behind task-independent interfaces.
4. Replace task-calibrated offsets with learned or geometry-derived estimates.
5. Connect the skill and scene-memory interfaces to ROS 2 actions.
