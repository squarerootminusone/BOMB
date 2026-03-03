# Project Rules

## Go VLA Benchmark — Absolute Rules
- **NEVER teleport stones.** All stone placement must happen through physics simulation. No `set_stone_pose` or `_place_stone_at_intersection` after the episode starts.
- **NEVER pin stone positions.** Stones must always remain fully dynamic rigid bodies.
- Use `conda run -n mujogo` to run all Python scripts in this project.
