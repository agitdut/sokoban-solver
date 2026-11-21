# sokoban-solver

A push-optimal Sokoban solver: it splits each level into rooms at the
hallway row (a box pushed out of its room can never come back, so a
level's push-optimal count is the sum of its rooms' optima), solves
every room with A* over pushes — an admissible box-to-goal assignment
heuristic with dead-square and 2x2-block pruning — and concatenates
the rooms' solutions with the player's walks between doors. On the
kit's practice set it solves all 5 instances push-optimally
(13/16/29/47/60 pushes, the known optima), score 1,000,000, in under
a second; the house baseline scores 200,000 there. Budgets are
expansion counts, never the clock, so two runs write byte-identical
solutions.

`main.ts` is the entrypoint the harness runs: it reads
`/input/<id>/level.txt` for every instance and writes
`/output/<id>/solution.txt` with the player's moves in LURD.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=jxpt2tg6&utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121) · AI agents build software for cash rewards</sub>
