// Sokoban solver: push-optimal per room, then concatenated.
//
// The generator lays a hallway one square wide across the board (the
// player starts in it) with one to four rooms off it, each room holding
// 3 or 4 boxes and as many goals, and a box pushed out of its room can
// never be pushed back — so a level's push-optimal count is the sum of
// its rooms' optima. This program exploits that: it splits the level
// into rooms at the hallway row, solves each room push-optimally with
// A* over pushes (admissible assignment heuristic, dead-square and
// 2x2-block pruning), and concatenates the rooms' solutions with the
// player's walks between doors. A room it cannot solve makes its level
// score 0, so a failed room is skipped rather than faked.
//
// The program runs as node /bot/main.ts on Node 22: it answers every
// instance of a set in one run, reading /input/<id>/level.txt and
// writing /output/<id>/solution.txt (the player's moves in LURD, one
// letter a step; the replay decides which steps push). It budgets by
// work (expansion counts), never by the clock, so two runs of it on
// the same set write byte-identical files.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expansion counts, never the clock, so runs are
// deterministic): per room and for the whole set.
const ROOM_BUDGET = 300_000;
const TOTAL_BUDGET = 3_000_000;
let spent = 0;

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  boxes: number[];
  player: number;
}

function parse(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// Breadth-first search over passable squares (a boolean array over
// the level's cells), avoiding the given blocked set. Returns the
// move string from start to target, or null when unreachable.
function bfsPath(
  lv: Level,
  passable: boolean[],
  blocked: ReadonlySet<number>,
  start: number,
  target: number,
): string | null {
  if (start === target) return '';
  const prev = new Int32Array(lv.w * lv.h).fill(-2);
  const prevMove = new Array<string>(lv.w * lv.h).fill('');
  const queue: number[] = [start];
  prev[start] = -1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi]!;
    const cx = cur % lv.w;
    const cy = Math.floor(cur / lv.w);
    for (const [dx, dy, letter] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const ni = ny * lv.w + nx;
      if (!passable[ni] || blocked.has(ni) || prev[ni] !== -2) continue;
      prev[ni] = cur;
      prevMove[ni] = letter;
      if (ni === target) {
        let moves = '';
        for (let t = ni; t !== start; t = prev[t]!) moves = prevMove[t] + moves;
        return moves;
      }
      queue.push(ni);
    }
  }
  return null;
}

// BFS distances from every goal (multi-source) over the given
// passable squares. dist[i] is the shortest path length from cell i
// to the nearest goal, or Infinity when unreachable.
function goalDistances(lv: Level, passable: boolean[], goals: number[]): number[] {
  const dist = new Array<number>(lv.w * lv.h).fill(Infinity);
  const queue: number[] = [];
  for (const g of goals) {
    dist[g] = 0;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi]!;
    const cx = cur % lv.w;
    const cy = Math.floor(cur / lv.w);
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
      const ni = ny * lv.w + nx;
      if (!passable[ni] || dist[ni] !== Infinity) continue;
      dist[ni] = dist[cur]! + 1;
      queue.push(ni);
    }
  }
  return dist;
}

// Dead squares: a box on one can never reach a goal. Computed by
// reverse reachability from the goals: a square is live when a box
// on it can be pushed onto a live neighbour, which needs the player
// to stand on the square beyond it (playerPassable there).
function deadSquares(
  lv: Level,
  boxPassable: boolean[],
  playerPassable: boolean[],
  goals: number[],
): boolean[] {
  const live = new Array<boolean>(lv.w * lv.h).fill(false);
  for (const g of goals) live[g] = true;
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < lv.w * lv.h; i++) {
      if (!boxPassable[i] || live[i]) continue;
      const x = i % lv.w;
      const y = Math.floor(i / lv.w);
      for (const [dx, dy] of DIRS) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
        const ni = ny * lv.w + nx;
        if (!live[ni]) continue;
        const bx = x - dx;
        const by = y - dy;
        if (bx < 0 || bx >= lv.w || by < 0 || by >= lv.h) continue;
        if (!playerPassable[by * lv.w + bx]) continue;
        live[i] = true;
        changed = true;
        break;
      }
    }
  }
  const dead = new Array<boolean>(lv.w * lv.h).fill(false);
  for (let i = 0; i < lv.w * lv.h; i++) dead[i] = boxPassable[i] && !live[i];
  return dead;
}

// A 2x2 block of walls/boxes with no goal in it traps its boxes.
function isFrozen2x2(
  lv: Level,
  wall: boolean[],
  boxSet: ReadonlySet<number>,
  goalSet: ReadonlySet<number>,
  at: number,
): boolean {
  const x = at % lv.w;
  const y = Math.floor(at / lv.w);
  for (let oy = -1; oy <= 0; oy++) {
    for (let ox = -1; ox <= 0; ox++) {
      const cells = [
        (y + oy) * lv.w + (x + ox),
        (y + oy) * lv.w + (x + ox + 1),
        (y + oy + 1) * lv.w + (x + ox),
        (y + oy + 1) * lv.w + (x + ox + 1),
      ];
      if (cells.some((c) => c < 0 || c >= lv.w * lv.h)) continue;
      if (cells.some((c) => goalSet.has(c))) continue;
      if (cells.every((c) => wall[c] || boxSet.has(c))) return true;
    }
  }
  return false;
}

interface SolverBoard {
  // Cells the player may stand on, and cells a box may be pushed onto.
  playerPassable: boolean[];
  boxPassable: boolean[];
  goals: number[];
  boxes: number[];
  player: number;
  // Push-optimal A* needs the assignment heuristic only for small
  // box counts; larger boards fall back to sum-of-min-distances.
  useAssignment: boolean;
}

// Push-optimal A* over pushes. Returns the move string (walks in
// lowercase, pushes in uppercase) from the board's player start to a
// state with every box on a goal, or null when the budget runs out.
// Cost is pushes; the heuristic is admissible, so a returned
// solution is push-optimal.
function solveBoard(lv: Level, board: SolverBoard, budget: number): string | null {
  const { playerPassable, boxPassable, goals, boxes, player } = board;
  const goalSet = new Set(goals);
  const dist = goalDistances(lv, boxPassable, goals);
  const dead = deadSquares(lv, boxPassable, playerPassable, goals);
  const nb = boxes.length;
  const ng = goals.length;
  if (nb === 0) return '';
  if (nb !== ng) return null;
  for (const b of boxes) if (dist[b] === Infinity) return null;

  // Per-goal distance tables (only built when the assignment
  // heuristic is used): goalDist[k][cell] is the BFS distance from
  // goal goals[k] to cell over box-passable squares.
  const goalDist: number[][] = board.useAssignment
    ? goals.map((g) => goalDistances(lv, boxPassable, [g]))
    : [];
  // Minimum-assignment heuristic: the cheapest bijection of boxes to
  // goals, brute-forced over goal permutations (boxes <= 4).
  const assignmentHeuristic = (bs: number[]): number => {
    let best = Infinity;
    const perm: number[] = [];
    for (let i = 0; i < ng; i++) perm.push(i);
    const rec = (i: number, acc: number): void => {
      if (acc >= best) return;
      if (i === nb) { best = acc; return; }
      for (let j = i; j < ng; j++) {
        const t = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = t;
        const d = goalDist[perm[i]!]![bs[i]!]!;
        if (d !== Infinity) rec(i + 1, acc + d);
        const t2 = perm[i]!;
        perm[i] = perm[j]!;
        perm[j] = t2;
      }
    };
    rec(0, 0);
    return best;
  };
  // Minimum-assignment heuristic (boxes <= 4) or sum of each box's
  // nearest-goal distance. Both are admissible: one push moves one
  // box one square, so the total can drop by at most 1 per push.
  const heuristic = (bs: number[]): number => {
    if (board.useAssignment) return assignmentHeuristic(bs);
    let s = 0;
    for (const b of bs) {
      const d = dist[b]!;
      if (d === Infinity) return Infinity;
      s += d;
    }
    return s;
  };
  // A cell counts as blocking in the 2x2 check only when a box can
  // neither be pushed onto it nor the player stand on it (the door
  // is player-passable, so a 2x2 holding it is not a trap).
  const wall2x2 = playerPassable.map((pp, i) => !pp && !boxPassable[i]);

  const keyOf = (p: number, bs: number[]): number | string => {
    if (nb <= 5) {
      let k = p;
      for (const b of bs) k = k * (lv.w * lv.h) + b;
      return k;
    }
    return `${p}:${bs.join(',')}`;
  };

  // Heap on [f, h, counter, state].
  const heap: [number, number, number, number][] = [];
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  const startBoxes = boxes.slice().sort((a, b) => a - b);
  const sp: number[] = [player];
  const sb: number[][] = [startBoxes];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<number | string, number>([[keyOf(player, startBoxes), 0]]);
  const h0 = heuristic(startBoxes);
  if (h0 === Infinity) return null;
  let counter = 0;
  heapPush([h0, h0, counter++, 0]);
  let expanded = 0;

  while (heap.length > 0 && expanded < budget && spent < TOTAL_BUDGET) {
    const [, , , s] = heapPop();
    if (best.get(keyOf(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goalSet.has(b))) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t] + moves;
      return moves;
    }
    expanded++;
    spent++;
    const p = sp[s]!;
    const boxSet = new Set(bs);
    for (let bi = 0; bi < nb; bi++) {
      const b = bs[bi]!;
      const bx = b % lv.w;
      const by = Math.floor(b / lv.w);
      for (const [dx, dy, letter] of DIRS) {
        const tx = bx + dx;
        const ty = by + dy;
        if (tx < 0 || tx >= lv.w || ty < 0 || ty >= lv.h) continue;
        const target = ty * lv.w + tx;
        if (!boxPassable[target] || boxSet.has(target)) continue;
        const px = bx - dx;
        const py = by - dy;
        if (px < 0 || px >= lv.w || py < 0 || py >= lv.h) continue;
        const pusher = py * lv.w + px;
        if (!playerPassable[pusher] || boxSet.has(pusher)) continue;
        const walk = bfsPath(lv, playerPassable, boxSet, p, pusher);
        if (walk === null) continue;
        if (dead[target]) continue;
        const nb2 = bs.slice();
        nb2[bi] = target;
        nb2.sort((a, b) => a - b);
        const boxSet2 = new Set(nb2);
        if (isFrozen2x2(lv, wall2x2, boxSet2, goalSet, target)) continue;
        const g2 = sg[s]! + 1;
        const k2 = keyOf(b, nb2);
        const seen = best.get(k2);
        if (seen !== undefined && sg[seen]! <= g2) continue;
        const h2 = board.useAssignment ? assignmentHeuristic(nb2) : heuristic(nb2);
        if (h2 === Infinity) continue;
        const t = sp.length;
        sp.push(b);
        sb.push(nb2);
        sg.push(g2);
        parent.push(s);
        step.push(walk + letter.toUpperCase());
        best.set(k2, t);
        heapPush([g2 + h2, h2, counter++, t]);
      }
    }
  }
  return null;
}

// Split a level into rooms at the hallway row (the player's row).
// Returns the rooms (each with its door) or null when the level does
// not split cleanly.
interface Room {
  door: number;
  boxes: number[];
  goals: number[];
  squares: number[];
}

function splitRooms(lv: Level): Room[] | null {
  const hallwayRow = Math.floor(lv.player / lv.w);
  // Floor without the hallway row: each component is one room.
  const rest = lv.wall.map((w, i) => w || Math.floor(i / lv.w) === hallwayRow);
  const seen = new Array<boolean>(lv.w * lv.h).fill(false);
  const rooms: Room[] = [];
  const boxRoom = new Map<number, number>();
  for (let i = 0; i < lv.w * lv.h; i++) {
    if (rest[i] || seen[i]) continue;
    const comp: number[] = [];
    const queue = [i];
    seen[i] = true;
    for (let qi = 0; qi < queue.length; qi++) {
      const cur = queue[qi]!;
      comp.push(cur);
      const cx = cur % lv.w;
      const cy = Math.floor(cur / lv.w);
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || nx >= lv.w || ny < 0 || ny >= lv.h) continue;
        const ni = ny * lv.w + nx;
        if (rest[ni] || seen[ni]) continue;
        seen[ni] = true;
        queue.push(ni);
      }
    }
    const boxes = comp.filter((c) => lv.boxes.includes(c));
    const goals = comp.filter((c) => lv.goals.includes(c));
    if (boxes.length === 0 && goals.length === 0) continue; // stray floor
    if (boxes.length === 0 || boxes.length !== goals.length) return null;
    // The door: hallway-row floor squares adjacent to the component.
    const doors = new Set<number>();
    for (const c of comp) {
      const cx = c % lv.w;
      const cy = Math.floor(c / lv.w);
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (ny !== hallwayRow || nx < 0 || nx >= lv.w) continue;
        const ni = ny * lv.w + nx;
        if (!lv.wall[ni]) doors.add(ni);
      }
    }
    if (doors.size !== 1) return null;
    const door = [...doors][0]!;
    for (const b of boxes) boxRoom.set(b, rooms.length);
    rooms.push({ door, boxes, goals, squares: comp });
  }
  if (boxRoom.size !== lv.boxes.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return rooms;
}

function solveLevel(lv: Level): string {
  const rooms = splitRooms(lv);
  const floorAll = lv.wall.map((w) => !w);
  if (rooms !== null) {
    let moves = '';
    let player = lv.player;
    const boxSet = new Set(lv.boxes);
    for (const room of rooms) {
      // Walk to the room's door through the hallway.
      const walkIn = bfsPath(lv, floorAll, boxSet, player, room.door);
      if (walkIn === null) return moves;
      moves += walkIn;
      player = room.door;
      // The room's board: the player may stand on its squares and
      // the door; a box may only be pushed onto its squares.
      const playerPassable = lv.wall.map((w, i) => !w && (room.squares.includes(i) || i === room.door));
      const boxPassable = lv.wall.map((w, i) => !w && room.squares.includes(i));
      const solved = solveBoard(lv, {        playerPassable,
        boxPassable,
        goals: room.goals,
        boxes: room.boxes,
        player: room.door,
        useAssignment: true,
      }, ROOM_BUDGET);
      if (solved === null) return moves;
      moves += solved;
      // The player ends inside the room; walk back out to the door.
      // (Recompute the player's square by replaying the room's moves
      // is overkill: the solver's last push leaves the player on the
      // pushed box's old square, so re-derive it from the board by
      // re-running the walk is avoided by tracking it below.)
      player = playerAfterRoom(lv, room, solved);
      // Boxes now sit on the room's goals; walking out treats them
      // as obstacles.
      const walkOut = bfsPath(lv, playerPassable, new Set(room.goals), player, room.door);
      if (walkOut === null) return moves;
      moves += walkOut;
      player = room.door;
      for (const g of room.goals) boxSet.add(g);
      for (const b of room.boxes) boxSet.delete(b);
    }
    return moves;
  }
  // Fallback: push-A* on the whole level.
  const solved = solveBoard(lv, {
    playerPassable: floorAll,
    boxPassable: floorAll,
    goals: lv.goals,
    boxes: lv.boxes,
    player: lv.player,
    useAssignment: lv.boxes.length <= 4,
  }, TOTAL_BUDGET);
  return solved ?? '';
}

// The player's square after a room's solution: replay the room's
// moves from the door over the room's floor with its boxes.
function playerAfterRoom(lv: Level, room: Room, moves: string): number {
  const boxes = new Set(room.boxes);
  let p = room.door;
  const passable = (i: number): boolean =>
    !lv.wall[i] && (room.squares.includes(i) || i === room.door);
  for (const m of moves) {
    const lower = m.toLowerCase();
    const d = DIRS.find(([, , l]) => l === lower)!;
    const q = p + d[1] * lv.w + d[0];
    if (!passable(q)) continue;
    if (boxes.has(q)) {
      const t = q + d[1] * lv.w + d[0];
      if (!room.squares.includes(t) || boxes.has(t)) continue;
      boxes.delete(q);
      boxes.add(t);
    }
    p = q;
  }
  return p;
}

for (const id of readdirSync(I).sort()) {
  const level = parse(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  const moves = solveLevel(level);
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves.toLowerCase()}\n`);
}
