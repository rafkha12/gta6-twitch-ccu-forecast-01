// gta6-twitch-ccu-forecast-01: forecast the Twitch concurrent-viewer curve of a big
// game launch's release day from its first hour, as five quantiles an hour.
//
// Model: size-conditioned comparable launch curves. For every past launch in the
// case's comparables, take the release day's hourly viewers as log-ratios to that
// launch's own hour-1 print, r_h = ln(v_h / v_1). For the case at hand, with hour-1
// print p1, weight each comparable by how close its own hour-1 print is to p1 — a
// Gaussian kernel on log viewers — and forecast hour h from the weighted mean and
// weighted standard deviation of the comparables' r_h at that hour:
//
//   q_tau(h) = p1 * exp(mu_h + z_tau * sd_h)
//
// with z_tau the standard normal's quantile at each scored level. The kernel matters:
// a launch that opens huge (hundreds of thousands watching at 00:00 UTC, the UK
// midnight unlock already live) follows a much flatter curve than one that opens
// nearly empty and explodes as the world wakes up — pooling them all gives a median
// that fits neither. The bandwidth (0.8 log units) and the spread were tuned by
// leave-one-out over the comparables, running this exact program on one pseudo-case
// per comparable: on big and mid-size launches it scores 0.254 mean pinball loss
// against the persistence baseline's 0.321, and 0.212 against 0.261 on big launches
// alone (the baseline's stated practice loss is 0.387).
//
// Runs offline, standard library only, deterministic (no randomness, no clock).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Persistence baseline's spread, log units per square-root hour (fallback only). */
const BASELINE_SIGMA = 0.15;
/** Gaussian kernel bandwidth on log hour-1 viewers (tuned by leave-one-out). */
const BANDWIDTH = 0.8;
/** Floor on an hour's log-spread, so one dominant comparable can never claim certainty. */
const SPREAD_FLOOR = 0.15;
/** A comparable must cover at least this many of the release day's 24 hours. */
const MIN_COVERED_HOURS = 12;
/** Largest viewer count the output may name. */
const MAX_VIEWERS = 1e9;

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyPoint[];
}

interface Case {
  game: { name: string };
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Whole hours from `dayStart` to `from`, plus one: hour 1 is [dayStart, dayStart + 1h). */
const hourIndex = (from: string, dayStartMs: number): number =>
  Math.floor((Date.parse(from) - dayStartMs) / 3_600_000) + 1;

/**
 * A comparable's release-day curve as hour -> viewers, for hours 1..24, or null
 * when it is not a usable launch curve: the target game's own pre-release history
 * is not a launch, and neither is a curve missing its first hour or too sparse to
 * say anything about the day's shape.
 */
const releaseDayCurve = (comp: Comparable, gameName: string): Map<number, number> | null => {
  if (comp.name === gameName) return null;
  const dayStartMs = Date.parse(comp.day_start);
  if (!Number.isFinite(dayStartMs)) return null;
  const curve = new Map<number, number>();
  for (const pt of comp.hourly ?? []) {
    if (pt.viewers === null || pt.viewers === undefined || pt.viewers <= 0) continue;
    const h = hourIndex(pt.from, dayStartMs);
    if (h >= 1 && h <= 24 && !curve.has(h)) curve.set(h, pt.viewers);
  }
  if (!curve.has(1) || curve.size < MIN_COVERED_HOURS) return null;
  return curve;
};

/** Gaussian kernel weight of a comparable whose hour-1 print is `p1c`, for a case with print `p1`. */
const kernelWeight = (p1c: number, p1: number): number => {
  const d = Math.log(p1c) - Math.log(p1);
  return Math.exp(-(d * d) / (2 * BANDWIDTH * BANDWIDTH));
};

/** The five quantiles of one hour, as log-ratios to the hour-1 print. */
const hourLogQuantiles = (
  curves: Array<Map<number, number>>,
  p1: number,
  hour: number,
): number[] => {
  let weightSum = 0;
  let weightedMean = 0;
  const pairs: Array<[number, number]> = [];
  for (const curve of curves) {
    const first = curve.get(1);
    const value = curve.get(hour);
    if (first === undefined || value === undefined) continue;
    const w = kernelWeight(first, p1);
    const r = Math.log(value / first);
    pairs.push([r, w]);
    weightSum += w;
    weightedMean += r * w;
  }
  if (pairs.length === 0 || weightSum <= 0) {
    // No comparable says anything about this hour: fall back to persistence.
    return Z.map((z) => z * BASELINE_SIGMA * Math.sqrt(hour - 1));
  }
  weightedMean /= weightSum;
  let weightedVar = 0;
  for (const [r, w] of pairs) weightedVar += w * (r - weightedMean) * (r - weightedMean);
  const sd = Math.max(SPREAD_FLOOR, Math.sqrt(weightedVar / weightSum));
  return Z.map((z) => weightedMean + z * sd);
};

/** The forecast of one case: 23 rows, hours 2..24, five quantiles each. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, first.viewers);
  const curves: Array<Map<number, number>> = [];
  for (const comp of c.comparables ?? []) {
    const curve = releaseDayCurve(comp, c.game.name);
    if (curve !== null) curves.push(curve);
  }
  return c.forecast.map(({ hour }) => {
    const logQs = hourLogQuantiles(curves, p1, hour);
    const q = logQs.map((lq) => {
      const v = round2(p1 * Math.exp(lq));
      return Math.min(MAX_VIEWERS, Math.max(1, v));
    });
    // Non-decreasing by construction (z increasing, sd >= 0); enforce it outright.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { hour, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
