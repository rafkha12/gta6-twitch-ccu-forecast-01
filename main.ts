// gta6-twitch-ccu-forecast-01: forecast the Twitch concurrent-viewer curve of a big
// game launch's release day from its first hour, as five quantiles an hour.
//
// Idea: a launch's day is almost entirely set by how big its first hour is. Across
// past launches, the log of the peak-to-first-hour ratio correlates about -0.94
// with the log of the first hour itself: a launch that opens nearly empty explodes
// a hundredfold as the world wakes up, while one that opens at 800k watchers is
// already at its peak and stays flat. So for each hour we look at every past
// launch's day as a log-ratio to its own first hour, and fit those ratios against
// log first-hour size with a Gaussian-weighted local line centred on this case's
// first hour: the line's intercept is the hour's centre, its slope corrects the
// bias a plain weighted mean has at the edges of the size range (the biggest
// launches sit past most of the training data, where a mean just flattens out).
// The centre is blended 70/30 with the plain weighted mean, which keeps the fit
// honest where the line would extrapolate too far past the largest launches seen.
// The spread is the weighted scatter of the ratios around the fitted line, floored
// so no hour ever claims more certainty than the data supports, and the five
// quantiles are the centre plus the normal quantiles times that spread.
//
// Nothing here looks at the case's own pre-release history: tested and dropped —
// the pre-day buzz, the ramp into hour one and the viewers-per-channel all add
// noise once first-hour size is known (their correlations with the peak ratio are
// near zero after conditioning on size), and every variant that used them scored
// worse on leave-one-out over the comparables. Likewise a global polynomial fit
// of ratio on size is far worse than the local line: the size-to-shape relation
// bends too sharply for one curve to hold it.
//
// Validation (leave-one-out over the 37 usable comparables, running this exact
// program on one pseudo-case per launch, plus the four practice cases scored
// against SullyGnome's published hourly averages): practice loss 0.1943 (was
// 0.2184 for the weighted-mean version), leave-one-out 0.2075 on big launches,
// 0.2532 on big-plus-mid and 0.3809 overall (was 0.2117 / 0.2539 / 0.3982),
// against the persistence baseline's 0.3871 practice / 0.3214 / 0.2611. Runs
// offline, standard library only, deterministic (no randomness, no clock).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantile at each scored level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Gaussian kernel width on log first-hour viewers. */
const KERNEL_WIDTH = 1.2;
/** Share of the local-line centre in the blend with the weighted mean. */
const LINE_SHARE = 0.7;
/** Largest slope the local line may use, in log-ratio per log-viewer. */
const MAX_SLOPE = 1.0;
/** Narrowest log-spread any hour may claim. */
const MIN_SPREAD = 0.15;
/** Fallback spread, log units per square-root hour, when a pool is empty. */
const FALLBACK_SPREAD = 0.15;
/** A launch only counts if it covers at least this many hours of its day. */
const MIN_DAY_HOURS = 12;
/** Largest viewer count an output row may name. */
const VIEWER_CAP = 1e9;

interface DayPoint {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface PastLaunch {
  name: string;
  day_start: string;
  hourly: DayPoint[];
}

interface CaseFile {
  game: { name: string };
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: PastLaunch[];
}

/** Two-decimal rounding, half up. */
const twoDecimals = (x: number): number => Math.round(x * 100) / 100;

/** Hour number (1-based) of `from` inside the UTC day that starts at `dayStartMs`. */
const hourOf = (from: string, dayStartMs: number): number =>
  Math.floor((Date.parse(from) - dayStartMs) / 3_600_000) + 1;

/**
 * One past launch's release day as hour -> viewers, or null when it cannot be
 * used: the target game's own pre-release history is not a launch day, and a
 * day missing its first hour or covering too few hours says nothing reliable
 * about the day's shape.
 */
const dayCurve = (launch: PastLaunch, targetName: string): Map<number, number> | null => {
  if (launch.name === targetName) return null;
  const startMs = Date.parse(launch.day_start);
  if (!Number.isFinite(startMs)) return null;
  const curve = new Map<number, number>();
  for (const point of launch.hourly ?? []) {
    if (point.viewers === null || point.viewers === undefined || point.viewers <= 0) continue;
    const h = hourOf(point.from, startMs);
    if (h >= 1 && h <= 24 && !curve.has(h)) curve.set(h, point.viewers);
  }
  if (!curve.has(1) || curve.size < MIN_DAY_HOURS) return null;
  return curve;
};

/** Gaussian weight of a past launch whose first hour was `first`, for a case whose first hour is `p1`. */
const sizeWeight = (first: number, p1: number): number => {
  const d = Math.log(first) - Math.log(p1);
  return Math.exp(-(d * d) / (2 * KERNEL_WIDTH * KERNEL_WIDTH));
};

interface RatioSample {
  /** log first-hour viewers, centred on this case's first hour. */
  u: number;
  /** log ratio of this hour's viewers to the launch's first hour. */
  r: number;
  /** Kernel weight. */
  w: number;
}

/**
 * Weighted least-squares line through the samples, returned as [intercept, slope]:
 * the intercept is the fitted log-ratio at this case's own first-hour size, the
 * slope how the ratio moves with size nearby. The slope is capped and the
 * intercept refitted under the cap, so a steep tail can never drag the centre
 * far past the sizes actually observed.
 */
const fittedLine = (samples: RatioSample[]): [number, number] => {
  let sw = 0;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const s of samples) {
    sw += s.w;
    sx += s.w * s.u;
    sy += s.w * s.r;
    sxx += s.w * s.u * s.u;
    sxy += s.w * s.u * s.r;
  }
  const det = sw * sxx - sx * sx;
  if (Math.abs(det) < 1e-12) return [sy / sw, 0];
  let intercept = (sxx * sy - sx * sxy) / det;
  let slope = (sw * sxy - sx * sy) / det;
  if (slope > MAX_SLOPE || slope < -MAX_SLOPE) {
    slope = Math.max(-MAX_SLOPE, Math.min(MAX_SLOPE, slope));
    intercept = (sy - slope * sx) / sw;
  }
  return [intercept, slope];
};

/** The five log-ratio quantiles of one hour, from the past launches' ratios at that hour. */
const hourLogQuantiles = (
  curves: Array<Map<number, number>>,
  p1: number,
  hour: number,
): number[] => {
  const samples: RatioSample[] = [];
  for (const curve of curves) {
    const first = curve.get(1);
    const value = curve.get(hour);
    if (first === undefined || value === undefined) continue;
    samples.push({
      u: Math.log(first) - Math.log(p1),
      r: Math.log(value / first),
      w: sizeWeight(first, p1),
    });
  }
  if (samples.length === 0) {
    // No past launch covers this hour: fall back to a widening persistence band.
    return Z.map((z) => z * FALLBACK_SPREAD * Math.sqrt(hour - 1));
  }
  const [intercept, slope] = fittedLine(samples);
  let weightSum = 0;
  let weightedMean = 0;
  for (const s of samples) {
    weightSum += s.w;
    weightedMean += s.r * s.w;
  }
  weightedMean /= weightSum;
  const centre = LINE_SHARE * intercept + (1 - LINE_SHARE) * weightedMean;
  let weightedVar = 0;
  for (const s of samples) {
    const resid = s.r - (intercept + slope * s.u);
    weightedVar += s.w * resid * resid;
  }
  const spread = Math.max(MIN_SPREAD, Math.sqrt(weightedVar / weightSum));
  return Z.map((z) => centre + z * spread);
};

/** One case's forecast: 23 rows, hours 2..24, five quantiles each. */
const forecastOne = (c: CaseFile): Array<{ hour: number; q: number[] }> => {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const firstRow = c.known.find((k) => k.hour === 1);
  if (firstRow === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(1, firstRow.viewers);
  const curves: Array<Map<number, number>> = [];
  for (const launch of c.comparables ?? []) {
    const curve = dayCurve(launch, c.game.name);
    if (curve !== null) curves.push(curve);
  }
  return c.forecast.map(({ hour }) => {
    const logQs = hourLogQuantiles(curves, p1, hour);
    const q = logQs.map((lq) => {
      const v = twoDecimals(p1 * Math.exp(lq));
      return Math.min(VIEWER_CAP, Math.max(1, v));
    });
    // Non-decreasing by construction (z increasing, spread >= 0); enforce it outright.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { hour, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const hours = forecastOne(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
