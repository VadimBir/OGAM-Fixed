'use strict';
/*
 * @offgrid/ui - LOCAL RECONSTRUCTION. Finite progress contract derived from
 * OGAM-Fixed/src/utils/progressPresentation.ts, src/stores/downloadStore.ts and
 * __tests__/unit/utils/progressPresentation.test.ts.
 */
Object.defineProperty(exports, '__esModule', { value: true });

var TERMINAL_STATUSES = ['completed', 'failed', 'cancelled'];
var RATE_MIN_INTERVAL_MS = 500;

function _finiteNonNegative(v) {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 ? v : undefined;
}

/**
 * input: { progress?: 0..1 fraction, bytesDownloaded?, totalBytes?, bytesPerSecond?, status? }
 * output: { percentage?: 0..100, currentBytes, totalBytes?, bytesPerSecond?, status?, terminal }
 */
function projectProgress(input) {
  var src = input || {};
  var status = typeof src.status === 'string' ? src.status : undefined;
  var terminal = status !== undefined && TERMINAL_STATUSES.indexOf(status) !== -1;
  var currentBytes = _finiteNonNegative(src.bytesDownloaded) || 0;
  var totalBytes = _finiteNonNegative(src.totalBytes);
  if (totalBytes === 0) totalBytes = undefined;
  var bytesPerSecond = _finiteNonNegative(src.bytesPerSecond);
  var percentage;
  if (status === 'completed') percentage = 100;
  else if (totalBytes !== undefined) percentage = Math.min(100, (currentBytes / totalBytes) * 100);
  else {
    var fraction = _finiteNonNegative(src.progress);
    if (fraction !== undefined && fraction <= 1) percentage = fraction * 100;
  }
  var out = { currentBytes: currentBytes, terminal: terminal };
  if (percentage !== undefined) out.percentage = percentage;
  if (totalBytes !== undefined) out.totalBytes = totalBytes;
  if (bytesPerSecond !== undefined && !terminal) out.bytesPerSecond = bytesPerSecond;
  if (status !== undefined) out.status = status;
  return out;
}

/**
 * sampleProgressRate(previousSample | undefined, { currentBytes, sampledAtMs })
 *   -> { sample, bytesPerSecond? }
 * Keeps the previous sample until RATE_MIN_INTERVAL_MS has elapsed so bursty progress events do
 * not produce a noisy rate; the last measured rate is carried meanwhile.
 */
function sampleProgressRate(previous, reading) {
  var bytes = _finiteNonNegative(reading && reading.currentBytes);
  var at = _finiteNonNegative(reading && reading.sampledAtMs);
  if (bytes === undefined || at === undefined) {
    return { sample: previous, bytesPerSecond: previous ? previous.bytesPerSecond : undefined };
  }
  if (!previous || bytes < previous.bytes || at < previous.atMs) {
    return { sample: { bytes: bytes, atMs: at }, bytesPerSecond: undefined };
  }
  var dt = at - previous.atMs;
  if (dt < RATE_MIN_INTERVAL_MS) {
    return { sample: previous, bytesPerSecond: previous.bytesPerSecond };
  }
  var rate = ((bytes - previous.bytes) * 1000) / dt;
  if (!Number.isFinite(rate) || rate < 0) rate = undefined;
  var sample = { bytes: bytes, atMs: at };
  if (rate !== undefined) sample.bytesPerSecond = rate;
  return { sample: sample, bytesPerSecond: rate };
}

exports.projectProgress = projectProgress;
exports.sampleProgressRate = sampleProgressRate;
