export interface ProgressLike {
  /** Fraction 0..1. */
  progress?: number;
  bytesDownloaded?: number;
  totalBytes?: number;
  bytesPerSecond?: number;
  status?: string;
}
export interface ProgressPresentation {
  /** 0..100, absent when unknown. */
  percentage?: number;
  currentBytes: number;
  totalBytes?: number;
  bytesPerSecond?: number;
  status?: string;
  terminal: boolean;
}
export interface ProgressRateSample {
  currentBytes: number;
  sampledAtMs: number;
}
export declare function projectProgress(input: ProgressLike): ProgressPresentation;
export declare function sampleProgressRate(
  previous: ProgressRateSample | undefined,
  reading: { currentBytes: number; sampledAtMs: number },
): { sample: ProgressRateSample | undefined; bytesPerSecond?: number };
