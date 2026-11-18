export type VoiceTurnMode = 'tap' | 'silence' | 'handsfree';
export declare const DEFAULT_SILENCE_AFTER_SPEECH_MS: number;
export declare const DEFAULT_SPEAKER_DRAIN_MS: number;
export declare const SILENCE_AFTER_SPEECH_CHOICES_MS: readonly number[];
export declare const SPEAKER_DRAIN_CHOICES_MS: readonly number[];
export declare const SPEECH_ONSET_LOOKBACK_MS: number;
export declare function secondsLabel(ms: number): string;
export declare const VOICE_TURN_LABELS: Record<VoiceTurnMode, { label: string; description: string }>;
export declare const VOICE_DELAY_LABELS: {
  silenceAfterSpeech: { label: string; description: string };
  speakerDrain: { label: string; description: string };
};

export declare class SpeechEndpointTimer {
  constructor(onSilence: () => void, log?: (line: string) => void);
  begin(atMs: number, opts: { handsFree: boolean; silenceAfterSpeechMs: number }): void;
  observeLevel(rms: number): { speech: boolean };
  quietTailSeconds(): number;
  cancel(): void;
}

export type VoiceSessionState = 'stop' | 'listen' | 'speak';
export type VoiceSessionPhase =
  | 'idle'
  | 'listening'
  | 'recording'
  | 'transcribing'
  | 'replying'
  | 'replaying';
export interface VoiceSession {
  state: VoiceSessionState;
  phase: VoiceSessionPhase;
  replayReturnsTo?: { state: VoiceSessionState; phase: VoiceSessionPhase };
}
export type VoiceSessionEvent =
  | 'userStart'
  | 'userStop'
  | 'cancel'
  | 'speechHeard'
  | 'voiceDetected'
  | 'speechStarted'
  | 'turnCaptured'
  | 'nothingHeard'
  | 'replyStarted'
  | 'replyFinished'
  | 'speechFinished'
  | 'dictationFinished'
  | 'replayStarted'
  | 'replayEnded';
export declare function initialVoiceSession(handsFree: boolean): VoiceSession;
export declare function nextVoiceSession(session: VoiceSession, event: VoiceSessionEvent, handsFree: boolean): VoiceSession;
export declare function micShouldBeOpen(session: VoiceSession): boolean;
export declare function speechMayPlay(session: VoiceSession): boolean;

export declare const WAV_HEADER_BYTES: number;
export declare const WAV_HEADER_SCAN_BYTES: number;
export interface WavTrimPlan {
  header: Uint8Array;
  copyFrom: number;
  copyBytes: number;
  droppedSeconds: number;
}
export declare function planWavTrim(
  head: Uint8Array,
  dropFrontSeconds: number,
  fileBytes: number,
  dropTailSeconds?: number,
): WavTrimPlan | null;

export declare function transcriptionLanguages(
  engine: string,
  modelId?: string | null,
): Array<{ code: string; label: string }>;
