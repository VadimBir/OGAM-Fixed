'use strict';
/*
 * @offgrid/speech - LOCAL RECONSTRUCTION. Contracts derived from OGAM-Fixed call sites:
 *   src/services/voiceSession.ts, src/services/recordingController.ts,
 *   src/components/ChatInput/{Voice.ts,useSilenceEndpoint.ts,useVoiceSessionDriver.ts},
 *   src/services/wavTrimmer.ts, src/components/settings/voiceSections.tsx,
 *   src/components/TranscriptionLanguageSelect.tsx.
 * Label/description copy and numeric defaults are RECONSTRUCTED (no source evidence beyond
 * __tests__/utils/testHelpers.ts:76-77 values 5000 / 2000).
 */
Object.defineProperty(exports, '__esModule', { value: true });

// ---- Turn timing ----------------------------------------------------------------------------
var DEFAULT_SILENCE_AFTER_SPEECH_MS = 5000;
var DEFAULT_SPEAKER_DRAIN_MS = 2000;
var SILENCE_AFTER_SPEECH_CHOICES_MS = [1000, 1500, 2000, 3000, 5000, 8000];
var SPEAKER_DRAIN_CHOICES_MS = [0, 500, 1000, 2000, 3000];
/** Loudness reports speech ~300ms after it began (src/services/wavTrimmer.ts comment). */
var SPEECH_ONSET_LOOKBACK_MS = 300;

function secondsLabel(ms) {
  var s = (Number(ms) || 0) / 1000;
  var text = Number.isInteger(s) ? String(s) : s.toFixed(1);
  return text + ' s';
}

var VOICE_TURN_LABELS = {
  tap: { label: 'Tap', description: 'Tap the mic to start and tap again to send.' },
  silence: { label: 'Auto-stop', description: 'Tap the mic to start; the turn ends when you stop speaking.' },
  handsfree: { label: 'Hands-free', description: 'The mic reopens after each reply, so you can keep talking.' },
};
var VOICE_DELAY_LABELS = {
  silenceAfterSpeech: { label: 'Silence before sending', description: 'How long a pause ends your turn.' },
  speakerDrain: { label: 'Speaker settle time', description: 'Wait after the reply finishes before listening again.' },
};

// ---- Speech endpoint (level-based VAD) ------------------------------------------------------
// rms is the float32 RMS of a ~100ms buffer (src/services/audioRecorderService.ts:96).
var SPEECH_RMS_THRESHOLD = 0.02;
/** Without speech, a tapped (non-hands-free) turn still ends after this long. */
var NO_SPEECH_TIMEOUT_MS = 15000;

function SpeechEndpointTimer(onSilence, log) {
  this._onSilence = typeof onSilence === 'function' ? onSilence : function () {};
  this._log = typeof log === 'function' ? log : function () {};
  this._timer = null;
  this._active = false;
  this._heard = false;
  this._lastSpeechAt = 0;
  this._lastLevelAt = 0;
  this._opts = { handsFree: false, silenceAfterSpeechMs: DEFAULT_SILENCE_AFTER_SPEECH_MS };
}
SpeechEndpointTimer.prototype._clear = function () {
  if (this._timer !== null) {
    clearTimeout(this._timer);
    this._timer = null;
  }
};
SpeechEndpointTimer.prototype._arm = function (ms) {
  var self = this;
  this._clear();
  this._timer = setTimeout(function () {
    self._timer = null;
    if (!self._active) return;
    self._active = false;
    self._onSilence();
  }, ms);
};
SpeechEndpointTimer.prototype.begin = function (atMs, opts) {
  this._active = true;
  this._heard = false;
  this._lastSpeechAt = 0;
  this._lastLevelAt = typeof atMs === 'number' ? atMs : Date.now();
  this._opts = {
    handsFree: !!(opts && opts.handsFree),
    silenceAfterSpeechMs:
      opts && opts.silenceAfterSpeechMs > 0 ? opts.silenceAfterSpeechMs : DEFAULT_SILENCE_AFTER_SPEECH_MS,
  };
  // Hands-free waits for a person indefinitely; a tapped turn gives up after NO_SPEECH_TIMEOUT_MS.
  if (!this._opts.handsFree) this._arm(NO_SPEECH_TIMEOUT_MS);
  this._log('[VAD] endpoint armed (handsFree=' + this._opts.handsFree + ', silence=' + this._opts.silenceAfterSpeechMs + 'ms)');
};
SpeechEndpointTimer.prototype.observeLevel = function (rms) {
  var now = Date.now();
  this._lastLevelAt = now;
  var speech = typeof rms === 'number' && rms >= SPEECH_RMS_THRESHOLD;
  if (!this._active) return { speech: speech };
  if (speech) {
    this._heard = true;
    this._lastSpeechAt = now;
    this._arm(this._opts.silenceAfterSpeechMs);
  }
  return { speech: speech };
};
/** Seconds of quiet since the last speech, i.e. dead air at the end of the turn. */
SpeechEndpointTimer.prototype.quietTailSeconds = function () {
  if (!this._heard || !this._lastSpeechAt) return 0;
  var tail = (Math.max(this._lastLevelAt, Date.now()) - this._lastSpeechAt) / 1000;
  return tail > 0 ? tail : 0;
};
SpeechEndpointTimer.prototype.cancel = function () {
  this._active = false;
  this._clear();
};

// ---- Voice session state machine ------------------------------------------------------------
// state: 'stop' | 'listen' | 'speak'. phase (what a person is shown):
//   stop: 'idle'; listen: 'listening' | 'recording'; speak: 'transcribing' | 'replying' | 'replaying'.
function _stopped() {
  return { state: 'stop', phase: 'idle' };
}
function _listening(phase) {
  return { state: 'listen', phase: phase || 'listening' };
}
function initialVoiceSession(handsFree) {
  return handsFree ? _listening('listening') : _stopped();
}
function _afterTurn(handsFree) {
  return handsFree ? _listening('listening') : _stopped();
}
function nextVoiceSession(session, event, handsFree) {
  var s = session || _stopped();
  switch (event) {
    case 'userStart':
      return _listening('recording');
    case 'voiceDetected':
    case 'speechStarted':
      return s.state === 'listen' ? _listening('recording') : s;
    case 'turnCaptured':
      return s.state === 'listen' ? { state: 'speak', phase: 'transcribing' } : s;
    case 'replyStarted':
      return s.state === 'speak' ? { state: 'speak', phase: 'replying' } : s;
    case 'nothingHeard':
      // Dispatched twice by Voice.ts so hands-free does NOT re-open the mic: speak -> listen -> stop.
      if (s.state === 'speak') return _afterTurn(handsFree);
      if (s.state === 'listen') return _stopped();
      return s;
    case 'replyFinished':
    case 'speechFinished':
    case 'dictationFinished':
      return _afterTurn(handsFree);
    case 'userStop':
    case 'cancel':
      return _stopped();
    case 'replayStarted':
      return {
        state: 'speak',
        phase: 'replaying',
        replayReturnsTo: { state: s.state, phase: s.phase },
      };
    case 'replayEnded':
      if (s.replayReturnsTo) return { state: s.replayReturnsTo.state, phase: s.replayReturnsTo.phase };
      return s;
    default:
      return s;
  }
}
function micShouldBeOpen(session) {
  return !!session && session.state === 'listen';
}
function speechMayPlay(session) {
  return !!session && session.state === 'speak';
}

// ---- WAV trimming plan ----------------------------------------------------------------------
var WAV_HEADER_BYTES = 44;
var WAV_HEADER_SCAN_BYTES = 4096;

function _u32(b, o) {
  return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
}
function _u16(b, o) {
  return b[o] | (b[o + 1] << 8);
}
function _tag(b, o) {
  return String.fromCharCode(b[o], b[o + 1], b[o + 2], b[o + 3]);
}
function _w32(b, o, v) {
  b[o] = v & 255;
  b[o + 1] = (v >>> 8) & 255;
  b[o + 2] = (v >>> 16) & 255;
  b[o + 3] = (v >>> 24) & 255;
}
function _w16(b, o, v) {
  b[o] = v & 255;
  b[o + 1] = (v >>> 8) & 255;
}
function _writeTag(b, o, t) {
  for (var i = 0; i < 4; i++) b[o + i] = t.charCodeAt(i);
}
/**
 * planWavTrim(head, dropFrontSeconds, fileBytes, dropTailSeconds?)
 *   -> { header: Uint8Array(44), copyFrom, copyBytes, droppedSeconds } | null
 * null = unreadable header, nothing to drop, or a trim that would leave no audio.
 */
function planWavTrim(head, dropFrontSeconds, fileBytes, dropTailSeconds) {
  if (!head || head.length < 12) return null;
  if (_tag(head, 0) !== 'RIFF' || _tag(head, 8) !== 'WAVE') return null;
  var fmt = null;
  var dataOffset = -1;
  var dataSize = 0;
  var o = 12;
  while (o + 8 <= head.length) {
    var id = _tag(head, o);
    var size = _u32(head, o + 4);
    if (id === 'fmt ' && o + 8 + 16 <= head.length) {
      fmt = {
        audioFormat: _u16(head, o + 8),
        channels: _u16(head, o + 10),
        sampleRate: _u32(head, o + 12),
        byteRate: _u32(head, o + 16),
        blockAlign: _u16(head, o + 20),
        bitsPerSample: _u16(head, o + 22),
      };
    } else if (id === 'data') {
      dataOffset = o + 8;
      dataSize = size;
      break;
    }
    o += 8 + size + (size % 2);
  }
  if (!fmt || dataOffset < 0) return null;
  if (!fmt.channels || !fmt.sampleRate || !fmt.bitsPerSample) return null;
  var blockAlign = fmt.blockAlign || (fmt.channels * fmt.bitsPerSample) / 8;
  var byteRate = fmt.byteRate || fmt.sampleRate * blockAlign;
  if (!blockAlign || !byteRate) return null;
  var total = Number(fileBytes);
  if (!Number.isFinite(total) || total <= dataOffset) return null;
  // Recorders that never finalised write 0 / 0xFFFFFFFF; trust the file size then.
  var available = total - dataOffset;
  if (!dataSize || dataSize === 0xffffffff || dataSize > available) dataSize = available;
  dataSize -= dataSize % blockAlign;

  var front = Number.isFinite(dropFrontSeconds) ? Math.max(0, dropFrontSeconds) : 0;
  var tail = Number.isFinite(dropTailSeconds) ? Math.max(0, dropTailSeconds) : 0;
  var frontBytes = Math.floor((front * byteRate) / blockAlign) * blockAlign;
  var tailBytes = Math.floor((tail * byteRate) / blockAlign) * blockAlign;
  if (frontBytes + tailBytes <= 0) return null;
  var copyBytes = dataSize - frontBytes - tailBytes;
  if (copyBytes <= 0) return null;

  var header = new Uint8Array(WAV_HEADER_BYTES);
  _writeTag(header, 0, 'RIFF');
  _w32(header, 4, 36 + copyBytes);
  _writeTag(header, 8, 'WAVE');
  _writeTag(header, 12, 'fmt ');
  _w32(header, 16, 16);
  _w16(header, 20, fmt.audioFormat || 1);
  _w16(header, 22, fmt.channels);
  _w32(header, 24, fmt.sampleRate);
  _w32(header, 28, byteRate);
  _w16(header, 32, blockAlign);
  _w16(header, 34, fmt.bitsPerSample);
  _writeTag(header, 36, 'data');
  _w32(header, 40, copyBytes);
  return {
    header: header,
    copyFrom: dataOffset + frontBytes,
    copyBytes: copyBytes,
    droppedSeconds: (frontBytes + tailBytes) / byteRate,
  };
}

// ---- Transcription languages ----------------------------------------------------------------
var WHISPER_LANGUAGES = [
  ['auto', 'Auto-detect'], ['en', 'English'], ['es', 'Spanish'], ['fr', 'French'], ['de', 'German'],
  ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['ru', 'Russian'], ['uk', 'Ukrainian'],
  ['pl', 'Polish'], ['tr', 'Turkish'], ['ar', 'Arabic'], ['he', 'Hebrew'], ['hi', 'Hindi'],
  ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean'], ['vi', 'Vietnamese'], ['th', 'Thai'],
  ['id', 'Indonesian'], ['sv', 'Swedish'], ['da', 'Danish'], ['no', 'Norwegian'], ['fi', 'Finnish'],
  ['cs', 'Czech'], ['el', 'Greek'], ['ro', 'Romanian'], ['hu', 'Hungarian'],
];
/** English-only whisper builds carry `.en` in their id (e.g. ggml-base.en). */
function transcriptionLanguages(engine, modelId) {
  var id = typeof modelId === 'string' ? modelId.toLowerCase() : '';
  if (/(^|[.\-_])en($|[.\-_])/.test(id)) return [{ code: 'en', label: 'English' }];
  return WHISPER_LANGUAGES.map(function (pair) {
    return { code: pair[0], label: pair[1] };
  });
}

exports.DEFAULT_SILENCE_AFTER_SPEECH_MS = DEFAULT_SILENCE_AFTER_SPEECH_MS;
exports.DEFAULT_SPEAKER_DRAIN_MS = DEFAULT_SPEAKER_DRAIN_MS;
exports.SILENCE_AFTER_SPEECH_CHOICES_MS = SILENCE_AFTER_SPEECH_CHOICES_MS;
exports.SPEAKER_DRAIN_CHOICES_MS = SPEAKER_DRAIN_CHOICES_MS;
exports.SPEECH_ONSET_LOOKBACK_MS = SPEECH_ONSET_LOOKBACK_MS;
exports.secondsLabel = secondsLabel;
exports.VOICE_TURN_LABELS = VOICE_TURN_LABELS;
exports.VOICE_DELAY_LABELS = VOICE_DELAY_LABELS;
exports.SpeechEndpointTimer = SpeechEndpointTimer;
exports.initialVoiceSession = initialVoiceSession;
exports.nextVoiceSession = nextVoiceSession;
exports.micShouldBeOpen = micShouldBeOpen;
exports.speechMayPlay = speechMayPlay;
exports.WAV_HEADER_BYTES = WAV_HEADER_BYTES;
exports.WAV_HEADER_SCAN_BYTES = WAV_HEADER_SCAN_BYTES;
exports.planWavTrim = planWavTrim;
exports.transcriptionLanguages = transcriptionLanguages;
