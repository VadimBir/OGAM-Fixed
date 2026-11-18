'use strict';
/*
 * @offgrid/rag - LOCAL RECONSTRUCTION. chunkText contract from
 * OGAM-Fixed/__tests__/unit/services/rag/chunking.test.ts (consumer: src/services/rag/chunking.ts).
 */
Object.defineProperty(exports, '__esModule', { value: true });

var DEFAULT_CHUNK_SIZE = 500;
var DEFAULT_OVERLAP = 50;
var DEFAULT_MIN_CHUNK_LENGTH = 20;

function slidingWindow(text, size, overlap) {
  var out = [];
  var step = Math.max(1, size - Math.max(0, Math.min(overlap, size - 1)));
  for (var start = 0; start < text.length; start += step) {
    var piece = text.slice(start, start + size).trim();
    if (piece) out.push(piece);
    if (start + size >= text.length) break;
  }
  return out;
}

function chunkText(text, options) {
  var opts = options || {};
  var chunkSize = opts.chunkSize > 0 ? opts.chunkSize : DEFAULT_CHUNK_SIZE;
  var overlap = opts.overlap >= 0 ? opts.overlap : DEFAULT_OVERLAP;
  var minLen = opts.minChunkLength >= 0 ? opts.minChunkLength : DEFAULT_MIN_CHUNK_LENGTH;
  var normalized = String(text).replace(/\r\n?/g, '\n').trim();
  if (!normalized) return [];
  // A whole document shorter than the minimum is all the user has: keep it as one chunk.
  if (normalized.length < minLen) return [{ content: normalized, position: 0 }];

  var paragraphs = normalized.split(/\n\s*\n+/).map(function (p) { return p.trim(); }).filter(Boolean);
  var pieces = [];
  var current = '';
  function flush() {
    if (current) pieces.push(current);
    current = '';
  }
  for (var i = 0; i < paragraphs.length; i++) {
    var para = paragraphs[i];
    if (para.length > chunkSize) {
      flush();
      var windows = slidingWindow(para, chunkSize, overlap);
      for (var w = 0; w < windows.length; w++) pieces.push(windows[w]);
      continue;
    }
    if (!current) current = para;
    else if (current.length + 2 + para.length <= chunkSize) current = current + '\n\n' + para;
    else {
      flush();
      current = para;
    }
  }
  flush();

  var chunks = [];
  for (var j = 0; j < pieces.length; j++) {
    if (pieces[j].length < minLen) continue;
    chunks.push({ content: pieces[j], position: chunks.length });
  }
  return chunks;
}

exports.DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE;
exports.DEFAULT_OVERLAP = DEFAULT_OVERLAP;
exports.DEFAULT_MIN_CHUNK_LENGTH = DEFAULT_MIN_CHUNK_LENGTH;
exports.chunkText = chunkText;
