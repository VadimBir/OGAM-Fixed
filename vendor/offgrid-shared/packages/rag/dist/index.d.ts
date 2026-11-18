export interface Chunk {
  content: string;
  position: number;
}
export interface ChunkOptions {
  chunkSize?: number;
  overlap?: number;
  minChunkLength?: number;
}
export declare const DEFAULT_CHUNK_SIZE: number;
export declare const DEFAULT_OVERLAP: number;
export declare const DEFAULT_MIN_CHUNK_LENGTH: number;
export declare function chunkText(text: string, options?: ChunkOptions): Chunk[];
