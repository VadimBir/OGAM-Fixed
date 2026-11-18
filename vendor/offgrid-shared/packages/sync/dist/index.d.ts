// @offgrid/sync - LOCAL RECONSTRUCTION types. Chat-path symbols are typed from call sites;
// mesh/pairing/transfer symbols are `any` (STUB) - see dist/index.js header.
export interface ReasoningDelimiter {
  open: string;
  close: string;
}
export declare const REASONING_DELIMITERS: readonly ReasoningDelimiter[];
export declare const TOOL_CALL_OPENERS: readonly string[];
export declare const TOOL_CALL_CLOSERS: readonly string[];
export declare const XML_TOOL_CALL_FUNCTION_MARKER: string;
export declare const XML_TOOL_CALL_PARAMETER_MARKER: string;
export declare function partialTagSuffix(text: string, tag: string): number;
export declare function maxPartialTagSuffix(text: string, tags: readonly string[]): number;
export interface ChatThinkingContent {
  thinking: string | null;
  response: string;
  isThinkingComplete: boolean;
  thinkingLabel?: string;
}
export declare function parseChatThinkingContent(content: string): ChatThinkingContent;
export interface ChatModelOutput {
  reasoning: string | null;
  answer: string;
  isReasoningComplete: boolean;
  reasoningLabel?: string;
}
export declare function parseChatModelOutput(content: string, reasoningContent?: string | null): ChatModelOutput;
export declare function stripChatControlTokens(content: string): string;
export declare function splitInlineReasoning(content: string): {
  answer: string;
  reasoning?: string;
  isReasoningComplete: boolean;
  reasoningLabel?: string;
};
export declare function preprocessChatMarkdown(text: string): string;
export declare function safeChatExternalUrl(url: string): string | null;
export declare function chatListPreviewLine(role: string | undefined, content: string | undefined): string;
export declare const PROMPT_ENHANCEMENT_REASONING_LABEL: string;
export declare const PROMPT_ENHANCEMENT_STATUS: string;
export declare function isRuntimeOnlyMessage(input: { role: string; content: string; notice?: boolean }): boolean;
export declare function isSupportingChatContext(input: {
  answer?: string | null;
  reasoning?: string | null;
  reasoningLabel?: string | null;
}): boolean;

export type ProDeviceAdmission = 'unknown' | 'active' | 'inactive';
export declare function hasProAccess(input: { hasCredential: boolean; admission: ProDeviceAdmission }): boolean;
export interface PersonalMeshActivationFailureProjection {
  reason: string;
  title: string;
  description: string;
}
export declare function projectPersonalMeshActivationFailure(reason: string): PersonalMeshActivationFailureProjection;

export type SyncedAssistantTimelineEntry =
  | { kind: 'thinking'; text: string }
  | { kind: 'tool'; toolIndex: number };
export interface SyncedToolArtifact {
  id?: string;
  name: string;
  result: string;
  status: 'running' | 'completed' | 'failed';
  [key: string]: any;
}
export type SyncedTurnStatus = 'streaming' | 'completed' | 'cancelled' | 'failed' | 'interrupted';
export interface RecordProvenance {
  originDeviceId?: string;
  [key: string]: any;
}
export interface ChatStreamTool {
  name: string;
  status: 'running' | 'completed' | 'failed';
  result?: string;
}
export interface ChatStreamPreview {
  [key: string]: any;
}
export interface ChatStreamPreviewRow {
  id: string;
  messageId: string;
  deviceId: string;
  conversationId?: string;
  phase: string;
  progress?: any;
  content: string;
  reasoning?: string;
  tools?: ChatStreamTool[];
  [key: string]: any;
}
export declare function chatStreamPhaseIsStatus(phase: string): boolean;
export declare function chatStreamPhaseLabel(phase: string, progress?: any): string | undefined;
export declare function chatStreamPreviewHasMessageBody(preview: { content?: string }): boolean;
export declare function chatStreamPreviewRows(previews: any, activeConversationId?: string | null): ChatStreamPreviewRow[];
export declare function startChatStreamTool(tools: readonly ChatStreamTool[], name: string): ChatStreamTool[];
export declare function completeChatStreamTool(tools: readonly ChatStreamTool[], name: string, result: string): ChatStreamTool[];

export declare const KNOWLEDGE_DOCUMENT_ENTITY: 'knowledge_document';
export declare const SHARED_FILE_ENTITY: 'shared_file';
export declare const TASK_LAUNCH_ENTITY: 'task_launch';
export declare const TASK_RUN_ENTITY: 'task_run';
export declare const TASK_CONTROL_ENTITY: 'task_control';

// Object types extended / implemented by app code (must be object types, not `any`).
export interface KnowledgeDocumentDescriptor { [key: string]: any }
export interface TransferredModelManifest { [key: string]: any }
export interface SyncConnection { [key: string]: any }
export interface TransportBridge { [key: string]: any }
export interface PairingEntitlementHostAdapter { [key: string]: any }
export interface PersonalMeshMembershipReplacementAdapter<T = any> { [key: string]: any }
export interface Materializer { [key: string]: any }
export interface PairingPersistence { [key: string]: any }
export interface MembershipRevocationPersistence { [key: string]: any }

export type AmbientDirectoryAccessCopy = any;
export type AmbientSharePolicy = any;
export type BlobChannelHost = any;
export type BlobEndpoint = any;
export type CompletedTransferRecord = any;
export type DeviceCap = any;
export type DeviceInfo = any;
export type DevicePlatform = any;
export type DiscoveredDevice = any;
export type DiscoveryScanSnapshot = any;
export type DiscoveryService = any;
export type DiscoverySource = any;
export type FileRequestMessage = any;
export type FileTransferProgress = any;
export type MembershipRevocationSnapshot = any;
export type MembershipRevocationTombstone = any;
export type Message = any;
export type ModelPackageTransferMetadata = any;
export type ModelTransferMetadata = any;
export type Op = any;
export type PairedDevice = any;
export type PairingAttemptSnapshot = any;
export type PairingEntitlementCredential = any;
export type PeerLink = any;
export type PeerLinkEvent = any;
export type PendingMembershipRevocation = any;
export type PersonalMeshActivationResult = any;
export type PersonalMeshInstallation = any;
export type PersonalMeshReconciliationReason = any;
export type PersonalMeshReconciliationSnapshot = any;
export type PersonalMeshRegistrationInput = any;
export type PersonalMeshRegistryAdapter = any;
export type ReceivePolicy = any;
export type SharedFileDescriptor = any;
export type StateMsg = any;
export type SyncControlAction = any;
export type SyncDiscoverabilityHealthInput = any;
export type SyncDiscoveryHealthSnapshot = any;
export type SyncEngineOptions = any;
export type SyncFileCompletionNotificationFact = any;
export type SyncHealthFact = any;
export type SyncManagedDevice = any;
export type SyncTransportHealthSnapshot = any;
export type SyncedMessageTurnProjection = any;
export type SyncedRetrievalSource = any;
export type SyncedTaskRun = any;
export type SyncedTaskVisualStep = any;
export type TransferFileSink = any;
export type TransferFileSource = any;
export type TransportRoute = any;
export declare const AMBIENT_SHARE_ANY_DESTINATION: any;
export declare const BLOB_FRAME_BYTES: any;
export declare const BLOB_TOKEN_TTL_MS: any;
export declare const CHUNK_SIZE: any;
export declare const CLIPBOARD_CHANNEL: any;
export declare const DEFAULT_COMPLETED_TRANSFER_HISTORY_LIMIT: any;
export declare const DEFAULT_RECEIVE_POLICY: any;
export declare const MAX_CLIPBOARD_TEXT_BYTES: any;
export declare const MAX_SHARED_FILE_BYTES: any;
export declare const MAX_SYNCED_TASK_VISUAL_STEPS: any;
export declare const MODEL_TRANSFER_MIME: any;
export declare const OFFGRID_SYNC_PORT: any;
export declare const PAIRING_CODE_ALPHABET: any;
export declare const PAIRING_CODE_LENGTH: any;
export declare const PAIRING_QR_VALIDITY_MS: any;
export declare const PERSONAL_MESH_DEVICE_CAP: any;
export declare const PERSONAL_MESH_ENTITLEMENT_REVALIDATION_INTERVAL_MS: any;
export declare const SHARED_FILE_MIME: any;
export declare const TASK_GUIDANCE_CHANNEL: any;
export declare const TASK_VISUAL_STEP_ENTITY: any;
export declare const blobKeyBase64: any;
export declare const checksumFromSha512Hex: any;
export declare const createBlobMaterial: any;
export declare const createKnowledgeDocumentStateFields: any;
export declare const createSharedFileStateFields: any;
export declare const createSharedFileTransferMetadata: any;
export declare const createTxtRecord: any;
export declare const encodePairingQrPayload: any;
export declare const isClipboardAckMessage: any;
export declare const isDialableAddress: any;
export declare const ogamModelTransferBlocker: any;
export declare const parsePairingCode: any;
export declare const parsePairingQrPayload: any;
export declare const projectSharedFilePreview: any;
export declare const projectSyncedMessageTurn: any;
export declare const serializeSyncedMessageContext: any;
export declare const sharedFileActivityId: any;
export declare const syncFileCompletionNotificationId: any;
export declare const taskVisualStepId: any;
export declare const countOps: any;
export declare const projectSyncReceiving: any;
export declare class SyncEngine { constructor(...args: any[]); [key: string]: any }
export declare class MultiTransportBridge { constructor(...args: any[]); [key: string]: any }
export declare class DiscoveryOrchestrator { constructor(...args: any[]); [key: string]: any }
export declare class CompositeDiscoveryService { constructor(...args: any[]); [key: string]: any }
export declare class FileTransferManager { constructor(...args: any[]); [key: string]: any }
export declare class OpLog { constructor(...args: any[]); [key: string]: any }
export declare class StateSync { constructor(...args: any[]); [key: string]: any }
export declare class IncrementalChecksum { constructor(...args: any[]); [key: string]: any }
export declare class PersonalMeshEntitlementError extends Error { constructor(message?: string, reason?: string); reason?: string; [key: string]: any }
export interface TransferredModelFile { name: string; sizeBytes: number; role?: string; [key: string]: any }
export interface TransferredModelManifest { id: string; name: string; kind: string; source?: string; origin?: any; files: TransferredModelFile[]; [key: string]: any }
export interface SyncedMessageTurnInput { role?: string; content: string | any[]; context?: unknown; [key: string]: any }
