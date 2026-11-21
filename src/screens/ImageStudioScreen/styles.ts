import type { ThemeColors, ThemeShadows } from '../../theme';
import { TYPOGRAPHY, SPACING } from '../../constants';

export const THUMB_SIZE = 64;

export const createStyles = (colors: ThemeColors, _shadows: ThemeShadows) => ({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scroll: {
    padding: SPACING.lg,
    paddingBottom: SPACING.xxl,
    gap: SPACING.md,
  },
  card: {
    backgroundColor: colors.surface,
    borderRadius: SPACING.md,
    padding: SPACING.md,
    gap: SPACING.sm,
  },
  sectionLabel: {
    ...TYPOGRAPHY.h3,
    color: colors.textSecondary,
  },
  modelRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: SPACING.sm,
  },
  modelName: {
    ...TYPOGRAPHY.body,
    color: colors.text,
    flex: 1,
  },
  modelNameMissing: {
    color: colors.textMuted,
  },
  input: {
    ...TYPOGRAPHY.body,
    color: colors.text,
    backgroundColor: colors.background,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: SPACING.sm,
    paddingHorizontal: SPACING.md,
    paddingVertical: SPACING.sm,
    minHeight: 44,
    textAlignVertical: 'top' as const,
  },
  inputTall: {
    minHeight: 88,
  },
  segmentRow: {
    flexDirection: 'row' as const,
    gap: SPACING.sm,
  },
  segment: {
    flex: 1,
    alignItems: 'center' as const,
    paddingVertical: SPACING.sm,
    borderRadius: SPACING.sm,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
  },
  segmentActive: {
    borderColor: colors.primary,
  },
  segmentText: {
    ...TYPOGRAPHY.meta,
    color: colors.textSecondary,
  },
  segmentTextActive: {
    color: colors.primary,
  },
  dimmed: {
    opacity: 0.45,
  },
  note: {
    ...TYPOGRAPHY.meta,
    color: colors.textMuted,
  },
  errorText: {
    ...TYPOGRAPHY.meta,
    color: colors.error,
  },
  referenceRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: SPACING.md,
  },
  thumb: {
    width: THUMB_SIZE,
    height: THUMB_SIZE,
    borderRadius: SPACING.sm,
    backgroundColor: colors.surfaceLight,
  },
  thumbSelected: {
    borderWidth: 2,
    borderColor: colors.primary,
  },
  strip: {
    gap: SPACING.sm,
  },
  checkboxRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: SPACING.sm,
    paddingVertical: SPACING.xs,
  },
  checkboxLabel: {
    ...TYPOGRAPHY.body,
    color: colors.text,
    flex: 1,
  },
  progressRow: {
    flexDirection: 'row' as const,
    alignItems: 'center' as const,
    gap: SPACING.md,
  },
  progressText: {
    ...TYPOGRAPHY.body,
    color: colors.text,
    flex: 1,
  },
  progressTrack: {
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.surfaceLight,
    overflow: 'hidden' as const,
  },
  progressFill: {
    height: 4,
    backgroundColor: colors.primary,
  },
  resultImage: {
    width: '100%' as const,
    aspectRatio: 1,
    borderRadius: SPACING.md,
    backgroundColor: colors.surfaceLight,
  },
  resultMeta: {
    ...TYPOGRAPHY.meta,
    color: colors.textMuted,
  },
  resultActions: {
    flexDirection: 'row' as const,
    gap: SPACING.sm,
  },
});
