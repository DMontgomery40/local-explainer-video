import { z } from "zod";

// ─── Shared design tokens ───

export const FONT_HEADLINE =
  'Georgia, "Times New Roman", serif';
export const FONT_BODY =
  '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif';
export const FONT_DATA =
  "ui-monospace, SFMono-Regular, Menlo, monospace";
export const FONT_CAPTION =
  '"SF Pro Text", -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif';

export const BODY_SIZE = 30;
export const BODY_LINE_HEIGHT = 1.45;
export const CAPTION_SIZE = 20;
export const LABEL_SIZE = 18;

export const COLOR_FG = "#f7efe6";
export const COLOR_BG = "#0a0a0f";
export const COLOR_ACCENT = "#5eead4";

// ─── Zod schemas for each composition's props ───

export const CoverHookSchema = z.object({
  headline: z.string(),
  subtitle: z.string().optional(),
  kicker: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type CoverHookProps = z.infer<typeof CoverHookSchema>;

export const BrainRegionFocusSchema = z.object({
  headline: z.string(),
  caption: z.string().optional(),
  regions: z.array(
    z.object({
      name: z.string(),
      value: z.string().optional(),
      status: z.enum(["improved", "stable", "declined", "flagged"]).optional(),
    }),
  ),
  backgroundImage: z.string().optional(),
});
export type BrainRegionFocusProps = z.infer<typeof BrainRegionFocusSchema>;

export const MetricCardSchema = z.object({
  headline: z.string(),
  metricName: z.string().optional(),
  beforeValue: z.string(),
  afterValue: z.string(),
  delta: z.string().optional(),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type MetricCardProps = z.infer<typeof MetricCardSchema>;

export const MetricComparisonSchema = z.object({
  headline: z.string(),
  leftLabel: z.string(),
  rightLabel: z.string(),
  leftValue: z.string(),
  rightValue: z.string(),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type MetricComparisonProps = z.infer<typeof MetricComparisonSchema>;

export const TimelineProgressionSchema = z.object({
  headline: z.string(),
  markers: z.array(
    z.object({
      label: z.string(),
      sublabel: z.string().optional(),
    }),
  ),
  backgroundImage: z.string().optional(),
});
export type TimelineProgressionProps = z.infer<typeof TimelineProgressionSchema>;

export const BulletStackSchema = z.object({
  headline: z.string(),
  items: z.array(z.string()),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type BulletStackProps = z.infer<typeof BulletStackSchema>;

export const DataStageSchema = z.object({
  headline: z.string(),
  items: z.array(
    z.object({
      label: z.string(),
      value: z.number(),
      displayValue: z.string().optional(),
    }),
  ),
  unit: z.string().optional(),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type DataStageProps = z.infer<typeof DataStageSchema>;

export const AnalogySplitSchema = z.object({
  headline: z.string(),
  left: z.object({
    title: z.string(),
    items: z.array(z.string()),
    accent: z.string().optional(),
  }),
  right: z.object({
    title: z.string(),
    items: z.array(z.string()),
    accent: z.string().optional(),
  }),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type AnalogySplitProps = z.infer<typeof AnalogySplitSchema>;

export const ClosingCtaSchema = z.object({
  headline: z.string(),
  bullets: z.array(z.string()).optional(),
  signoff: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type ClosingCtaProps = z.infer<typeof ClosingCtaSchema>;

export const NarrationSlideSchema = z.object({
  headline: z.string(),
  body: z.string().optional(),
  caption: z.string().optional(),
  backgroundImage: z.string().optional(),
});
export type NarrationSlideProps = z.infer<typeof NarrationSlideSchema>;

// Union of all family names
export const COMPOSITION_FAMILIES = [
  "cover_hook",
  "brain_region_focus",
  "metric_card",
  "metric_comparison",
  "timeline_progression",
  "bullet_stack",
  "data_stage",
  "analogy_split",
  "closing_cta",
  "narration_slide",
] as const;
export type CompositionFamily = (typeof COMPOSITION_FAMILIES)[number];

// Per-scene input props passed via Remotion's inputProps
export const SceneInputSchema = z.object({
  family: z.enum(COMPOSITION_FAMILIES),
  props: z.record(z.string(), z.unknown()),
  durationInFrames: z.number().int().positive().optional(),
});
export type SceneInput = z.infer<typeof SceneInputSchema>;
