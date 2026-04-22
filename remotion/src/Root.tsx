import React from "react";
import { Composition } from "remotion";
import { z } from "zod";
import type { CompositionFamily } from "./types";
import { DynamicScene } from "./DynamicScene";
import {
  CoverHookSchema,
  BrainRegionFocusSchema,
  MetricCardSchema,
  MetricComparisonSchema,
  TimelineProgressionSchema,
  BulletStackSchema,
  DataStageSchema,
  AnalogySplitSchema,
  ClosingCtaSchema,
  NarrationSlideSchema,
} from "./types";
import {
  CoverHook,
  BrainRegionFocus,
  MetricCard,
  MetricComparison,
  TimelineProgression,
  BulletStack,
  DataStage,
  AnalogySplit,
  ClosingCta,
  NarrationSlide,
} from "./compositions";

const WIDTH = 1664;
const HEIGHT = 928;
const FPS = 30;
const DEFAULT_DURATION = 150; // 5s at 30fps

export const Root: React.FC = () => (
  <>
    <Composition
      id="cover-hook"
      component={CoverHook}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={CoverHookSchema}
      defaultProps={{
        headline: "Your Brain Health Report",
        subtitle: "A personalized deep-dive into your qEEG data",
        kicker: "QEEG ANALYSIS",
      }}
    />
    <Composition
      id="brain-region-focus"
      component={BrainRegionFocus}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={BrainRegionFocusSchema}
      defaultProps={{
        headline: "Brain Region Activity",
        regions: [
          { name: "Frontal", value: "+15%", status: "improved" },
          { name: "Parietal", value: "stable", status: "stable" },
        ],
      }}
    />
    <Composition
      id="metric-card"
      component={MetricCard}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={MetricCardSchema}
      defaultProps={{
        headline: "Executive Function",
        metricName: "Trail Making Test B",
        beforeValue: "161s",
        afterValue: "80s",
        delta: "50% improvement",
      }}
    />
    <Composition
      id="metric-comparison"
      component={MetricComparison}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={MetricComparisonSchema}
      defaultProps={{
        headline: "Session Comparison",
        leftLabel: "Session 1",
        rightLabel: "Session 3",
        leftValue: "4.8",
        rightValue: "2.1",
      }}
    />
    <Composition
      id="timeline-progression"
      component={TimelineProgression}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={TimelineProgressionSchema}
      defaultProps={{
        headline: "Treatment Timeline",
        markers: [
          { label: "June", sublabel: "Baseline" },
          { label: "September", sublabel: "Mid-treatment" },
          { label: "November", sublabel: "Follow-up" },
        ],
      }}
    />
    <Composition
      id="bullet-stack"
      component={BulletStack}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={BulletStackSchema}
      defaultProps={{
        headline: "Key Findings",
        items: [
          "Cognitive software received a significant upgrade",
          "Brain hardware adapted through neurocompensation",
          "Session 3 anomaly explained by drowsiness",
          "Overall trajectory strongly positive",
        ],
      }}
    />
    <Composition
      id="data-stage"
      component={DataStage}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={DataStageSchema}
      defaultProps={{
        headline: "Alpha Power by Region",
        items: [
          { label: "Frontal", value: 12.5, displayValue: "12.5 µV²" },
          { label: "Central", value: 9.8, displayValue: "9.8 µV²" },
          { label: "Parietal", value: 14.2, displayValue: "14.2 µV²" },
        ],
        unit: "µV²",
      }}
    />
    <Composition
      id="analogy-split"
      component={AnalogySplit}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={AnalogySplitSchema}
      defaultProps={{
        headline: "Software vs Hardware",
        left: {
          title: "Software (Function)",
          items: ["Task switching speed", "Working memory", "Executive function"],
          accent: "teal",
        },
        right: {
          title: "Hardware (Biology)",
          items: ["Alpha frequency", "Processing speed", "Raw neural power"],
          accent: "blue",
        },
      }}
    />
    <Composition
      id="closing-cta"
      component={ClosingCta}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={ClosingCtaSchema}
      defaultProps={{
        headline: "What This Means For You",
        bullets: [
          "Continue current treatment protocol",
          "Schedule follow-up in 8 weeks",
          "Track sleep quality between sessions",
        ],
        signoff: "We're excited about this progress and look forward to seeing even more gains ahead.",
      }}
    />
    <Composition
      id="narration-slide"
      component={NarrationSlide}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={NarrationSlideSchema}
      defaultProps={{
        headline: "Understanding Your Results",
        body: "Let's walk through what your brain data is telling us...",
      }}
    />
    <Composition
      id="dynamic-scene"
      component={DynamicScene}
      width={WIDTH}
      height={HEIGHT}
      fps={FPS}
      durationInFrames={DEFAULT_DURATION}
      schema={z.object({ code: z.string(), durationInFrames: z.number().optional() })}
      defaultProps={{
        code: 'const frame = useCurrentFrame();\nreturn <AbsoluteFill style={{background:"#0a0e1a",justifyContent:"center",alignItems:"center"}}><div style={{color:"#fff",fontSize:48}}>Dynamic Scene</div></AbsoluteFill>;',
        durationInFrames: DEFAULT_DURATION,
      }}
      calculateMetadata={({ props }) => ({
        durationInFrames: props.durationInFrames || DEFAULT_DURATION,
      })}
    />
  </>
);
