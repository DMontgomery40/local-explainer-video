import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig, Easing } from "remotion";
import type { CoverHookProps } from "../types";
import { FONT_HEADLINE, FONT_CAPTION, FONT_DATA, COLOR_ACCENT } from "../types";
import { SceneLayout, useReveal } from "./shared";

export const CoverHook: React.FC<CoverHookProps> = ({
  headline,
  subtitle,
  kicker,
  backgroundImage,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const reveal = useReveal();

  const kickerReveal = spring({
    frame: Math.max(frame - 4, 0),
    fps,
    config: { damping: 200 },
  });
  const headlineReveal = spring({
    frame: Math.max(frame - 8, 0),
    fps,
    config: { damping: 18, stiffness: 80, mass: 1.2 },
  });
  const subtitleReveal = spring({
    frame: Math.max(frame - 18, 0),
    fps,
    config: { damping: 200 },
  });

  const lineWidth = interpolate(headlineReveal, [0, 1], [0, 120], {
    extrapolateRight: "clamp",
  });

  return (
    <SceneLayout backgroundImage={backgroundImage} brightness={0.25}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          height: "100%",
          gap: 20,
        }}
      >
        {kicker && (
          <div
            style={{
              fontFamily: FONT_DATA,
              fontSize: 18,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: COLOR_ACCENT,
              opacity: kickerReveal,
              transform: `translateY(${interpolate(kickerReveal, [0, 1], [8, 0])}px)`,
            }}
          >
            {kicker}
          </div>
        )}
        <div
          style={{
            fontFamily: FONT_HEADLINE,
            fontSize: 84,
            lineHeight: 0.92,
            fontWeight: "bold",
            opacity: headlineReveal,
            transform: `translateY(${interpolate(headlineReveal, [0, 1], [40, 0])}px)`,
            textShadow: "0 4px 30px rgba(0,0,0,0.6), 0 1px 3px rgba(0,0,0,0.4)",
          }}
        >
          {headline}
        </div>
        {/* accent line */}
        <div
          style={{
            width: lineWidth,
            height: 3,
            background: `linear-gradient(90deg, ${COLOR_ACCENT}, transparent)`,
            borderRadius: 2,
            opacity: headlineReveal,
          }}
        />
        {subtitle && (
          <div
            style={{
              fontFamily: FONT_CAPTION,
              fontSize: 30,
              color: "rgba(255,244,234,0.65)",
              opacity: subtitleReveal,
              lineHeight: 1.45,
              transform: `translateY(${interpolate(subtitleReveal, [0, 1], [12, 0])}px)`,
              textShadow: "0 2px 12px rgba(0,0,0,0.4)",
              maxWidth: "65%",
            }}
          >
            {subtitle}
          </div>
        )}
      </div>
    </SceneLayout>
  );
};
