import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { TimelineProgressionProps } from "../types";
import { FONT_DATA, FONT_BODY, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, useReveal } from "./shared";

export const TimelineProgression: React.FC<TimelineProgressionProps> = ({
  headline,
  markers,
  backgroundImage,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const reveal = useReveal();

  const trackTop = 520;
  const trackLeft = 160;
  const trackWidth = 1344;

  return (
    <SceneLayout backgroundImage={backgroundImage} brightness={0.3}>
      <Headline text={headline} />

      {/* track line */}
      <div
        style={{
          position: "absolute",
          top: trackTop,
          left: trackLeft,
          width: trackWidth,
          height: 4,
          background: "rgba(255,255,255,0.15)",
          borderRadius: 2,
          opacity: reveal,
        }}
      />

      {markers.map((m, i) => {
        const x = trackLeft + (trackWidth / Math.max(markers.length - 1, 1)) * i;
        const dotReveal = spring({
          frame: Math.max(frame - 8 - i * 6, 0),
          fps,
          config: { damping: 16, stiffness: 110, mass: 0.9 },
        });
        return (
          <div
            key={i}
            style={{
              position: "absolute",
              top: trackTop - 60,
              left: x - 60,
              width: 120,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 8,
              opacity: dotReveal,
              transform: `translateY(${interpolate(dotReveal, [0, 1], [12, 0])}px)`,
            }}
          >
            <div style={{ fontFamily: FONT_BODY, fontSize: 22, fontWeight: 600, textAlign: "center" }}>
              {m.label}
            </div>
            {m.sublabel && (
              <div style={{ fontFamily: FONT_DATA, fontSize: 16, color: COLOR_ACCENT, textAlign: "center" }}>
                {m.sublabel}
              </div>
            )}
            {/* dot */}
            <div
              style={{
                width: 16,
                height: 16,
                borderRadius: "50%",
                background: COLOR_ACCENT,
                transform: `scale(${interpolate(dotReveal, [0, 1], [0, 1])})`,
              }}
            />
          </div>
        );
      })}
    </SceneLayout>
  );
};
