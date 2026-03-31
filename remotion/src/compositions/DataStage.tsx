import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { DataStageProps } from "../types";
import { FONT_DATA, FONT_BODY, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption } from "./shared";

const Bar: React.FC<{
  label: string;
  value: number;
  displayValue?: string;
  maxValue: number;
  index: number;
  total: number;
}> = ({ label, value, displayValue, maxValue, index, total }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const barReveal = spring({
    frame: Math.max(frame - 12 - index * 6, 0),
    fps,
    config: { damping: 18, stiffness: 80, mass: 0.9 },
  });
  const labelReveal = spring({
    frame: Math.max(frame - 8 - index * 6, 0),
    fps,
    config: { damping: 200 },
  });

  const barPercent = maxValue > 0 ? (value / maxValue) * 100 : 0;
  const barWidth = interpolate(barReveal, [0, 1], [0, barPercent]);
  const isTop = index === 0;

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 16, height: 44,
      opacity: labelReveal,
      transform: `translateX(${interpolate(labelReveal, [0, 1], [-12, 0])}px)`,
    }}>
      <div style={{
        fontFamily: FONT_BODY, fontSize: 20, minWidth: 200, textAlign: "right",
        color: "rgba(255,244,234,0.85)",
        textShadow: "0 1px 4px rgba(0,0,0,0.5)",
      }}>
        {label}
      </div>
      <div style={{
        flex: 1, height: 36, background: "rgba(255,255,255,0.05)",
        borderRadius: 8, overflow: "hidden", position: "relative",
      }}>
        <div style={{
          width: `${barWidth}%`, height: "100%", borderRadius: 8,
          background: isTop
            ? `linear-gradient(90deg, ${COLOR_ACCENT}cc, ${COLOR_ACCENT})`
            : `linear-gradient(90deg, rgba(96,165,250,0.6), rgba(96,165,250,0.9))`,
          boxShadow: isTop ? `0 0 20px ${COLOR_ACCENT}44` : "none",
        }} />
      </div>
      <div style={{
        fontFamily: FONT_DATA, fontSize: 20, fontWeight: 700, minWidth: 120,
        color: isTop ? COLOR_ACCENT : "rgba(255,244,234,0.8)",
        textShadow: isTop ? `0 0 12px ${COLOR_ACCENT}33` : "none",
        opacity: barReveal,
      }}>
        {displayValue ?? String(value)}
      </div>
    </div>
  );
};

export const DataStage: React.FC<DataStageProps> = ({
  headline,
  items,
  unit,
  caption,
  backgroundImage,
}) => {
  const maxValue = Math.max(...items.map((i) => i.value), 1);

  return (
    <SceneLayout backgroundImage={backgroundImage} brightness={0.28}>
      <div style={{ display: "flex", flexDirection: "column", gap: 20, justifyContent: "center", height: "100%" }}>
        <Headline text={headline} />
        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 12 }}>
          {items.map((item, i) => (
            <Bar key={i} label={item.label} value={item.value} displayValue={item.displayValue} maxValue={maxValue} index={i} total={items.length} />
          ))}
        </div>
        {unit && (
          <div style={{ fontFamily: FONT_DATA, fontSize: 13, color: "rgba(255,244,234,0.35)", textAlign: "right", marginTop: 4 }}>
            {unit}
          </div>
        )}
        {caption && <Caption text={caption} />}
      </div>
    </SceneLayout>
  );
};
