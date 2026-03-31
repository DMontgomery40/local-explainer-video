import React from "react";
import { interpolate } from "remotion";
import type { ClosingCtaProps } from "../types";
import { FONT_BODY, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, useReveal, useStaggeredReveal } from "./shared";

const Bullet: React.FC<{ text: string; index: number }> = ({ text, index }) => {
  const itemReveal = useStaggeredReveal(index);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 12,
        opacity: itemReveal,
        transform: `translateX(${interpolate(itemReveal, [0, 1], [-12, 0])}px)`,
      }}
    >
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: COLOR_ACCENT, flexShrink: 0, marginTop: 10 }} />
      <div style={{ fontFamily: FONT_BODY, fontSize: 26, lineHeight: 1.4 }}>{text}</div>
    </div>
  );
};

export const ClosingCta: React.FC<ClosingCtaProps> = ({
  headline,
  bullets,
  signoff,
  backgroundImage,
}) => {
  const reveal = useReveal();
  return (
    <SceneLayout backgroundImage={backgroundImage}>
      <div style={{ display: "flex", flexDirection: "column", gap: 24, justifyContent: "center", height: "100%" }}>
        <Headline text={headline} />
        {bullets && bullets.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 12 }}>
            {bullets.map((b, i) => (
              <Bullet key={i} text={b} index={i} />
            ))}
          </div>
        )}
        {signoff && (
          <div
            style={{
              fontFamily: FONT_BODY,
              fontSize: 24,
              color: "rgba(255,244,234,0.6)",
              fontStyle: "italic",
              marginTop: "auto",
              opacity: interpolate(reveal, [0.5, 1], [0, 1]),
            }}
          >
            {signoff}
          </div>
        )}
      </div>
    </SceneLayout>
  );
};
