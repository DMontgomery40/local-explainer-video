import React from "react";
import { interpolate } from "remotion";
import type { BulletStackProps } from "../types";
import { FONT_BODY, FONT_DATA, COLOR_ACCENT } from "../types";
import { SceneLayout, Headline, Caption, useStaggeredReveal, useReveal } from "./shared";

const BulletItem: React.FC<{ text: string; index: number }> = ({ text, index }) => {
  const itemReveal = useStaggeredReveal(index);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 16,
        opacity: itemReveal,
        transform: `translateX(${interpolate(itemReveal, [0, 1], [-16, 0])}px)`,
      }}
    >
      <div style={{ fontFamily: FONT_DATA, fontSize: 20, color: COLOR_ACCENT, fontWeight: 700, minWidth: 32 }}>
        {String(index + 1).padStart(2, "0")}
      </div>
      <div style={{ fontFamily: FONT_BODY, fontSize: 28, lineHeight: 1.4 }}>{text}</div>
    </div>
  );
};

export const BulletStack: React.FC<BulletStackProps> = ({
  headline,
  items,
  caption,
  backgroundImage,
}) => (
  <SceneLayout backgroundImage={backgroundImage}>
    <div style={{ display: "flex", flexDirection: "column", gap: 24, justifyContent: "center", height: "100%" }}>
      <Headline text={headline} />
      <div style={{ display: "flex", flexDirection: "column", gap: 18, marginTop: 16 }}>
        {items.map((item, i) => (
          <BulletItem key={i} text={item} index={i} />
        ))}
      </div>
      {caption && <Caption text={caption} />}
    </div>
  </SceneLayout>
);
