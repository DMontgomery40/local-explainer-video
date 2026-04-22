import React from "react";
import { AbsoluteFill, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { FONT_HEADLINE, FONT_BODY, FONT_DATA, FONT_CAPTION, COLOR_FG, BODY_SIZE } from "../types";

function resolveImageSrc(src: string): string {
  if (!src) return "";
  if (src.startsWith("http://") || src.startsWith("https://") || src.startsWith("data:")) {
    return src;
  }
  const normalized = src.startsWith("backgrounds/") ? src : `backgrounds/${src}`;
  return staticFile(normalized);
}

/* Reusable background image layer with scrim, brightness, and reveal animation. */
export const BackgroundImage: React.FC<{
  src: string;
  reveal?: number;
  brightness?: number;
  scrimGradient?: string;
}> = ({ src, reveal = 1, brightness = 0.4, scrimGradient }) => (
  <AbsoluteFill>
    <Img
      src={resolveImageSrc(src)}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "cover",
        filter: `brightness(${brightness})`,
        opacity: interpolate(reveal, [0, 1], [0.4, 1]),
      }}
    />
    {scrimGradient && (
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: scrimGradient,
        }}
      />
    )}
  </AbsoluteFill>
);

/* Standard spring reveal used across all templates. */
export function useReveal() {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({ frame, fps, config: { damping: 20, stiffness: 100, mass: 0.9 } });
}

/* Staggered reveal for list items. */
export function useStaggeredReveal(index: number, delayPerItem = 8) {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return spring({
    frame: Math.max(frame - 12 - index * delayPerItem, 0),
    fps,
    config: { damping: 16, stiffness: 110, mass: 0.9 },
  });
}

/* Standard padded content wrapper with fade-in headline. */
export const SceneLayout: React.FC<{
  backgroundImage?: string;
  brightness?: number;
  scrimGradient?: string;
  children: React.ReactNode;
}> = ({ backgroundImage, brightness, scrimGradient, children }) => {
  const reveal = useReveal();
  return (
    <AbsoluteFill style={{ color: COLOR_FG, fontFamily: FONT_BODY }}>
      {backgroundImage && (
        <BackgroundImage
          src={backgroundImage}
          reveal={reveal}
          brightness={brightness ?? 0.35}
          scrimGradient={
            scrimGradient ??
            "linear-gradient(180deg, rgba(3,5,10,0.75) 0%, rgba(3,5,10,0.4) 40%, rgba(3,5,10,0.4) 60%, rgba(3,5,10,0.75) 100%)"
          }
        />
      )}
      {!backgroundImage && (
        <AbsoluteFill style={{ background: "linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%)" }} />
      )}
      <AbsoluteFill style={{ padding: 88 }}>{children}</AbsoluteFill>
    </AbsoluteFill>
  );
};

/* Animated headline text. */
export const Headline: React.FC<{ text: string; fontSize?: number }> = ({
  text,
  fontSize = 60,
}) => {
  const reveal = useReveal();
  return (
    <div
      style={{
        fontFamily: FONT_HEADLINE,
        fontSize,
        lineHeight: 0.96,
        fontWeight: "bold",
        opacity: reveal,
        transform: `translateY(${interpolate(reveal, [0, 1], [20, 0])}px)`,
        textShadow: "0 3px 20px rgba(0,0,0,0.5), 0 1px 3px rgba(0,0,0,0.3)",
      }}
    >
      {text}
    </div>
  );
};

/* Caption text at the bottom. */
export const Caption: React.FC<{ text: string }> = ({ text }) => {
  const reveal = useReveal();
  if (!text) return null;
  return (
    <div
      style={{
        fontFamily: FONT_CAPTION,
        fontSize: 22,
        color: "rgba(255,244,234,0.6)",
        opacity: interpolate(reveal, [0.5, 1], [0, 1]),
        marginTop: "auto",
      }}
    >
      {text}
    </div>
  );
};

/* Monospace data label. */
export const DataLabel: React.FC<{
  text: string;
  color?: string;
  fontSize?: number;
}> = ({ text, color = "#5eead4", fontSize = 18 }) => (
  <span
    style={{
      fontFamily: FONT_DATA,
      fontSize,
      letterSpacing: "0.06em",
      textTransform: "uppercase" as const,
      color,
      fontWeight: 700,
    }}
  >
    {text}
  </span>
);
