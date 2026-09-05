/**
 * Brain3DScene — Full composition template wrapping the 3D brain
 * with headline, region labels, and caption in a cinematic layout.
 *
 * Labels are projected from 3D electrode positions to 2D screen coordinates
 * so they track the brain regardless of camera angle or rotation.
 */

import React, { useMemo } from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { Brain3DSceneProps, ViewAngle } from "../types";
import {
  FONT_HEADLINE,
  FONT_BODY,
  FONT_DATA,
  FONT_CAPTION,
  COLOR_FG,
  COLOR_ACCENT,
} from "../types";
import { Brain3D, VIEW_PRESETS } from "../brain";
import { getElectrodePosition, project3Dto2D } from "../brain";
import type { Vec3 } from "../brain";

const STATUS_COLORS: Record<string, string> = {
  improved: COLOR_ACCENT,
  stable: "#60a5fa",
  declined: "#fbbf24",
  flagged: "#f87171",
};

const FOV = 35;

function resolveCamera(viewAngle: ViewAngle): { position: Vec3; target: Vec3 } {
  if (typeof viewAngle === "string") {
    return VIEW_PRESETS[viewAngle] ?? VIEW_PRESETS.topdown;
  }
  const { azimuth, elevation, distance = 3 } = viewAngle;
  const az = (azimuth * Math.PI) / 180;
  const el = (elevation * Math.PI) / 180;
  return {
    position: [
      distance * Math.cos(el) * Math.sin(az),
      distance * Math.sin(el),
      distance * Math.cos(el) * Math.cos(az),
    ],
    target: [0, 0, 0],
  };
}

export const Brain3DScene: React.FC<Brain3DSceneProps> = ({
  headline,
  caption,
  viewAngle = "topdown",
  regions = [],
  rotateSpeed = 0,
  accentColor = COLOR_ACCENT,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const reveal = spring({ frame, fps, config: { damping: 20, stiffness: 100, mass: 0.9 } });
  const rotation =
    rotateSpeed !== 0 ? (frame / fps) * rotateSpeed * (Math.PI / 180) : 0;
  const camera = useMemo(() => resolveCamera(viewAngle), [viewAngle]);

  // Build highlight list for the 3D layer
  const highlights = useMemo(
    () =>
      regions.map((r) => ({
        name: r.name,
        color: STATUS_COLORS[r.status ?? "stable"] ?? "#60a5fa",
      })),
    [regions],
  );

  // Project region electrode positions to 2D for label placement
  const labelPositions = useMemo(() => {
    return regions.map((r, i) => {
      const pos3d = getElectrodePosition(r.name);
      if (!pos3d) return { region: r, screen: null, index: i };
      const screen = project3Dto2D(
        pos3d,
        camera.position,
        camera.target,
        FOV,
        width,
        height,
        rotation,
      );
      return { region: r, screen, index: i };
    });
  }, [regions, camera, width, height, rotation]);

  return (
    <AbsoluteFill style={{ color: COLOR_FG }}>
      {/* Background: dark radial gradient matching the Qwen brain aesthetic */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at 50% 42%, #0e1a25 0%, #080e14 50%, #040709 100%)",
        }}
      />

      {/* 3D brain layer */}
      <AbsoluteFill style={{ opacity: interpolate(reveal, [0, 0.6], [0, 1]) }}>
        <Brain3D
          viewAngle={viewAngle}
          highlights={highlights}
          accentColor={accentColor}
          rotateSpeed={rotateSpeed}
        />
      </AbsoluteFill>

      {/* 2D overlay: headline */}
      <AbsoluteFill style={{ padding: 72, pointerEvents: "none" }}>
        <div
          style={{
            fontFamily: FONT_HEADLINE,
            fontSize: 52,
            lineHeight: 0.96,
            fontWeight: "bold",
            opacity: reveal,
            transform: `translateY(${interpolate(reveal, [0, 1], [20, 0])}px)`,
            textShadow: "0 3px 24px rgba(0,0,0,0.7), 0 1px 4px rgba(0,0,0,0.5)",
            maxWidth: "70%",
          }}
        >
          {headline}
        </div>
      </AbsoluteFill>

      {/* 2D overlay: projected region labels */}
      {labelPositions.map(({ region: r, screen, index: i }) => {
        if (!screen || !screen.visible) return null;

        const labelReveal = spring({
          frame: Math.max(frame - 18 - i * 10, 0),
          fps,
          config: { damping: 14, stiffness: 100, mass: 0.8 },
        });
        const color = STATUS_COLORS[r.status ?? "stable"] ?? "#60a5fa";

        return (
          <div
            key={i}
            style={{
              position: "absolute",
              left: screen.x,
              top: screen.y,
              transform: `translate(-50%, -50%) scale(${interpolate(labelReveal, [0, 1], [0.7, 1])})`,
              opacity: labelReveal,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 2,
              pointerEvents: "none",
            }}
          >
            <div
              style={{
                fontFamily: FONT_BODY,
                fontSize: 24,
                fontWeight: 700,
                color: "#fff",
                textShadow:
                  "0 2px 10px rgba(0,0,0,0.8), 0 0 6px rgba(0,0,0,0.5)",
                whiteSpace: "nowrap",
              }}
            >
              {r.name}
            </div>
            {r.value && (
              <div
                style={{
                  fontFamily: FONT_DATA,
                  fontSize: 19,
                  color,
                  textShadow: `0 1px 8px rgba(0,0,0,0.6), 0 0 14px ${color}44`,
                  whiteSpace: "nowrap",
                }}
              >
                {r.value}
              </div>
            )}
          </div>
        );
      })}

      {/* 2D overlay: caption */}
      {caption && (
        <div
          style={{
            position: "absolute",
            bottom: 72,
            left: 72,
            fontFamily: FONT_CAPTION,
            fontSize: 22,
            color: "rgba(255,244,234,0.6)",
            opacity: interpolate(reveal, [0.5, 1], [0, 1]),
            textShadow: "0 1px 6px rgba(0,0,0,0.5)",
          }}
        >
          {caption}
        </div>
      )}
    </AbsoluteFill>
  );
};
