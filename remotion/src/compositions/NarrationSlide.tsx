import React from "react";
import { interpolate } from "remotion";
import type { NarrationSlideProps } from "../types";
import { FONT_BODY } from "../types";
import { SceneLayout, Headline, Caption, useReveal } from "./shared";

export const NarrationSlide: React.FC<NarrationSlideProps> = ({
  headline,
  body,
  caption,
  backgroundImage,
}) => {
  const reveal = useReveal();
  return (
    <SceneLayout backgroundImage={backgroundImage}>
      <div style={{ display: "flex", flexDirection: "column", gap: 20, justifyContent: "center", height: "100%" }}>
        <Headline text={headline} />
        {body && (
          <div
            style={{
              fontFamily: FONT_BODY,
              fontSize: 28,
              lineHeight: 1.5,
              color: "rgba(255,244,234,0.85)",
              maxWidth: "70%",
              opacity: interpolate(reveal, [0.2, 1], [0, 1]),
            }}
          >
            {body}
          </div>
        )}
        {caption && <Caption text={caption} />}
      </div>
    </SceneLayout>
  );
};
