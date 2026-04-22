import React, { useMemo, useState, useEffect, useRef, useCallback } from "react";
import * as Babel from "@babel/standalone";
import {
  AbsoluteFill,
  Img,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  Easing,
} from "remotion";

type DynamicSceneProps = Record<string, unknown>;

export const DynamicScene: React.FC<DynamicSceneProps> = (props) => {
  const code = typeof props.code === "string" ? props.code : "";
  const Component = useMemo(() => {
    if (!code?.trim()) return null;

    try {
      const codeWithoutImports = code.replace(/^import\s+.*$/gm, "").trim();

      const exportMatch = codeWithoutImports.match(
        /export\s+(?:default\s+)?(?:const|function)\s+(\w+)\s*(?::\s*React\.FC\s*)?=?\s*\(?.*?\)?\s*(?:=>)?\s*\{([\s\S]*)\};?\s*$/,
      );

      let wrappedSource: string;
      if (exportMatch) {
        const body = exportMatch[2].trim();
        wrappedSource = `const DynamicComponent = () => {\n${body}\n};`;
      } else {
        wrappedSource = `const DynamicComponent = () => {\n${codeWithoutImports}\n};`;
      }

      const transpiled = Babel.transform(wrappedSource, {
        presets: ["react", "typescript"],
        filename: "dynamic.tsx",
      });

      if (!transpiled.code) return null;

      const createComponent = new Function(
        "React",
        "useState",
        "useEffect",
        "useMemo",
        "useRef",
        "useCallback",
        "AbsoluteFill",
        "Img",
        "Sequence",
        "interpolate",
        "spring",
        "staticFile",
        "useCurrentFrame",
        "useVideoConfig",
        "Easing",
        `${transpiled.code}\nreturn DynamicComponent;`,
      );

      return createComponent(
        React,
        useState,
        useEffect,
        useMemo,
        useRef,
        useCallback,
        AbsoluteFill,
        Img,
        Sequence,
        interpolate,
        spring,
        staticFile,
        useCurrentFrame,
        useVideoConfig,
        Easing,
      ) as React.FC;
    } catch (err) {
      console.error("Dynamic compilation failed:", err);
      return null;
    }
  }, [code]);

  if (!Component) {
    return (
      <AbsoluteFill
        style={{
          background: "#0a0a0f",
          justifyContent: "center",
          alignItems: "center",
          color: "#f87171",
          fontFamily: "monospace",
          fontSize: 24,
        }}
      >
        Scene compilation error
      </AbsoluteFill>
    );
  }

  return <Component />;
};
