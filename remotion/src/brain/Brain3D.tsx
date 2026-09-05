/**
 * Brain3D — Real anatomical brain rendered with Three.js inside Remotion.
 *
 * Uses FreeSurfer pial surface meshes (lh.glb + rh.glb) for anatomically
 * accurate gyri/sulci detail. Supports arbitrary camera angles, animated
 * rotation, glowing electrode markers, and cinematic Fresnel rim glow.
 */

import React, { useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import { ThreeCanvas } from "@remotion/three";
import { useCurrentFrame, useVideoConfig, staticFile, delayRender, continueRender } from "remotion";
import { useThree } from "@react-three/fiber";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import {
  brainVertexShader,
  brainFragmentShader,
  atmosphereVertexShader,
  atmosphereFragmentShader,
  createBrainUniforms,
  hexToVec3,
} from "./shaders";
import { getElectrodePosition, type Vec3 } from "./electrodes";

// ─── View presets ───

export const VIEW_PRESETS: Record<string, { position: Vec3; target: Vec3 }> = {
  topdown: { position: [0, 3.2, 0.4], target: [0, 0, 0] },
  frontal: { position: [0, 0.2, 3.0], target: [0, 0, 0] },
  "lateral-left": { position: [-3.0, 0.4, 0.3], target: [0, 0, 0] },
  "lateral-right": { position: [3.0, 0.4, 0.3], target: [0, 0, 0] },
  "three-quarter": { position: [-2.2, 1.4, 2.0], target: [0, 0, 0] },
  "three-quarter-right": { position: [2.2, 1.4, 2.0], target: [0, 0, 0] },
  posterior: { position: [0, 0.3, -3.0], target: [0, 0, 0] },
};

export type ViewAngle =
  | keyof typeof VIEW_PRESETS
  | { azimuth: number; elevation: number; distance?: number };

// ─── Props ───

export interface Brain3DProps {
  viewAngle?: ViewAngle;
  highlights?: Array<{ name: string; color?: string; intensity?: number }>;
  accentColor?: string;
  rotateSpeed?: number;
  width?: number;
  height?: number;
}

// ─── Mesh processing ───

interface BrainMeshData {
  left: THREE.BufferGeometry;
  right: THREE.BufferGeometry;
}

function processBrainMeshes(
  lhScene: THREE.Group,
  rhScene: THREE.Group,
): BrainMeshData {
  const lhGeo = findFirstGeometry(lhScene);
  const rhGeo = findFirstGeometry(rhScene);

  // Combined bounding box
  lhGeo.computeBoundingBox();
  rhGeo.computeBoundingBox();
  const bbox = new THREE.Box3();
  bbox.union(lhGeo.boundingBox!);
  bbox.union(rhGeo.boundingBox!);

  const center = bbox.getCenter(new THREE.Vector3());
  const size = bbox.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const s = 1.6 / maxDim;

  // Reorient: FreeSurfer (+Y=anterior, +Z=superior) → Three.js (+Y=up, +Z=front)
  // Swap Y↔Z so superior goes up and anterior faces the camera.
  const swapYZ = new THREE.Matrix4().set(
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
  );

  for (const geo of [lhGeo, rhGeo]) {
    geo.translate(-center.x, -center.y, -center.z);
    geo.scale(s, s, s);
    geo.applyMatrix4(swapYZ);
    geo.computeVertexNormals();

    // The Y↔Z swap changes handedness, so computed normals point inward.
    // Flip them so Fresnel and diffuse shading work correctly.
    const normals = geo.attributes.normal;
    for (let i = 0; i < normals.count; i++) {
      normals.setXYZ(i, -normals.getX(i), -normals.getY(i), -normals.getZ(i));
    }
    normals.needsUpdate = true;
  }

  return { left: lhGeo, right: rhGeo };
}

function findFirstGeometry(scene: THREE.Group): THREE.BufferGeometry {
  const meshes: THREE.Mesh[] = [];
  scene.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) meshes.push(child as THREE.Mesh);
  });
  if (meshes.length === 0) throw new Error("No mesh found in GLB");
  return meshes[0].geometry.clone();
}

// ─── Main component ───

export const Brain3D: React.FC<Brain3DProps> = ({
  viewAngle = "topdown",
  highlights = [],
  accentColor = "#5eead4",
  rotateSpeed = 0,
  width: widthOverride,
  height: heightOverride,
}) => {
  const frame = useCurrentFrame();
  const { fps, width: compWidth, height: compHeight } = useVideoConfig();
  const width = widthOverride ?? compWidth;
  const height = heightOverride ?? compHeight;

  // Load brain meshes with Remotion's delayRender pattern
  const [meshData, setMeshData] = useState<BrainMeshData | null>(null);
  const [handle] = useState(() => delayRender("Loading FreeSurfer brain meshes"));

  useEffect(() => {
    const loader = new GLTFLoader();
    const lhUrl = staticFile("brain/lh.glb");
    const rhUrl = staticFile("brain/rh.glb");

    Promise.all([loader.loadAsync(lhUrl), loader.loadAsync(rhUrl)])
      .then(([lhGltf, rhGltf]) => {
        const data = processBrainMeshes(lhGltf.scene, rhGltf.scene);
        setMeshData(data);
        continueRender(handle);
      })
      .catch((err) => {
        console.error("Failed to load brain meshes:", err);
        continueRender(handle);
      });
  }, [handle]);

  const camera = useMemo(() => {
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
      ] as Vec3,
      target: [0, 0, 0] as Vec3,
    };
  }, [viewAngle]);

  const rotation = rotateSpeed !== 0 ? (frame / fps) * rotateSpeed * (Math.PI / 180) : 0;

  if (!meshData) return null;

  return (
    <ThreeCanvas
      width={width}
      height={height}
      camera={{
        position: camera.position as unknown as [number, number, number],
        fov: 35,
        near: 0.1,
        far: 100,
      }}
      style={{ backgroundColor: "transparent" }}
    >
      <CameraLookAt target={camera.target} />
      <BrainScene
        meshData={meshData}
        rotation={rotation}
        accentColor={accentColor}
        highlights={highlights}
        frame={frame}
      />
    </ThreeCanvas>
  );
};

// ─── Camera helper ───

function CameraLookAt({ target }: { target: Vec3 }) {
  const { camera } = useThree();
  camera.lookAt(target[0], target[1], target[2]);
  return null;
}

// ─── Scene content ───

interface BrainSceneProps {
  meshData: BrainMeshData;
  rotation: number;
  accentColor: string;
  highlights: Array<{ name: string; color?: string; intensity?: number }>;
  frame: number;
}

function BrainScene({ meshData, rotation, accentColor, highlights, frame }: BrainSceneProps) {
  // Brain shader material
  const brainMaterial = useMemo(() => {
    const uniforms = createBrainUniforms(accentColor) as unknown as Record<
      string,
      THREE.IUniform
    >;
    return new THREE.ShaderMaterial({
      vertexShader: brainVertexShader,
      fragmentShader: brainFragmentShader,
      uniforms,
      side: THREE.DoubleSide,
    });
  }, [accentColor]);

  // Atmosphere material
  const atmosphereMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader: atmosphereVertexShader,
        fragmentShader: atmosphereFragmentShader,
        uniforms: {
          uGlowColor: { value: new THREE.Vector3(...hexToVec3(accentColor)) },
          uIntensity: { value: 0.06 },
        },
        transparent: true,
        side: THREE.BackSide,
        depthWrite: false,
      }),
    [accentColor],
  );

  // Glow texture for electrode markers
  const glowTexture = useMemo(() => {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const gradient = ctx.createRadialGradient(
      size / 2, size / 2, 0,
      size / 2, size / 2, size / 2,
    );
    gradient.addColorStop(0, "rgba(255,255,255,1.0)");
    gradient.addColorStop(0.15, "rgba(255,255,255,0.8)");
    gradient.addColorStop(0.4, "rgba(94,234,212,0.4)");
    gradient.addColorStop(0.7, "rgba(94,234,212,0.1)");
    gradient.addColorStop(1, "rgba(94,234,212,0.0)");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(canvas);
    tex.needsUpdate = true;
    return tex;
  }, []);

  // Resolve highlight electrode positions
  const electrodeMarkers = useMemo(() => {
    return highlights
      .map((h) => {
        const pos = getElectrodePosition(h.name);
        if (!pos) return null;
        return { position: pos, color: h.color ?? accentColor, intensity: h.intensity ?? 1 };
      })
      .filter(Boolean) as Array<{ position: Vec3; color: string; intensity: number }>;
  }, [highlights, accentColor]);

  const pulse = Math.sin(frame * 0.1) * 0.3 + 0.8;

  return (
    <>
      <ambientLight intensity={0.08} color="#8090a0" />
      <directionalLight position={[2, 4, 3]} intensity={0.6} color="#c8d8e8" />
      <directionalLight position={[-3, -1, 1]} intensity={0.15} color="#607080" />
      <directionalLight position={[0, 1, -4]} intensity={0.3} color={accentColor} />

      <group rotation={[0, rotation, 0]}>
        <mesh geometry={meshData.left} material={brainMaterial} />
        <mesh geometry={meshData.right} material={brainMaterial} />

        {/* Atmospheric glow shell */}
        <mesh material={atmosphereMaterial}>
          <sphereGeometry args={[0.95, 32, 24]} />
        </mesh>

        {/* Electrode glow markers */}
        {electrodeMarkers.map((marker, i) => {
          const scale = 0.12 * pulse * marker.intensity;
          return (
            <sprite
              key={`electrode-${i}`}
              position={marker.position}
              scale={[scale, scale, 1]}
            >
              <spriteMaterial
                map={glowTexture}
                transparent
                blending={THREE.AdditiveBlending}
                depthWrite={false}
                color={marker.color}
                opacity={0.9 * marker.intensity}
              />
            </sprite>
          );
        })}
      </group>
    </>
  );
}
