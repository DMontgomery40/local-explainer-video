/**
 * Project 3D brain-space positions to 2D screen coordinates.
 * Used for placing electrode labels in the 2D overlay layer.
 */

import type { Vec3 } from "./electrodes";

export interface ScreenPoint {
  x: number;
  y: number;
  depth: number;
  /** True when the point is facing the camera and within frame. */
  visible: boolean;
}

/**
 * Project a 3D point (in brain coordinate space) to pixel coordinates
 * on the composition canvas.
 */
export function project3Dto2D(
  point: Vec3,
  cameraPos: Vec3,
  cameraTarget: Vec3,
  fov: number,
  width: number,
  height: number,
  brainRotationY = 0,
): ScreenPoint {
  // Rotate point around Y axis (brain turntable rotation)
  const cos = Math.cos(brainRotationY);
  const sin = Math.sin(brainRotationY);
  const px = point[0] * cos - point[2] * sin;
  const py = point[1];
  const pz = point[0] * sin + point[2] * cos;

  // Forward direction (camera → target)
  const fwd: Vec3 = [
    cameraTarget[0] - cameraPos[0],
    cameraTarget[1] - cameraPos[1],
    cameraTarget[2] - cameraPos[2],
  ];
  const fwdLen = Math.sqrt(fwd[0] ** 2 + fwd[1] ** 2 + fwd[2] ** 2) || 1;
  fwd[0] /= fwdLen;
  fwd[1] /= fwdLen;
  fwd[2] /= fwdLen;

  // Right = forward x worldUp
  const right: Vec3 = [
    fwd[1] * 0 - fwd[2] * 1,  // cross with (0, 1, 0) → simplified
    fwd[2] * 0 - fwd[0] * 0,
    fwd[0] * 1 - fwd[1] * 0,
  ];
  // Actually: cross(fwd, [0,1,0])
  right[0] = fwd[1] * 0 - fwd[2] * 1; // fy*0 - fz*1 → wrong sign
  // Let's compute correctly
  // right = fwd x up = (fy*uz - fz*uy, fz*ux - fx*uz, fx*uy - fy*ux)
  // with up=(0,1,0): right = (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0) = (-fz, 0, fx)
  right[0] = -fwd[2];
  right[1] = 0;
  right[2] = fwd[0];
  const rLen = Math.sqrt(right[0] ** 2 + right[2] ** 2) || 1;
  right[0] /= rLen;
  right[2] /= rLen;

  // Up = right x forward
  const up: Vec3 = [
    right[1] * fwd[2] - right[2] * fwd[1],
    right[2] * fwd[0] - right[0] * fwd[2],
    right[0] * fwd[1] - right[1] * fwd[0],
  ];

  // Point relative to camera
  const rel: Vec3 = [px - cameraPos[0], py - cameraPos[1], pz - cameraPos[2]];

  // Project onto view axes
  const viewX = rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2];
  const viewY = rel[0] * up[0] + rel[1] * up[1] + rel[2] * up[2];
  const viewZ = rel[0] * fwd[0] + rel[1] * fwd[1] + rel[2] * fwd[2];

  if (viewZ <= 0.01) return { x: 0, y: 0, depth: -1, visible: false };

  // Perspective divide
  const aspect = width / height;
  const tanHalf = Math.tan(((fov * Math.PI) / 180) / 2);

  const ndcX = viewX / (viewZ * tanHalf * aspect);
  const ndcY = viewY / (viewZ * tanHalf);

  const screenX = (ndcX * 0.5 + 0.5) * width;
  const screenY = (-ndcY * 0.5 + 0.5) * height;

  // Visibility: point must face camera and be within frame
  const normalDot =
    px * (cameraPos[0] - px) + py * (cameraPos[1] - py) + pz * (cameraPos[2] - pz);
  const inFrame = Math.abs(ndcX) < 1.15 && Math.abs(ndcY) < 1.15;

  return { x: screenX, y: screenY, depth: viewZ, visible: normalDot > 0 && inFrame };
}
