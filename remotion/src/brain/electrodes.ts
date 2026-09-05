/**
 * 10-20 EEG electrode system positions mapped to 3D brain coordinates.
 *
 * Coordinate system (matches Three.js default):
 *   +X = right,  -X = left
 *   +Y = dorsal (up),  -Y = ventral (down)
 *   +Z = anterior (front),  -Z = posterior (back)
 *
 * Positions approximate the electrode locations on the brain surface
 * after ellipsoid scaling (~0.65 x 0.72 x 0.88).
 */

export type Vec3 = [number, number, number];

const ELECTRODES: Record<string, Vec3> = {
  // Frontal pole
  fp1: [-0.22, 0.48, 0.62],
  fp2: [0.22, 0.48, 0.62],
  // Frontal
  f7: [-0.52, 0.26, 0.42],
  f3: [-0.35, 0.50, 0.45],
  fz: [0.00, 0.56, 0.48],
  f4: [0.35, 0.50, 0.45],
  f8: [0.52, 0.26, 0.42],
  // Central
  t3: [-0.58, 0.06, 0.00],
  c3: [-0.42, 0.52, 0.00],
  cz: [0.00, 0.66, 0.00],
  c4: [0.42, 0.52, 0.00],
  t4: [0.58, 0.06, 0.00],
  // Parietal
  t5: [-0.48, 0.10, -0.38],
  p3: [-0.35, 0.48, -0.38],
  pz: [0.00, 0.54, -0.38],
  p4: [0.35, 0.48, -0.38],
  t6: [0.48, 0.10, -0.38],
  // Occipital
  o1: [-0.18, 0.34, -0.66],
  oz: [0.00, 0.36, -0.66],
  o2: [0.18, 0.34, -0.66],
};

/** Lobe-level aggregate positions (centroid of relevant electrodes). */
const LOBES: Record<string, Vec3> = {
  frontal: [0.00, 0.52, 0.48],
  prefrontal: [0.00, 0.48, 0.64],
  central: [0.00, 0.60, 0.00],
  parietal: [0.00, 0.52, -0.38],
  temporal: [-0.55, 0.10, 0.00],
  occipital: [0.00, 0.35, -0.66],
  "central-parietal": [0.00, 0.58, -0.20],
};

/** Resolve a region name to its 3D position, checking electrodes then lobes. */
export function getElectrodePosition(name: string): Vec3 | undefined {
  const key = name.toLowerCase().replace(/[\s_-]+/g, "");
  // Exact match
  if (ELECTRODES[key]) return ELECTRODES[key];
  if (LOBES[key]) return LOBES[key];
  // Fuzzy: check if any key contains this name or vice-versa
  for (const [k, v] of Object.entries(ELECTRODES)) {
    if (key.includes(k) || k.includes(key)) return v;
  }
  for (const [k, v] of Object.entries(LOBES)) {
    const normalized = k.replace(/-/g, "");
    if (key.includes(normalized) || normalized.includes(key)) return v;
  }
  return undefined;
}

export function getAllElectrodes(): Record<string, Vec3> {
  return { ...ELECTRODES };
}

export function getAllLobes(): Record<string, Vec3> {
  return { ...LOBES };
}
