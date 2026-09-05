/**
 * Custom GLSL shaders for the cinematic glass-brain look.
 * Matches the Qwen brain aesthetic: dark teal-grey surface, bright cyan Fresnel rim.
 */

export const brainVertexShader = /* glsl */ `
varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vViewDirection;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vWorldPosition = worldPos.xyz;
  vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
  vViewDirection = normalize(cameraPosition - worldPos.xyz);
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const brainFragmentShader = /* glsl */ `
uniform vec3 uBaseColor;
uniform vec3 uDeepColor;
uniform vec3 uRimColor;
uniform float uRimPower;
uniform float uRimStrength;

varying vec3 vNormal;
varying vec3 vWorldPosition;
varying vec3 vViewDirection;

void main() {
  vec3 N = normalize(vNormal);
  vec3 V = normalize(vViewDirection);

  // Key light: above-right, standard Lambert (no wrapping)
  vec3 keyDir = normalize(vec3(0.3, 0.8, 0.5));
  float NdotL = max(dot(N, keyDir), 0.0);
  float diffuse = NdotL * 0.45;

  // Fill: very faint from opposite side
  vec3 fillDir = normalize(vec3(-0.4, -0.2, 0.3));
  float fill = max(dot(N, fillDir), 0.0) * 0.06;

  // Dark body — sulci stay black, gyri crests get subtle shading
  vec3 color = mix(uDeepColor, uBaseColor, diffuse + fill);

  // Fresnel rim glow — narrow bright edge only
  float fresnel = pow(1.0 - max(dot(V, N), 0.0), uRimPower);
  color += uRimColor * fresnel * uRimStrength;

  // Tight specular
  vec3 H = normalize(keyDir + V);
  float spec = pow(max(dot(N, H), 0.0), 120.0);
  color += uRimColor * spec * 0.08;

  gl_FragColor = vec4(color, 1.0);
}
`;

export const atmosphereVertexShader = /* glsl */ `
varying vec3 vNormal;
varying vec3 vViewDirection;

void main() {
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vNormal = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
  vViewDirection = normalize(cameraPosition - worldPos.xyz);
  gl_Position = projectionMatrix * viewMatrix * worldPos;
}
`;

export const atmosphereFragmentShader = /* glsl */ `
uniform vec3 uGlowColor;
uniform float uIntensity;

varying vec3 vNormal;
varying vec3 vViewDirection;

void main() {
  float rim = pow(0.72 - dot(normalize(vNormal), normalize(vViewDirection)), 2.0);
  gl_FragColor = vec4(uGlowColor, rim * uIntensity);
}
`;

export interface BrainUniforms {
  uBaseColor: { value: [number, number, number] };
  uDeepColor: { value: [number, number, number] };
  uRimColor: { value: [number, number, number] };
  uRimPower: { value: number };
  uRimStrength: { value: number };
}

export function hexToVec3(hex: string): [number, number, number] {
  const c = hex.replace("#", "");
  return [
    parseInt(c.substring(0, 2), 16) / 255,
    parseInt(c.substring(2, 4), 16) / 255,
    parseInt(c.substring(4, 6), 16) / 255,
  ];
}

export function createBrainUniforms(accentColor = "#5eead4"): BrainUniforms {
  return {
    uBaseColor: { value: hexToVec3("#182428") },
    uDeepColor: { value: hexToVec3("#050a0e") },
    uRimColor: { value: hexToVec3(accentColor) },
    uRimPower: { value: 3.5 },
    uRimStrength: { value: 0.75 },
  };
}
