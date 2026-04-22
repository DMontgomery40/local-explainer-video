declare module "@babel/standalone" {
  export interface TransformResult {
    code?: string | null;
  }

  export interface TransformOptions {
    presets?: string[];
    filename?: string;
  }

  export function transform(source: string, options?: TransformOptions): TransformResult;
}
