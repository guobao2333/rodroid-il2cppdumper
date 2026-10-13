export type MethodBodyMode = "off" | "assembly" | "csharp";

export interface DumperConfig {
  dumpMethod: boolean;
  dumpField: boolean;
  dumpProperty: boolean;
  dumpAttribute: boolean;
  dumpMethodOffset: boolean;
  dumpFieldOffset: boolean;
  dumpTypeDefIndex: boolean;
  dumpAssemblyName: boolean;
  generateStruct: boolean;
  generateDummyDll: boolean;
  dummyDllAddToken: boolean;
  forceIl2cppVersion: boolean;
  forceVersion: number;
  forceDump: boolean;
  noRedirectedPointer: boolean;
  splitDumpPerType: boolean;
  generateGenericsDump: boolean;
  dumpGenericsRgctx: boolean;
  dumpGenericsMethodSpecs: boolean;
  dumpGenericsCustomAttributes: boolean;
  dumpGenericsStringLiterals: boolean;
  dumpGenericsMetadataUsages: boolean;
  dumpGenericsVtables: boolean;
  dumpGenericsInterfaces: boolean;
  methodBodyMode: MethodBodyMode;
  dumpDisassembly: boolean;
  assemblyTarget: number;
  dumpPseudoC: boolean;
  csharpTarget: number;
  csharpInteractive: boolean;
  csharpMaxBodySize: number;
  csharpIncludeAssemblies: string[];
  csharpIncludeTypes: string[];
  assemblyHexBytes: boolean;
  assemblyFieldNames: boolean;
  assemblyAnnotations: boolean;
  assemblyCfg: boolean;
  maxAssemblyInstructions: number;
  generateCppScaffold: boolean;
  mangleNames: boolean;
  enhancedIdaMetadata: boolean;
  generateUnityHeaders: boolean;
  compilerLayout: string;
  useTopologicalSort: boolean;
  codm: boolean;
  dumpStaticFieldMetadata: boolean;
  dumpFieldRvaData: boolean;
  maxFieldRvaDumpBytes: number;
  ilRecovery: boolean;
  pseudoCEngine: string;
  annotateRecovery: boolean;
}

export interface TypeTreeType {
  fullName: string;
  methodCount: number;
}

export interface TypeTreeNamespace {
  namespace: string;
  methodCount: number;
  types: TypeTreeType[];
}

export interface TypeTreeAssembly {
  assembly: string;
  methodCount: number;
  namespaces: TypeTreeNamespace[];
}

export function methodBodyModeOf(config: DumperConfig): MethodBodyMode {
  const fromField = (config.methodBodyMode || "off").toLowerCase();
  // Asm and C# are mutually exclusive — legacy "both" coerces to off.
  if (fromField === "assembly" || fromField === "asm" || fromField === "disassembly" || fromField === "disasm") {
    return "assembly";
  }
  if (fromField === "csharp" || fromField === "c#" || fromField === "cs" || fromField === "pseudo") {
    return "csharp";
  }
  if (config.dumpDisassembly && !config.dumpPseudoC) return "assembly";
  if (config.dumpPseudoC && !config.dumpDisassembly) return "csharp";
  return "off";
}

export function withMethodBodyMode(config: DumperConfig, mode: MethodBodyMode): DumperConfig {
  switch (mode) {
    case "off":
      return { ...config, methodBodyMode: "off", dumpDisassembly: false, dumpPseudoC: false };
    case "assembly":
      return { ...config, methodBodyMode: "assembly", dumpDisassembly: true, dumpPseudoC: false };
    case "csharp":
      return { ...config, methodBodyMode: "csharp", dumpDisassembly: false, dumpPseudoC: true };
  }
}

/** Keep `methodBodyMode` and legacy bools aligned (UI may show C# from bools alone). */
export function normalizeDumperConfig(config: DumperConfig): DumperConfig {
  return withMethodBodyMode(config, methodBodyModeOf(config));
}

export interface BinaryInfo {
  format: string;
  unity_version: string;
}

export interface DumpCompleteEvent {
  success: boolean;
  output_path: string;
  error_message: string;
}

export interface InputRequestEvent {
  prompt_type: string;
}

export type AppState = "idle" | "dumping" | "result" | "error";

export const DEFAULT_CONFIG: DumperConfig = {
  dumpMethod: true,
  dumpField: true,
  dumpProperty: true,
  dumpAttribute: true,
  dumpMethodOffset: true,
  dumpFieldOffset: true,
  dumpTypeDefIndex: true,
  dumpAssemblyName: true,
  generateStruct: true,
  generateDummyDll: true,
  dummyDllAddToken: true,
  forceIl2cppVersion: false,
  forceVersion: 24.3,
  forceDump: false,
  noRedirectedPointer: false,
  splitDumpPerType: false,
  generateGenericsDump: true,
  dumpGenericsRgctx: true,
  dumpGenericsMethodSpecs: true,
  dumpGenericsCustomAttributes: true,
  dumpGenericsStringLiterals: true,
  dumpGenericsMetadataUsages: true,
  dumpGenericsVtables: true,
  dumpGenericsInterfaces: true,
  methodBodyMode: "off",
  dumpDisassembly: false,
  assemblyTarget: 0,
  dumpPseudoC: false,
  csharpTarget: 0,
  csharpInteractive: false,
  csharpMaxBodySize: 0,
  csharpIncludeAssemblies: [],
  csharpIncludeTypes: [],
  assemblyHexBytes: true,
  assemblyFieldNames: true,
  assemblyAnnotations: true,
  assemblyCfg: true,
  maxAssemblyInstructions: 512,
  generateCppScaffold: true,
  mangleNames: true,
  enhancedIdaMetadata: true,
  generateUnityHeaders: true,
  compilerLayout: "GCC",
  useTopologicalSort: true,
  codm: false,
  dumpStaticFieldMetadata: false,
  dumpFieldRvaData: false,
  maxFieldRvaDumpBytes: 4096,
  ilRecovery: false,
  pseudoCEngine: "native",
  annotateRecovery: true,
};
