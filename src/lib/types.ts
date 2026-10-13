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

function asNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function asBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  return fallback;
}

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((s) => String(s)).filter((s) => s.length > 0);
}

/**
 * Rebuild a clean DumperConfig from known keys only.
 *
 * Old localStorage may still carry serde alias keys (`dumpDisassemblyHexBytes`,
 * `dumpDisassemblyTarget`, …). Spreading those into JSON makes Rust reject the
 * whole Config with "duplicate field". Allow-listing avoids that forever.
 */
export function normalizeDumperConfig(config: DumperConfig): DumperConfig {
  const raw = config as DumperConfig & Record<string, unknown>;
  const pick = <T>(primary: unknown, legacy: unknown, fallback: T, coerce: (v: unknown, f: T) => T): T =>
    coerce(primary !== undefined ? primary : legacy, fallback);

  const cleaned: DumperConfig = {
    ...DEFAULT_CONFIG,
    dumpMethod: asBool(raw.dumpMethod, DEFAULT_CONFIG.dumpMethod),
    dumpField: asBool(raw.dumpField, DEFAULT_CONFIG.dumpField),
    dumpProperty: asBool(raw.dumpProperty, DEFAULT_CONFIG.dumpProperty),
    dumpAttribute: asBool(raw.dumpAttribute, DEFAULT_CONFIG.dumpAttribute),
    dumpMethodOffset: asBool(raw.dumpMethodOffset, DEFAULT_CONFIG.dumpMethodOffset),
    dumpFieldOffset: asBool(raw.dumpFieldOffset, DEFAULT_CONFIG.dumpFieldOffset),
    dumpTypeDefIndex: asBool(raw.dumpTypeDefIndex, DEFAULT_CONFIG.dumpTypeDefIndex),
    dumpAssemblyName: asBool(raw.dumpAssemblyName, DEFAULT_CONFIG.dumpAssemblyName),
    generateStruct: asBool(raw.generateStruct, DEFAULT_CONFIG.generateStruct),
    generateDummyDll: asBool(raw.generateDummyDll, DEFAULT_CONFIG.generateDummyDll),
    dummyDllAddToken: asBool(raw.dummyDllAddToken, DEFAULT_CONFIG.dummyDllAddToken),
    forceIl2cppVersion: asBool(raw.forceIl2cppVersion, DEFAULT_CONFIG.forceIl2cppVersion),
    forceVersion: asNumber(raw.forceVersion, DEFAULT_CONFIG.forceVersion),
    forceDump: asBool(raw.forceDump, DEFAULT_CONFIG.forceDump),
    noRedirectedPointer: asBool(raw.noRedirectedPointer, DEFAULT_CONFIG.noRedirectedPointer),
    splitDumpPerType: asBool(raw.splitDumpPerType, DEFAULT_CONFIG.splitDumpPerType),
    generateGenericsDump: asBool(raw.generateGenericsDump, DEFAULT_CONFIG.generateGenericsDump),
    dumpGenericsRgctx: asBool(raw.dumpGenericsRgctx, DEFAULT_CONFIG.dumpGenericsRgctx),
    dumpGenericsMethodSpecs: asBool(raw.dumpGenericsMethodSpecs, DEFAULT_CONFIG.dumpGenericsMethodSpecs),
    dumpGenericsCustomAttributes: asBool(
      raw.dumpGenericsCustomAttributes,
      DEFAULT_CONFIG.dumpGenericsCustomAttributes,
    ),
    dumpGenericsStringLiterals: asBool(
      raw.dumpGenericsStringLiterals,
      DEFAULT_CONFIG.dumpGenericsStringLiterals,
    ),
    dumpGenericsMetadataUsages: asBool(
      raw.dumpGenericsMetadataUsages,
      DEFAULT_CONFIG.dumpGenericsMetadataUsages,
    ),
    dumpGenericsVtables: asBool(raw.dumpGenericsVtables, DEFAULT_CONFIG.dumpGenericsVtables),
    dumpGenericsInterfaces: asBool(raw.dumpGenericsInterfaces, DEFAULT_CONFIG.dumpGenericsInterfaces),
    methodBodyMode: (typeof raw.methodBodyMode === "string"
      ? raw.methodBodyMode
      : DEFAULT_CONFIG.methodBodyMode) as MethodBodyMode,
    dumpDisassembly: asBool(raw.dumpDisassembly, DEFAULT_CONFIG.dumpDisassembly),
    dumpPseudoC: asBool(raw.dumpPseudoC, DEFAULT_CONFIG.dumpPseudoC),
    assemblyTarget: pick(raw.assemblyTarget, raw.dumpDisassemblyTarget, DEFAULT_CONFIG.assemblyTarget, asNumber),
    csharpTarget: pick(raw.csharpTarget, raw.dumpPseudoCTarget, DEFAULT_CONFIG.csharpTarget, asNumber),
    csharpInteractive: pick(
      raw.csharpInteractive,
      raw.dumpPseudoCInteractive,
      DEFAULT_CONFIG.csharpInteractive,
      asBool,
    ),
    csharpMaxBodySize: pick(
      raw.csharpMaxBodySize,
      raw.dumpPseudoCMaxBodySize,
      DEFAULT_CONFIG.csharpMaxBodySize,
      asNumber,
    ),
    csharpIncludeAssemblies: asStringList(
      raw.csharpIncludeAssemblies ?? raw.dumpPseudoCIncludeAssemblies,
    ),
    csharpIncludeTypes: asStringList(raw.csharpIncludeTypes ?? raw.dumpPseudoCIncludeTypes),
    assemblyHexBytes: pick(
      raw.assemblyHexBytes,
      raw.dumpDisassemblyHexBytes,
      DEFAULT_CONFIG.assemblyHexBytes,
      asBool,
    ),
    assemblyFieldNames: pick(
      raw.assemblyFieldNames,
      raw.dumpDisassemblyFieldNames,
      DEFAULT_CONFIG.assemblyFieldNames,
      asBool,
    ),
    assemblyAnnotations: pick(
      raw.assemblyAnnotations,
      raw.dumpDisassemblyAnnotations,
      DEFAULT_CONFIG.assemblyAnnotations,
      asBool,
    ),
    assemblyCfg: pick(raw.assemblyCfg, raw.dumpDisassemblyCfg, DEFAULT_CONFIG.assemblyCfg, asBool),
    maxAssemblyInstructions: pick(
      raw.maxAssemblyInstructions,
      raw.maxDisassemblyInstructions,
      DEFAULT_CONFIG.maxAssemblyInstructions,
      asNumber,
    ),
    generateCppScaffold: asBool(raw.generateCppScaffold, DEFAULT_CONFIG.generateCppScaffold),
    mangleNames: asBool(raw.mangleNames, DEFAULT_CONFIG.mangleNames),
    enhancedIdaMetadata: asBool(raw.enhancedIdaMetadata, DEFAULT_CONFIG.enhancedIdaMetadata),
    generateUnityHeaders: asBool(raw.generateUnityHeaders, DEFAULT_CONFIG.generateUnityHeaders),
    compilerLayout:
      typeof raw.compilerLayout === "string" ? raw.compilerLayout : DEFAULT_CONFIG.compilerLayout,
    useTopologicalSort: asBool(raw.useTopologicalSort, DEFAULT_CONFIG.useTopologicalSort),
    codm: asBool(raw.codm, DEFAULT_CONFIG.codm),
    dumpStaticFieldMetadata: asBool(raw.dumpStaticFieldMetadata, DEFAULT_CONFIG.dumpStaticFieldMetadata),
    dumpFieldRvaData: asBool(raw.dumpFieldRvaData, DEFAULT_CONFIG.dumpFieldRvaData),
    maxFieldRvaDumpBytes: asNumber(raw.maxFieldRvaDumpBytes, DEFAULT_CONFIG.maxFieldRvaDumpBytes),
    ilRecovery: asBool(raw.ilRecovery, DEFAULT_CONFIG.ilRecovery),
    pseudoCEngine:
      typeof raw.pseudoCEngine === "string" ? raw.pseudoCEngine : DEFAULT_CONFIG.pseudoCEngine,
    annotateRecovery: asBool(raw.annotateRecovery, DEFAULT_CONFIG.annotateRecovery),
  };

  return withMethodBodyMode(cleaned, methodBodyModeOf(cleaned));
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
