import { writable, derived } from "svelte/store";
import type { AppState, DumperConfig, BinaryInfo, TypeTreeAssembly } from "./types";
import { DEFAULT_CONFIG, normalizeDumperConfig } from "./types";
import { type AppLanguage, type ThemeMode, getTranslations } from "./i18n";
import { setDumpInProgress } from "./dumpState";

export type ScreenState = "idle" | "settings" | "about" | "dumping" | "result" | "error" | "crash" | "splash";

const STORAGE_KEY = "il2cpp_dumper_prefs";

interface PersistedPrefs {
  themeMode: ThemeMode;
  language: AppLanguage;
  config: DumperConfig;
  outputDir: string;
}

function loadPrefs(): PersistedPrefs {
  const defaults = defaultPrefs();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Partial<PersistedPrefs>;
      return {
        ...defaults,
        ...parsed,
        config: normalizeDumperConfig({ ...DEFAULT_CONFIG, ...parsed.config }),
      };
    }
  } catch {}
  return defaults;
}

function defaultPrefs(): PersistedPrefs {
  return {
    themeMode: "system",
    language: "en",
    config: { ...DEFAULT_CONFIG },
    outputDir: "IL2CppDumper",
  };
}

function savePrefs(prefs: Partial<PersistedPrefs>) {
  try {
    const current = loadPrefs();
    const next = { ...current, ...prefs };
    if (next.config) {
      next.config = normalizeDumperConfig(next.config);
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {}
}

const saved = loadPrefs();

export const themeMode = writable<ThemeMode>(saved.themeMode);
export const language = writable<AppLanguage>(saved.language);
export const config = writable<DumperConfig>({ ...DEFAULT_CONFIG, ...saved.config });
export const configDialogOpen = writable(false);
export const outputDir = writable<string>(saved.outputDir);

themeMode.subscribe(v => savePrefs({ themeMode: v }));
language.subscribe(v => savePrefs({ language: v }));
config.subscribe(v => savePrefs({ config: v }));
outputDir.subscribe(v => savePrefs({ outputDir: v }));

export const t = derived(language, ($lang) => getTranslations($lang));

export const defaultOutputDir = writable<string>("IL2CppDumper");
export const currentScreen = writable<ScreenState>("splash");
export const crashLog = writable<string>("");
export const appState = writable<AppState>("idle");
export const logs = writable<string[]>([]);
export const binaryPath = writable("");
export const metadataPath = writable("");
export const binaryInfo = writable<BinaryInfo | null>(null);
export const typeTree = writable<TypeTreeAssembly[]>([]);
export const typeTreeLoading = writable(false);
export const typeTreeError = writable("");
export const classSearchQuery = writable("");
export const classPickerOpen = writable(false);
export const outputPath = writable("");
export const errorMessage = writable("");
export const inputRequest = writable<string | null>(null);
export const elapsedSeconds = writable(0);

export function resetAll() {
  setDumpInProgress(false);
  appState.set("idle");
  currentScreen.set("idle");
  logs.set([]);
  binaryPath.set("");
  metadataPath.set("");
  binaryInfo.set(null);
  typeTree.set([]);
  typeTreeLoading.set(false);
  typeTreeError.set("");
  classSearchQuery.set("");
  classPickerOpen.set(false);
  outputPath.set("");
  errorMessage.set("");
  inputRequest.set(null);
  elapsedSeconds.set(0);
  crashLog.set("");
}

export function resetForNewDump() {
  setDumpInProgress(false);
  appState.set("idle");
  currentScreen.set("idle");
  logs.set([]);
  outputPath.set("");
  errorMessage.set("");
  inputRequest.set(null);
  elapsedSeconds.set(0);
}

export function applyTheme(mode: ThemeMode) {
  const html = document.documentElement;
  const resolved =
    mode === "system"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light"
      : mode;
  html.setAttribute("data-theme", resolved);
  html.style.colorScheme = resolved;
}
