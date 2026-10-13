import { get } from "svelte/store";
import { invoke } from "@tauri-apps/api/core";
import {
  appState,
  binaryPath,
  config,
  currentScreen,
  elapsedSeconds,
  errorMessage,
  logs,
  metadataPath,
  outputDir,
} from "$lib/stores";
import { normalizeDumperConfig, type DumperConfig } from "$lib/types";
import { setDumpInProgress, isDumpInProgress } from "$lib/dumpState";
import { setupDumpEvents } from "$lib/dumpEvents";

export async function beginDump(): Promise<void> {
  if (isDumpInProgress()) return;

  const bp = get(binaryPath);
  const mp = get(metadataPath);
  if (!bp || !mp) return;

  await setupDumpEvents();

  setDumpInProgress(true);
  logs.set([]);
  elapsedSeconds.set(0);
  appState.set("dumping");
  currentScreen.set("dumping");

  const cfg = normalizeDumperConfig(get(config));
  // Guarantee include lists are real string arrays (never mysteriously lost /
  // stringified) before they hit Rust.
  cfg.csharpIncludeAssemblies = (cfg.csharpIncludeAssemblies ?? [])
    .map((s) => String(s))
    .filter((s) => s.length > 0);
  cfg.csharpIncludeTypes = (cfg.csharpIncludeTypes ?? [])
    .map((s) => String(s))
    .filter((s) => s.length > 0);
  // Persist the normalized shape so localStorage cannot keep stringy numbers /
  // a stale methodBodyMode that the dialog looked like it had applied.
  config.set(cfg);
  const outDir = get(outputDir);

  try {
    await invoke("start_dump", {
      binaryPath: bp,
      metadataPath: mp,
      outputDir: outDir,
      // Send mode + legacy bools + picker filters.
      configJson: JSON.stringify(cfg),
    });
  } catch (e) {
    setDumpInProgress(false);
    errorMessage.set(String(e));
    appState.set("error");
    currentScreen.set("error");
  }
}

