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
  const outDir = get(outputDir);
  // Engine wire format uses methodBodyMode only; omit legacy mode bools.
  const { dumpDisassembly: _disasm, dumpPseudoC: _pseudo, ...engineCfg } = cfg;

  try {
    await invoke("start_dump", {
      binaryPath: bp,
      metadataPath: mp,
      outputDir: outDir,
      configJson: JSON.stringify(engineCfg),
    });
  } catch (e) {
    setDumpInProgress(false);
    errorMessage.set(String(e));
    appState.set("error");
    currentScreen.set("error");
  }
}

