import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  appState,
  crashLog,
  currentScreen,
  errorMessage,
  inputRequest,
  outputPath,
} from "$lib/stores";
import type { DumpCompleteEvent, InputRequestEvent } from "$lib/types";
import { appendDumpLog } from "$lib/dumpLogs";
import { setDumpInProgress } from "$lib/dumpState";

let unlisteners: UnlistenFn[] = [];
let setupPromise: Promise<void> | null = null;

async function registerDumpEvents(): Promise<void> {
  if (unlisteners.length > 0) return;

  unlisteners.push(
    await listen<{ message: string }>("dump-log", (event) => {
      appendDumpLog(event.payload.message);
    }),
    await listen<DumpCompleteEvent>("dump-complete", (event) => {
      setDumpInProgress(false);
      const payload = event.payload;
      if (payload.success) {
        outputPath.set(payload.output_path ?? "");
        appState.set("result");
        currentScreen.set("result");
      } else {
        errorMessage.set(payload.error_message ?? "Dump failed");
        appState.set("error");
        currentScreen.set("error");
      }
    }),
    await listen<InputRequestEvent>("dump-input-request", (event) => {
      inputRequest.set(event.payload.prompt_type);
    }),
    await listen<{ crash_log: string }>("dump-crash", (event) => {
      setDumpInProgress(false);
      crashLog.set(event.payload.crash_log);
      currentScreen.set("crash");
    }),
  );
}

/** Register Tauri dump listeners once; safe to await from multiple callers. */
export function setupDumpEvents(): Promise<void> {
  if (!setupPromise) {
    setupPromise = registerDumpEvents().catch((err) => {
      setupPromise = null;
      throw err;
    });
  }
  return setupPromise;
}

export function teardownDumpEvents(): void {
  for (const unlisten of unlisteners) unlisten();
  unlisteners = [];
  setupPromise = null;
}
