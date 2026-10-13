import { logs } from "$lib/stores";

const MAX_LOG_LINES = 1000;

export function appendDumpLog(message: string): void {
  logs.update((lines) => {
    if (lines.length >= MAX_LOG_LINES) {
      return [...lines.slice(-(MAX_LOG_LINES - 1)), message];
    }
    return [...lines, message];
  });
}
