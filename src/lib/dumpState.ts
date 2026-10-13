let dumpInProgress = false;

export function isDumpInProgress(): boolean {
  return dumpInProgress;
}

export function setDumpInProgress(value: boolean): void {
  dumpInProgress = value;
}
