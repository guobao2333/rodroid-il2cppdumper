<script lang="ts">
  import { CircularProgress, LinearProgress } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  import { logs, elapsedSeconds, t } from "$lib/stores";
  import { onMount, onDestroy, tick } from "svelte";

  let logContainer: HTMLDivElement;
  let timer: ReturnType<typeof setInterval>;

  onMount(() => {
    timer = setInterval(() => elapsedSeconds.update((n) => n + 1), 1000);
  });

  onDestroy(() => {
    if (timer) clearInterval(timer);
  });

  function formatTime(s: number): string {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }

  function getLogStyle(log: string): { color: string; icon: string } {
    if (log.startsWith("ERROR")) return { color: "var(--np-color-error)", icon: "error" };
    if (log.startsWith("Done!")) return { color: "var(--np-color-primary)", icon: "check_circle" };
    if (log.includes("generated")) return { color: "var(--np-color-tertiary)", icon: "check" };
    if (log.startsWith("Warning")) return { color: "var(--np-color-secondary)", icon: "warning" };
    if (log.includes("Detected") || log.includes("Found") || log.includes("Registration"))
      return { color: "var(--np-color-primary)", icon: "info" };
    return { color: "var(--np-color-on-surface-variant)", icon: "terminal" };
  }

  $effect(() => {
    if ($logs.length > 0 && logContainer) {
      tick().then(() => {
        logContainer.scrollTop = logContainer.scrollHeight;
      });
    }
  });
</script>

<div class="flex flex-col h-full p-4 gap-3 animate-slide-up">
  <div class="m3-surface p-4 space-y-4">
    <div class="flex items-center justify-between gap-3">
      <div class="flex items-center gap-3 min-w-0">
        <CircularProgress fourColor indeterminate --np-circular-progress-size="2.5rem" />
        <div class="min-w-0">
          <p class="text-sm font-semibold">{$t.status_processing}</p>
          <p class="text-xs m3-secondary">{$logs.length} operations</p>
        </div>
      </div>
      <div
        class="px-3 py-1.5 rounded-xl font-mono shrink-0"
        style="background: var(--np-color-surface-container-highest);"
      >
        <span class="text-sm font-bold tabular-nums" style="color: var(--np-color-primary);">{formatTime($elapsedSeconds)}</span>
      </div>
    </div>
    <LinearProgress fourColor indeterminate />
  </div>

  <div class="m3-surface flex-1 flex flex-col min-h-0 overflow-hidden">
    <div class="py-3 px-4 border-b flex items-center justify-between" style="border-color: var(--np-color-outline-variant);">
      <div class="flex items-center gap-2">
        <span class="size-2 rounded-full animate-pulse" style="background: var(--np-color-tertiary);"></span>
        <span class="text-xs font-semibold uppercase tracking-widest m3-secondary">Live Output</span>
      </div>
      <span class="m3-badge m3-badge-muted">{$logs.length} lines</span>
    </div>
    <div bind:this={logContainer} class="flex-1 overflow-y-auto p-3 space-y-px">
      {#each $logs as log, i}
        {@const style = getLogStyle(log)}
        <div class="flex items-start gap-2 py-0.5 px-2 -mx-2 rounded-lg">
          <span class="text-[10px] font-mono mt-1 w-5 text-right shrink-0 tabular-nums m3-secondary opacity-60">{i + 1}</span>
          <span class="mt-0.5 shrink-0" style="color: {style.color};"><Icon>{style.icon}</Icon></span>
          <span class="text-[13px] font-mono break-all leading-relaxed" style="color: {style.color};">{log}</span>
        </div>
      {/each}
      {#if $logs.length === 0}
        <div class="flex items-center justify-center h-full">
          <p class="text-sm m3-secondary">Waiting for output...</p>
        </div>
      {/if}
    </div>
  </div>
</div>
