<script lang="ts">
  import lottie from "lottie-web";
  import { onMount } from "svelte";
  import { Icon } from "noph-ui/icons";
  import { logs, outputPath, resetAll, t } from "$lib/stores";

  let lottieContainer: HTMLDivElement;
  let headerVisible = $state(false);
  let pathVisible = $state(false);
  let logsVisible = $state(false);
  let actionsVisible = $state(false);

  function getLogStyle(log: string): { color: string; showCheck: boolean } {
    if (log.startsWith("Done!")) return { color: "var(--np-color-primary)", showCheck: true };
    if (log.includes("generated")) return { color: "var(--np-color-tertiary)", showCheck: true };
    return { color: "var(--np-color-on-surface-variant)", showCheck: false };
  }

  onMount(() => {
    lottie.loadAnimation({
      container: lottieContainer,
      path: "/success.json",
      loop: false,
      autoplay: true,
      rendererSettings: { preserveAspectRatio: "xMidYMid slice" },
    });

    setTimeout(() => (headerVisible = true), 100);
    setTimeout(() => (pathVisible = true), 220);
    setTimeout(() => (logsVisible = true), 340);
    setTimeout(() => (actionsVisible = true), 460);
  });
</script>

<div class="flex flex-col h-full p-4 overflow-hidden animate-slide-up">
  <div class="flex-1 min-h-0 overflow-y-auto flex flex-col gap-4">
    <div
      class="transition-all duration-400 ease-out"
      style:opacity={headerVisible ? 1 : 0}
      style:transform="translateY({headerVisible ? 0 : 20}px)"
    >
      <div class="m3-surface p-8 text-center space-y-4">
        <div
          class="size-28 mx-auto rounded-3xl flex items-center justify-center"
          style="background: var(--np-color-tertiary-container);"
        >
          <div bind:this={lottieContainer} class="size-20"></div>
        </div>
        <h2 class="text-xl font-bold" style="color: var(--np-color-tertiary);">{$t.dump_complete}</h2>
      </div>
    </div>

    <div
      class="transition-all duration-400 ease-out"
      style:opacity={pathVisible ? 1 : 0}
      style:transform="translateY({pathVisible ? 0 : 15}px)"
    >
      <div class="m3-surface p-4">
        <div class="flex items-center gap-3">
          <div class="m3-icon-tile size-10 rounded-full">
            <Icon>folder_open</Icon>
          </div>
          <div class="min-w-0 flex-1">
            <p class="m3-label">{$t.label_output_dir}</p>
            <p class="text-sm font-mono truncate select-text">{$outputPath}</p>
          </div>
        </div>
      </div>
    </div>

    {#if $logs.length > 0}
      <div
        class="transition-all duration-400 ease-out"
        style:opacity={logsVisible ? 1 : 0}
        style:transform="translateY({logsVisible ? 0 : 15}px)"
      >
        <div class="m3-surface p-4 space-y-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <Icon>terminal</Icon>
              <span class="text-xs font-semibold uppercase tracking-widest m3-secondary">{$t.label_log}</span>
            </div>
            <span class="m3-badge m3-badge-muted">{$logs.length} entries</span>
          </div>
          <div class="m3-log-panel max-h-64 overflow-y-auto p-3 space-y-1">
            {#each $logs as log}
              {@const style = getLogStyle(log)}
              <div class="flex items-start gap-2 py-0.5">
                {#if style.showCheck}
                  <Icon --np-icon-size="14px" style="color: {style.color}; margin-top: 2px;">check</Icon>
                {:else}
                  <span class="size-1.5 rounded-full mt-1.5 shrink-0 m3-secondary opacity-40"></span>
                {/if}
                <p class="text-[11px] font-mono break-all select-text leading-relaxed" style="color: {style.color};">{log}</p>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>

  <div
    class="shrink-0 pt-3 pb-1 transition-all duration-400 ease-out"
    style:opacity={actionsVisible ? 1 : 0}
    style:transform="translateY({actionsVisible ? 0 : 10}px)"
  >
    <button type="button" class="m3-start-dump m3-start-dump--success" onclick={resetAll}>
      <Icon>refresh</Icon>
      <span>{$t.new_dump}</span>
    </button>
  </div>
</div>
