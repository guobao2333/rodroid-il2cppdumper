<script lang="ts">
  import lottie from "lottie-web";
  import { onMount } from "svelte";
  import { Icon } from "noph-ui/icons";
  import { errorMessage, logs, resetAll, resetForNewDump, t } from "$lib/stores";

  let lottieContainer: HTMLDivElement;
  let headerVisible = $state(false);
  let logsVisible = $state(false);
  let actionsVisible = $state(false);

  onMount(() => {
    lottie.loadAnimation({
      container: lottieContainer,
      path: "/error.json",
      loop: false,
      autoplay: true,
      rendererSettings: { preserveAspectRatio: "xMidYMid slice" },
    });

    setTimeout(() => (headerVisible = true), 100);
    setTimeout(() => (logsVisible = true), 250);
    setTimeout(() => (actionsVisible = true), 400);
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
          style="background: var(--np-color-error-container);"
        >
          <div bind:this={lottieContainer} class="size-20"></div>
        </div>
        <h2 class="text-xl font-bold" style="color: var(--np-color-error);">{$t.dump_failed}</h2>
        <div class="m3-error-detail max-w-sm mx-auto">
          <p class="select-text break-all leading-relaxed">{$errorMessage}</p>
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
            <span class="m3-badge m3-badge-error">{$logs.length} entries</span>
          </div>
          <div class="m3-log-panel max-h-64 overflow-y-auto p-3 space-y-1">
            {#each $logs as log}
              <div class="flex items-start gap-2 py-0.5">
                <span
                  class="size-1.5 rounded-full mt-1.5 shrink-0"
                  style="background: {log.startsWith('ERROR') ? 'var(--np-color-error)' : 'var(--np-color-on-surface-variant)'}; opacity: {log.startsWith('ERROR') ? '1' : '0.35'};"
                ></span>
                <p
                  class="text-[11px] font-mono break-all select-text leading-relaxed"
                  style="color: {log.startsWith('ERROR') ? 'var(--np-color-error)' : 'var(--np-color-on-surface-variant)'};"
                >
                  {log}
                </p>
              </div>
            {/each}
          </div>
        </div>
      </div>
    {/if}
  </div>

  <div
    class="shrink-0 pt-3 pb-1 space-y-3 transition-all duration-400 ease-out"
    style:opacity={actionsVisible ? 1 : 0}
    style:transform="translateY({actionsVisible ? 0 : 10}px)"
  >
    <button
      type="button"
      class="m3-start-dump m3-start-dump--error"
      onclick={resetForNewDump}
    >
      <Icon>refresh</Icon>
      <span>{$t.dump_again}</span>
    </button>
    <button
      type="button"
      class="m3-start-dump m3-start-dump--tonal"
      onclick={resetAll}
    >
      <Icon>restart_alt</Icon>
      <span>{$t.try_again}</span>
    </button>
  </div>
</div>
