<script lang="ts">
  import lottie from "lottie-web";
  import { onMount } from "svelte";
  import { Button } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  let { crashLog, onrestart }: { crashLog: string; onrestart: () => void } = $props();

  let lottieContainer: HTMLDivElement;
  let headerVisible = $state(false);
  let detailsVisible = $state(false);
  let actionsVisible = $state(false);

  let crashLines = $derived(crashLog.split("\n"));
  let exceptionLine = $derived(
    crashLines.find((l) => l.includes("panicked") || l.includes("Error") || l.includes("Exception") || l.includes("Panic"))
  );
  let threadLine = $derived(crashLines.find((l) => l.startsWith("Thread:") || l.startsWith("thread")));

  function copyLog() {
    navigator.clipboard.writeText(crashLog);
  }

  onMount(() => {
    lottie.loadAnimation({
      container: lottieContainer,
      path: "/error.json",
      loop: false,
      autoplay: true,
      rendererSettings: { preserveAspectRatio: "xMidYMid slice" },
    });

    setTimeout(() => (headerVisible = true), 100);
    setTimeout(() => (detailsVisible = true), 250);
    setTimeout(() => (actionsVisible = true), 400);
  });
</script>

<div class="flex flex-col h-full overflow-y-auto p-4 gap-4 animate-slide-up">
  <div
    class="transition-all duration-400 ease-out"
    style:opacity={headerVisible ? 1 : 0}
    style:transform="translateY({headerVisible ? 0 : 20}px)"
  >
    <div class="m3-surface p-8 text-center space-y-4">
      <div
        class="size-24 mx-auto rounded-3xl flex items-center justify-center"
        style="background: var(--np-color-error-container);"
      >
        <div bind:this={lottieContainer} class="size-20"></div>
      </div>
      <h2 class="text-xl font-bold" style="color: var(--np-color-error);">Application Crashed</h2>
      <p class="text-sm max-w-xs mx-auto m3-secondary">
        An unexpected error occurred. You can copy the crash report and restart.
      </p>
    </div>
  </div>

  {#if exceptionLine || threadLine}
    <div
      class="transition-all duration-400 ease-out"
      style:opacity={detailsVisible ? 1 : 0}
      style:transform="translateY({detailsVisible ? 0 : 15}px)"
    >
      <div class="m3-surface overflow-hidden">
        {#if exceptionLine}
          <div class="flex items-center gap-3 p-4">
            <div class="m3-icon-tile size-9 rounded-full" style="background: var(--np-color-error-container); color: var(--np-color-error);">
              <Icon>error</Icon>
            </div>
            <div class="min-w-0">
              <p class="text-[10px] uppercase tracking-widest font-semibold" style="color: var(--np-color-error);">Exception</p>
              <p class="text-xs font-mono break-all">{exceptionLine.trim()}</p>
            </div>
          </div>
        {/if}
        {#if exceptionLine && threadLine}
          <hr style="border: none; height: 1px; background: var(--np-color-outline-variant);" />
        {/if}
        {#if threadLine}
          <div class="flex items-center gap-3 p-4">
            <div class="m3-icon-tile size-9 rounded-full" style="background: var(--np-color-secondary-container); color: var(--np-color-on-secondary-container);">
              <Icon>warning</Icon>
            </div>
            <div class="min-w-0">
              <p class="text-[10px] uppercase tracking-widest font-semibold" style="color: var(--np-color-secondary);">Thread</p>
              <p class="text-xs font-mono break-all">{threadLine.replace("Thread:", "").replace("thread", "").trim()}</p>
            </div>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <div
    class="transition-all duration-400 ease-out"
    style:opacity={detailsVisible ? 1 : 0}
    style:transform="translateY({detailsVisible ? 0 : 15}px)"
  >
    <div class="m3-surface p-4 space-y-3">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <Icon>error</Icon>
          <span class="text-xs font-semibold uppercase tracking-widest" style="color: var(--np-color-error);">Stack Trace</span>
        </div>
        <span class="m3-badge m3-badge-error">{crashLines.length} lines</span>
      </div>
      <div class="m3-log-panel max-h-64 overflow-y-auto overflow-x-auto p-3">
        <pre class="text-[11px] font-mono leading-relaxed select-text whitespace-pre m3-secondary">{crashLog}</pre>
      </div>
    </div>
  </div>

  <div
    class="flex gap-3 pt-2 transition-all duration-400 ease-out"
    style:opacity={actionsVisible ? 1 : 0}
    style:transform="translateY({actionsVisible ? 0 : 10}px)"
  >
    <Button variant="outlined" class="flex-1" onclick={copyLog}>
      {#snippet start()}
        <Icon>content_copy</Icon>
      {/snippet}
      Copy Log
    </Button>
    <Button
      variant="filled"
      class="flex-1"
      --np-filled-button-container-color="var(--np-color-error)"
      --np-filled-button-label-text-color="var(--np-color-on-error)"
      onclick={onrestart}
    >
      {#snippet start()}
        <Icon>restart_alt</Icon>
      {/snippet}
      Restart
    </Button>
  </div>
</div>
