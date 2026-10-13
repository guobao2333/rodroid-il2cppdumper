<script lang="ts">
  import { downloadDir, documentDir, join } from "@tauri-apps/api/path";
  import { type } from "@tauri-apps/plugin-os";
  import { onMount } from "svelte";
  import { IconButton } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  import { get } from "svelte/store";
  import {
    config, configDialogOpen, classPickerOpen, currentScreen, themeMode, applyTheme, t, crashLog, resetAll,
    outputDir, defaultOutputDir, typeTree, typeTreeLoading, typeTreeError,
  } from "$lib/stores";
  import { DEFAULT_CONFIG, type DumperConfig } from "$lib/types";
  import ConfigDialog from "$lib/components/ConfigDialog.svelte";
  import ClassPickerDialog from "$lib/components/ClassPickerDialog.svelte";
  import IdleScreen from "$lib/components/IdleScreen.svelte";
  import DumpingScreen from "$lib/components/DumpingScreen.svelte";
  import ResultScreen from "$lib/components/ResultScreen.svelte";
  import ErrorScreen from "$lib/components/ErrorScreen.svelte";
  import SettingsScreen from "$lib/components/SettingsScreen.svelte";
  import AboutScreen from "$lib/components/AboutScreen.svelte";
  import SplashScreen from "$lib/components/SplashScreen.svelte";
  import CrashScreen from "$lib/components/CrashScreen.svelte";
  import InputDialog from "$lib/components/InputDialog.svelte";
  import { setupDumpEvents } from "$lib/dumpEvents";

  let currentOs = $state("Desktop");
  let draftConfig = $state<DumperConfig>({ ...DEFAULT_CONFIG });

  $effect(() => {
    if ($configDialogOpen) {
      draftConfig = { ...DEFAULT_CONFIG, ...get(config) };
    }
  });

  function confirmConfigDialog() {
    config.set({ ...DEFAULT_CONFIG, ...draftConfig });
    configDialogOpen.set(false);
  }

  function cancelConfigDialog() {
    configDialogOpen.set(false);
  }

  onMount(async () => {
    try {
      const osType = await type();
      let baseDir = "";

      if (osType === "ios") {
        currentOs = "iOS";
        baseDir = await documentDir();
      } else if (osType === "android") {
        currentOs = "Android";
        baseDir = await documentDir();
      } else {
        currentOs = "Desktop";
        baseDir = await downloadDir();
      }

      const defaultPath = await join(baseDir, "IL2CppDumper");
      defaultOutputDir.set(defaultPath);
      let currentOutDir = "IL2CppDumper";
      outputDir.subscribe(v => currentOutDir = v)();
      if (currentOutDir === "IL2CppDumper") {
        outputDir.set(defaultPath);
      }
    } catch (e) {
      // Ignore if API fails
    }

    applyTheme($themeMode);

    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", () => { if ($themeMode === "system") applyTheme("system"); });

    await setupDumpEvents();
  });

  function handleSplashFinished() {
    currentScreen.set("idle");
  }

  function handleCrashRestart() {
    crashLog.set("");
    resetAll();
  }
</script>

<svelte:head>
  <title>Rodroid IL2CPP Dumper</title>
</svelte:head>

{#if $currentScreen === "splash"}
  <SplashScreen onfinished={handleSplashFinished} />
{:else if $currentScreen === "crash"}
  <main class="h-[100dvh] flex flex-col overflow-hidden pb-[env(safe-area-inset-bottom)] m3-app">
    <CrashScreen crashLog={$crashLog} onrestart={handleCrashRestart} />
  </main>
{:else}
  <main class="h-[100dvh] flex flex-col overflow-hidden pb-[env(safe-area-inset-bottom)] m3-app">
    {#if $currentScreen !== "settings" && $currentScreen !== "about"}
      <header class="m3-top-bar shrink-0 px-5 py-3" data-tauri-drag-region>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="m3-icon-tile size-9">
              <Icon>code</Icon>
            </div>
            <div>
              <h1 class="text-sm font-semibold tracking-tight">{$t.app_name}</h1>
              <p class="text-[10px] m3-secondary">v8.0 {currentOs}</p>
            </div>
          </div>
          <div class="flex items-center gap-2">
            {#if $currentScreen === "dumping"}
              <span class="m3-badge m3-badge-primary">
                <span class="size-1.5 rounded-full animate-pulse" style="background: var(--np-color-primary);"></span>
                {$t.status_processing}
              </span>
            {:else if $currentScreen === "result"}
              <span class="m3-badge m3-badge-tertiary">
                <span class="size-1.5 rounded-full" style="background: var(--np-color-tertiary);"></span>
                {$t.dump_complete}
              </span>
            {:else if $currentScreen === "error"}
              <span class="m3-badge m3-badge-error">
                <span class="size-1.5 rounded-full" style="background: var(--np-color-error);"></span>
                {$t.dump_failed}
              </span>
            {/if}
            {#if $currentScreen === "idle"}
              <IconButton aria-label="Settings" onclick={() => currentScreen.set("settings")}>
                <Icon>settings</Icon>
              </IconButton>
            {/if}
          </div>
        </div>
      </header>
    {/if}

    <div class="flex-1 min-h-0">
      {#if $currentScreen === "idle"}
        <IdleScreen />
      {:else if $currentScreen === "dumping"}
        <DumpingScreen />
      {:else if $currentScreen === "result"}
        <ResultScreen />
      {:else if $currentScreen === "error"}
        <ErrorScreen />
      {:else if $currentScreen === "settings"}
        <SettingsScreen />
      {:else if $currentScreen === "about"}
        <AboutScreen />
      {/if}
    </div>
  </main>

  <InputDialog />
  {#if $configDialogOpen}
    <ConfigDialog bind:config={draftConfig} onconfirm={confirmConfigDialog} oncancel={cancelConfigDialog} />
  {/if}
  {#if $classPickerOpen}
    <ClassPickerDialog
      typeTree={$typeTree}
      loading={$typeTreeLoading}
      error={$typeTreeError}
      onclose={() => classPickerOpen.set(false)}
    />
  {/if}
{/if}
