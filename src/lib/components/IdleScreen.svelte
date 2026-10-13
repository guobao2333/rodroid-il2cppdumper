<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";
  import { type } from "@tauri-apps/plugin-os";
  import { copyFile, mkdir } from "@tauri-apps/plugin-fs";
  import { appDataDir, join } from "@tauri-apps/api/path";
  import { Button, IconButton, ChipSet, AssistChip } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  import { get } from "svelte/store";
  import { beginDump } from "$lib/dumpRunner";
  import {
    binaryPath,
    metadataPath,
    binaryInfo,
    config,
    configDialogOpen,
    classPickerOpen,
    typeTree,
    typeTreeLoading,
    typeTreeError,
    t,
  } from "$lib/stores";
  import type { BinaryInfo, TypeTreeAssembly } from "$lib/types";
  import { methodBodyModeOf } from "$lib/types";
  import AnimatedExpand from "./AnimatedExpand.svelte";
  import PathInput from "./PathInput.svelte";

  let binaryValue = $state("");
  let metadataValue = $state("");

  $effect(() => {
    binaryValue = $binaryPath;
  });
  $effect(() => {
    metadataValue = $metadataPath;
  });

  let showBinaryInfo = $derived(
    !!$binaryInfo && ($binaryInfo.format.length > 0 || ($binaryInfo.unity_version?.length ?? 0) > 0)
  );

  let selectedAssemblies = $derived($config.csharpIncludeAssemblies?.length ?? 0);
  let selectedTypes = $derived($config.csharpIncludeTypes?.length ?? 0);
  // Class filter only applies to Asm / C# method bodies.
  let classFilterVisible = $derived(methodBodyModeOf($config) !== "off");

  async function loadTypeTree(path: string) {
    typeTreeLoading.set(true);
    typeTreeError.set("");
    typeTree.set([]);
    try {
      const codm = get(config).codm;
      const raw = await invoke<string>("list_type_tree", { metadataPath: path, codm });
      const parsed = JSON.parse(raw) as TypeTreeAssembly[];
      typeTree.set(Array.isArray(parsed) ? parsed : []);
    } catch (e: any) {
      typeTreeError.set(e?.message || String(e) || "Failed to load class tree");
      typeTree.set([]);
    } finally {
      typeTreeLoading.set(false);
    }
  }

  async function pickBinary() {
    const osType = await type();
    const file = await open({
      multiple: false,
      ...(osType === "ios" ? {} : { filters: [{ name: "IL2CPP Binary", extensions: ["so", "dll", "exe", "dylib", "nso", "wasm", "*"] }] }),
    });
    if (file) {
      let finalPath = file;
      if (osType === "ios") {
        try {
          const appData = await appDataDir();
          await mkdir(appData, { recursive: true }).catch(() => {});
          const targetPath = await join(appData, "target_binary.bin");
          await copyFile(file, targetPath);
          finalPath = targetPath;
        } catch (e: any) {
          console.error("iOS copy error (binary):", e);
          alert("iOS Copy Error: " + (e?.message || e));
        }
      }

      binaryPath.set(finalPath);
      try {
        const info: BinaryInfo = await invoke("detect_binary", { path: finalPath });
        binaryInfo.set(info);
      } catch {
        binaryInfo.set(null);
      }
    }
  }

  async function pickMetadata() {
    const osType = await type();
    const file = await open({
      multiple: false,
      ...(osType === "ios" ? {} : { filters: [{ name: "Metadata", extensions: ["dat", "*"] }] }),
    });
    if (file) {
      let finalPath = file;
      if (osType === "ios") {
        try {
          const appData = await appDataDir();
          await mkdir(appData, { recursive: true }).catch(() => {});
          const targetPath = await join(appData, "target_metadata.dat");
          await copyFile(file, targetPath);
          finalPath = targetPath;
        } catch (e: any) {
          console.error("iOS copy error (metadata):", e);
          alert("iOS Copy Error: " + (e?.message || e));
        }
      }
      metadataPath.set(finalPath);
      await loadTypeTree(finalPath);
    }
  }

  function onBinaryChange(val: string) {
    binaryPath.set(val);
    if (val) {
      invoke("detect_binary", { path: val })
        .then((info: any) => binaryInfo.set(info))
        .catch(() => binaryInfo.set(null));
    } else {
      binaryInfo.set(null);
    }
  }

  function onMetadataChange(val: string) {
    metadataPath.set(val);
    if (val) {
      void loadTypeTree(val);
    } else {
      typeTree.set([]);
      typeTreeError.set("");
      typeTreeLoading.set(false);
    }
  }

  function startDump() {
    void beginDump();
  }
</script>

<div class="flex flex-col h-full p-4 gap-3 overflow-y-auto">
  <div class="m3-surface p-4 space-y-3 m3-stagger-in" style="animation-delay: 0ms">
    <div class="flex items-center justify-between gap-2">
      <div class="flex items-center gap-2 min-w-0">
        <div class="m3-icon-tile size-7">
          <Icon>description</Icon>
        </div>
        <span class="m3-label">{$t.label_binary}</span>
      </div>
      <IconButton variant="tonal" aria-label={$t.select_binary} onclick={pickBinary}>
        <Icon>folder_open</Icon>
      </IconButton>
    </div>
    <PathInput
      bind:value={binaryValue}
      placeholder="No file selected"
      title={binaryValue || undefined}
      onchange={() => onBinaryChange(binaryValue)}
    />
    <AnimatedExpand show={showBinaryInfo}>
      <ChipSet>
        {#if $binaryInfo?.format}
          <AssistChip label="{$t.label_format}: {$binaryInfo.format}" />
        {/if}
        {#if $binaryInfo?.unity_version && $binaryInfo.unity_version !== "null"}
          <AssistChip label="{$t.label_unity}: {$binaryInfo.unity_version}" />
        {/if}
      </ChipSet>
    </AnimatedExpand>
  </div>

  <div class="m3-surface p-4 space-y-3 m3-stagger-in" style="animation-delay: 80ms">
    <div class="flex items-center justify-between gap-2">
      <div class="flex items-center gap-2 min-w-0">
        <div class="m3-icon-tile size-7">
          <Icon>storage</Icon>
        </div>
        <span class="m3-label">{$t.label_metadata}</span>
      </div>
      <IconButton variant="tonal" aria-label={$t.select_metadata} onclick={pickMetadata}>
        <Icon>folder_open</Icon>
      </IconButton>
    </div>
    <PathInput
      bind:value={metadataValue}
      placeholder="No file selected"
      title={metadataValue || undefined}
      onchange={() => onMetadataChange(metadataValue)}
    />
  </div>

  <AnimatedExpand show={!!$metadataPath && classFilterVisible}>
    <div class="m3-surface p-4 space-y-2 m3-stagger-in" style="animation-delay: 120ms">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 min-w-0">
          <div class="m3-icon-tile size-7">
            <Icon>filter_list</Icon>
          </div>
          <div class="min-w-0">
            <div class="m3-label">{$t.class_picker_open}</div>
            <div class="text-xs m3-secondary truncate">
              {#if $typeTreeLoading}
                {$t.class_picker_loading}
              {:else if $typeTreeError}
                {$typeTreeError}
              {:else if selectedAssemblies === 0 && selectedTypes === 0}
                {$t.class_picker_summary_all}
              {:else}
                {selectedAssemblies} assemblies · {selectedTypes} types
              {/if}
            </div>
          </div>
        </div>
        <Button
          variant="tonal"
          disabled={$typeTreeLoading || !$metadataPath}
          onclick={() => classPickerOpen.set(true)}
        >
          {$t.class_picker_open}
        </Button>
      </div>
      {#if !$typeTreeLoading && !$typeTreeError && $typeTree.length}
        <p class="text-[11px] m3-secondary">{$typeTree.length} assemblies loaded</p>
      {/if}
    </div>
  </AnimatedExpand>

  <div class="m3-stagger-in" style="animation-delay: 200ms">
    <Button variant="tonal" class="w-full" onclick={() => configDialogOpen.set(true)}>
      {#snippet start()}
        <Icon>tune</Icon>
      {/snippet}
      {$t.dump_options}
    </Button>
  </div>

  <div class="flex-1 min-h-4"></div>

  <div class="m3-stagger-in pb-4" style="animation-delay: 240ms">
    <button
      type="button"
      class="m3-start-dump"
      disabled={!$binaryPath || !$metadataPath}
      onclick={startDump}
    >
      <Icon>play_arrow</Icon>
      <span>{$t.start_dump}</span>
    </button>
  </div>
</div>
