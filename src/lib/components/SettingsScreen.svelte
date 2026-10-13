<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { Button, IconButton, SegmentedButton, List, ListItem, Divider } from "noph-ui";
  import PathInput from "./PathInput.svelte";
  import { BrightnessMediumIcon, DarkModeIcon, LightModeIcon, LanguageIcon, Icon, CheckIcon } from "noph-ui/icons";
  import { themeMode, language, outputDir, defaultOutputDir, currentScreen, t, applyTheme } from "$lib/stores";
  import { LANGUAGES, type ThemeMode } from "$lib/i18n";

  let langExpanded = $state(false);
  let outputValue = $state("");

  $effect(() => {
    outputValue = $outputDir;
  });

  function setTheme(mode: ThemeMode) {
    themeMode.set(mode);
    applyTheme(mode);
  }

  async function pickOutputDir() {
    const dir = await open({ directory: true, title: "Select Output Directory" });
    if (dir) outputDir.set(dir);
  }
</script>

{#snippet darkIcon()}
  <DarkModeIcon />
{/snippet}
{#snippet systemIcon()}
  <BrightnessMediumIcon />
{/snippet}
{#snippet lightIcon()}
  <LightModeIcon />
{/snippet}

<div class="flex flex-col h-full overflow-y-auto animate-slide-up m3-app">
  <div class="m3-top-bar flex items-center gap-3 px-5 py-4">
    <IconButton aria-label="Back" onclick={() => currentScreen.set("idle")}>
      <Icon>arrow_back</Icon>
    </IconButton>
    <h2 class="text-lg font-semibold">{$t.settings}</h2>
  </div>

  <div class="flex-1 overflow-y-auto p-4 space-y-3">
    <div class="m3-surface p-4 space-y-4">
      <div class="flex items-center gap-2">
        <Icon>brightness_medium</Icon>
        <span class="m3-label">{$t.label_appearance}</span>
      </div>
      <p class="text-sm m3-secondary">{$t.label_theme}</p>
      <SegmentedButton
        name="theme-mode"
        options={[
          {
            label: $t.theme_dark,
            icon: darkIcon,
            selected: $themeMode === "dark",
            onclick: () => setTheme("dark"),
          },
          {
            label: $t.theme_system,
            icon: systemIcon,
            selected: $themeMode === "system",
            onclick: () => setTheme("system"),
          },
          {
            label: $t.theme_light,
            icon: lightIcon,
            selected: $themeMode === "light",
            onclick: () => setTheme("light"),
          },
        ]}
      />
    </div>

    <div class="m3-surface overflow-hidden">
      <button type="button" class="w-full text-left cursor-pointer p-4" onclick={() => (langExpanded = !langExpanded)}>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <LanguageIcon />
            <div>
              <p class="m3-label">{$t.label_language}</p>
              <p class="text-sm m3-secondary">{LANGUAGES.find((l) => l.code === $language)?.displayName}</p>
            </div>
          </div>
          <Icon class={langExpanded ? "rotate-180" : ""}>expand_more</Icon>
        </div>
      </button>
      {#if langExpanded}
        <Divider />
        <List>
          {#each LANGUAGES as lang}
            {@const isActive = $language === lang.code}
            <ListItem
              variant="button"
              selected={isActive}
              onclick={() => {
                language.set(lang.code);
                langExpanded = false;
              }}
            >
              {#snippet start()}
                <LanguageIcon />
              {/snippet}
              {lang.displayName}
              {#snippet end()}
                {#if isActive}
                  <CheckIcon />
                {/if}
              {/snippet}
            </ListItem>
          {/each}
        </List>
      {/if}
    </div>

    <div class="m3-surface p-4 space-y-3">
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-2 min-w-0">
          <Icon>folder_open</Icon>
          <span class="m3-label">{$t.label_output_dir}</span>
        </div>
        <IconButton variant="tonal" aria-label="Browse output directory" onclick={pickOutputDir}>
          <Icon>upload_file</Icon>
        </IconButton>
      </div>
      <p class="text-xs m3-secondary">{$t.setting_output_dir_desc}</p>
      <PathInput
        bind:value={outputValue}
        title={outputValue || undefined}
        onchange={() => outputDir.set(outputValue)}
      />
      {#if $outputDir !== $defaultOutputDir}
        <Button variant="text" class="w-full" onclick={() => outputDir.set($defaultOutputDir)}>
          {#snippet start()}
            <Icon>restart_alt</Icon>
          {/snippet}
          {$t.output_reset}
        </Button>
      {/if}
    </div>

    <Button variant="tonal" class="w-full" onclick={() => currentScreen.set("about")}>
      {#snippet start()}
        <Icon>info</Icon>
      {/snippet}
      {$t.label_about}
    </Button>
  </div>
</div>
