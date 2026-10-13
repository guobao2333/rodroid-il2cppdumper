<script lang="ts">
  import lottie from "lottie-web";
  import { onMount } from "svelte";
  import { IconButton, List, ListItem, Divider, AssistChip, ChipSet } from "noph-ui";
  import { Icon, GroupsIcon, SendIcon, ArrowForwardIcon } from "noph-ui/icons";
  import { currentScreen, t } from "$lib/stores";
  import { openUrl as tauriOpenUrl } from "@tauri-apps/plugin-opener";

  const LINKS = {
    channel1: "https://t.me/+WmudnO0-xoNhMDQ8",
    channel2: "https://t.me/+WLLFw3pr-aVmMjBk",
    group: "https://t.me/+QylrYL1GNsJiYjc0",
    bugs: "https://t.me/rodroidbugreporter_bot",
  };

  async function openUrl(url: string) {
    try {
      await tauriOpenUrl(url);
    } catch (e) {
      console.error("Failed to open URL", e);
    }
  }

  let logoContainer: HTMLDivElement;
  let likeContainer: HTMLDivElement;

  let heroVisible = $state(false);
  let devVisible = $state(false);
  let linksVisible = $state(false);

  onMount(() => {
    lottie.loadAnimation({
      container: logoContainer,
      path: "/android_logo.json",
      loop: true,
      autoplay: true,
      rendererSettings: { preserveAspectRatio: "xMidYMid slice" },
    });

    lottie.loadAnimation({
      container: likeContainer,
      path: "/like.json",
      loop: false,
      autoplay: true,
      rendererSettings: { preserveAspectRatio: "xMidYMid slice" },
    });

    setTimeout(() => (heroVisible = true), 100);
    setTimeout(() => (devVisible = true), 220);
    setTimeout(() => (linksVisible = true), 340);
  });
</script>

<div class="flex flex-col h-full overflow-y-auto animate-slide-up m3-app">
  <div class="m3-top-bar flex items-center gap-3 px-5 py-4">
    <IconButton aria-label="Back" onclick={() => currentScreen.set("settings")}>
      <Icon>arrow_back</Icon>
    </IconButton>
    <h2 class="text-lg font-semibold">{$t.label_about}</h2>
  </div>

  <div class="flex-1 overflow-y-auto p-4 space-y-4">
    <div
      class="transition-all duration-400 ease-out"
      style:opacity={heroVisible ? 1 : 0}
      style:transform="translateY({heroVisible ? 0 : 20}px)"
    >
      <div class="m3-surface p-8 text-center space-y-4">
        <div
          class="size-32 mx-auto rounded-[32px] flex items-center justify-center overflow-hidden"
          style="background: var(--np-color-tertiary-container);"
        >
          <div bind:this={logoContainer} class="size-[100px]"></div>
        </div>
        <div>
          <h3 class="text-2xl font-bold">{$t.app_name}</h3>
          <div class="mt-2 flex justify-center">
            <ChipSet>
              <AssistChip label={$t.about_version} />
            </ChipSet>
          </div>
        </div>
        <p class="text-sm px-2 m3-secondary">{$t.about_description}</p>
      </div>
    </div>

    <div
      class="transition-all duration-400 ease-out"
      style:opacity={devVisible ? 1 : 0}
      style:transform="translateY({devVisible ? 0 : 15}px)"
    >
      <div class="m3-surface p-4">
        <div class="flex items-center gap-4">
          <Icon>code</Icon>
          <div>
            <p class="m3-label leading-tight">{$t.about_developer}</p>
            <p class="text-[15px] font-semibold mt-0.5">Rodroid Mods</p>
            <p class="text-sm m3-secondary">{$t.about_powered_by}</p>
          </div>
        </div>
      </div>
    </div>

    <div
      class="transition-all duration-400 ease-out"
      style:opacity={linksVisible ? 1 : 0}
      style:transform="translateY({linksVisible ? 0 : 15}px)"
    >
      <div class="m3-surface overflow-hidden">
        <div class="flex items-center gap-3 px-4 py-4">
          <SendIcon />
          <span class="m3-label">{$t.about_community}</span>
        </div>
        <Divider />
        <List>
          {#each [
            { url: LINKS.channel1, title: $t.about_channel_1, desc: $t.about_channel_1_desc, group: false },
            { url: LINKS.channel2, title: $t.about_channel_2, desc: $t.about_channel_2_desc, group: false },
            { url: LINKS.group, title: $t.about_group, desc: $t.about_group_desc, group: true },
          ] as link}
            <ListItem variant="button" onclick={() => openUrl(link.url)}>
              {#snippet start()}
                <div class="m3-icon-tile size-10 rounded-full">
                  {#if link.group}
                    <GroupsIcon />
                  {:else}
                    <SendIcon />
                  {/if}
                </div>
              {/snippet}
              {link.title}
              {#snippet supportingText()}
                {link.desc}
              {/snippet}
              {#snippet end()}
                <ArrowForwardIcon />
              {/snippet}
            </ListItem>
          {/each}
          <Divider />
          <ListItem variant="button" onclick={() => openUrl(LINKS.bugs)}>
            {#snippet start()}
              <div class="m3-icon-tile size-10 rounded-full">
                <Icon>error</Icon>
              </div>
            {/snippet}
            {$t.about_report_bugs}
            {#snippet supportingText()}
              @rodroidbugreporter_bot
            {/snippet}
            {#snippet end()}
              <ArrowForwardIcon />
            {/snippet}
          </ListItem>
        </List>
      </div>

      <div
        class="flex items-center justify-center gap-1 mt-4 transition-all duration-700 ease-out delay-500"
        style:opacity={linksVisible ? 1 : 0}
      >
        <div bind:this={likeContainer} class="size-8"></div>
        <p class="text-xs m3-secondary">Made with Rust by Rodroid Mods</p>
      </div>
    </div>
  </div>
</div>
