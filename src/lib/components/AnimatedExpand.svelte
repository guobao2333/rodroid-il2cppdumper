<script lang="ts">
  import type { Snippet } from "svelte";

  let { show = false, children }: { show: boolean; children: Snippet } = $props();
</script>

<div class="m3-collapsible" class:open={show} aria-hidden={!show}>
  <div class="m3-collapsible-inner">
    {@render children()}
  </div>
</div>
