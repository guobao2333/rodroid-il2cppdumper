<script lang="ts">
  import { Switch } from "noph-ui";

  let { label, checked = $bindable(false) }: { label: string; checked: boolean } = $props();
</script>

<div class="m3-config-row">
  <span class="text-sm flex-1 min-w-0">{label}</span>
  <div class="m3-config-switch">
    <Switch bind:selected={checked} />
  </div>
</div>
