<script lang="ts">
  import { Icon } from "noph-ui/icons";

  let {
    value = $bindable(""),
    placeholder = "",
    title = "",
    onchange,
  }: {
    value?: string;
    placeholder?: string;
    title?: string;
    onchange?: (e: Event) => void;
  } = $props();
</script>

<div class="m3-path-input" {title}>
  <input
    type="text"
    class="m3-path-input-field"
    bind:value
    {placeholder}
    spellcheck="false"
    {onchange}
  />
  {#if value}
    <span class="m3-path-input-check" aria-hidden="true">
      <Icon>check_circle</Icon>
    </span>
  {/if}
</div>
