<script>
  import { onMount } from "svelte";
  import "noph-ui/defaultTheme";
  import "../app.css";
  import InputDialog from "$lib/components/InputDialog.svelte";
  import { setupDumpEvents } from "$lib/dumpEvents";

  let { children } = $props();

  onMount(() => {
    void setupDumpEvents();
  });
</script>

{@render children()}
<InputDialog />
