<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { Button, TextField } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  import { inputRequest, t } from "$lib/stores";

  let dumpAddress = $state("");
  let codeReg = $state("");
  let metaReg = $state("");

  function isValidHex(s: string): boolean {
    if (!s.trim()) return false;
    const stripped = s.trim().replace(/^0[xX]/, "");
    return stripped.length > 0 && /^[0-9a-fA-F]+$/.test(stripped);
  }

  async function submitDumpAddress() {
    await invoke("submit_input", { response: dumpAddress || "0" });
    inputRequest.set(null);
    dumpAddress = "";
  }

  async function skipDumpAddress() {
    await invoke("submit_input", { response: "0" });
    inputRequest.set(null);
    dumpAddress = "";
  }

  async function submitManualAddresses() {
    await invoke("submit_input", { response: `${codeReg},${metaReg}` });
    inputRequest.set(null);
    codeReg = "";
    metaReg = "";
  }

  async function cancelManual() {
    await invoke("submit_input", { response: "" });
    inputRequest.set(null);
    codeReg = "";
    metaReg = "";
  }

  let dumpOpen = $derived($inputRequest === "dump_address");
  let manualOpen = $derived($inputRequest === "manual_addresses");
</script>

{#if dumpOpen}
  <div class="m3-dialog-scrim" role="presentation">
    <button type="button" class="absolute inset-0 cursor-default" aria-label="Close dialog" onclick={skipDumpAddress}></button>
    <div class="m3-dialog p-6 space-y-5 max-w-md">
      <div class="space-y-2">
        <div class="flex items-center gap-2.5">
          <div class="m3-icon-tile size-9">
            <Icon>refresh</Icon>
          </div>
          <h3 class="text-lg font-semibold">{$t.dialog_dump_address_title}</h3>
        </div>
        <p class="text-sm m3-secondary">{$t.dialog_dump_address_desc}</p>
      </div>
      <TextField variant="filled" placeholder="0x10000" bind:value={dumpAddress} />
      <div class="m3-dialog-actions">
        <Button variant="text" onclick={skipDumpAddress}>{$t.dialog_skip}</Button>
        <Button variant="filled" onclick={submitDumpAddress}>OK</Button>
      </div>
    </div>
  </div>
{/if}

{#if manualOpen}
  <div class="m3-dialog-scrim" role="presentation">
    <button type="button" class="absolute inset-0 cursor-default" aria-label="Close dialog" onclick={cancelManual}></button>
    <div class="m3-dialog p-6 space-y-5 max-w-md">
      <div class="space-y-2">
        <div class="flex items-center gap-2.5">
          <div class="m3-icon-tile size-9">
            <Icon>code</Icon>
          </div>
          <h3 class="text-lg font-semibold">{$t.dialog_manual_title}</h3>
        </div>
        <p class="text-sm m3-secondary">{$t.dialog_manual_desc}</p>
      </div>
      <div class="space-y-4">
        <TextField
          label={$t.setting_code_registration}
          variant="filled"
          placeholder="0x654aef0"
          bind:value={codeReg}
          aria-invalid={codeReg ? !isValidHex(codeReg) : false}
        />
        <TextField
          label={$t.setting_metadata_registration}
          variant="filled"
          placeholder="0x66c4998"
          bind:value={metaReg}
          aria-invalid={metaReg ? !isValidHex(metaReg) : false}
        />
      </div>
      <div class="m3-dialog-actions">
        <Button variant="text" onclick={cancelManual}>Cancel</Button>
        <Button variant="filled" disabled={!isValidHex(codeReg) || !isValidHex(metaReg)} onclick={submitManualAddresses}>OK</Button>
      </div>
    </div>
  </div>
{/if}
