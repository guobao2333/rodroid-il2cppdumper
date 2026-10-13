<script lang="ts">
  import { Button, IconButton, TextField } from "noph-ui";
  import { Icon } from "noph-ui/icons";
  import type { TypeTreeAssembly, TypeTreeNamespace } from "$lib/types";
  import { config, t } from "$lib/stores";

  /** Cap type rows when expanding a DLL — 10k–20k+ classes freeze the DOM. */
  const MAX_TYPES_EXPANDED = 80;

  let {
    typeTree = [],
    loading = false,
    error = "",
    onclose,
  }: {
    typeTree: TypeTreeAssembly[];
    loading: boolean;
    error: string;
    onclose: () => void;
  } = $props();

  let expanded = $state<Record<string, boolean>>({});
  let search = $state("");
  let selectedAssemblies = $derived(new Set($config.csharpIncludeAssemblies ?? []));
  let selectedTypes = $derived(new Set($config.csharpIncludeTypes ?? []));
  let query = $derived(search.trim().toLowerCase());

  let filtered = $derived.by(() => {
    if (!query) return typeTree;
    return typeTree
      .map((asm) => {
        const namespaces = asm.namespaces
          .map((ns) => {
            const types = ns.types.filter(
              (ty) =>
                ty.fullName.toLowerCase().includes(query) ||
                ns.namespace.toLowerCase().includes(query) ||
                asm.assembly.toLowerCase().includes(query),
            );
            if (
              types.length ||
              ns.namespace.toLowerCase().includes(query) ||
              asm.assembly.toLowerCase().includes(query)
            ) {
              return { ...ns, types: types.length ? types : ns.types };
            }
            return null;
          })
          .filter(Boolean) as TypeTreeAssembly["namespaces"];
        if (namespaces.length || asm.assembly.toLowerCase().includes(query)) {
          return { ...asm, namespaces: namespaces.length ? namespaces : asm.namespaces };
        }
        return null;
      })
      .filter(Boolean) as TypeTreeAssembly[];
  });

  /** Cap types for expand; keep already-selected types visible. */
  function cappedNamespaces(
    namespaces: TypeTreeNamespace[],
    selected: Set<string>,
  ): { namespaces: TypeTreeNamespace[]; hidden: number } {
    const total = namespaces.reduce((n, ns) => n + ns.types.length, 0);
    if (total <= MAX_TYPES_EXPANDED) return { namespaces, hidden: 0 };

    const out: TypeTreeNamespace[] = [];
    let used = 0;
    let hidden = 0;
    for (const ns of namespaces) {
      if (ns.types.length === 0) {
        out.push(ns);
        continue;
      }
      const pinned = ns.types.filter((ty) => selected.has(ty.fullName));
      const rest = ns.types.filter((ty) => !selected.has(ty.fullName));
      const room = Math.max(0, MAX_TYPES_EXPANDED - used - pinned.length);
      const kept = [...pinned, ...rest.slice(0, room)];
      hidden += ns.types.length - kept.length;
      used += kept.length;
      if (kept.length > 0) out.push({ ...ns, types: kept });
    }
    return { namespaces: out, hidden };
  }

  function writeSelection(assemblies: string[], types: string[]) {
    config.update((c) => ({
      ...c,
      csharpIncludeAssemblies: [...new Set(assemblies)].sort(),
      csharpIncludeTypes: [...new Set(types)].sort(),
    }));
  }

  function toggleAssembly(name: string) {
    const next = new Set(selectedAssemblies);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    writeSelection([...next], [...selectedTypes]);
  }

  function toggleType(name: string) {
    const next = new Set(selectedTypes);
    if (next.has(name)) next.delete(name);
    else next.add(name);
    writeSelection([...selectedAssemblies], [...next]);
  }

  function presetAll() {
    writeSelection([], []);
  }

  function presetCSharp() {
    const names = typeTree
      .map((a) => a.assembly)
      .filter((a) => a.toLowerCase().includes("assembly-csharp"));
    writeSelection(names, []);
  }

  function presetClear() {
    writeSelection([], []);
  }
</script>

<div class="m3-dialog-scrim m3-scrim-enter" role="presentation">
  <button type="button" class="absolute inset-0 cursor-default" aria-label="Close" onclick={onclose}></button>
  <div class="m3-dialog m3-dialog-enter max-w-3xl h-[min(860px,92vh)]" role="dialog" aria-labelledby="class-picker-title">
    <div class="p-6 pb-3 shrink-0 space-y-3">
      <div class="flex items-center gap-3">
        <div class="m3-icon-tile size-10">
          <Icon>filter_list</Icon>
        </div>
        <div class="flex-1 min-w-0">
          <h3 id="class-picker-title" class="text-lg font-semibold">{$t.class_picker_title}</h3>
          <p class="text-xs m3-secondary">{$t.class_picker_subtitle}</p>
        </div>
        <IconButton aria-label="Close" onclick={onclose}>
          <Icon>close</Icon>
        </IconButton>
      </div>

      <TextField
        variant="filled"
        label={$t.class_picker_search}
        bind:value={search}
      />

      <div class="flex flex-wrap gap-2">
        <Button variant="tonal" onclick={presetAll}>{$t.class_picker_preset_all}</Button>
        <Button variant="tonal" onclick={presetCSharp}>{$t.class_picker_preset_csharp}</Button>
        <Button variant="tonal" onclick={presetClear}>{$t.class_picker_preset_clear}</Button>
      </div>

      <p class="text-xs font-medium" style="color: var(--md-sys-color-primary)">
        {#if selectedAssemblies.size === 0 && selectedTypes.size === 0}
          {$t.class_picker_summary_all}
        {:else}
          {selectedAssemblies.size} assemblies · {selectedTypes.size} types selected
        {/if}
      </p>
    </div>

    <div class="flex-1 overflow-y-auto px-6 pb-4 space-y-2">
      {#if loading}
        <div class="py-16 text-center m3-secondary">{$t.class_picker_loading}</div>
      {:else if error}
        <div class="py-8 text-center" style="color: var(--md-sys-color-error)">{error}</div>
      {:else if filtered.length === 0}
        <div class="py-8 text-center m3-secondary">{$t.class_picker_empty}</div>
      {:else}
        {#each filtered as asm (asm.assembly)}
          <div class="m3-surface rounded-2xl overflow-hidden">
            <div class="flex items-center gap-2 p-3">
              <input
                type="checkbox"
                class="size-4 accent-[var(--md-sys-color-primary)]"
                checked={selectedAssemblies.has(asm.assembly)}
                onchange={() => toggleAssembly(asm.assembly)}
              />
              <button
                type="button"
                class="flex-1 text-left min-w-0"
                onclick={() => (expanded[asm.assembly] = !expanded[asm.assembly])}
              >
                <div class="font-medium truncate">{asm.assembly}</div>
                <div class="text-xs m3-secondary">{asm.methodCount} methods</div>
              </button>
              <Icon>{expanded[asm.assembly] ? "expand_less" : "expand_more"}</Icon>
            </div>
            {#if expanded[asm.assembly]}
              {@const capped = cappedNamespaces(asm.namespaces, selectedTypes)}
              <div class="px-3 pb-3 space-y-3">
                {#each capped.namespaces as ns (ns.namespace + asm.assembly)}
                  <div>
                    <div class="text-xs font-semibold mb-1" style="color: var(--md-sys-color-tertiary)">
                      {ns.namespace || "<global>"}
                    </div>
                    {#each ns.types as ty (ty.fullName)}
                      <label class="flex items-start gap-2 py-1.5 px-1 rounded-lg hover:bg-[color-mix(in_srgb,var(--md-sys-color-primary)_8%,transparent)]">
                        <input
                          type="checkbox"
                          class="mt-1 size-4 accent-[var(--md-sys-color-primary)]"
                          checked={selectedTypes.has(ty.fullName)}
                          onchange={() => toggleType(ty.fullName)}
                        />
                        <span class="min-w-0">
                          <span class="block text-sm break-all">{ty.fullName}</span>
                          <span class="block text-xs m3-secondary">{ty.methodCount} methods</span>
                        </span>
                      </label>
                    {/each}
                  </div>
                {/each}
                {#if capped.hidden > 0}
                  <p class="text-xs m3-secondary px-1">
                    {$t.class_picker_types_capped.replace("{n}", String(capped.hidden))}
                  </p>
                {/if}
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

    <div class="p-4 pt-2 shrink-0 flex justify-end">
      <Button variant="filled" onclick={onclose}>{$t.class_picker_done}</Button>
    </div>
  </div>
</div>
