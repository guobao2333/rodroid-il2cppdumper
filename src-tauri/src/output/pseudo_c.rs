//! Interactive pseudo-C assembly/namespace/class selection tree.
//!
//! Engine-independent UI support used by the CLI before running the native
//! recovery-based pseudo-C# generator (`crate::recovery::csharp`). The old
//! Ghidra-backed pipeline was removed when the native engine became the
//! default for ARM64 and x64.

use std::collections::{HashMap, HashSet};
use crate::executor::Il2CppExecutor;
use crate::il2cpp::base::Il2Cpp;
use crate::il2cpp::enums::method_attributes;

/// Methods smaller than this are not worth decompiling (alignment / stub
/// thunks with no useful body).
const MIN_BODY_SIZE: usize = 4;

use crate::il2cpp::metadata::Metadata;


/// One type with its concrete (decompilable) method count.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PseudoTypeEntry {
    pub full_name: String,
    pub method_count: usize,
}

/// One namespace with its aggregate method count and the types it contains.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PseudoNamespaceEntry {
    pub namespace: String,
    pub method_count: usize,
    pub types: Vec<PseudoTypeEntry>,
}

/// One assembly (metadata image) with its aggregate method count, namespaces
/// and types. Mirror of the `collect_methods` walk (same concrete-method
/// criteria and RVA de-duplication) so the counts shown in the picker match
/// exactly what will be decompiled.
#[derive(Clone, Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PseudoAssemblyEntry {
    pub assembly: String,
    pub method_count: usize,
    pub namespaces: Vec<PseudoNamespaceEntry>,
}

/// Walk every image -> type -> method exactly like `collect_methods` and
/// group the concrete (decompilable) method counts into an assembly ->
/// namespace -> type tree. Used by the CLI to present a checkbox selection
/// before the pseudo-C pass.
pub fn list_filter_tree(
    il2cpp: &Il2Cpp,
    metadata: &Metadata,
    executor: &mut Il2CppExecutor,
) -> Vec<PseudoAssemblyEntry> {
    let sorted_addrs = il2cpp.build_sorted_method_addresses();
    let mut seen_rva: HashSet<u64> = HashSet::new();
    let mut out: Vec<PseudoAssemblyEntry> = Vec::new();

    for image_def in &metadata.image_defs {
        let image_name = metadata
            .get_string_from_index(image_def.name_index)
            .unwrap_or_default();
        let mut ns_map: HashMap<String, (usize, HashMap<String, usize>)> = HashMap::new();
        let mut asm_count = 0usize;
        let type_end = image_def.type_start as usize + image_def.type_count as usize;

        for type_def_index in image_def.type_start as usize..type_end {
            let td = &metadata.type_defs[type_def_index];
            let full_type =
                executor.get_type_def_name(td, type_def_index, metadata, il2cpp, true, false);
            let namespace = metadata
                .get_string_from_index(td.namespace_index)
                .unwrap_or_default();
            let method_end = td.method_start as usize + td.method_count as usize;
            let mut type_count = 0usize;

            for mi in td.method_start as usize..method_end {
                let md = &metadata.method_defs[mi];
                if (md.flags as u32 & method_attributes::ABSTRACT) != 0 {
                    continue;
                }
                let method_pointer = il2cpp.get_method_pointer(&image_name, md);
                if method_pointer == 0 {
                    continue;
                }
                let rva = il2cpp.get_rva(method_pointer);
                if !seen_rva.insert(rva) {
                    continue;
                }
                let body_size = il2cpp.get_method_body_size(rva, &sorted_addrs);
                if body_size < MIN_BODY_SIZE {
                    continue;
                }
                type_count += 1;
                asm_count += 1;
            }

            if type_count == 0 {
                continue;
            }
            let ns_entry = ns_map
                .entry(namespace.clone())
                .or_insert_with(|| (0, HashMap::new()));
            ns_entry.0 += type_count;
            ns_entry.1.insert(full_type, type_count);
        }

        if asm_count == 0 {
            continue;
        }
        let mut namespaces: Vec<PseudoNamespaceEntry> = ns_map
            .into_iter()
            .map(|(namespace, (count, types))| PseudoNamespaceEntry {
                namespace,
                method_count: count,
                types: types
                    .into_iter()
                    .map(|(full_name, method_count)| PseudoTypeEntry {
                        full_name,
                        method_count,
                    })
                    .collect(),
            })
            .collect();
        for ns in &mut namespaces {
            ns.types.sort_by(|a, b| {
                b.method_count
                    .cmp(&a.method_count)
                    .then_with(|| a.full_name.cmp(&b.full_name))
            });
        }
        namespaces.sort_by(|a, b| {
            b.method_count
                .cmp(&a.method_count)
                .then_with(|| a.namespace.cmp(&b.namespace))
        });
        out.push(PseudoAssemblyEntry {
            assembly: image_name,
            method_count: asm_count,
            namespaces,
        });
    }

    out.sort_by(|a, b| {
        b.method_count
            .cmp(&a.method_count)
            .then_with(|| a.assembly.cmp(&b.assembly))
    });
    out
}

/// Fast, binary-free class tree for GUI pickers. Built from metadata image /
/// type-def tables only so it can appear as soon as `global-metadata.dat` is
/// chosen. Method counts are metadata `method_count` totals (not the stricter
/// decompilable-body census from [`list_filter_tree`]).
pub fn list_metadata_type_tree(metadata: &Metadata) -> Vec<PseudoAssemblyEntry> {
    let mut out: Vec<PseudoAssemblyEntry> = Vec::new();

    for image_def in &metadata.image_defs {
        let image_name = metadata
            .get_string_from_index(image_def.name_index)
            .unwrap_or_default();
        let mut ns_map: HashMap<String, (usize, HashMap<String, usize>)> = HashMap::new();
        let mut asm_count = 0usize;
        let type_end = image_def.type_start as usize + image_def.type_count as usize;

        for type_def_index in image_def.type_start as usize..type_end {
            let td = &metadata.type_defs[type_def_index];
            let type_count = td.method_count as usize;
            if type_count == 0 {
                continue;
            }
            let name = metadata
                .get_string_from_index(td.name_index)
                .unwrap_or_default();
            let namespace = metadata
                .get_string_from_index(td.namespace_index)
                .unwrap_or_default();
            let full_type = if namespace.is_empty() {
                name
            } else {
                format!("{namespace}.{name}")
            };
            asm_count += type_count;
            let ns_entry = ns_map
                .entry(namespace)
                .or_insert_with(|| (0, HashMap::new()));
            ns_entry.0 += type_count;
            ns_entry.1.insert(full_type, type_count);
        }

        if asm_count == 0 {
            continue;
        }
        let mut namespaces: Vec<PseudoNamespaceEntry> = ns_map
            .into_iter()
            .map(|(namespace, (count, types))| PseudoNamespaceEntry {
                namespace,
                method_count: count,
                types: types
                    .into_iter()
                    .map(|(full_name, method_count)| PseudoTypeEntry {
                        full_name,
                        method_count,
                    })
                    .collect(),
            })
            .collect();
        for ns in &mut namespaces {
            ns.types.sort_by(|a, b| {
                b.method_count
                    .cmp(&a.method_count)
                    .then_with(|| a.full_name.cmp(&b.full_name))
            });
        }
        namespaces.sort_by(|a, b| {
            b.method_count
                .cmp(&a.method_count)
                .then_with(|| a.namespace.cmp(&b.namespace))
        });
        out.push(PseudoAssemblyEntry {
            assembly: image_name,
            method_count: asm_count,
            namespaces,
        });
    }

    out.sort_by(|a, b| {
        b.method_count
            .cmp(&a.method_count)
            .then_with(|| a.assembly.cmp(&b.assembly))
    });
    out
}

/// Does an image/assembly name pass the include filter? Empty = all assemblies.
/// Matches the recovery engine's rule: strip a trailing `.dll` and allow an
/// exact or prefix match (so `Assembly-CSharp` also matches
/// `Assembly-CSharp-firstpass`).
pub fn assembly_matches(image_name: &str, include_assemblies: &[String]) -> bool {
    if include_assemblies.is_empty() {
        return true;
    }
    let base = image_name.trim_end_matches(".dll");
    include_assemblies.iter().any(|f| {
        let f = f.trim_end_matches(".dll");
        base == f || base.starts_with(f)
    })
}

/// Does `full_type` match any of the `include_types` entries? An entry is
/// either an exact full type name (`Namespace.Type`) or a namespace/type
/// prefix (`Namespace` matches every `Namespace.*` type; a nested type is
/// matched through its declaring type prefix). Trailing dots are tolerated.
///
/// The comparison runs against `get_type_def_name(.., add_namespace = true)`, so
/// the strings the user writes match the names `dump.cs` prints.
pub fn type_matches(full_type: &str, include_types: &[String]) -> bool {
    if include_types.is_empty() {
        return true;
    }
    include_types.iter().any(|entry| {
        let entry = entry.trim_end_matches('.');
        full_type == entry || full_type.starts_with(&format!("{entry}."))
    })
}

/// Shared assembly + type gate for C# pseudo **and** native disassembly bodies.
/// Empty filters mean "all". Used so the class picker applies to both modes.
pub fn method_body_filter_allows(
    image_name: &str,
    full_type: &str,
    include_assemblies: &[String],
    include_types: &[String],
) -> bool {
    assembly_matches(image_name, include_assemblies)
        && type_matches(full_type, include_types)
}
