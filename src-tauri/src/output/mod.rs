/// Print an informational note to stderr.
///
/// Ports do not depend on the CLI `console` crate — keep this unstyled so a
/// sync from `il2cpp-dumper-rs` only needs this one-line adaptation.
pub fn note(msg: &str) {
    eprintln!("  ℹ {msg}");
}

pub mod script_json;
pub mod decompiler;
pub mod pseudo_c;
pub mod struct_generator;
pub mod dummy_assembly_generator;
pub mod static_field_exporter;
pub mod header_constants;
pub mod embedded_scripts;
pub mod generics;
pub mod name_mangler;
pub mod cpp_scaffolding;

pub use script_json::*;
pub use decompiler::Il2CppDecompiler;
pub use struct_generator::StructGenerator;
pub use name_mangler::MangledNameBuilder;
pub use cpp_scaffolding::CppScaffolding;
pub mod cpp_type_dependency_graph;
pub mod cpp_ast;
pub mod cpp_type_model;

pub mod unity_version;
pub mod header_manager;
pub mod recovery_integration;
