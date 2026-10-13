// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

pub mod config;
pub mod disassembler;
pub mod error;
pub mod executor;
pub mod formats;
pub mod il2cpp;
pub mod io;
pub mod output;
pub mod recovery;
pub mod search;
pub mod utils;

use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::config::Config;
use crate::executor::Il2CppExecutor;
use crate::formats::elf::Elf;
use crate::formats::macho::MachO;
use crate::formats::nso::Nso;
use crate::formats::pe::Pe;
use crate::formats::wasm::Wasm;
use crate::il2cpp::base::{Il2Cpp, VaSegment};
use crate::il2cpp::metadata::{Metadata, MetadataVariant};
use crate::output::decompiler::Il2CppDecompiler;
use crate::output::struct_generator::StructGenerator;

const MAGIC_METADATA: u32 = 0xFAB11BAF;
const MAGIC_ELF: u32 = 0x464C457F;
const MAGIC_PE: u16 = 0x5A4D;
const MAGIC_MACHO32: u32 = 0xFEEDFACE;
const MAGIC_MACHO64: u32 = 0xFEEDFACF;
const MAGIC_MACHOFAT: u32 = 0xBEBAFECA;
const MAGIC_NSO: u32 = 0x304F534E;
const MAGIC_WASM: u32 = 0x6D736100;

#[derive(Clone, Serialize)]
struct LogEvent {
    message: String,
}

#[derive(Clone, Serialize)]
struct InputRequestEvent {
    prompt_type: String,
}

#[derive(Clone, Serialize)]
struct DumpCompleteEvent {
    success: bool,
    output_path: String,
    error_message: String,
}

#[derive(Clone, Serialize)]
struct CrashEvent {
    crash_log: String,
}

#[derive(Serialize)]
struct BinaryInfo {
    format: String,
    unity_version: String,
}

struct AppState {
    input_sender: Mutex<Option<std::sync::mpsc::Sender<String>>>,
    dump_running: Mutex<bool>,
}

fn emit_log(app: &AppHandle, message: &str) {
    let _ = app.emit(
        "dump-log",
        LogEvent {
            message: message.to_string(),
        },
    );
}

fn request_input(app: &AppHandle, state: &AppState, prompt_type: &str) -> String {
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let mut sender = state.input_sender.lock().unwrap();
        *sender = Some(tx);
    }

    let _ = app.emit(
        "dump-input-request",
        InputRequestEvent {
            prompt_type: prompt_type.to_string(),
        },
    );

    rx.recv().unwrap_or_default()
}

fn parse_hex(s: &str) -> u64 {
    let trimmed = s.trim().trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(trimmed, 16).unwrap_or(0)
}

fn prompt_dump_address(app: &AppHandle, state: &AppState) -> Option<u64> {
    let response = request_input(app, state, "dump_address");
    let addr = parse_hex(&response);
    if addr != 0 { Some(addr) } else { None }
}

fn prompt_manual_addresses(app: &AppHandle, state: &AppState) -> Option<(u64, u64)> {
    let response = request_input(app, state, "manual_addresses");
    let parts: Vec<&str> = response.split(',').collect();
    if parts.len() == 2 {
        let cr = parse_hex(parts[0].trim());
        let mr = parse_hex(parts[1].trim());
        if cr != 0 && mr != 0 {
            return Some((cr, mr));
        }
    }
    None
}

fn read_magic_u32(data: &[u8]) -> u32 {
    if data.len() < 4 {
        return 0;
    }
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn read_magic_u16(data: &[u8]) -> u16 {
    if data.len() < 2 {
        return 0;
    }
    u16::from_le_bytes([data[0], data[1]])
}

fn is_valid_metadata_version(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    let ver = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    ver > 0 && ver < 200
}

fn try_decrypt_metadata(data: &mut Vec<u8>) -> Option<String> {
    if data.len() < 16 {
        return None;
    }
    let magic = MAGIC_METADATA.to_le_bytes();

    let k1 = magic[0] ^ data[0];
    if k1 != 0 && (1..4).all(|i| (magic[i] ^ data[i]) == k1) {
        let mut test = data.clone();
        for b in test.iter_mut() {
            *b ^= k1;
        }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!("Single-byte XOR (key: 0x{k1:02X})"));
        }
    }

    let key4: [u8; 4] = std::array::from_fn(|i| magic[i] ^ data[i]);
    if key4 != [0u8; 4] && !key4.windows(2).all(|w| w[0] == w[1]) {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4];
        }
        if is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "4-byte XOR (key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    let key8: [u8; 8] = std::array::from_fn(|i| if i < 4 { magic[i] ^ data[i] } else { data[i] });
    if key8[0..4] != [0u8; 4] {
        let expected_ver_bytes: Vec<u8> = (4..8).map(|i| data[i] ^ key8[i]).collect();
        let test_ver = i32::from_le_bytes([
            expected_ver_bytes[0],
            expected_ver_bytes[1],
            expected_ver_bytes[2],
            expected_ver_bytes[3],
        ]);
        if test_ver > 0 && test_ver < 200 {
            let mut test = data.clone();
            for (i, b) in test.iter_mut().enumerate() {
                *b ^= key8[i % 8];
            }
            if is_valid_metadata_version(&test) {
                *data = test;
                return Some(format!(
                    "8-byte XOR (key: {:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X})",
                    key8[0], key8[1], key8[2], key8[3], key8[4], key8[5], key8[6], key8[7]
                ));
            }
        }
    }

    for key_len in [16usize, 32, 64, 128, 256] {
        if data.len() < key_len * 2 {
            continue;
        }
        let key: Vec<u8> = (0..key_len)
            .map(|i| {
                if i < 4 {
                    magic[i] ^ data[i]
                } else {
                    data[i] ^ 0
                }
            })
            .collect();
        if key[0..4] == [0u8; 4] {
            continue;
        }
        let mut test = data[..8].to_vec();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key[i % key_len];
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            for (i, b) in data.iter_mut().enumerate() {
                *b ^= key[i % key_len];
            }
            return Some(format!("{key_len}-byte rolling XOR key"));
        }
    }

    {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4] ^ (i as u8);
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Position-dependent XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    {
        let mut test = data.clone();
        for (i, b) in test.iter_mut().enumerate() {
            *b ^= key4[i % 4] ^ ((i & 0xFF) as u8);
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Masked position XOR (base key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    {
        let header_size = 256usize.min(data.len());
        let mut test = data.clone();
        for i in 0..header_size {
            test[i] ^= key4[i % 4];
        }
        if &test[0..4] == &magic && is_valid_metadata_version(&test) {
            *data = test;
            return Some(format!(
                "Header-only XOR ({header_size} bytes, key: {:02X}{:02X}{:02X}{:02X})",
                key4[0], key4[1], key4[2], key4[3]
            ));
        }
    }

    None
}

fn detect_unity_version(data: &[u8]) -> Option<String> {
    let mut best: Option<String> = None;
    let mut i = 0;
    while i + 12 < data.len() {
        let is_20xx = data[i] == b'2'
            && data[i + 1] == b'0'
            && data[i + 2].is_ascii_digit()
            && data[i + 3].is_ascii_digit()
            && data[i + 4] == b'.';
        let is_6xxx = data[i] == b'6'
            && data[i + 1] == b'0'
            && data[i + 2] == b'0'
            && data[i + 3] == b'0'
            && data[i + 4] == b'.';

        if (is_20xx || is_6xxx) && data[i + 5].is_ascii_digit() {
            let max_len = std::cmp::min(24, data.len() - i);
            let end = data[i..i + max_len]
                .iter()
                .position(|&b| !b.is_ascii_alphanumeric() && b != b'.')
                .unwrap_or(max_len);
            let candidate = &data[i..i + end];
            if candidate.len() >= 8 && candidate.len() <= 20 {
                if let Ok(s) = std::str::from_utf8(candidate) {
                    let dot_count = s.chars().filter(|c| *c == '.').count();
                    let has_suffix = s.contains('f')
                        || s.contains('b')
                        || s.contains('a')
                        || s.contains('p')
                        || s.contains("rc");
                    if dot_count == 2 && has_suffix && s.ends_with(|c: char| c.is_ascii_digit()) {
                        if best.as_ref().map_or(true, |prev| s > prev.as_str()) {
                            best = Some(s.to_string());
                        }
                    }
                }
            }
            i += end.max(1);
        } else {
            i += 1;
        }
    }
    best
}

fn detect_format(data: &[u8]) -> &'static str {
    let magic32 = read_magic_u32(data);
    let magic16 = read_magic_u16(data);
    match magic32 {
        MAGIC_ELF => "ELF",
        MAGIC_MACHO32 | MAGIC_MACHO64 => "Mach-O",
        MAGIC_MACHOFAT => "Fat Mach-O",
        MAGIC_NSO => "NSO",
        MAGIC_WASM => "WASM",
        _ if magic16 == MAGIC_PE => "PE",
        _ => "Unknown",
    }
}

fn resolve_codm(config: &Config, metadata: &Metadata) -> bool {
    config.codm || metadata.variant == MetadataVariant::Codm
}

fn init_elf(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    let is_64 = data.len() > 4 && data[4] == 2;
    emit_log(
        app,
        &format!("Detected ELF{} format", if is_64 { "64" } else { "32" }),
    );

    let mut elf = if metadata.variant == MetadataVariant::Codm {
        Elf::new_with_codm_diag(data, !is_64, true)?
    } else {
        Elf::new(data, !is_64)?
    };
    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };
    elf.set_properties(version, metadata.metadata_usages_count as u64);
    emit_log(app, &format!("IL2CPP Version: {}", elf.stream.version));

    if config.force_dump || elf.check_dump() {
        emit_log(app, "Detected this may be a dump file.");
        if let Some(addr) = prompt_dump_address(app, state) {
            elf.stream.image_base = addr;
            elf.is_dumped = true;
            if !config.no_redirected_pointer {
                elf.load()?;
            }
            emit_log(app, &format!("Using dump address: 0x{:x}", addr));
        }
    }

    let mut force_cr = 0u64;
    let mut force_mr = 0u64;
    if config.force_dump {
        emit_log(app, "Force dump mode enabled, entering manual addresses...");
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            force_cr = cr;
            force_mr = mr;
        }
    }

    if force_cr != 0 && force_mr != 0 {
        elf.init_with_auto_plus(force_cr, force_mr)?;
    } else {
        emit_log(app, "Searching...");
        let method_count = metadata
            .method_defs
            .iter()
            .filter(|m| m.method_index >= 0)
            .count();
        let type_count = metadata.type_defs.len();
        let image_count = metadata.image_defs.len();

        let mut helper = elf.get_section_helper(method_count, type_count, image_count);
        let code_reg = helper.find_code_registration();
        let metadata_reg = if metadata.variant == MetadataVariant::Codm {
            helper.find_metadata_registration_codm().or_else(|| helper.find_metadata_registration())
        } else {
            helper.find_metadata_registration()
        };

        // Log CR/MR only once — after the strategy that actually succeeds.
        let mut found = false;
        let mut final_cr: Option<u64> = None;
        let mut final_mr: Option<u64> = None;
        let mut how = "";

        if elf.auto_plus_init(code_reg, metadata_reg).unwrap_or(false) {
            found = true;
            final_cr = code_reg;
            final_mr = metadata_reg;
            how = "Section scan";
        }

        if !found {
            if let Ok(Some((cr, mr))) = elf.symbol_search() {
                if elf.init_with_auto_plus(cr, mr).is_ok() {
                    found = true;
                    final_cr = Some(cr);
                    final_mr = Some(mr);
                    how = "Symbol table";
                }
            }
        }

        if !found {
            if let Some((cr, mr)) = elf.search_arm32(version) {
                if elf.init_with_auto_plus(cr, mr).is_ok() {
                    found = true;
                    final_cr = Some(cr);
                    final_mr = Some(mr);
                    how = "ARM32 search pattern";
                }
            }
        }

        if !found {
            emit_log(app, "Auto mode failed, requesting manual addresses...");
            if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
                elf.init_with_auto_plus(cr, mr)?;
            } else {
                return Err(error::Error::Other("Manual mode cancelled.".into()));
            }
        } else {
            if !how.is_empty() {
                emit_log(app, &format!("Found via {how}"));
            }
            if let Some(cr) = final_cr {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            }
            if let Some(mr) = final_mr {
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            }
        }
    }

    let elf_exports = elf.list_exported_symbols().unwrap_or_default();
    let mut il2cpp = Il2Cpp::from_elf(&elf);
    il2cpp.codm = il2cpp.codm || resolve_codm(config, metadata);
    il2cpp.exported_symbols = elf_exports.iter().map(|(n, _)| n.clone()).collect();
    for (name, addr) in elf_exports {
        if name.starts_with("il2cpp_") || name.starts_with("mono_") {
            let rva = il2cpp.get_rva(addr);
            il2cpp.api_export_rvas.insert(name, rva);
        }
    }
    emit_log(
        app,
        &format!(
            "Found {} exported symbols, {} IL2CPP APIs",
            il2cpp.exported_symbols.len(),
            il2cpp.api_export_rvas.len()
        ),
    );
    Ok(il2cpp)
}

fn init_pe(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    let mut pe = Pe::new(data)?;
    emit_log(
        app,
        &format!(
            "Detected PE{} format",
            if pe.is_32bit { "32" } else { "64" }
        ),
    );

    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };
    pe.stream.version = version;
    pe.stream.is_32bit = pe.is_32bit;
    emit_log(app, &format!("IL2CPP Version: {version}"));

    if config.force_dump || pe.check_dump() {
        emit_log(app, "Detected this may be a dump file.");
        if let Some(addr) = prompt_dump_address(app, state) {
            pe.stream.image_base = addr;
            emit_log(app, &format!("Using dump address: 0x{:x}", addr));
        }
    }

    let mut force_cr = 0u64;
    let mut force_mr = 0u64;
    if config.force_dump {
        emit_log(app, "Force dump mode enabled.");
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            force_cr = cr;
            force_mr = mr;
        }
    }

    let mut cr_addr = force_cr;
    let mut mr_addr = force_mr;

    if cr_addr == 0 || mr_addr == 0 {
        emit_log(app, "Searching...");
        let method_count = metadata
            .method_defs
            .iter()
            .filter(|m| m.method_index >= 0)
            .count();
        let type_count = metadata.type_defs.len();
        let image_count = metadata.image_defs.len();
        let mu_count = metadata.metadata_usages_count;

        if let Ok(Some((cr, mr))) = pe.symbol_search() {
            emit_log(app, "Detected Symbol!");
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        }

        if cr_addr == 0 || mr_addr == 0 {
            let mut helper =
                pe.get_section_helper(method_count, type_count, mu_count, image_count, version);
            let code_reg = helper.find_code_registration();
            let metadata_reg = helper.find_metadata_registration();
            if let (Some(cr), Some(mr)) = (code_reg, metadata_reg) {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
                cr_addr = cr;
                mr_addr = mr;
            }
        }

        if cr_addr == 0 || mr_addr == 0 {
            emit_log(app, "Auto mode failed, requesting manual addresses...");
            if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
                cr_addr = cr;
                mr_addr = mr;
            } else {
                return Err(error::Error::Other("Manual mode cancelled.".into()));
            }
        }
    }

    let pe_image_base = pe.image_base();
    let va_segments: Vec<VaSegment> = pe
        .sections
        .iter()
        .map(|s| VaSegment {
            vaddr: s.virtual_address as u64 + pe_image_base,
            memsz: s.virtual_size as u64,
            offset: s.pointer_to_raw_data as u64,
        })
        .collect();

    let mut il2cpp = Il2Cpp::new(pe.stream.clone(), version, pe.is_32bit);
    il2cpp.va_segments = va_segments;
    il2cpp.image_base = pe_image_base;
    il2cpp.is_pe = true;
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.arch = Some(if pe.is_32bit {
        crate::disassembler::Architecture::X86
    } else {
        crate::disassembler::Architecture::X64
    });
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| pe.map_vatr(addr), false)?;
    il2cpp.data_sections = pe.data_search_sections();
    if let Ok(exports) = pe.list_exported_symbols() {
        il2cpp.exported_symbols = exports.iter().map(|(n, _)| n.clone()).collect();
        for (name, rva) in exports {
            if name.starts_with("il2cpp_") || name.starts_with("mono_") {
                il2cpp.api_export_rvas.insert(name, rva);
            }
        }
    }
    emit_log(
        app,
        &format!(
            "Found {} exported symbols, {} IL2CPP APIs",
            il2cpp.exported_symbols.len(),
            il2cpp.api_export_rvas.len()
        ),
    );
    Ok(il2cpp)
}

fn init_macho_fat(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    use crate::formats::macho::{MH_MAGIC_64, extract_fat_slice, parse_fat};
    let arches = parse_fat(&data)?;
    emit_log(
        app,
        &format!("Detected Fat Mach-O with {} architectures", arches.len()),
    );
    let index = arches
        .iter()
        .position(|a| a.magic == MH_MAGIC_64)
        .unwrap_or(0);
    emit_log(
        app,
        &format!(
            "Auto-selected architecture {} ({})",
            index + 1,
            if arches[index].magic == MH_MAGIC_64 {
                "64-bit"
            } else {
                "32-bit"
            }
        ),
    );
    let slice = extract_fat_slice(&data, &arches[index])?;
    init_macho(slice, metadata, config, app, state)
}

fn init_macho(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    let magic = read_magic_u32(&data);
    let is_64 = magic == MAGIC_MACHO64;
    emit_log(
        app,
        &format!(
            "Detected Mach-O{} format",
            if is_64 { " 64-bit" } else { " 32-bit" }
        ),
    );

    let mut macho = if metadata.variant == MetadataVariant::Codm {
        MachO::new_with_codm_fixups(data, !is_64, true)?
    } else {
        MachO::new(data, !is_64)?
    };
    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };
    macho.stream.version = version;
    macho.stream.is_32bit = macho.is_32bit;
    emit_log(app, &format!("IL2CPP Version: {version}"));

    if macho.check_dump() {
        emit_log(app, "Detected this may be a dump file.");
        if let Some(addr) = prompt_dump_address(app, state) {
            macho.stream.image_base = addr;
            emit_log(app, &format!("Using dump address: 0x{:x}", addr));
        }
    }

    emit_log(app, "Searching...");
    let method_count = metadata
        .method_defs
        .iter()
        .filter(|m| m.method_index >= 0)
        .count();
    let type_count = metadata.type_defs.len();
    let image_count = metadata.image_defs.len();
    let mu_count = metadata.metadata_usages_count;

    let mut cr_addr = 0u64;
    let mut mr_addr = 0u64;

    if config.force_dump {
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        if let Some((cr, mr)) = macho.symbol_search() {
            emit_log(app, "Detected Symbol!");
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        if let Some((cr, mr)) = macho.search_mod_init_func(version) {
            emit_log(app, "Found via __mod_init_func search");
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        let mut helper =
            macho.get_section_helper(method_count, type_count, mu_count, image_count, version);
        let metadata_reg = if metadata.variant == MetadataVariant::Codm {
            helper.find_metadata_registration_codm().or_else(|| helper.find_metadata_registration())
        } else {
            helper.find_metadata_registration()
        };
        if let (Some(cr), Some(mr)) = (helper.find_code_registration(), metadata_reg) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        emit_log(app, "Auto mode failed, requesting manual addresses...");
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        } else {
            return Err(error::Error::Other("Manual mode cancelled.".into()));
        }
    }

    let va_segments: Vec<VaSegment> = macho
        .segments
        .iter()
        .map(|s| VaSegment {
            vaddr: s.vmaddr,
            memsz: s.vmsize,
            offset: s.fileoff,
        })
        .collect();

    let try_init = |cr: u64, mr: u64| -> error::Result<Il2Cpp> {
        let mut il2cpp = Il2Cpp::new(macho.stream.clone(), version, macho.is_32bit);
        il2cpp.va_segments = va_segments.clone();
        il2cpp.codm = resolve_codm(config, metadata);
        il2cpp.init_with_auto_plus(cr, mr, &|addr| macho.map_vatr(addr), false)?;
        Ok(il2cpp)
    };

    let mut il2cpp = match try_init(cr_addr, mr_addr) {
        Ok(i) => i,
        Err(e) => {
            emit_log(
                app,
                &format!(
                    "Auto-detected CodeRegistration / MetadataRegistration failed to parse ({}). \
                     Likely a byte-pattern false positive. Please enter the correct addresses \
                     (e.g. from IDA / Ghidra cross-references on s_Il2CppCodeRegistration / \
                     s_Il2CppMetadataRegistration).",
                    e
                ),
            );
            let (cr2, mr2) = match prompt_manual_addresses(app, state) {
                Some(v) => v,
                None => return Err(error::Error::Other("Manual mode cancelled.".into())),
            };
            cr_addr = cr2;
            mr_addr = mr2;
            try_init(cr_addr, mr_addr)?
        }
    };

    il2cpp.data_sections = macho.data_search_sections();

    if macho.is_32bit {
        for ptr in il2cpp.method_pointers.iter_mut() {
            if *ptr > 0 {
                *ptr -= 1;
            }
        }
        for ptr in il2cpp.custom_attribute_generators.iter_mut() {
            if *ptr > 0 {
                *ptr -= 1;
            }
        }
    }

    let macho_exports = macho.list_exported_symbols();
    il2cpp.exported_symbols = macho_exports.iter().map(|(n, _)| n.clone()).collect();
    for (name, addr) in macho_exports {
        if name.starts_with("il2cpp_") || name.starts_with("mono_") {
            let rva = il2cpp.get_rva(addr);
            il2cpp.api_export_rvas.insert(name, rva);
        }
    }
    emit_log(
        app,
        &format!(
            "Found {} exported symbols, {} IL2CPP APIs",
            il2cpp.exported_symbols.len(),
            il2cpp.api_export_rvas.len()
        ),
    );

    Ok(il2cpp)
}

fn init_nso(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    emit_log(app, "Detected NSO format");
    let mut nso = Nso::new(data)?;
    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };
    nso.stream.version = version;
    nso.stream.is_32bit = nso.is_32bit;
    emit_log(app, &format!("IL2CPP Version: {version}"));

    if nso.check_dump() {
        emit_log(app, "Detected this may be a dump file.");
        if let Some(addr) = prompt_dump_address(app, state) {
            nso.stream.image_base = addr;
        }
    }

    let mut cr_addr = 0u64;
    let mut mr_addr = 0u64;

    if config.force_dump {
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        emit_log(app, "Searching...");
        let method_count = metadata
            .method_defs
            .iter()
            .filter(|m| m.method_index >= 0)
            .count();
        let mut helper = nso.get_section_helper(
            method_count,
            metadata.type_defs.len(),
            metadata.metadata_usages_count,
            metadata.image_defs.len(),
            version,
        );
        if let (Some(cr), Some(mr)) = (
            helper.find_code_registration(),
            helper.find_metadata_registration(),
        ) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        } else {
            emit_log(app, "Auto mode failed, requesting manual addresses...");
            if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
                cr_addr = cr;
                mr_addr = mr;
            } else {
                return Err(error::Error::Other("Manual mode cancelled.".into()));
            }
        }
    }

    let stream_len = nso.stream.data().len() as u64;
    let mut il2cpp = Il2Cpp::new(nso.stream.clone(), version, nso.is_32bit);
    il2cpp.va_segments = vec![VaSegment {
        vaddr: 0,
        memsz: stream_len,
        offset: 0,
    }];
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| nso.map_vatr(addr), false)?;
    il2cpp.data_sections = nso.data_search_sections();

    if let Ok(nso_exports) = nso.list_exported_symbols() {
        il2cpp.exported_symbols = nso_exports.iter().map(|(n, _)| n.clone()).collect();
        for (name, addr) in nso_exports {
            if name.starts_with("il2cpp_") || name.starts_with("mono_") {
                let rva = il2cpp.get_rva(addr);
                il2cpp.api_export_rvas.insert(name, rva);
            }
        }
    }
    emit_log(
        app,
        &format!(
            "Found {} exported symbols, {} IL2CPP APIs",
            il2cpp.exported_symbols.len(),
            il2cpp.api_export_rvas.len()
        ),
    );

    Ok(il2cpp)
}

fn init_wasm(
    data: Vec<u8>,
    metadata: &Metadata,
    config: &Config,
    app: &AppHandle,
    state: &AppState,
) -> error::Result<Il2Cpp> {
    emit_log(app, "Detected WASM format");
    let mut wasm = Wasm::new(data)?;
    let version = if config.force_il2cpp_version {
        config.force_version
    } else {
        metadata.version
    };
    wasm.stream.version = version;
    wasm.stream.is_32bit = wasm.is_32bit;
    emit_log(app, &format!("IL2CPP Version: {version}"));

    if wasm.check_dump() {
        emit_log(app, "Detected this may be a dump file.");
        if let Some(addr) = prompt_dump_address(app, state) {
            wasm.stream.image_base = addr;
        }
    }

    let mut cr_addr = 0u64;
    let mut mr_addr = 0u64;

    if config.force_dump {
        if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
            cr_addr = cr;
            mr_addr = mr;
        }
    }

    if cr_addr == 0 || mr_addr == 0 {
        emit_log(app, "Searching...");
        let method_count = metadata
            .method_defs
            .iter()
            .filter(|m| m.method_index >= 0)
            .count();
        let mut helper = wasm.get_section_helper(
            method_count,
            metadata.type_defs.len(),
            metadata.metadata_usages_count,
            metadata.image_defs.len(),
            version,
        );
        if let (Some(cr), Some(mr)) = (
            helper.find_code_registration(),
            helper.find_metadata_registration(),
        ) {
            emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
            emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
            cr_addr = cr;
            mr_addr = mr;
        } else {
            emit_log(app, "Auto mode failed, requesting manual addresses...");
            if let Some((cr, mr)) = prompt_manual_addresses(app, state) {
                emit_log(app, &format!("CodeRegistration : 0x{cr:x}"));
                emit_log(app, &format!("MetadataRegistration : 0x{mr:x}"));
                cr_addr = cr;
                mr_addr = mr;
            } else {
                return Err(error::Error::Other("Manual mode cancelled.".into()));
            }
        }
    }

    let stream_len = wasm.stream.data().len() as u64;
    let mut il2cpp = Il2Cpp::new(wasm.stream.clone(), version, wasm.is_32bit);
    il2cpp.va_segments = vec![VaSegment {
        vaddr: 0,
        memsz: stream_len,
        offset: 0,
    }];
    // WebAssembly bytecode, not machine code — see `Architecture::Wasm`.
    // Without this the 32-bit/non-PE default (`Arm32`) applies and the ARM
    // disassembler decodes wasm bytes as instructions.
    il2cpp.arch = Some(crate::disassembler::Architecture::Wasm);
    il2cpp.codm = resolve_codm(config, metadata);
    il2cpp.init_with_auto_plus(cr_addr, mr_addr, &|addr| wasm.map_vatr(addr), true)?;
    il2cpp.data_sections = wasm.data_search_sections();
    Ok(il2cpp)
}

fn run_dump(
    app: &AppHandle,
    state: &AppState,
    binary_path: &str,
    metadata_path: &str,
    output_dir: &str,
    config: &Config,
) -> std::result::Result<String, String> {
    let start_time = Instant::now();
    let mut config = config.clone();
    config.reconcile_method_body_mode();
    if config.il_recovery && !config.generate_dummy_dll {
        config.generate_dummy_dll = true;
        emit_log(app, "IL recovery requires DummyDLL — enabling generateDummyDll");
    }
    let config = &config;
    if config.dump_pseudo_c || config.dump_disassembly {
        let mode = match (config.dump_pseudo_c, config.dump_disassembly) {
            (true, true) => "C# pseudo + assembly",
            (true, false) => "C# pseudo",
            (false, true) => "assembly",
            _ => unreachable!(),
        };
        emit_log(app, &format!("Method body output: {mode}"));
    }
    if !config.dump_pseudo_c_include_assemblies.is_empty() {
        emit_log(
            app,
            &format!(
                "Class filter assemblies ({}): {}",
                config.dump_pseudo_c_include_assemblies.len(),
                config.dump_pseudo_c_include_assemblies.join(", ")
            ),
        );
    }
    if !config.dump_pseudo_c_include_types.is_empty() {
        let preview: Vec<_> = config
            .dump_pseudo_c_include_types
            .iter()
            .take(8)
            .cloned()
            .collect();
        let more = config.dump_pseudo_c_include_types.len().saturating_sub(preview.len());
        let mut msg = format!(
            "Class filter types ({}): {}",
            config.dump_pseudo_c_include_types.len(),
            preview.join(", ")
        );
        if more > 0 {
            msg.push_str(&format!(" … (+{more} more)"));
        }
        emit_log(app, &msg);
    }

    let base_dir = std::path::Path::new(output_dir);
    let mut dump_num = 0u32;
    while base_dir.join(format!("Dump{dump_num}")).exists() {
        dump_num += 1;
    }
    let final_output = base_dir
        .join(format!("Dump{dump_num}"))
        .to_string_lossy()
        .to_string();
    fs::create_dir_all(&final_output).map_err(|e| format!("Failed to create output dir: {e}"))?;

    emit_log(app, "Initializing IL2CPP binary...");
    let il2cpp_bytes = fs::read(binary_path).map_err(|e| format!("Failed to read binary: {e}"))?;

    let unity_version_str = detect_unity_version(&il2cpp_bytes);
    if let Some(ref uv) = unity_version_str {
        emit_log(app, &format!("Unity Version: {uv}"));
    }

    emit_log(app, "Initializing metadata...");
    let mut metadata_bytes =
        fs::read(metadata_path).map_err(|e| format!("Failed to read metadata: {e}"))?;
    let metadata_magic = read_magic_u32(&metadata_bytes);
    if metadata_magic != MAGIC_METADATA {
        match try_decrypt_metadata(&mut metadata_bytes) {
            Some(scheme) => emit_log(
                app,
                &format!("Encrypted metadata detected ({scheme}), decrypting..."),
            ),
            None => {
                let b0 = (metadata_magic & 0xFF) as u8;
                let b1 = ((metadata_magic >> 8) & 0xFF) as u8;
                let b2 = ((metadata_magic >> 16) & 0xFF) as u8;
                let b3 = ((metadata_magic >> 24) & 0xFF) as u8;
                return Err(format!(
                    "Metadata error: wrong magic 0x{metadata_magic:08X} \
                     (bytes {b0:02X} {b1:02X} {b2:02X} {b3:02X}). Expected AF 1B B1 FA. \
                     File is likely encrypted, obfuscated, truncated, or not global-metadata.dat. \
                     Protected games often need a runtime memory dump of decrypted metadata."
                ));
            }
        }
    }

    let mut metadata = Metadata::new_with_options(
        metadata_bytes,
        unity_version_str.as_deref(),
        config.codm,
    )
    .map_err(|e| e.user_message())?;
    emit_log(app, &format!("Metadata Version: {}", metadata.version));

    let format_name = detect_format(&il2cpp_bytes);
    emit_log(app, &format!("Binary format: {format_name}"));

    let mut il2cpp = match format_name {
        "ELF" => init_elf(il2cpp_bytes, &metadata, config, app, state),
        "PE" => init_pe(il2cpp_bytes, &metadata, config, app, state),
        "Mach-O" => init_macho(il2cpp_bytes, &metadata, config, app, state),
        "Fat Mach-O" => init_macho_fat(il2cpp_bytes, &metadata, config, app, state),
        "NSO" => init_nso(il2cpp_bytes, &metadata, config, app, state),
        "WASM" => init_wasm(il2cpp_bytes, &metadata, config, app, state),
        _ => {
            let magic = read_magic_u32(&il2cpp_bytes);
            return Err(format!("Unsupported binary format (magic: 0x{magic:08X})"));
        }
    }
    .map_err(|e| format!("{e}"))?;

    if il2cpp.version >= 27.0 && il2cpp.is_dumped {
        if let Some(type_def) = metadata.type_defs.first() {
            let byval_idx = type_def.byval_type_index as usize;
            if byval_idx < il2cpp.types.len() {
                let il2cpp_type = &il2cpp.types[byval_idx];
                let type_handle = il2cpp_type.type_handle();
                il2cpp.image_base =
                    type_handle.wrapping_sub(metadata.header.type_definitions_offset as u64);
            }
        }
    }

    emit_log(app, "Dumping...");
    let mut executor = Il2CppExecutor::new(&metadata, &mut il2cpp).map_err(|e| format!("{e}"))?;

    let app_clone = app.clone();
    Il2CppDecompiler::decompile(
        &mut executor,
        &mut metadata,
        &mut il2cpp,
        config,
        binary_path,
        &final_output,
        |msg| {
            emit_log(&app_clone, msg);
        },
    )
    .map_err(|e| format!("{e}"))?;
    emit_log(app, "dump.cs generated");

    let static_catalog = if config.dump_static_field_metadata {
        emit_log(app, "Analyzing thread-static / FieldRVA metadata...");
        let catalog = crate::output::static_field_exporter::StaticFieldCatalog::collect(
            &mut executor, &mut metadata, &mut il2cpp, config,
        ).map_err(|e| format!("{e}"))?;
        crate::output::static_field_exporter::StaticFieldExporter::write_document(
            &catalog, &il2cpp, &final_output,
        ).map_err(|e| format!("{e}"))?;
        emit_log(
            app,
            &format!(
                "static_metadata.json generated ({} thread-static, {} FieldRVA)",
                catalog.summary.thread_static_count,
                catalog.summary.field_rva_count,
            ),
        );
        Some(catalog)
    } else {
        None
    };

    if config.generate_struct {
        emit_log(app, "Generating struct...");
        StructGenerator::write_all(
            &mut executor, &mut metadata, &mut il2cpp, config, &final_output,
            static_catalog.as_ref(),
        ).map_err(|e| format!("{e}"))?;
        crate::output::embedded_scripts::write_scripts(std::path::Path::new(&final_output))
            .map_err(|e| format!("{e}"))?;
        emit_log(app, "script.json, il2cpp.h, stringliteral.json generated");
    }

    if config.generate_dummy_dll {
        emit_log(app, "Generating dummy dll...");
        crate::output::dummy_assembly_generator::generate_dummy_dlls(
            &mut executor,
            &mut metadata,
            &mut il2cpp,
            config,
            &final_output,
        )
        .map_err(|e| format!("{e}"))?;
        emit_log(app, "Dummy dll files generated");
    }

    if config.generate_generics_dump {
        emit_log(app, "Generating generics dump...");
        let generics_path = std::path::Path::new(&final_output).join("generics_dump.txt");
        if let Err(e) = crate::output::generics::dump_generics(
            &generics_path.to_string_lossy(),
            &mut metadata,
            &mut il2cpp,
            &mut executor,
            config,
        ) {
            emit_log(
                app,
                &format!("Warning: Failed to generate generics_dump.txt: {}", e),
            );
        } else {
            emit_log(app, "generics_dump.txt generated");
        }
    }

    let elapsed = start_time.elapsed();
    emit_log(app, &format!("Done! ({:.2}s)", elapsed.as_secs_f64()));

    Ok(final_output)
}

#[tauri::command]
fn detect_binary(path: String) -> std::result::Result<BinaryInfo, String> {
    let data = fs::read(&path).map_err(|e| format!("Failed to read file: {e}"))?;
    let format = detect_format(&data).to_string();
    let unity_version = detect_unity_version(&data).unwrap_or_default();
    Ok(BinaryInfo {
        format,
        unity_version,
    })
}

#[tauri::command]
fn start_dump(
    app: AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    binary_path: String,
    metadata_path: String,
    output_dir: String,
    config_json: String,
) -> Result<(), String> {
    {
        let mut running = state.dump_running.lock().unwrap();
        if *running {
            return Err("A dump is already in progress".into());
        }
        *running = true;
    }

    let state = Arc::clone(&state);
    std::thread::spawn(move || {
        struct DumpGuard(Arc<AppState>);
        impl Drop for DumpGuard {
            fn drop(&mut self) {
                *self.0.dump_running.lock().unwrap() = false;
            }
        }
        let _guard = DumpGuard(Arc::clone(&state));

        let mut config: Config = match serde_json::from_str(&config_json) {
            Ok(c) => c,
            Err(e) => {
                // Last-chance salvage: pull methodBodyMode / legacy bools out of
                // the raw JSON so a single bad field cannot silently disable C#.
                let mut salvaged = Config::default();
                if let Ok(raw) = serde_json::from_str::<serde_json::Value>(&config_json) {
                    if let Some(mode) = raw.get("methodBodyMode").and_then(|v| v.as_str()) {
                        if let Some(m) = crate::config::MethodBodyMode::parse(mode) {
                            salvaged.set_method_body_mode(m);
                        }
                    } else {
                        let pseudo = raw
                            .get("dumpPseudoC")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        let disasm = raw
                            .get("dumpDisassembly")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        salvaged.set_method_body_mode(
                            crate::config::MethodBodyMode::from_flags(disasm, pseudo),
                        );
                    }
                }
                let _ = app.emit(
                    "dump-log",
                    LogEvent {
                        message: format!(
                            "Config JSON parse failed ({e}); salvaged methodBodyMode={:?} — check dump options",
                            salvaged.method_body_mode
                        ),
                    },
                );
                salvaged
            }
        };
        config.reconcile_method_body_mode();
        let app_for_panic = app.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_dump(
                &app,
                &state,
                &binary_path,
                &metadata_path,
                &output_dir,
                &config,
            )
        }));
        match result {
            Ok(Ok(output_path)) => {
                let _ = app_for_panic.emit(
                    "dump-complete",
                    DumpCompleteEvent {
                        success: true,
                        output_path,
                        error_message: String::new(),
                    },
                );
            }
            Ok(Err(error_message)) => {
                let _ = app_for_panic.emit(
                    "dump-complete",
                    DumpCompleteEvent {
                        success: false,
                        output_path: String::new(),
                        error_message,
                    },
                );
            }
            Err(panic_info) => {
                let panic_msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                let timestamp = {
                    let dur = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let secs = dur % 60;
                    let mins = (dur / 60) % 60;
                    let hrs = (dur / 3600) % 24;
                    format!("{hrs:02}:{mins:02}:{secs:02} UTC")
                };
                let crash_log = format!(
                    "=== IL2CPP Dumper Crash Report ===\nTime: {}\nThread: dump-worker\nOS: {} {}\n\n=== Exception ===\nRust Panic\n{}\n",
                    timestamp,
                    std::env::consts::OS,
                    std::env::consts::ARCH,
                    panic_msg
                );
                let _ = app_for_panic.emit("dump-crash", CrashEvent { crash_log });
            }
        }
    });
    Ok(())
}

#[tauri::command]
fn submit_input(state: tauri::State<'_, Arc<AppState>>, response: String) {
    let sender = state.input_sender.lock().unwrap();
    if let Some(ref tx) = *sender {
        let _ = tx.send(response);
    }
}

#[tauri::command]
fn get_default_config() -> String {
    serde_json::to_string_pretty(&Config::default()).unwrap_or_default()
}

/// Load `global-metadata.dat` and return the assembly → namespace → type tree
/// for the GUI class picker. Does not require the IL2CPP binary.
#[tauri::command]
fn list_type_tree(metadata_path: String, codm: Option<bool>) -> Result<String, String> {
    let mut metadata_bytes =
        fs::read(&metadata_path).map_err(|e| format!("Failed to read metadata: {e}"))?;
    let metadata_magic = read_magic_u32(&metadata_bytes);
    if metadata_magic != MAGIC_METADATA {
        try_decrypt_metadata(&mut metadata_bytes).ok_or_else(|| {
            format!(
                "Metadata error: wrong magic 0x{metadata_magic:08X}. Expected AF 1B B1 FA."
            )
        })?;
    }
    let metadata = Metadata::new_with_options(metadata_bytes, None, codm.unwrap_or(false))
        .map_err(|e| e.user_message())?;
    let tree = crate::output::pseudo_c::list_metadata_type_tree(&metadata);
    serde_json::to_string(&tree).map_err(|e| format!("JSON encode error: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_state = Arc::new(AppState {
        input_sender: Mutex::new(None),
        dump_running: Mutex::new(false),
    });

    tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            detect_binary,
            start_dump,
            submit_input,
            get_default_config,
            list_type_tree,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
