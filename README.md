<p align="center">
  <img src="https://img.shields.io/badge/Tauri-FFC131?style=for-the-badge&logo=tauri&logoColor=black" />
  <img src="https://img.shields.io/badge/SvelteKit-FF3E00?style=for-the-badge&logo=svelte&logoColor=white" />
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/License-GPLv3-green?style=for-the-badge" />
</p>

<h1 align="center">🛡️ Rodroid Il2CppDumper — Desktop & Mobile GUI</h1>

<p align="center">
  <b>A modern, cross-platform GUI for the Rodroid Il2CppDumper engine.</b><br/>
  Built with <a href="https://tauri.app/">Tauri 2</a>, <a href="https://kit.svelte.dev/">SvelteKit</a>, and <a href="https://www.typescriptlang.org/">TypeScript</a> — runs natively on Windows, macOS, Linux, Android, and iOS.
</p>

<p align="center">
  <a href="https://t.me/+WmudnO0-xoNhMDQ8">📢 Telegram Channel</a> &nbsp;·&nbsp;
  <a href="https://t.me/+QylrYL1GNsJiYjc0">💬 Telegram Group</a> &nbsp;·&nbsp;
  <b>Dev:</b> <a href="https://t.me/rodroidmods"><code>@rodroidmods</code></a>
</p>

---
## ✨ What's New in V8 — The C# Decompiler

**V8 brings the C# decompiler to the desktop and mobile app**, after 1.5 months
of focused development. It recovers native IL2CPP method bodies into readable
C#-style output, alongside the metadata and type information in `dump.cs`.

### Native method recovery

- Recover C#-style pseudocode for supported native architectures.
- Process methods in parallel while retaining the same recovery features in
  single-worker mode.
- Focus a dump on selected assemblies or types.

### Broader game and metadata compatibility

- **Free Fire and Free Fire MAX memory dumps:** fixed crashes while processing
  memory-dump files.
- **Unity 6000.1:** fixed a metadata parsing issue.
- **Metadata versions 106.1, 107, and 108:** added support.
- Includes further fixes to parsing, native analysis, and dump generation.

V8 makes the app useful for exploring recovered method logic as well as IL2CPP
types, fields, and metadata.

## ✨ What's New in V7

V7 introduced multi-core dumping, clearer errors, and stability improvements to
the shared engine. V8 builds on that foundation with native C# method recovery.

## ✨ What's New in V6.1

### Engine
- 🐛 **Fixed CODM disk-based dumps crashing with "Auto mode failed"** — obfuscated internal pointers in `CodeRegistration`/`MetadataRegistration` structs no longer abort the entire `init()`. Bad pointers are skipped with a `[WARN]` when CODM mode is active; non-CODM games retain strict error propagation.
- 🛡️ **CODM-conditional resilience** in `load_pointers`, `load_types`, `load_generics` (ELF) and `Il2Cpp::init` (Mach-O/PE/NSO/WASM)
- 🔄 **All targets synced**: Tauri desktop/mobile, CLI, Android JNI — same engine fix

---

## ✨ What's New in V6

### Engine (same as CLI v6)
- Thread-static + FieldRVA export (`static_metadata.json`, dump.cs annotations, DummyDll) — **off by default**, enable in Dump Options
- Nested FieldRVA options when static metadata is on: hex data toggle + max bytes cap
- iOS CODM auto-flag parity with Android
- Unity 27+ FieldRva binary scan on PE / Mach-O / NSO / WASM
- Memory-dumped CODM binary fixes (image-base / dump-file path alongside CODM decryption)

### UI transformation (Material 3)
The desktop/mobile GUI was rebuilt on **[noph-ui](https://www.npmjs.com/package/noph-ui)** — a Material 3 component kit for Svelte — to match the Android Jetpack Compose look and feel:

- **Material 3 theming** — `data-theme` dark / light / system, dynamic surface containers, M3 typography and elevation
- **Migrated screens** — Idle, Dumping (live log), Result, Error, Settings, About, Splash, Crash, Config dialog, input prompts
- **Android-parity layout** — full-width Start Dump bar, tonal Dump Options button, labeled path cards with folder pickers
- **Config dialog sections** — Output · Generation · Advanced Generics · C++ Headers · Disassembly · **Static Field Metadata** · Advanced
- **Animations** — staggered card entrance, collapsible nested toggles (generics / disassembly / static metadata), dialog scrim fade
- **Custom components** — `PathInput` (full-width monospace paths), `ConfigSwitch`, `AnimatedExpand`
- **i18n** — 7 languages (EN, SQ, AR, ES, HI, ID, JV)

> Skeleton UI was removed in favour of noph-ui M3 primitives (`Button`, `Switch`, `TextField`, `SegmentedButton`, `ChipSet`, etc.).

## ✨ What's New in V5

- Live streaming logs with per-step registration addresses
- Full disassembly engine exposed in the config panel
- CODM toggle, mobile file pickers, auto Unity version chips

---

## 🖼️ Screens

```
┌─────────────────────────────────────┐
│  Rodroid IL2CPP Dumper    v8 Desktop│
│  ─────────────────────────────────  │
│  IL2CPP BINARY    [path]      [📁]  │
│  METADATA         [path]      [📁]  │
│  [ Dump Options ]                   │
│  ┌───────────────────────────────┐  │
│  │  ▶  Start Dump                │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

**Dump Options** dialog sections:

- **Output** — methods, fields, properties, attributes, offsets, typedef indices, assembly names, split-per-type
- **Generation** — structs, DummyDLLs, tokens
- **Advanced Generics** — master toggle + 7 sub-options (RGCTX, MethodSpecs, …)
- **C++ Headers** — scaffold, mangling, IDA metadata, Unity headers, topological sort, GCC/MSVC
- **Disassembly** — target (dump.cs / DiffableCs / both), hex bytes, field names, annotations, CFG, max instructions
- **Static Field Metadata** — thread-static / FieldRVA export + nested FieldRVA hex + max bytes
- **Advanced** — force IL2CPP version, force dump, no-redirected-pointer, CODM

---

## 🚀 Features

All capabilities of the [il2cpp_dumper](../il2cpp_dumper/README.md) CLI are available through the GUI:

### Core
- IL2CPP **v16 – v39** (Unity 5.3 → Unity 6), plus metadata formats v104, v106, v106.1, v107, and v108
- Auto XOR metadata decryption (1-byte, 4-byte, 8-byte, rolling, position-dependent)
- ELF / PE / Mach-O / Fat Mach-O / NSO / WASM
- Variable-width indices for v39 / Unity 6
- Native C# decompiler for recovering method bodies, with assembly/type filters
- Auto-numbered output directories (Dump0/, Dump1/...)

### Disassembly Engine (V5 highlights)
- **Backward slicing** — resolves `BLR Xn` virtual calls into `// virtual call: TypeName.MethodName`
- **Init-check folding** — collapses `il2cpp_codegen_initialize_method` / `Il2CppCodeGenWriteBarrier` / TBZ-on-bit-0 prologues into a single `// [init check]` annotation
- **String literal indirect resolution** — annotates `il2cpp_string_new_wrapper` calls with the actual literal content
- **Generic instantiation tracking** — annotates calls with the concrete specialization (`// → List<int>.Add(this, item)`)
- **Switch table reconstruction** — detects ARM64 jump tables and emits `switch (var)` blocks in the CFG
- **Boxing / unboxing detection** — annotates `il2cpp_codegen_box` / `il2cpp_unbox` with the resolved type
- **Static field access annotation** — resolves the `ADRP+ADD → static_fields → field_offset` chain
- Plus everything from V4: forward constant propagation, register-pair memory access, semantic variable tracking, ARM64/ARM32/x86/x64 multi-arch decoders

### CODM (Call of Duty Mobile)
- Custom v23 metadata layout with two-slot `type_definitions_count` fingerprint anchor
- Android packed relocations (`DT_ANDROID_RELA` / `DT_ANDROID_REL`, APS2 + SLEB128) — 32-bit and 64-bit ELF
- iOS chained fixups (`LC_DYLD_CHAINED_FIXUPS`) and legacy rebase opcodes (`LC_DYLD_INFO_ONLY`) — 32-bit and 64-bit Mach-O
- Pointer formats: `DYLD_CHAINED_PTR_64`, `_64_OFFSET`, ARM64E
- Toggle from the **Advanced** section of the config dialog — additive code path, leaves standard Unity games untouched

---

## 📦 Installation

### Prebuilt Binaries
Grab the latest installer for your platform from the [Releases](https://github.com/rodroidmods/il2cpp-dumper-rs/releases) page.

| Platform | Artifact |
|----------|----------|
| Windows | `Rodroid-Il2CppDumper-setup.exe` / `.msi` |
| macOS | `Rodroid-Il2CppDumper.dmg` |
| Linux | `.AppImage` / `.deb` / `.rpm` |
| Android | `.apk` |
| iOS | `.ipa` (sideload via AltStore / TrollStore) |

### From Source

Requirements:
- [Node.js](https://nodejs.org/) ≥ 18 + [pnpm](https://pnpm.io/)
- [Rust](https://rustup.rs/) stable toolchain
- Tauri prerequisites for your OS — see the [Tauri prerequisites guide](https://tauri.app/start/prerequisites/)

```bash
git clone https://github.com/rodroidmods/il2cpp-dumper-rs.git
cd il2cpp-dumper-rs/rodroid-il2cppdumper
pnpm install
```

Run in dev mode:
```bash
pnpm tauri dev
```

Build a release bundle:
```bash
pnpm tauri build
```

Mobile targets:
```bash
pnpm tauri android init
pnpm tauri android dev      # or: pnpm tauri android build

pnpm tauri ios init
pnpm tauri ios dev          # or: pnpm tauri ios build
```

---

## 🔧 Usage

1. Launch the app
2. **Pick a binary** — `libil2cpp.so`, `GameAssembly.dll`, `UnityFramework`, `.nso`, or `.wasm`
3. **Pick a metadata file** — usually `global-metadata.dat`
4. *(Optional)* Open **Dump Options** — enable **Static Field Metadata**, disassembly, CODM, etc.
5. Hit **Start Dump** — watch the live log and find the output in an auto-numbered `Dump0/`, `Dump1/`, … folder

---

## 🏗️ Architecture

```
rodroid-il2cppdumper/
├── src/                              # SvelteKit + noph-ui (Material 3)
│   ├── lib/
│   │   ├── components/               # IdleScreen, ConfigDialog, DumpingScreen, …
│   │   ├── dumpRunner.ts             # single-shot dump start + log cap
│   │   ├── dumpEvents.ts             # Tauri event listeners
│   │   ├── stores.ts                 # config, i18n, theme
│   │   └── types.ts                  # DumperConfig (incl. v6 static keys)
│   ├── app.css                       # M3 tokens, dialog, animations
│   └── routes/+page.svelte           # screen router + config dialog host
├── src-tauri/
│   ├── src/lib.rs                    # Tauri commands + v8 dump pipeline
│   ├── config.json                   # default Rust config reference
│   └── Cargo.toml
└── package.json                      # frontend scripts and dependencies
```

The backend wraps the [`il2cpp_dumper`](../il2cpp_dumper) Rust crate as a library and streams progress events back to the frontend via Tauri's event system.

---

## 📜 License

GNU GPLv3

---

## 🙏 Credits

- [Perfare/Il2CppDumper](https://github.com/Perfare/Il2CppDumper) — Original C# implementation
- [SamboyCoding/Cpp2IL](https://github.com/SamboyCoding/Cpp2IL) — Advanced IL2CPP analysis tool
- [tauri-apps](https://tauri.app/) — Cross-platform native shell
- [SvelteKit](https://kit.svelte.dev/) — Frontend framework
- [noph-ui](https://www.npmjs.com/package/noph-ui) — Material 3 components for Svelte
- [console-rs](https://github.com/console-rs) — Terminal styling ecosystem reused for the CLI

---

## 📬 Community

| | Link |
|---|---|
| 📢 **Telegram Channel** | [Join Channel](https://t.me/+WmudnO0-xoNhMDQ8) |
| 💬 **Telegram Group** | [Join Group](https://t.me/+QylrYL1GNsJiYjc0) |
| 👤 **Developer** | [`@rodroidmods`](https://t.me/rodroidmods) |

---

> **⚠️ Disclaimer**: This tool is for educational and research purposes only. Respect game developers' rights and terms of service.
