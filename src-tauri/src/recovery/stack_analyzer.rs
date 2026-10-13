//! Native stack analysis (port of Cpp2IL `Analysis/StackAnalyzer.cs`).
//!
//! Tracks net push depth per block via worklist fixpoint, then rewrites
//! sp-relative memory operands into position-independent virtual registers
//! named `stack_N` — exactly Cpp2IL's scheme. ARM64 prologues mostly address
//! locals off a frame pointer copied from sp, so we also resolve those aliases.

use super::cfg::Cfg;
use super::ir::{Insn, Op, Operand};
use std::collections::{HashMap, HashSet};

pub struct StackAnalysis {
    /// Net stack delta at entry to each block.
    pub in_depth: Vec<i64>,
}

pub fn analyze(insns: &mut [Insn], cfg: &Cfg) -> StackAnalysis {
    analyze_with_ptr_size(insns, cfg, 8)
}

/// `ptr_size` selects the slot scale (8 for ARM64/x64, 4 for ARM32).
pub fn analyze_with_ptr_size(insns: &mut [Insn], cfg: &Cfg, ptr_size: u64) -> StackAnalysis {
    let mut in_state: Vec<Option<i64>> = vec![None; cfg.blocks.len()];
    let mut out_state: Vec<i64> = vec![0; cfg.blocks.len()];
    let mut insn_depth: HashMap<usize, i64> = HashMap::new();

    // Convergence budget: normal dataflow needs a few visits per block, but
    // a cycle with non-zero net ShiftStack (mislifted prologues, LDP/STP
    // offset loss) diverges forever — the old fixed 500k-per-block cap let
    // that burn CPU for seconds per method. Budget the TOTAL work instead;
    // when it runs out the depths are simply approximate.
    let total_budget: u64 = (cfg.blocks.len() as u64).saturating_mul(64) + 256;
    let mut budget = total_budget;

    // only revisit successors when the incoming depth actually changed
    // (standard dataflow convergence; prevents loop-thrash)
    let mut worklist: Vec<(usize, i64)> = vec![(0, 0)];
    while let Some((bi, incoming)) = worklist.pop() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        if in_state[bi].is_some() {
            if in_state[bi] == Some(incoming) {
                continue;
            }
        }
        in_state[bi] = Some(incoming);

        let b = &cfg.blocks[bi];
        let mut depth = incoming;
        for i in b.start..b.end {
            insn_depth.insert(i, depth);
            if let Op::ShiftStack(d) = insns[i].op {
                depth += d;
            }
        }
        out_state[bi] = depth;

        for &succ in cfg.successors[bi].iter() {
            if in_state[succ] != Some(depth) {
                worklist.push((succ, depth));
            }
        }
    }

    excise_callee_save_round_trips(insns, &insn_depth);
    resolve_frame_aliases(insns, &insn_depth, ptr_size);
    rewrite_sp_operands(insns, &insn_depth, ptr_size);

    StackAnalysis { in_depth: out_state }
}

/// Excise callee-saved register save/restore round-trips: the prologue's
/// `stp d9,d8,[sp,#-96]!` and every matching epilogue `ldp d9,d8,[sp],#96`
/// are pure ABI machinery — the saved register's incoming value is stored
/// and restored untouched, so the pair has NO source-level semantics. Left
/// in, they poison the pipeline two ways: the store READS the incoming
/// register (an SSA entry version never defined in-body → renders as
/// `v41 = v10` with v10 undefined), and the restore defines a version
/// nobody uses. Standard decompilers hide these entirely.
///
/// Pairing is by REGISTER IDENTITY (not stack slot): the epilogue's sp
/// writeback can be lifted before the loads, making depth-relative slot
/// arithmetic unreliable — but the register is exact on both sides.
/// Safety rules:
/// 1. Whitelist only CALLEE-SAVED registers (x19–x30, d/v8–15, rbx/rbp/
///    r12–r15, XMM6–15, …). Caller-saved spills (argument spills like
///    `str x0,[sp]`) carry live values across calls and must stay.
/// 2. A register with TWO prologue stores is skipped (frame-slot reuse).
/// 3. EVERY stack load of that register must sit in an epilogue envelope
///    (terminator within the next 16 insns, no call in between): a load
///    whose value stays live is a real spill-restore, not ABI machinery.
/// 4. A register with zero stack loads (saved, never restored — early-
///    exit paths) loses just the store. Safe either way: a restore's
///    value is dead at exit, and mid-body uses read the SSA entry
///    version, which survives (only the store's own read disappears).
/// Runs BEFORE sp-rewrite so operands are still raw `Memory{base:sp}`;
/// erased slots never become `stack_N` locals at all.
pub(crate) fn excise_callee_save_round_trips(
    insns: &mut [Insn],
    _insn_depth: &HashMap<usize, i64>,
) -> usize {
    use super::verify::{ledger_note, LedgerKind};

    /// Callee-saved register names across the three supported ISAs, as the
    /// lifters name them. Matching is exact-name; FP views included per view
    /// width (arm64 names SIMD regs "vd8"/"vs8" by width, x86 "vf6").
    const CALLEE_SAVED: &[&str] = &[
        // ARM64 integer: x19–x28 + FP(x29) + LR(x30)
        "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
        "x29", "x30",
        // ARM64 SIMD callee-saved v8–v15, per width view
        "vd8", "vd9", "vd10", "vd11", "vd12", "vd13", "vd14", "vd15",
        "vs8", "vs9", "vs10", "vs11", "vs12", "vs13", "vs14", "vs15",
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
        "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15",
        // x86-64 (SysV + MSVC union): rbx, rbp, r12–r15 (+rdi/rsi on MSVC)
        "rbx", "rbp", "r12", "r13", "r14", "r15", "rdi", "rsi",
        // …and the ABI-NORMALIZED names the x86 lifter actually emits
        // (`gpr_name_for_ordinal`): ordinal 3 = rbx → `g3`, 5 = rbp → `fp`,
        // 12–15 = r12–r15 → `g12`–`g15`, and 6/7 = rsi/rdi → `g6`/`g7` under
        // MSVC. SysV names ordinals 6/7 `x1`/`x0` (argument registers) and they
        // are deliberately NOT listed. Without these twins the raw names above
        // matched nothing x86 produced, so no x64 callee-save pair was ever
        // excised and every saved register stayed undefined (`g3` was the #1
        // residual origin at 603).
        "g3", "fp", "g6", "g7", "g12", "g13", "g14", "g15",
        // XMM6–15 callee-saved on Windows, harmless to include elsewhere
        "vf6", "vf7", "vf8", "vf9", "vf10", "vf11", "vf12", "vf13", "vf14",
        "vf15",
        // ARM32: r4–r11 + lr
        "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "lr",
        // x86-32
        "esi", "edi", "ebx", "ebp",
    ];
    let is_callee_saved = |r: &str| CALLEE_SAVED.contains(&r);
    let min_addr = insns.first().map(|i| i.address).unwrap_or(0);
    // Epilogue envelope: a terminator within the next 16 insns with no
    // call in between (a load feeding a call is live code, not a restore).
    let in_epilogue = |from: usize| -> bool {
        let end = (from + 17).min(insns.len());
        for n in &insns[from + 1..end] {
            match n.op {
                Op::Return | Op::Throw | Op::Jump(_) | Op::CondJump(_) | Op::IndirectCall
                | Op::Switch(_) => return true,
                Op::Call | Op::CallVoid | Op::NewObj(_) => return false,
                _ => {}
            }
        }
        false
    };
    // A STACK-relative access, by frame register. This must accept the FRAME
    // POINTER as well as the stack pointer: x64 saves a callee-saved register
    // as `mov [rbp+0x20], rbx` once `lea rbp,[rsp+0x20]` has established the
    // frame, and arm64 as `str x19, [x29, #…]`. Recognising only `sp` meant the
    // pairs went unseen and the SAVE survived with its read of the incoming
    // (undefined) register — `g3`/`g6`/`g7` were the top three residual
    // undefined-local origins (603/257/238) for exactly this reason. The
    // frame register is only ever a frame base in this IR (`gpr_name_for_ordinal`
    // names RBP `fp`, and arm64 names x29 `x29`), so accepting it is exact.
    // A STACK-relative access, as `(is_frame_based, byte_addend)`. It must
    // accept the FRAME POINTER as well as the stack pointer: x64 saves a
    // callee-saved register as `mov [rbp+0x20], rbx` once `lea rbp,[rsp+0x20]`
    // has established the frame, and arm64 as `str x19, [x29, #…]`. Recognising
    // only `sp` meant the pairs went unseen and the SAVE survived with its read
    // of the incoming (undefined) register — `g3`/`g6`/`g7` were the top three
    // residual undefined-local origins (603/257/238). The frame register is
    // only ever a frame base in this IR (`gpr_name_for_ordinal` names RBP `fp`,
    // and arm64 names x29 `x29`), so accepting it is exact.
    let stack_ref = |o: &Operand| -> Option<(bool, i64)> {
        match o {
            Operand::Memory { base, index: None, addend, .. } => match base.as_ref() {
                Operand::Register(b) if matches!(b.as_str(), "sp" | "esp" | "rsp") => {
                    Some((false, *addend))
                }
                Operand::Register(b) if matches!(b.as_str(), "fp" | "rbp" | "x29") => {
                    Some((true, *addend))
                }
                _ => None,
            },
            _ => None,
        }
    };

    // Prologue stores: callee-saved register spilled to the stack within the
    // first 64 bytes of the body. One store per register, else skip it.
    let mut store_of: HashMap<String, (usize, bool, i64)> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    // Registers already (re)written in-body. A SAVE stores the register's
    // INCOMING value, so it must precede any in-body write; a store of a value
    // the body just computed is an ordinary local store, not a save. Counting
    // it as a second "prologue store" marked the register AMBIGUOUS and blocked
    // the excision entirely — `mov [rbp+0x20],rbx` (save) followed 0x20 bytes
    // later by `mov [rbp+0x28],ebx` (store the just-zeroed value) left the save
    // in place, reading the undefined incoming `rbx` (`g3` was 603 findings).
    let mut written: HashSet<String> = HashSet::new();
    for (i, insn) in insns.iter().enumerate() {
        if let Some(pos) = Insn::dest_position(&insn.op) {
            if let Some(Operand::Register(d)) = insn.operands.get(pos) {
                written.insert(d.clone());
            }
        }
        if insn.op != Op::Move {
            continue;
        }
        let (Some(dest), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            continue;
        };
        let Some((save_frame, save_addend)) = stack_ref(dest) else { continue };
        let Operand::Register(r) = src else { continue };
        if !is_callee_saved(r) || insn.address.saturating_sub(min_addr) >= 64 {
            continue;
        }
        if written.contains(r) {
            continue;
        }
        if store_of.contains_key(r) {
            ambiguous.insert(r.clone());
        } else {
            store_of.insert(r.clone(), (i, save_frame, save_addend));
        }
    }
    if store_of.is_empty() {
        return 0;
    }

    // All stack loads per whitelisted register, anywhere in the body.
    let mut loads_of: HashMap<String, Vec<(usize, bool, i64)>> = HashMap::new();
    for (i, insn) in insns.iter().enumerate() {
        if insn.op != Op::Move {
            continue;
        }
        let (Some(dest), Some(src)) = (insn.operands.first(), insn.operands.get(1)) else {
            continue;
        };
        let Operand::Register(r) = dest else { continue };
        if !is_callee_saved(r) {
            continue;
        }
        let Some((load_frame, load_addend)) = stack_ref(src) else { continue };
        loads_of
            .entry(r.clone())
            .or_default()
            .push((i, load_frame, load_addend));
    }

    let mut to_erase: HashSet<usize> = HashSet::new();
    for (r, &(si, save_frame, save_addend)) in &store_of {
        if ambiguous.contains(r) {
            continue;
        }
        let Some(idxs) = loads_of.get(r) else {
            // saved but never restored (early-exit paths): drop the store
            to_erase.insert(si);
            continue;
        };
        // A load from a DIFFERENT frame slot is a reload of another spill, not
        // this save's restore — it must not block the excision. Only frame-based
        // accesses are compared: an `sp`-based epilogue goes through a writeback
        // whose addend cannot be matched against the prologue's.
        let considered: Vec<usize> = idxs
            .iter()
            .filter(|(_, load_frame, load_addend)| {
                !(*load_frame && save_frame && *load_addend != save_addend)
            })
            .map(|(i, _, _)| *i)
            .collect();
        if considered.iter().all(|&i| in_epilogue(i)) {
            to_erase.insert(si);
            to_erase.extend(considered);
        }
    }

    let n = to_erase.len();
    for i in to_erase {
        let addr = insns[i].address;
        ledger_note(addr, LedgerKind::CalleeSaveRoundTrip);
        insns[i] = Insn::nop(addr);
    }
    n
}

/// `mov x29, sp` / `add x29, sp, #k` records an alias; later `[x29 + N]`
/// becomes stack slot `(k + N - depth_at_copy) / ptr_size`.
///
/// WHAT COUNTS AS AN ALIAS. Only an instruction that puts a POINTER INTO THE
/// FRAME into a register:
///
///   * `mov xD, sp`              → `Move xD, Register("sp")`
///   * `add xD, sp, #k`          → `Move xD, AddressOf(Memory{base: sp, k})`
///     (and the plain `xD = sp` degenerate form, `AddressOf(sp)`)
///
/// A `Move xD, Memory { base: sp, addend: N }` is **not** that: it is a LOAD
/// of a stack VALUE — `ldr xD, [sp, #N]`, and in particular every `ldp`/
/// `pop` epilogue restore of a callee-saved register. Treating a load as an
/// alias says "this register holds sp", so from that point on EVERY
/// `[xD + N]` access was rebased into a bogus `stack_(N/8)` slot. Registers
/// are reused: the frame slot `ldp x24, x23, [sp]` restores x23 from the
/// frame, but x23 held a heap OBJECT pointer for the whole body, so each
/// `[x23 + 0x4c]` field access (the real one at 0x2530a0 of method 0x252d8c)
/// became `[stack_9 + 4]` — a different address entirely, which the verifier
/// reports as `memory offset mismatch: native +76, IR [4]` and which also
/// invented a never-defined `stack_9` local. The bare `Memory` form is
/// unambiguous in this IR: `lift.rs` wraps the address of a slot in
/// `AddressOf` precisely because `Memory` on its own is a load everywhere
/// else, so an `AddressOf` source is the only non-register shape that
/// yields a frame pointer.
///
/// An alias is dropped as soon as its register is REdefined by anything
/// else (a fresh `ldr x23, [x0]` moves the register off the frame). Without
/// that invalidation a single prologue alias would claim the register for
/// the rest of the body, including blocks where it holds a live value.
fn resolve_frame_aliases(
    insns: &mut [Insn],
    insn_depth: &HashMap<usize, i64>,
    ptr_size: u64,
) {
    // reg → (bias in bytes, stack depth at the copy)
    let mut aliases: HashMap<String, (i64, i64)> = HashMap::new();
    for i in 0..insns.len() {
        let here = insn_depth.get(&i).copied().unwrap_or(0);
        // 1. Consume: rewrite reads through a LIVE alias (uses in an
        //    instruction refer to the value that instruction reads, so this
        //    runs before the instruction's own definition is applied).
        for op in &mut insns[i].operands {
            rewrite_alias_operand(op, &aliases, here, ptr_size);
        }
        // 2. Update the alias set for this instruction's own destination.
        match frame_alias_def(&insns[i]) {
            Some((dst, bias)) => {
                aliases.insert(dst, (bias, here));
            }
            None => {
                if let Some(pos) = Insn::dest_position(&insns[i].op) {
                    if let Some(Operand::Register(d)) = insns[i].operands.get(pos) {
                        aliases.remove(d);
                    }
                }
            }
        }
    }
}

/// `xD = sp` (bias 0) or `xD = sp + k` (bias `k`) — see
/// [`resolve_frame_aliases`] for why the bare-`Memory` load form is excluded.
/// The register name is returned owned so the caller can mutate `insns`.
fn frame_alias_def(insn: &Insn) -> Option<(String, i64)> {
    if !matches!(insn.op, Op::Move) {
        return None;
    }
    let Operand::Register(dst) = insn.operands.first()? else {
        return None;
    };
    let bias = match insn.operands.get(1)? {
        // `mov xD, sp` — the value of sp itself.
        Operand::Register(b) if b == "sp" => 0,
        // `add xD, sp, #k` — sp plus a frame offset (address of a slot).
        Operand::AddressOf(inner) => match inner.as_ref() {
            Operand::Memory { base, index: None, addend, .. } => match base.as_ref() {
                Operand::Register(b) if b == "sp" => *addend,
                _ => return None,
            },
            Operand::Register(b) if b == "sp" => 0,
            _ => return None,
        },
        _ => return None,
    };
    Some((dst.clone(), bias))
}

/// A stack operand for an access at byte offset `slot*scale + off`.
///
/// `stack_N` names a memory CELL, and cells are `scale` bytes wide. An access
/// that lands INSIDE a cell — a narrower field of a slot whose address was
/// taken, e.g. `add x8, sp, #8` (the indirect-result buffer) followed by
/// `ldr w19, [sp, #0xc]` (4 bytes INTO that buffer) — must stay a memory
/// access relative to the cell.
///
/// Flooring the offset to the cell name instead made `sp+8` and `sp+12` the
/// SAME operand, so the value read out of the buffer was provably equal to the
/// buffer's own ADDRESS: the address printed as the value, the callee's store
/// through x8 became invisible, and the read that `ldr w19,[sp,#0xc]` performs
/// was eliminated as a redundant copy.
fn slot_operand(slot: i64, off: i64, scale: i64, width: u8) -> Operand {
    if off == 0 {
        return Operand::reg(format!("stack_{slot}"));
    }
    Operand::Memory {
        base: Box::new(Operand::reg(format!("stack_{slot}"))),
        index: None,
        scale: scale as u8,
        addend: off,
        width,
    }
}

fn rewrite_alias_operand(
    op: &mut Operand,
    aliases: &HashMap<String, (i64, i64)>,
    depth_here: i64,
    ptr_size: u64,
) {
    match op {
        Operand::Memory { base, addend, width, .. } => {
            if let Operand::Register(b) = base.as_ref() {
                if let Some(&(bias, copy_depth)) = aliases.get(b) {
                    let scale = ptr_size as i64;
                    // `[xD + addend]` where `xD == sp + bias` at depth
                    // `copy_depth`, and sp is at `depth_here` now: the access
                    // sits `bias + addend` bytes above the copy's sp, and the
                    // frame has moved `depth_here - copy_depth` bytes since.
                    let num = bias + *addend - (depth_here - copy_depth);
                    *op = slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                }
            }
        }
        Operand::AddressOf(inner) => rewrite_alias_operand(inner, aliases, depth_here, ptr_size),
        _ => {}
    }
}

/// Remaining `[sp + N]` operands become absolute slots `stack_(N/8 - depth)`;
/// ShiftStack ops are removed (their effect is baked into depths already).
fn rewrite_sp_operands(insns: &mut [Insn], insn_depth: &HashMap<usize, i64>, ptr_size: u64) {
    let scale = ptr_size as i64;
    for (i, insn) in insns.iter_mut().enumerate() {
        if matches!(insn.op, Op::ShiftStack(_)) {
            insn.op = Op::Move;
            insn.operands.clear();
            continue;
        }
        let here = insn_depth.get(&i).copied().unwrap_or(0);
        for op in &mut insn.operands {
            match op {
                Operand::Memory { base, addend, index: None, width, .. } => {
                    if let Operand::Register(b) = base.as_ref() {
                        if b == "sp" {
                            let num = (*addend) - here * scale;
                            *op = slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                        }
                    }
                }
                Operand::AddressOf(inner) => {
                    if let Operand::Memory { base, addend, index: None, width, .. } = inner.as_mut() {
                        if let Operand::Register(b) = base.as_ref() {
                            if b == "sp" {
                                let num = (*addend) - here * scale;
                                **inner =
                                    slot_operand(num / scale, num.rem_euclid(scale), scale, *width);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
