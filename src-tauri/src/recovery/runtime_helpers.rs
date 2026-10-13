//! Identity for the IL2CPP runtime helpers that managed methods call.
//!
//! A managed method regularly calls into the IL2CPP **runtime** rather than into
//! other managed code: the static-constructor guard the compiler emits before
//! the first use of a type's statics, and the generic/virtual dispatch
//! trampolines that rgctx-driven calls go through.
//!
//! Those functions live in the executable's own `.text` section — the runtime's
//! C++ compiled into the same image — so they are NOT in the metadata's method
//! table and are NOT exported. `methods_by_rva` therefore cannot name them and
//! every one renders as `fn_<address>`. On the v24.5 reference build that is
//! 53,620 call sites from only **489** addresses, and the distribution is very
//! top-heavy: **50% of all sites are 4 addresses and 90% are 15**. One function
//! alone (`0x11c7f0`) accounts for 18% of them.
//!
//! Naming them from a symbol table is impossible by construction, so they are
//! identified from the SHAPE of their call sites instead — the same evidence
//! discipline the rest of the pipeline uses. The static-init helper is the
//! tractable one, because the guard it implements leaves three independent
//! traces that must all agree:
//!
//! ```text
//!   Compare(NotEqual)[c, *guard_slot, 0]      // the guard test …
//!   CondJump(c)                               // … folded into a branch
//!   Call  Immediate(<helper rva>) …           // the helper
//!   Move  [*guard_slot, 1]                    // … and the guard being set
//! ```
//!
//! The load, the branch and the store must all name the SAME slot, and the call
//! target must be an address no map could name. Requiring all four is what keeps
//! a coincidental call-next-to-a-guard from being mislabelled.
//!
//! ARM32 does not keep the slot literal: the guard byte is read out of the
//! metadata-usage table (`*(byte*)(tbl + *(nint*)(…))`), so the compare operand
//! is the load's local and `slot_of` names nothing. There the store half is
//! checked by shape instead — it must write an INDEXED byte
//! (`indexed_byte_store_to_one`) — which is the same evidence seen from the
//! other end. A bare compare-against-0 next to a plain `x = 1` is a null-check
//! raise or a type-init helper, not a static-init guard, and must stay unnamed.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::ir::{CompareKind, Insn, Op, Operand};

/// What a runtime helper does, as far as the call sites prove.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuntimeHelper {
    /// `if (guard == 0) { ClassInit(cls); guard = 1; }` — the static-constructor
    /// guard. `ClassInit` here is the runtime entry point that runs the type's
    /// `.cctor` once; the loaded class pointer travels in the helper's
    /// arguments (a metadata-slot load), not in this name.
    ClassInit,
}

impl RuntimeHelper {
    /// The name the pseudo-C# shows for this helper. Deliberately spelled as a
    /// runtime call so a reader can tell "the runtime did this" from "a managed
    /// method with an unknown name did this" (`fn_…`).
    pub fn display_name(self) -> &'static str {
        match self {
            RuntimeHelper::ClassInit => "IL2CPP_ClassInit",
        }
    }
}

fn table() -> &'static Mutex<HashMap<u64, RuntimeHelper>> {
    static T: OnceLock<Mutex<HashMap<u64, RuntimeHelper>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Record that `target` behaves as `role`. Called once per agreeing call site;
/// repeated notes for the same target are idempotent.
pub fn note(target: u64, role: RuntimeHelper) {
    if let Ok(mut t) = table().lock() {
        t.insert(target, role);
    }
}

/// The identified role of an unresolved call target, if any.
pub fn role_of(target: u64) -> Option<RuntimeHelper> {
    table().lock().ok().and_then(|t| t.get(&target).copied())
}

/// The name to render for an unresolved call target: the runtime helper's name
/// when it is identified, otherwise `fn_<address>` as before.
pub fn display_name(target: u64) -> String {
    match role_of(target) {
        Some(r) => r.display_name().to_string(),
        None => format!("fn_{target:x}"),
    }
}

/// How many helpers were identified (for the `ILR_TIMING` line).
pub fn identified_count() -> usize {
    table().lock().map(|t| t.len()).unwrap_or(0)
}

/// The guard slot a memory operand addresses, as `(base name, addend)`.
///
/// The base is carried as a name rather than an index so a slot versioned by
/// SSA still matches the store that sets it: at the point this runs the guard
/// is addressed through a stable local (`Local(0)`, the image base), and both
/// the load and the store name the SAME local id.
fn slot_of(op: &Operand) -> Option<(String, i64)> {
    match op {
        Operand::Memory { base, index: None, addend, .. } => match &**base {
            Operand::Local(v) => Some((format!("l{v}"), *addend)),
            Operand::Register(r) => Some((format!("r{r}"), *addend)),
            _ => None,
        },
        _ => None,
    }
}

/// Is this a zero-compare guard test whose compared operand `slot_of` cannot
/// name?
///
/// `slot_of` only matches a literal `base+addend` memory operand. On ARM32 the
/// guard byte is reached through the metadata-usage table (`*(byte*)(tbl +
/// *(nint*)(…))`), so the compared operand is the local that load landed in and
/// `slot_of` refuses it — which is why the strict rule never fired there. The
/// SHAPE is the same comparison against `0`; it is recognised as a FALLBACK
/// only, and the store that must close the shape is checked far more narrowly
/// (`indexed_byte_store_to_one`).
fn zero_guard_test(insn: &Insn) -> bool {
    matches!(
        &insn.op,
        Op::Compare(k)
            if matches!(k, CompareKind::NotEqual | CompareKind::Equal)
                && insn.operands.get(2) == Some(&Operand::Immediate(0))
    )
}

/// STRICT (arm64/x64): `insn` sets exactly the guard slot `g` to the literal
/// 1. Unchanged from the original rule.
fn sets_slot_to_one(insn: &Insn, g: &(String, i64)) -> bool {
    matches!(insn.op, Op::Move)
        && insn.operands.get(1) == Some(&Operand::Immediate(1))
        && insn.operands.first().and_then(slot_of).as_ref() == Some(g)
}

/// ARM32: `insns[j]` writes the literal 1 — held either as the immediate or in
/// a register set by a nearby `mov rN, #1` — to an INDEXED byte cell,
/// `*(byte*)(<table base> + <index>)`.
///
/// That indexed metadata-usage byte is what makes the guard the static-init
/// guard rather than just "some compare against 0 near some store of 1". Both
/// halves are required: the loose shape alone also matches the null-check raise
/// (`fn_5ae014`: `if (p == 0) panic(…)`, whose following `x = 1` lands in a
/// plain `base+off` field) and the type-init helper (`fn_5906b0`). Measured on
/// the 4k ARM32 dump: all 2,079 static-init sites carry the indexed write and
/// NONE of the 18 look-alike sites do.
fn indexed_byte_store_to_one(insns: &[Insn], j: usize) -> bool {
    if !matches!(insns[j].op, Op::Move) {
        return false;
    }
    let Some(Operand::Memory { base, index, addend, width, .. }) = insns[j].operands.first()
    else {
        return false;
    };
    if index.is_none() || *width != 1 || *addend != 0 || !matches!(**base, Operand::Immediate(_)) {
        return false;
    }
    let Some(val) = insns[j].operands.get(1) else {
        return false;
    };
    if *val == Operand::Immediate(1) {
        return true;
    }
    for k in (0..j).rev().take(4) {
        if insns[k].is_terminator() {
            break;
        }
        if matches!(insns[k].op, Op::Move)
            && insns[k].operands.get(1) == Some(&Operand::Immediate(1))
            && insns[k].operands.first() == Some(val)
        {
            return true;
        }
    }
    false
}

/// Is this the `[dest, *slot, 0]` guard test, and if so, which slot?
fn guard_test(insn: &Insn) -> Option<(String, i64)> {
    if let Op::Compare(k) = insn.op {
        if matches!(k, CompareKind::NotEqual | CompareKind::Equal) {
            // operands are [condition, lhs, rhs]
            if insn.operands.get(2) == Some(&Operand::Immediate(0)) {
                return insn.operands.get(1).and_then(slot_of);
            }
        }
    }
    None
}

/// How many instructions back the guard test may sit. The compiler emits the
/// test, the branch and then the helper; allowing a few slots of address
/// computation between them keeps the rule from depending on instruction
/// selection noise, while staying short enough that an unrelated comparison
/// earlier in the block cannot be reached.
const GUARD_LOOKBACK: usize = 8;
/// How many instructions forward the `slot = 1` may sit.
const GUARD_LOOKAHEAD: usize = 8;

/// Classify this method's unresolved call targets by call-site shape.
///
/// Returns the number of call sites that matched. Only targets that NO map
/// resolved are considered — a call whose operand is `Operand::Method` already
/// has a better name than anything here.
pub fn detect(insns: &[Insn]) -> usize {
    let mut found = 0usize;
    for i in 0..insns.len() {
        let insn = &insns[i];
        if !matches!(insn.op, Op::Call | Op::CallVoid) {
            continue;
        }
        // An unresolved target is still a bare address; a resolved one carries
        // its `MethodRef` and is not our business.
        let Some(Operand::Immediate(target)) = insn.operands.first() else {
            continue;
        };
        let target = *target as u64;
        if target == 0 {
            continue;
        }

        // (1) the guard test: the nearest preceding Compare against 0.
        //
        // The guard's OWN branch sits between the test and the call, so exactly
        // one `CondJump` is stepped over; a second control transfer means the
        // comparison found beyond it belongs to some other block.
        let mut guard: Option<(String, i64)> = None;
        // ARM32 reaches the guard byte through an INDEXED metadata-usage load
        // (`*(byte*)(base + *(nint*)(…))`), so `slot_of` — which needs a literal
        // base+addend — refuses it and the whole rule never fired on ARM32
        // (`IL2CPP_ClassInit` appears ZERO times on the corpus, while
        // `fn_5827e4` carries the shape at 99% of its 13,060 sites). The SHAPE
        // is unchanged: a byte compared against 0, the guarded call, then the
        // byte set to 1.
        let mut indexed_guard = false;
        let mut crossed_branch = false;
        for j in (0..i).rev().take(GUARD_LOOKBACK) {
            if let Some(s) = guard_test(&insns[j]) {
                guard = Some(s);
                break;
            }
            if zero_guard_test(&insns[j]) {
                indexed_guard = true;
                break;
            }
            if insns[j].is_terminator() {
                if !crossed_branch && matches!(insns[j].op, Op::CondJump(_)) {
                    crossed_branch = true;
                    continue;
                }
                break;
            }
        }
        if guard.is_none() && !indexed_guard {
            continue;
        }

        // (2) the guard being set. STRICT: the same slot the test named, set to
        // the literal 1 (arm64/x64, unchanged). ARM32: the matching INDEXED byte
        // write — the only form that separates the static-init guard from a
        // null-check raise or a type-init helper (see
        // `indexed_byte_store_to_one`).
        let mut set = false;
        for j in i + 1..(i + 1 + GUARD_LOOKAHEAD).min(insns.len()) {
            let hit = match &guard {
                Some(g) => sets_slot_to_one(&insns[j], g),
                None => indexed_byte_store_to_one(insns, j),
            };
            if hit {
                set = true;
                break;
            }
            if insns[j].is_terminator() {
                break;
            }
        }
        if !set {
            continue;
        }

        note(target, RuntimeHelper::ClassInit);
        super::timing::RT_CLASSINIT_SITES
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        found += 1;
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::ir::Insn;

    fn mem(base: Operand, addend: i64) -> Operand {
        Operand::Memory { base: Box::new(base), index: None, scale: 1, addend, width: 0 }
    }

    /// The full guard shape: test, branch, helper call, `guard = 1`.
    fn guard_body(target: i64, slot_base: Operand, addend: i64) -> Vec<Insn> {
        vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(50), mem(slot_base.clone(), addend), Operand::Immediate(0)],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![
                    Operand::Immediate(target),
                    Operand::Local(9),
                    Operand::Local(3),
                    mem(slot_base.clone(), addend + 16),
                ],
                0x18,
            ),
            Insn::new(Op::Move, vec![mem(slot_base, addend), Operand::Immediate(1)], 0x20),
        ]
    }

    #[test]
    fn a_guard_that_is_tested_called_and_set_names_the_helper() {
        let insns = guard_body(0x11c7f0, Operand::Local(0), 9909776);
        assert_eq!(detect(&insns), 1);
        assert_eq!(role_of(0x11c7f0), Some(RuntimeHelper::ClassInit));
        assert_eq!(display_name(0x11c7f0), "IL2CPP_ClassInit");
        // and an address that was never classified keeps the old rendering
        assert_eq!(display_name(0xdeadbeef), "fn_deadbeef");
    }

    /// The ARM32 guard shape: the byte is reached through an INDEXED load, so
    /// `slot_of` names no slot, and the guard's read and write cells differ
    /// textually (`fn_5827e4` is 99% this shape on the corpus).
    #[test]
    fn an_indexed_guard_still_names_the_class_init_helper() {
        let idx = |base: i64, v: u32| Operand::Memory {
            base: Box::new(Operand::Immediate(base)),
            index: Some(Box::new(Operand::Local(v))),
            scale: 1,
            addend: 0,
            width: 1,
        };
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::Equal),
                vec![Operand::Local(50), idx(0x12eb194, 60), Operand::Immediate(0)],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x5827e4), Operand::Local(9), Operand::Local(3)],
                0x18,
            ),
            // The write names a DIFFERENT cell (0x12eb1bc / local 61) — the
            // indexed form has no textual slot to compare, so the shape is what
            // carries the identification.
            Insn::new(Op::Move, vec![idx(0x12eb1bc, 61), Operand::Immediate(1)], 0x20),
        ];
        assert_eq!(detect(&insns), 1);
        assert_eq!(display_name(0x5827e4), "IL2CPP_ClassInit");
    }

    /// A bare `== 0` compare with no `= 1` store is a null check, not a guard.
    #[test]
    fn an_indexed_zero_compare_without_a_store_is_not_a_class_init() {
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![
                    Operand::Local(50),
                    Operand::Memory {
                        base: Box::new(Operand::Immediate(0x10)),
                        index: Some(Box::new(Operand::Local(60))),
                        scale: 1,
                        addend: 0,
                        width: 1,
                    },
                    Operand::Immediate(0),
                ],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x5ae014), Operand::Local(9)],
                0x18,
            ),
        ];
        assert_eq!(detect(&insns), 0);
        assert_eq!(display_name(0x5ae014), "fn_5ae014");
    }

    /// The measured ARM32 FALSE positive: a null-check raise (`fn_5ae014`:
    /// `if (p == 0) panic(…)`) followed by an unrelated `field = 1`. The store
    /// is a plain `base+off` byte, NOT the indexed metadata-usage cell, so the
    /// shape must not close.
    #[test]
    fn a_null_check_raise_with_a_plain_byte_store_is_not_a_class_init() {
        let insns = vec![
            Insn::new(Op::Move, vec![Operand::Local(70), Operand::Immediate(1)], 0x08),
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(50), Operand::Local(60), Operand::Immediate(0)],
                0x10,
            ),
            Insn::new(Op::CondJump(3), vec![Operand::Local(50)], 0x10),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x5ae014), Operand::Local(9)],
                0x18,
            ),
            Insn::new(
                Op::Move,
                vec![mem(Operand::Local(9), 88), Operand::Local(70)],
                0x20,
            ),
        ];
        assert_eq!(detect(&insns), 0);
        assert_eq!(display_name(0x5ae014), "fn_5ae014");
    }

    #[test]
    fn a_call_before_a_guard_is_not_a_class_init() {
        // The store sets a DIFFERENT slot, so the shape does not close.
        let mut insns = guard_body(0x2222, Operand::Local(0), 9909776);
        insns[3] = Insn::new(
            Op::Move,
            vec![mem(Operand::Local(0), 4242), Operand::Immediate(1)],
            0x20,
        );
        assert_eq!(detect(&insns), 0);
    }

    #[test]
    fn a_guarded_call_is_not_the_guard() {
        // Test and store agree, but the CALL follows the store instead of
        // sitting between them.
        let insns = vec![
            Insn::new(
                Op::Compare(CompareKind::NotEqual),
                vec![Operand::Local(50), mem(Operand::Local(0), 88), Operand::Immediate(0)],
                0x10,
            ),
            Insn::new(Op::CondJump(4), vec![Operand::Local(50)], 0x10),
            Insn::new(Op::Move, vec![mem(Operand::Local(0), 88), Operand::Immediate(1)], 0x14),
            Insn::new(
                Op::Call,
                vec![Operand::Immediate(0x3333), Operand::Local(9)],
                0x18,
            ),
        ];
        assert_eq!(detect(&insns), 0);
    }

    #[test]
    fn a_resolved_target_is_never_relabelled() {
        use crate::recovery::types::{ManagedType, MethodRef, MethodSigInfo, PrimKind};
        let m = MethodRef {
            image_index: 0,
            type_def_index: 1,
            method_index: 2,
            name: "M".into(),
            declaring_namespace: "N".into(),
            declaring_type_name: "T".into(),
            assembly: "a".into(),
            sig: MethodSigInfo {
                ret: ManagedType::Primitive(PrimKind::Void),
                params: vec![],
                is_static: true,
            },
        };
        let mut insns = guard_body(0x11c7f0, Operand::Local(0), 9909776);
        insns[2].operands[0] = Operand::Method(m);
        assert_eq!(detect(&insns), 0);
    }
}
