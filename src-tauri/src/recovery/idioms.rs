//! High-level C# idiom recognition over the FINISHED statement tree.
//!
//! ## Why this is a separate module
//!
//! Everything upstream reasons about the *machine*: instructions, CFG edges, SSA
//! versions, reaching definitions. That is the right level for recovering what
//! the code does, but the wrong level for recognising what it was *written as*.
//! Once the structurer has produced a statement tree the machine detail is gone,
//! and what remains are C# shapes — a guard branching to the statement right
//! after it, a dead tail after a `return`, the same structure-field load spelled
//! out five times because the native loaded it once.
//!
//! Those shapes have nowhere to live among the IR-level passes, and the
//! unstructured fallback representation is exactly what this module exists to
//! avoid needing. So they live here: post-structuring, whole-tree, local
//! rewrites that can be reasoned about on their own.
//!
//! ## The two rules that keep this module honest
//!
//! 1. **Never delete an observation.** One idiom was written, measured, and
//!    then REMOVED for violating this rule: deleting `if (c) goto L;` when `L`
//!    is the next statement dropped 79 real native branches on a 4000-method
//!    sample. The shape only *looks* vacuous — it appears when the branch's
//!    fall block rendered no statements, while the machine's branch still skips
//!    it. See `dead_code` step 2 for the full reasoning; the rule that survives
//!    is that an idiom may only delete control flow that evaluates nothing
//!    (`goto L; L:`), never a CONDITION.
//! 2. **Never move a write.** Structure-field propagation hoists a load, which
//!    is sound only if nothing between the first and last use can change the
//!    value: no call, no construction, and no write to any local the address
//!    expression names. Anything not provable is left alone (a failed match is
//!    a no-op, exactly like the rewrite catalog).
//!
//! Every counter is process-wide and reported beside the structural census, so
//! the effect of each idiom on a real corpus is a number rather than a claim.

use super::csharp::{cleanup, BinOp, CmpKind, Expr, Stmt};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-run tally for the idioms. Reported with the structural census.
pub struct IdiomCensus {
    /// `goto L;` with `L:` as the very next statement — control reaches `L`
    /// either way, so the jump is a no-op and is removed.
    pub dead_goto_to_next_label: AtomicU64,
    /// Statements after a `return`/`throw`/`goto` with no label between them:
    /// the machine can never execute them.
    pub dead_statements_after_terminator: AtomicU64,
    /// Labels nothing jumps to and no `switch` arm names.
    pub dead_unreferenced_label: AtomicU64,
    /// Structure-field loads hoisted into a named temporary.
    pub fields_propagated: AtomicU64,
    /// Individual field-read expressions replaced by that temporary.
    pub field_reads_collapsed: AtomicU64,
    /// IL2CPP static-constructor / thread-static initialization guards
    /// (`if (*(byte*)(<static>) == 0) { ClassInit(…); *(...) = 1; }`) folded
    /// away — runtime bookkeeping managed C# does not contain.
    pub class_init_guards: AtomicU64,
    /// `goto Lx;` inside a loop whose `Lx` labels the statement right after that
    /// loop — the machine's jump out of the loop, i.e. the loop's own `break`.
    pub goto_to_loop_break: AtomicU64,
    /// `goto Lx;` inside a `while` loop whose `Lx` labels the statement right
    /// before that loop — the machine's jump back to the loop head, i.e. the
    /// loop's own `continue`.
    pub goto_to_loop_continue: AtomicU64,
    /// `if (a) { if (b) { … } }` folded to `if (a && b) { … }`: one branch, one
    /// brace level, the compound predicate the source wrote.
    pub nested_ifs_combined: AtomicU64,
    /// `if (c) { v = a; } else { v = b; }` folded to `v = c ? a : b;`.
    pub ifs_to_ternary: AtomicU64,
    /// `v = v;` — a self-dead store, dropped.
    pub self_assignments_dropped: AtomicU64,
    /// `V = C.GetEnumerator(); while (V.MoveNext()) { … V.Current … }`
    /// collapsed to `foreach (var e in C) { … e … }`.
    pub foreach_collapsed: AtomicU64,
}

pub static IDIOMS: IdiomCensus = IdiomCensus {
    dead_goto_to_next_label: AtomicU64::new(0),
    dead_statements_after_terminator: AtomicU64::new(0),
    dead_unreferenced_label: AtomicU64::new(0),
    fields_propagated: AtomicU64::new(0),
    field_reads_collapsed: AtomicU64::new(0),
    class_init_guards: AtomicU64::new(0),
    goto_to_loop_break: AtomicU64::new(0),
    goto_to_loop_continue: AtomicU64::new(0),
    nested_ifs_combined: AtomicU64::new(0),
    ifs_to_ternary: AtomicU64::new(0),
    self_assignments_dropped: AtomicU64::new(0),
    foreach_collapsed: AtomicU64::new(0),
};

/// Run every idiom over one method body.
///
/// Ordering is deliberate: dead code first (so a temporary is never created for
/// a load that was about to be deleted), then field propagation, then dead code
/// again (propagation inserts an assignment, which can itself orphan the
/// statement that used to compute the value inline).
pub fn apply(stmts: &mut Vec<Stmt>) {
    // The enumerator collapse runs FIRST: it removes the acquisition, the
    // `MoveNext` test and the `.Current` read as a UNIT, and every later rewrite
    // would see those three separately and could fold a guard between them into
    // something the shape test no longer recognises.
    collapse_enumerator_loops(stmts);
    // Masked-assembly folding runs next: it removes whole statements, so
    // doing it before dead-code means the orphaned labels/gotos it can leave
    // are swept by the same pass rather than surviving to the printer.
    collapse_class_init_guards(stmts);
    dead_code(stmts);
    propagate_fields(stmts);
    dead_code(stmts);
    // Restatement runs LAST, on the cleaned tree: a folded ternary or a
    // combined guard produces an `if`/assign the earlier passes have already
    // had their chance at, and a `goto` turned into `break` can newly orphan
    // the label it used to name (swept by `drop_unreferenced_labels` below).
    drop_self_assignments(stmts);
    fold_if_assign_to_ternary(stmts);
    combine_nested_ifs(stmts);
    collapse_goto_to_loop_break(stmts);
    // A converted `break` terminates its statement list, so whatever followed
    // it in the same arm is now unreachable.
    dead_code(stmts);
    drop_unreferenced_labels(stmts);
}

/// The idiom counts for the run report, or an empty string if nothing fired.
pub fn report() -> String {
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let dead = g(&IDIOMS.dead_goto_to_next_label)
        + g(&IDIOMS.dead_statements_after_terminator)
        + g(&IDIOMS.dead_unreferenced_label);
    let restated = g(&IDIOMS.goto_to_loop_break)
        + g(&IDIOMS.goto_to_loop_continue)
        + g(&IDIOMS.nested_ifs_combined)
        + g(&IDIOMS.ifs_to_ternary)
        + g(&IDIOMS.self_assignments_dropped)
        + g(&IDIOMS.foreach_collapsed);
    if dead == 0
        && restated == 0
        && g(&IDIOMS.fields_propagated) == 0
        && g(&IDIOMS.class_init_guards) == 0
    {
        return String::new();
    }
    let mut s = String::new();
    s.push_str(&format!(
        "[idioms] dead-end block / dead code: {dead} sites\n"
    ));
    s.push_str(&format!(
        "[idioms]   `goto L; L:` — jump to the very next statement: {}\n",
        g(&IDIOMS.dead_goto_to_next_label)
    ));
    s.push_str(&format!(
        "[idioms]   unreachable statements after a terminator: {}\n",
        g(&IDIOMS.dead_statements_after_terminator)
    ));
    s.push_str(&format!(
        "[idioms]   labels nothing can reach: {}\n",
        g(&IDIOMS.dead_unreferenced_label)
    ));
    s.push_str(&format!(
        "[idioms] structure fields: {} loads hoisted, {} read expressions collapsed into them\n",
        g(&IDIOMS.fields_propagated),
        g(&IDIOMS.field_reads_collapsed)
    ));
    s.push_str(&format!(
        "[idioms] masked assembly: {} static-init guard(s) folded away\n",
        g(&IDIOMS.class_init_guards)
    ));
    s.push_str(&format!(
        "[idioms] restatement: {} loop-exit goto(s) → `break`, {} loop-head goto(s) → `continue`, {} nested if(s) combined, {} if/else → `?:`, {} self-assignment(s) dropped\n",
        g(&IDIOMS.goto_to_loop_break),
        g(&IDIOMS.goto_to_loop_continue),
        g(&IDIOMS.nested_ifs_combined),
        g(&IDIOMS.ifs_to_ternary),
        g(&IDIOMS.self_assignments_dropped)
    ));
    s.push_str(&format!(
        "[idioms] enumerator loops: {} collapsed to `foreach`\n",
        g(&IDIOMS.foreach_collapsed)
    ));
    s
}

// ---------------------------------------------------------------------------
// Masked-assembly folding: the IL2CPP static-constructor guard
// ---------------------------------------------------------------------------

/// Fold away the IL2CPP **static-constructor / thread-static initialization
/// guard** — runtime bookkeeping the source language never contained.
///
/// Before the first use of a type's statics the compiler emits:
///
/// ```text
///   if (*(byte*)(<static slot>) == 0) {
///       <runtime class-init>(<class>);
///       *(byte*)(<static slot>) = 1;
///   }
/// ```
///
/// On the ARM32 reference build the guard is 2 statements on the guarded arm,
/// a zero test on a **statically-rooted** byte load, and an empty else arm. In
/// managed C# the access simply names the static — the CLR runs the `.cctor`
/// on demand — so the whole statement is dropped:
///
/// ```text
///   CharacterRenderer.SomeStatic = …;      // no guard in the C#
/// ```
///
/// This is the one idiom that removes a CONDITION, which
/// [`dead_code`]'s rule 1 otherwise forbids. It is permitted here for the
/// reason the rule exists: the rule protects a branch that is the only trace of
/// machine code the text would then not contain. This branch guards a runtime
/// call and a runtime flag write that are NOT user code — the shape is fully
/// pinned (a memory-load-vs-zero test whose entire taken arm is one call and
/// one immediate-`1` store into memory, with no else), so nothing the source
/// wrote can match it. Custom `get`/`set` where a user typed this exact shape
/// would have to read a STATIC word and store the literal 1 — measured below.
pub fn collapse_class_init_guards(stmts: &mut Vec<Stmt>) {
    // Depth-first so a nested guard is folded before its parent's else-arm
    // emptiness is judged (a folded inner guard leaves the parent identical).
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                collapse_class_init_guards(then_body);
                collapse_class_init_guards(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                collapse_class_init_guards(body)
            }
            Stmt::Try { try_body, catches, finally_body } => {
                collapse_class_init_guards(try_body);
                for (_, b) in catches.iter_mut() {
                    collapse_class_init_guards(b);
                }
                if let Some(f) = finally_body {
                    collapse_class_init_guards(f);
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases.iter_mut() {
                    collapse_class_init_guards(b);
                }
                collapse_class_init_guards(tail);
            }
            _ => {}
        }
    }
    let mut kept: Vec<Stmt> = Vec::with_capacity(stmts.len());
    let mut removed = 0u64;
    // The fold ERASES a native `bl`, and `concord` scores every native call it
    // cannot find in the text as DROPPED — so the erasure has to reach the
    // verify ledger (`LedgerKind::DroppedRuntimeCall`). This pass cannot write
    // it: it works on statement TREES, which carry no instruction address, and
    // the address on the call expression is the CALLEE's, not the call site's.
    // Ledgering the callee there would be worse than not ledgering at all — the
    // same number is another instruction's address in the same method often
    // enough to excuse a genuinely dropped call. The renderer owns both facts
    // (`ir.insns` and the statement list) and does it, against a count taken
    // here — see `csharp::render_method_csharp_full_chains`.
    for s in stmts.drain(..) {
        if is_class_init_guard(&s) {
            removed += 1;
        } else {
            kept.push(s);
        }
    }
    *stmts = kept;
    if removed > 0 {
        IDIOMS
            .class_init_guards
            .fetch_add(removed, Ordering::Relaxed);
    }
}

/// Class-init guards standing in a statement list, keyed by the address of the
/// runtime call each one is about to erase.
///
/// The renderer needs this BEFORE folding to excuse the erased call sites (see
/// the note in `collapse_class_init_guards`): the erase happens on the
/// statement tree, the call SITE's address lives only in the IR, and the one
/// thing both sides can be counted against is the call TARGET. A target whose
/// erased-guard count equals its call-site count in the IR means every one of
/// those sites was erased, so each can be excused exactly once — and a target
/// that does not match is left alone, because excusing a call that is still in
/// the text would hide a real drop.
#[derive(Default, Debug, Clone)]
pub struct GuardFolds {
    /// Runtime call target → guards whose armed arm calls it.
    pub by_target: std::collections::BTreeMap<u64, u64>,
    /// Guards whose arm call carries no address (`Expr::Call` — the call was
    /// prettified into a member access). They cannot be tied to a site, so a
    /// list containing one is never excused at all.
    pub unmappable: u64,
}

impl GuardFolds {
    /// Guards in this list — i.e. calls the fold is about to erase.
    pub fn total(&self) -> u64 {
        self.by_target.values().sum::<u64>() + self.unmappable
    }

    pub fn merge(&mut self, other: &GuardFolds) {
        for (k, v) in &other.by_target {
            *self.by_target.entry(*k).or_insert(0) += *v;
        }
        self.unmappable += other.unmappable;
    }
}

/// The runtime call an already-recognised class-init guard would erase.
fn class_init_guard_call(s: &Stmt) -> Option<u64> {
    let Stmt::If { then_body, .. } = s else { return None };
    then_body.iter().find_map(|st| match st {
        Stmt::Eval(Expr::RawCall(a, ..)) => Some(*a),
        Stmt::Assign(_, _, Expr::RawCall(a, ..)) => Some(*a),
        _ => None,
    })
}

/// Class-init guards this statement list currently holds, at every nesting
/// level — exactly where the fold looks.
pub fn count_class_init_guards(stmts: &[Stmt]) -> GuardFolds {
    let mut f = GuardFolds::default();
    for s in stmts {
        if is_class_init_guard(s) {
            match class_init_guard_call(s) {
                Some(a) => *f.by_target.entry(a).or_insert(0) += 1,
                None => f.unmappable += 1,
            }
        }
        let mut inner = GuardFolds::default();
        match s {
            Stmt::If { then_body, else_body, .. } => {
                inner.merge(&count_class_init_guards(then_body));
                inner.merge(&count_class_init_guards(else_body));
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                inner.merge(&count_class_init_guards(body));
            }
            Stmt::Try { try_body, catches, finally_body } => {
                inner.merge(&count_class_init_guards(try_body));
                for (_, b) in catches.iter() {
                    inner.merge(&count_class_init_guards(b));
                }
                if let Some(f) = finally_body {
                    inner.merge(&count_class_init_guards(f));
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    inner.merge(&count_class_init_guards(b));
                }
                inner.merge(&count_class_init_guards(tail));
            }
            _ => {}
        }
        f.merge(&inner);
    }
    f
}

/// Label / comment — carries no execution.
fn is_trivia(s: &Stmt) -> bool {
    matches!(s, Stmt::Label(_) | Stmt::Comment(_))
}

/// A call evaluated for its side effect, in either the bare or the
/// assigned form (`v = f(…)` where the assigned value is the call).
fn is_call_stmt(s: &Stmt) -> bool {
    fn is_call_expr(e: &Expr) -> bool {
        matches!(e, Expr::RawCall(..) | Expr::Call { .. })
    }
    match s {
        Stmt::Eval(e) => is_call_expr(e),
        Stmt::Assign(_, _, e) => is_call_expr(e),
        _ => false,
    }
}

/// Does this expression read a word at a FIXED, image-relative address — an
/// IL2CPP static slot / metadata word — rather than a value that can change?
///
/// The check exists to keep the collapse from firing on a user field: the
/// guard always tests a static word (`*(byte*)(0x… + *(nint*)(0x… ))`), and
/// ordinary instance reads (`this._flag == 0`) never do.
fn reads_static_word(e: &Expr) -> bool {
    match e {
        Expr::StaticField(..) | Expr::Int(_) => true,
        Expr::RawDeref(b, _, _) | Expr::RawAddr(b, _) => reads_static_word(b) || roots_in_immediate(e),
        Expr::Field(b, _) => reads_static_word(b),
        _ => roots_in_immediate(e),
    }
}

/// Recognise the static-init guard at the statement level.
fn is_class_init_guard(s: &Stmt) -> bool {
    let Stmt::If { cond, then_body, else_body } = s else {
        return false;
    };
    // The else arm must carry no execution — the guard has no false branch.
    if else_body.iter().any(|s| !is_trivia(s)) {
        return false;
    }
    // Condition: `<static word> == 0` / `!= 0` (either operand order).
    let Expr::Cmp(k, a, b) = cond else { return false };
    if !matches!(k, CmpKind::Eq | CmpKind::Ne) {
        return false;
    }
    let load = if matches!(**b, Expr::Int(0)) {
        &**a
    } else if matches!(**a, Expr::Int(0)) {
        &**b
    } else {
        return false;
    };
    if !reads_static_word(load) {
        return false;
    }
    // Guarded arm: EXACTLY one call and one `*… = 1` store, in either order.
    // Any third real statement means the arm does more than initialize.
    let mut call = false;
    let mut store_one = false;
    for st in then_body {
        if is_trivia(st) {
            continue;
        }
        if !call && is_call_stmt(st) {
            call = true;
            continue;
        }
        if !store_one
            && matches!(st, Stmt::Assign(lhs, _, Expr::Int(1) | Expr::Bool(true)) if lhs.starts_with('*'))
        {
            store_one = true;
            continue;
        }
        return false;
    }
    call && store_one
}

// ---------------------------------------------------------------------------
// Dead-end blocks and dead code
// ---------------------------------------------------------------------------

/// Does this statement leave without falling through to the next one?
fn is_terminator(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue
    )
}

/// Index of the next statement after `i` that carries control flow: comments
/// are transparent, a label is the answer (it can be jumped to from elsewhere).
fn next_control_index(list: &[Stmt], i: usize) -> Option<usize> {
    let mut j = i + 1;
    while j < list.len() {
        if matches!(list[j], Stmt::Comment(_)) {
            j += 1;
            continue;
        }
        return Some(j);
    }
    None
}

fn dead_code(stmts: &mut Vec<Stmt>) {
    // Bottom-up: a nested block's dead tail must go before the parent's
    // adjacency checks look at it.
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                dead_code(then_body);
                dead_code(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => dead_code(body),
            Stmt::Try { try_body, catches, finally_body } => {
                dead_code(try_body);
                for (_, b) in catches.iter_mut() {
                    dead_code(b);
                }
                if let Some(f) = finally_body {
                    dead_code(f);
                }
            }
            _ => {}
        }
    }

    // --- 1. `goto L;` immediately followed by `L:` --------------------------
    //
    // A jump to the statement textually right after it is a no-op: fall-through
    // reaches the same label. Removed outright, because deleting a `goto`
    // evaluates nothing — this idiom can never change what the program observes.
    // This is what a dead-end block looks like once structured: the compiler's
    // empty block survives only as the jump.
    let old = std::mem::take(stmts);
    let mut keep = vec![true; old.len()];
    for i in 0..old.len() {
        let Stmt::Goto(l) = &old[i] else { continue };
        let Some(j) = next_control_index(&old, i) else { continue };
        if matches!(&old[j], Stmt::Label(x) if x == l) {
            keep[i] = false;
            IDIOMS.dead_goto_to_next_label.fetch_add(1, Ordering::Relaxed);
        }
    }

    // --- 2. DELIBERATELY NOT DONE: the "vacuous" guard -----------------------
    //
    // `if (c) goto L; L:` LOOKS like a branch that branches nowhere, and deleting
    // it is tempting — it is 81 sites on the 4000-method sample and makes the
    // output cleaner. It is also WRONG, and the metric caught it: deleting them
    // dropped 79 conditions that the concordance counts as real native branches.
    //
    // The reason is in `emit_if`: this shape appears when the branch's FALL block
    // rendered no statements (`else_body.is_empty()`), not when the two edges
    // reach the same block. In the machine the branch still skips that block —
    // `cmp …; b.eq L` with an instruction between the branch and `L`. So the
    // taken/fall edges diverge, the block in between was simply emitted as
    // nothing, and deleting the guard removes the only remaining trace that
    // anything was there at all. The text gets prettier and quietly less faithful.
    //
    // That is the module's first rule doing its job: never delete an observation.
    // A guard is dead only when its arms provably reconverge, which this pass
    // cannot see and the structurer already handles. `dead_goto_to_next_label`
    // below is the safe half of the same idea — it drops the JUMP, which
    // evaluates nothing, and keeps every condition in the text.
    *stmts = old
        .into_iter()
        .zip(keep)
        .filter_map(|(s, k)| k.then_some(s))
        .collect();

    // --- 3. statements after a terminator ----------------------------------
    //
    // Within ONE list, once control has left, nothing until the next `Label`
    // can run — a label is reachable from elsewhere, so it stops the sweep.
    let mut dropped = false;
    stmts.retain(|s| {
        if dropped {
            if !matches!(s, Stmt::Label(_)) {
                IDIOMS
                    .dead_statements_after_terminator
                    .fetch_add(1, Ordering::Relaxed);
                return false;
            }
            dropped = false;
        }
        if is_terminator(s) {
            dropped = true;
        }
        true
    });
}

/// Drop labels that no `goto` and no `switch` arm can reach.
///
/// A label is only ever a jump target, so removing an unreferenced one cannot
/// change control flow. Collected over the WHOLE body first, because the only
/// reference to a label may sit in a sibling subtree — the structurer emits
/// `goto Lx` in one arm for a label defined in another.
fn drop_unreferenced_labels(stmts: &mut Vec<Stmt>) {
    let mut referenced: HashSet<String> = HashSet::new();
    collect_jump_targets(stmts, &mut referenced);
    remove_labels(stmts, &referenced);
}

fn collect_jump_targets(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Goto(l) => {
                out.insert(l.clone());
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, body) in cases {
                    collect_jump_targets(body, out);
                }
                collect_jump_targets(tail, out);
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_jump_targets(then_body, out);
                collect_jump_targets(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_jump_targets(body, out),
            Stmt::Try { try_body, catches, finally_body } => {
                collect_jump_targets(try_body, out);
                for (_, b) in catches {
                    collect_jump_targets(b, out);
                }
                if let Some(f) = finally_body {
                    collect_jump_targets(f, out);
                }
            }
            _ => {}
        }
    }
}

fn remove_labels(stmts: &mut Vec<Stmt>, referenced: &HashSet<String>) {
    let before = stmts.len();
    stmts.retain(|s| !matches!(s, Stmt::Label(l) if !referenced.contains(l)));
    IDIOMS
        .dead_unreferenced_label
        .fetch_add((before - stmts.len()) as u64, Ordering::Relaxed);
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                remove_labels(then_body, referenced);
                remove_labels(else_body, referenced);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => remove_labels(body, referenced),
            Stmt::Try { try_body, catches, finally_body } => {
                remove_labels(try_body, referenced);
                for (_, b) in catches.iter_mut() {
                    remove_labels(b, referenced);
                }
                if let Some(f) = finally_body {
                    remove_labels(f, referenced);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Structure-field propagation
// ---------------------------------------------------------------------------

/// Hoist a repeatedly-READ structure-field load into one named local.
///
/// The renderer inlines a single-def local at its use site, which is what makes
/// the C# readable — but when the same address is computed from different
/// reaching definitions, the SAME native load ends up spelled out once per use.
/// A hot method then reads
///
/// ```text
/// if (*(nint*)(*(nint*)(obj) + 224) != 0) …
/// … *(nint*)(*(nint*)(obj) + 224) …
/// ```
///
/// where the machine loaded it once and branched on the register. Naming it
/// restores both the shape and the fact that it is ONE read:
///
/// ```text
/// sf0 = *(nint*)(*(nint*)(obj) + 224);
/// if (sf0 != 0) …
/// ```
///
/// Scope is deliberately narrow, because this is the one idiom that MOVES code:
///
///   * only `RawDeref(_, addend, _)` with a NON-ZERO addend qualifies — that is
///     the un-named structure-field load this idiom exists for; a zero addend is
///     usually a header/vtable word whose repetition is meaningful;
///   * only expression positions directly in ONE statement list are counted
///     (`Eval`, `Assign`, `Return`, `Throw`, `If.cond`), never inside a nested
///     body: a load guarded by a branch must not be hoisted out of it;
///   * loop conditions are excluded — hoisting out of a loop is a different and
///     much easier-to-get-wrong transformation;
///   * the load must be side-effect-free (no call anywhere inside it);
///   * nothing between its first and last read may call, construct, or ASSIGN a
///     local the address expression mentions.
fn propagate_fields(stmts: &mut Vec<Stmt>) {
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                propagate_fields(then_body);
                propagate_fields(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => propagate_fields(body),
            Stmt::Try { try_body, catches, finally_body } => {
                propagate_fields(try_body);
                for (_, b) in catches.iter_mut() {
                    propagate_fields(b);
                }
                if let Some(f) = finally_body {
                    propagate_fields(f);
                }
            }
            _ => {}
        }
    }
    propagate_fields_one_list(stmts);
}

fn propagate_fields_one_list(stmts: &mut Vec<Stmt>) {
    // Occurrences per canonical key, with the index of the statement holding
    // each one, in text order.
    let mut hits: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, s) in stmts.iter().enumerate() {
        let mut here: Vec<String> = Vec::new();
        // DESCEND into the expression: a structure-field load is almost never
        // the whole condition — it is buried in `(*(nint*)(obj + 224) != 0)`.
        // Counting only the top-level expression found nothing at all, which is
        // how this pass silently did no work until a test pinned it down.
        for_each_source_expr(s, &mut |e| {
            walk_expr(e, &mut |sub| {
                if let Some(k) = field_load_key(sub) {
                    here.push(k);
                }
            });
        });
        for k in here {
            hits.entry(k).or_default().push(i);
        }
    }

    let mut cands: Vec<(String, Vec<usize>)> =
        hits.into_iter().filter(|(_, v)| v.len() >= 2).collect();
    // DETERMINISTIC ORDER. Two different loads can first appear inside the SAME
    // statement, so sorting on the first index alone leaves ties, and `HashMap`
    // iteration order then decides which one is hoisted first. That made the
    // pass non-deterministic — `scripts/det_gate.sh` caught 45 methods whose two
    // runs rendered differently — and a decompiler that emits different text for
    // the same bytes is not usable. Ties are broken on the key, which is derived
    // from the rendered expression, so the order is total and reproducible.
    cands.sort_by(|a, b| a.1[0].cmp(&b.1[0]).then_with(|| a.0.cmp(&b.0)));

    let mut used_names: HashSet<String> = HashSet::new();
    collect_var_names(stmts, &mut used_names);
    let mut next_temp: u64 = 0;

    for (key, sites) in cands {
        let first = sites[0];
        let last = *sites.last().unwrap();
        let shadowed = vars_in_key(&key);
        // Nothing between the first and last read may be able to change the
        // value. `first + 1..last` excludes the reads themselves, but a read
        // statement can also contain a call — that is what it WOULD do, not what
        // it changes before the read, so it is correctly outside the range.
        if (first + 1..last).any(|i| stmt_may_write(&stmts[i], &shadowed)) {
            continue;
        }
        let name = loop {
            let n = format!("sf{next_temp}");
            next_temp += 1;
            if !used_names.contains(&n) {
                break n;
            }
        };
        used_names.insert(name.clone());

        // Materialize the load ONCE, immediately before its first use — exactly
        // where the machine evaluated it.
        let mut value = None;
        for_each_source_expr(&stmts[first], &mut |e| {
            if value.is_none() && find_field_load(e, &key).is_some() {
                value = find_field_load(e, &key);
            }
        });
        let Some(value) = value else { continue };
        stmts.insert(first, Stmt::Assign(name.clone(), None, value));
        // Indices at and after the insertion point shift by one.
        let last = last + 1;
        let mut collapsed = 0u64;
        for s in stmts[first + 1..=last].iter_mut() {
            for_each_source_expr_mut(s, &mut |e| {
                if field_load_key(e).as_deref() == Some(key.as_str()) {
                    *e = Expr::Var(name.clone());
                    collapsed += 1;
                }
            });
        }
        IDIOMS.fields_propagated.fetch_add(1, Ordering::Relaxed);
        IDIOMS
            .field_reads_collapsed
            .fetch_add(collapsed, Ordering::Relaxed);
        // The list changed shape; rescan so an overlapping key is reconsidered
        // against the new layout. Each pass eliminates at least one key (a
        // substituted `Var` can never match `field_load_key`), so this ends.
        return propagate_fields_one_list(stmts);
    }
}

/// First sub-expression of `e` (depth-first, parents first) whose canonical key
/// matches, so the exact load node can be moved to a temporary.
fn find_field_load(e: &Expr, key: &str) -> Option<Expr> {
    if field_load_key(e).as_deref() == Some(key) {
        return Some(e.clone());
    }
    let mut found = None;
    walk_expr(e, &mut |sub| {
        if found.is_none() && field_load_key(sub).as_deref() == Some(key) {
            found = Some(sub.clone());
        }
    });
    found
}

/// Canonical text for a structure-field load, or `None` if this is not one.
/// Rendering is the same canonical form the printer uses, so two occurrences
/// that print identically are treated as the same load.
fn field_load_key(e: &Expr) -> Option<String> {
    match e {
        Expr::RawDeref(base, addend, _) if *addend != 0 => {
            if cleanup::expr_has_side_effect(base) {
                return None;
            }
            // NOT a load rooted at an immediate address. `*(nint*)(*(nint*)(0x6054568)
            // + 224)` is an IL2CPP static / class-init word: the second read can
            // legitimately see a DIFFERENT value than the first, because the
            // class initializer may run in between on another thread. Instance
            // structure fields are what this idiom is for; lazily-initialized
            // statics are the case where two reads are two observations.
            if roots_in_immediate(base) {
                return None;
            }
            let mut t = String::new();
            e.render(&mut t);
            Some(t)
        }
        _ => None,
    }
}

/// Does this address expression bottom out in a fixed immediate address (a
/// static/metadata word) rather than in a live value?
fn roots_in_immediate(e: &Expr) -> bool {
    match e {
        Expr::Int(_) => true,
        // A second load is the usual shape: `*(nint*)(*(nint*)(obj) + 176)`.
        Expr::RawDeref(b, _, _) | Expr::RawAddr(b, _) => roots_in_immediate(b),
        Expr::Field(b, _) | Expr::Not(b) | Expr::Neg(b) | Expr::Cast(_, b) => roots_in_immediate(b),
        Expr::Bin(_, a, b) => roots_in_immediate(a) || roots_in_immediate(b),
        _ => false,
    }
}

/// Can this statement change the value a load over `names` would read?
fn stmt_may_write(s: &Stmt, names: &HashSet<String>) -> bool {
    match s {
        // A call can write any memory the callee can reach.
        Stmt::Eval(e) => cleanup::expr_has_side_effect(e),
        Stmt::Assign(target, _, rhs) => {
            cleanup::expr_has_side_effect(rhs)
                || names.contains(target)
                || expr_mentions_any_var(rhs, names)
        }
        Stmt::Return(Some(e)) | Stmt::Throw(e) => cleanup::expr_has_side_effect(e),
        Stmt::If { cond, then_body, else_body } => {
            cleanup::expr_has_side_effect(cond)
                || then_body.iter().any(|s| stmt_may_write(s, names))
                || else_body.iter().any(|s| stmt_may_write(s, names))
        }
        Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
            cleanup::expr_has_side_effect(cond)
                || body.iter().any(|s| stmt_may_write(s, names))
        }
        Stmt::Try { try_body, catches, finally_body } => {
            try_body.iter().any(|s| stmt_may_write(s, names))
                || catches
                    .iter()
                    .any(|(_, b)| b.iter().any(|s| stmt_may_write(s, names)))
                || finally_body
                    .as_ref()
                    .is_some_and(|f| f.iter().any(|s| stmt_may_write(s, names)))
        }
        _ => false,
    }
}

/// Identifier-shaped tokens of a rendered expression, so `stmt_may_write` can ask
/// whether a write rebinds one of them. Only used to be CONSERVATIVE: a spurious
/// match merely declines the optimisation.
fn vars_in_key(key: &str) -> HashSet<String> {
    key.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

fn expr_mentions_any_var(e: &Expr, names: &HashSet<String>) -> bool {
    match e {
        Expr::Var(n) => names.contains(n),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_mentions_any_var(a, names) || expr_mentions_any_var(b, names)
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Field(a, _) => {
            expr_mentions_any_var(a, names)
        }
        Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) => expr_mentions_any_var(a, names),
        Expr::AddressOf(a) | Expr::Await(a) => expr_mentions_any_var(a, names),
        Expr::Cond(a, b, c) => {
            expr_mentions_any_var(a, names)
                || expr_mentions_any_var(b, names)
                || expr_mentions_any_var(c, names)
        }
        Expr::Call { recv, args, .. } => {
            recv.as_ref()
                .is_some_and(|r| expr_mentions_any_var(r, names))
                || args.iter().any(|a| expr_mentions_any_var(a, names))
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(|a| expr_mentions_any_var(a, names))
        }
        _ => false,
    }
}

fn collect_var_names(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Assign(t, _, _) => {
                out.insert(t.clone());
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_var_names(then_body, out);
                collect_var_names(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_var_names(body, out),
            Stmt::Try { try_body, catches, finally_body } => {
                collect_var_names(try_body, out);
                for (_, b) in catches {
                    collect_var_names(b, out);
                }
                if let Some(f) = finally_body {
                    collect_var_names(f, out);
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Structured restatement
// ---------------------------------------------------------------------------
//
// These rewrites do not recover anything the structurer could not already see;
// they change the SHAPE of what it emitted into the shape the source had. Each
// one is a bijection on the machine's behaviour — a fold that cannot be proven
// is a no-op, exactly like the rewrite catalog — and each counts itself, so its
// reach over a real corpus is a number rather than a claim.

/// Drop `v = v;` — an assignment that stores a local's own value back into it.
///
/// The lifter/SSA pipeline leaves these where a register shuffle was folded to
/// a self-move (`mov r4, r4`, or a phi whose arms both resolved to the same
/// version). A local self-store evaluates nothing and changes nothing, so this
/// is the purest of the safe rewrites.
///
/// Deliberately narrow: only a whole-variable target (`*…` targets are memory,
/// where the ADDRESS expression is re-evaluated and may itself be volatile) and
/// only when the right-hand side is that same variable and nothing else.
fn drop_self_assignments(stmts: &mut Vec<Stmt>) {
    for s in stmts.iter_mut() {
        recurse_into_bodies(s, &mut |b| drop_self_assignments(b));
    }
    let before = stmts.len();
    stmts.retain(|s| !is_self_assignment(s));
    let removed = (before - stmts.len()) as u64;
    if removed > 0 {
        IDIOMS
            .self_assignments_dropped
            .fetch_add(removed, Ordering::Relaxed);
    }
}

fn is_self_assignment(s: &Stmt) -> bool {
    match s {
        Stmt::Assign(t, ty, Expr::Var(n)) => {
            ty.is_none() && t == n && !t.starts_with('*')
        }
        _ => false,
    }
}

/// Fold a guarded pair of assignments into one conditional expression.
///
/// ```text
///   if (c) { v = a; } else { v = b; }   →   v = c ? a : b;
/// ```
///
/// This is `?:` spelled out, and the compiler's lowering of a ternary is
/// exactly this pair of arms, so folding it back is a restatement rather than a
/// guess. The two forms evaluate the SAME things in the SAME order: the
/// condition once, then exactly one arm (the other not at all), with the store
/// performed last in both. Nothing is reordered, so nothing observable changes.
///
/// Requirements, each of which only declines the fold:
///   * both arms carry exactly ONE real statement, an assignment;
///   * both assign the SAME whole local (never a `*…` memory target — there the
///     address expression would be printed twice and re-evaluated);
///   * neither arm declares a type (a `T v = …` inside one arm is a scope, not
///     a select);
///   * no label anywhere in either arm — a label is a jump target, and the
///     folded form has no place to define it.
fn fold_if_assign_to_ternary(stmts: &mut Vec<Stmt>) {
    for s in stmts.iter_mut() {
        recurse_into_bodies(s, &mut |b| fold_if_assign_to_ternary(b));
    }
    let mut n = 0u64;
    for s in stmts.iter_mut() {
        let Stmt::If { cond, then_body, else_body } = s else {
            continue;
        };
        let (Some((tv, tt, te)), Some((ev, et, ee))) =
            (sole_assignment(then_body), sole_assignment(else_body))
        else {
            continue;
        };
        if tv != ev || tt != et {
            continue;
        }
        *s = Stmt::Assign(
            tv,
            tt,
            Expr::Cond(Box::new(cond.clone()), Box::new(te), Box::new(ee)),
        );
        n += 1;
    }
    if n > 0 {
        IDIOMS.ifs_to_ternary.fetch_add(n, Ordering::Relaxed);
    }
}

/// The single non-trivia assignment of `body`, if that is all it holds.
fn sole_assignment(body: &[Stmt]) -> Option<(String, Option<String>, Expr)> {
    let mut found: Option<(String, Option<String>, Expr)> = None;
    for s in body {
        match s {
            Stmt::Comment(_) => continue,
            // A label is a jump target: the folded form cannot hold one.
            Stmt::Label(_) => return None,
            Stmt::Assign(t, ty, e) => {
                if found.is_some() || t.starts_with('*') {
                    return None;
                }
                found = Some((t.clone(), ty.clone(), e.clone()));
            }
            _ => return None,
        }
    }
    found
}

/// Fold `if (a) { if (b) { B } }` into `if (a && b) { B }`.
///
/// Both the machine and the source evaluate `b` only when `a` held, which is
/// `&&`'s short-circuit rule, so the compound predicate is the same program
/// with one branch and one brace level fewer. This is the compiler's test-chain
/// lowering read back: a nested guard chain produced the staircase, and the
/// source wrote one `&&` chain.
///
/// Declines unless the outer arm holds nothing but the inner `if`, the outer
/// has no else at all, and the inner has no else either — with an inner `else`
/// the folded form would need `else if`, which is a different (and larger)
/// rewrite.
fn combine_nested_ifs(stmts: &mut Vec<Stmt>) {
    for s in stmts.iter_mut() {
        recurse_into_bodies(s, &mut |b| combine_nested_ifs(b));
    }
    let mut n = 0u64;
    for s in stmts.iter_mut() {
        let Stmt::If { cond, then_body, else_body } = s else {
            continue;
        };
        if else_body.iter().any(|x| !is_trivia(x)) {
            continue;
        }
        // Exactly one real statement, and it is the inner `if`.
        let mut inner: Option<(Expr, Vec<Stmt>)> = None;
        let mut ok = true;
        for x in then_body.iter() {
            match x {
                Stmt::Comment(_) => continue,
                Stmt::Label(_) => {
                    ok = false;
                    break;
                }
                Stmt::If { cond: c2, then_body: t2, else_body: e2 } => {
                    if inner.is_some() || e2.iter().any(|y| !is_trivia(y)) {
                        ok = false;
                        break;
                    }
                    inner = Some((c2.clone(), t2.clone()));
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        let Some((inner_cond, inner_body)) = inner.filter(|_| ok) else {
            continue;
        };
        *s = Stmt::If {
            cond: Expr::Bin(BinOp::AndAlso, Box::new(cond.clone()), Box::new(inner_cond)),
            then_body: inner_body,
            else_body: Vec::new(),
        };
        n += 1;
    }
    if n > 0 {
        IDIOMS.nested_ifs_combined.fetch_add(n, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Enumerator loops → `foreach`
// ---------------------------------------------------------------------------
//
// `foreach (var x in c) { … }` is not a loop shape the machine has. C# lowers
// it to the enumerator protocol:
//
//     V = c.GetEnumerator();
//     try { while (V.MoveNext()) { x = V.Current;  … } }
//     finally { (V as IDisposable)?.Dispose(); }
//
// so the structurer, which sees only the lowered form, renders exactly that.
// Recognising it back is not a guess about the machine: the three statements
// ARE the iterator protocol, and the source had no other way to produce them.
//
// This is the one statement-tree idiom besides the static-init guard that
// REMOVES A CONDITION, so it is pinned the same way that one is — the whole
// shape must hold, and anything not provable is left alone:
//
//   * the `MoveNext` test must be the loop's ONLY exit. It may be the loop
//     condition (`while (e.MoveNext())`) or the body's first statement
//     (`t = e.MoveNext(); if (t == 0) break;`); in the latter case the
//     enclosing loop's own condition must never exit (`while (true)` /
//     `do … while (true);`), which is exactly the compiler's spelling for "the
//     `MoveNext` break is the only way out".
//   * a `do … while (c)` with a real `c` is REFUSED, even with a `MoveNext`
//     break in the body: `c` is a second exit the `foreach` protocol does not
//     explain, and collapsing would silently drop it.
//
//     Measured example of that refusal: `GetExistingParameters` @0x12EA65C has
//     `v77 = v71.MoveNext(); if (v77 == 0) break;` in the body AND an
//     independent latch test (`v89 == 0`), i.e. a secondary exit the protocol
//     does not explain (the `DSA`-key scan). It is left structured.
//   * every read of the enumerator must be inside that loop — an enumerator
//     mentioned after it, or in a sibling `finally`, is also disposed there and
//     the foreach would not define it;
//   * after the `MoveNext` test and the `.Current` read are accounted for,
//     nothing may still mention the enumerator. The IL2CPP null-guard chains
//     around `V` (`if (V != 0) goto L; helper(…); L:`) are such mentions —
//     they are the runtime's null check, not the program, so they are STRIPPED
//     first (see `strip_enumerator_guards`); any OTHER surviving mention of `V`
//     refuses the fold;
//   * the element must be a single `x = V.Current;` at the top of the body
//     (which becomes the loop variable) or absent entirely (a fresh name is
//     introduced and every `V.Current` read is renamed to it).

/// The collection an enumerator is acquired from, for the acquisition shape
/// `«collection».GetEnumerator()` / `.get_Enumerator()`.
fn enumerator_acquire(e: &Expr) -> Option<Expr> {
    match e {
        Expr::Call { recv: Some(c), name, args }
            if args.is_empty() && (name == "GetEnumerator" || name == "get_Enumerator") =>
        {
            Some((**c).clone())
        }
        _ => None,
    }
}

/// The local an `«local».MoveNext()` expression is called on.
fn move_next_receiver(e: &Expr) -> Option<&str> {
    match e {
        Expr::Call { recv: Some(r), name, args } if args.is_empty() && name == "MoveNext" => {
            match &**r {
                Expr::Var(v) => Some(v),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Is this expression the element read `«v».Current` / `«v».get_Current()`?
fn is_current_of(e: &Expr, v: &str) -> bool {
    match e {
        Expr::Field(b, name) if name == "Current" => {
            matches!(&**b, Expr::Var(x) if x == v)
        }
        Expr::Call { recv: Some(r), name, args }
            if args.is_empty() && (name == "get_Current" || name == "Current") =>
        {
            matches!(&**r, Expr::Var(x) if x == v)
        }
        _ => false,
    }
}

/// Collapse every reconstructible enumerator loop, reporting how many were.
pub fn collapse_enumerator_loops(stmts: &mut Vec<Stmt>) -> u64 {
    // Read counts for the WHOLE body, so the shape test can insist that the
    // enumerator is mentioned nowhere outside the loop it belongs to.
    let mut reads: HashMap<String, usize> = HashMap::new();
    for_each_expr_node(stmts, &mut |e| {
        if let Expr::Var(v) = e {
            *reads.entry(v.clone()).or_insert(0) += 1;
        }
    });
    let n = collapse_enumerator_loops_here(stmts, &reads);
    if n > 0 {
        IDIOMS.foreach_collapsed.fetch_add(n, Ordering::Relaxed);
    }
    n
}

fn collapse_enumerator_loops_here(stmts: &mut Vec<Stmt>, reads: &HashMap<String, usize>) -> u64 {
    // Deepest first, so a loop nested in an already-collapsed body still sees
    // its own acquisition.
    for s in stmts.iter_mut() {
        recurse_into_bodies(s, &mut |b| {
            collapse_enumerator_loops_here(b, reads);
        });
    }
    let mut n = 0u64;
    let mut buf: std::collections::VecDeque<Stmt> = std::mem::take(stmts).into();
    let mut out: Vec<Stmt> = Vec::with_capacity(buf.len());
    while let Some(cur) = buf.pop_front() {
        if let Stmt::Assign(v, None, rhs) = &cur {
            if let Some(coll) = enumerator_acquire(rhs) {
                // Comments may sit between the acquisition and the loop; a
                // LABEL may not (a jump could enter the loop, bypassing the
                // foreach header).
                let mut k = 0usize;
                while k < buf.len() && matches!(buf[k], Stmt::Comment(_)) {
                    k += 1;
                }
                if k < buf.len() {
                    let lp = buf[k].clone();
                    if let Some(fe) = try_collapse_enumerator(v, &coll, &lp, reads) {
                        for _ in 0..=k {
                            buf.pop_front();
                        }
                        out.push(fe);
                        n += 1;
                        continue;
                    }
                }
            }
        }
        out.push(cur);
    }
    *stmts = out;
    n
}

fn try_collapse_enumerator(
    v: &str,
    coll: &Expr,
    lp: &Stmt,
    reads: &HashMap<String, usize>,
) -> Option<Stmt> {
    // Which loop form is this, and does the `MoveNext` test sit in the loop
    // CONDITION or open the BODY? See the module note above for the shapes.
    let (mut body, test_outside) = match lp {
        Stmt::While { cond, body } => {
            if move_next_receiver(cond) == Some(v) {
                (body.clone(), true)
            } else if is_trivially_true(cond) {
                (body.clone(), false)
            } else {
                return None;
            }
        }
        Stmt::DoWhile { body, cond } if is_trivially_true(cond) => (body.clone(), false),
        _ => return None,
    };

    // Every read of the enumerator must be inside this loop: `inside` covers the
    // body, `cond_reads` the `while (e.MoveNext())` test when it lives there.
    let inside = var_reads(&body, v);
    let cond_reads = usize::from(test_outside);
    if reads.get(v).copied().unwrap_or(0) != inside + cond_reads {
        // Anything else means the enumerator escapes the loop.
        return None;
    }

    // The runtime's null checks around calls on the enumerator evaluate nothing
    // a `foreach` needs; without stripping them `inside` includes mentions that
    // would refuse the fold — which is why those loops stay lowered today.
    strip_enumerator_guards(&mut body, v);

    // When the test was not the loop's condition it opens the body:
    //   `t = e.MoveNext(); if (t == 0) break;`  or  `if (!e.MoveNext()) break;`
    if !test_outside && !take_body_move_next_test(&mut body, v) {
        return None;
    }

    // The element: a single top-level `x = V.Current;`, or none at all.
    let mut named: Option<String> = None;
    for s in body.iter() {
        if let Stmt::Assign(e, None, rhs) = s {
            if is_current_of(rhs, v) && !e.starts_with('*') {
                if named.is_some() {
                    return None; // two element locals: ambiguous
                }
                named = Some(e.clone());
            }
        }
    }
    // The chosen local must belong to this loop alone.
    if let Some(e) = &named {
        if count_writes(&body, e) != 1 {
            return None;
        }
    }
    let named_elem = named.is_some();
    let elem = named.unwrap_or_else(|| fresh_loop_var(&body));
    if named_elem {
        remove_first_assign(&mut body, v);
    }
    // Every remaining `.Current` read becomes the loop variable.
    for_each_expr_node_mut(&mut body, &mut |e| {
        if is_current_of(e, v) {
            *e = Expr::Var(elem.clone());
        }
    });
    // Nothing may still mention the enumerator: the `foreach` is the only
    // definition it has now.
    if var_reads(&body, v) > 0 {
        return None;
    }
    let mut coll_s = String::new();
    coll.render(&mut coll_s);
    Some(Stmt::ForEach { var_name: elem, collection: coll_s, body })
}

/// `true` / `1` / `0 == 0` — a loop condition that never exits.
///
/// The structurer spells an infinite loop `while (true)`, so that is the shape
/// the `MoveNext` break form is recognised under.
fn is_trivially_true(e: &Expr) -> bool {
    match e {
        Expr::Bool(true) | Expr::Int(1) => true,
        Expr::Cmp(CmpKind::Eq, a, b) => {
            matches!(&**a, Expr::Int(0)) && matches!(&**b, Expr::Int(0))
        }
        Expr::Cmp(CmpKind::Ge, a, b) | Expr::Cmp(CmpKind::Le, a, b) => {
            matches!(&**a, Expr::Int(0)) && matches!(&**b, Expr::Int(0))
        }
        _ => false,
    }
}

/// `t == 0` / `!t` — the local bound to `MoveNext()` says "no more elements".
fn is_move_next_failure(cond: &Expr, t: &str) -> bool {
    match cond {
        Expr::Not(inner) => matches!(&**inner, Expr::Var(x) if x == t),
        Expr::Cmp(CmpKind::Eq, a, b) => {
            (matches!(&**a, Expr::Var(x) if x == t) && matches!(&**b, Expr::Int(0)))
                || (matches!(&**b, Expr::Var(x) if x == t) && matches!(&**a, Expr::Int(0)))
        }
        _ => false,
    }
}

/// `!e.MoveNext()` / `e.MoveNext() == 0` — the test written inline in a guard.
fn cond_is_move_next_failure(cond: &Expr, v: &str) -> bool {
    match cond {
        Expr::Not(inner) => move_next_receiver(inner) == Some(v),
        Expr::Cmp(CmpKind::Eq, a, b) => {
            (move_next_receiver(a) == Some(v) && matches!(&**b, Expr::Int(0)))
                || (move_next_receiver(b) == Some(v) && matches!(&**a, Expr::Int(0)))
        }
        _ => false,
    }
}

/// Remove a leading `t = e.MoveNext(); if (t == 0) break;` (or the inline
/// `if (!e.MoveNext()) break;`) from a loop body, skipping comments first.
///
/// Returns false when the body does not open with such a test, or when the
/// temporary it binds is read anywhere the fold would no longer define it.
fn take_body_move_next_test(body: &mut Vec<Stmt>, v: &str) -> bool {
    let mut i = 0usize;
    while i < body.len() && matches!(body[i], Stmt::Comment(_)) {
        i += 1;
    }
    // `t = e.MoveNext(); if (t == 0) break;`
    if let Some(Stmt::Assign(t, _, rhs)) = body.get(i) {
        if move_next_receiver(rhs) == Some(v) && !t.starts_with('*') {
            let t = t.clone();
            if let Some(Stmt::If { cond, then_body, else_body }) = body.get(i + 1) {
                if else_body.is_empty()
                    && then_body.len() == 1
                    && matches!(then_body[0], Stmt::Break)
                    && is_move_next_failure(cond, &t)
                {
                    body.drain(i..i + 2);
                    return var_reads(body, &t) == 0;
                }
            }
        }
    }
    // `if (!e.MoveNext()) break;`
    if let Some(Stmt::If { cond, then_body, else_body }) = body.get(i) {
        if else_body.is_empty()
            && then_body.len() == 1
            && matches!(then_body[0], Stmt::Break)
            && cond_is_move_next_failure(cond, v)
        {
            body.remove(i);
            return true;
        }
    }
    false
}

/// How many expressions in `stmts` read local `name`.
fn var_reads(stmts: &[Stmt], name: &str) -> usize {
    let mut n = 0usize;
    for_each_expr_node(stmts, &mut |e| {
        if matches!(e, Expr::Var(x) if x == name) {
            n += 1;
        }
    });
    n
}

/// Does `e` mention variable `v` at all?
fn mentions_var(e: &Expr, v: &str) -> bool {
    let mut found = false;
    walk_expr(e, &mut |x| {
        if matches!(x, Expr::Var(n) if n == v) {
            found = true;
        }
    });
    found
}

/// Does `e` mention any variable OTHER than `v`?
fn mentions_only_var(e: &Expr, v: &str) -> bool {
    let mut ok = true;
    walk_expr(e, &mut |x| {
        if let Expr::Var(n) = x {
            if n != v {
                ok = false;
            }
        }
    });
    ok
}

/// How many `goto`s target each label.
fn count_goto_targets(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in stmts {
        match s {
            Stmt::Goto(l) => *out.entry(l.clone()).or_insert(0) += 1,
            Stmt::If { then_body, else_body, .. } => {
                count_goto_targets(then_body, out);
                count_goto_targets(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => count_goto_targets(body, out),
            Stmt::ForEach { body, .. } => count_goto_targets(body, out),
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    count_goto_targets(b, out);
                }
                count_goto_targets(tail, out);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                count_goto_targets(try_body, out);
                for (_, b) in catches {
                    count_goto_targets(b, out);
                }
                if let Some(f) = finally_body {
                    count_goto_targets(f, out);
                }
            }
            _ => {}
        }
    }
}

/// Length of the IL2CPP null-guard starting at `i` in `body`, or 0.
///
/// The runtime's null check before a call on the enumerator is one of
///
///     if (v != 0) goto L;   helper(…);   L:     // branch form
///     if (v == 0) helper(…);                    // guard form
///
/// where `helper` is the null-reference thrower (an unresolved raw call). A
/// `foreach` over `v` cannot be null, so neither evaluates anything the program
/// needs. The branch form is only recognised when THIS guard is the label's one
/// and only target, so no `goto` is left pointing at a deleted label.
fn guard_len_at(body: &[Stmt], i: usize, v: &str, targets: &HashMap<String, usize>) -> usize {
    let Stmt::If { cond, then_body, else_body } = &body[i] else {
        return 0;
    };
    if !else_body.is_empty()
        || then_body.len() != 1
        || !mentions_var(cond, v)
        || !mentions_only_var(cond, v)
    {
        return 0;
    }
    match &then_body[0] {
        Stmt::Goto(l) if targets.get(l).copied().unwrap_or(0) == 1 => {
            if matches!(body.get(i + 1), Some(Stmt::Eval(Expr::RawCall(..))))
                && matches!(body.get(i + 2), Some(Stmt::Label(x)) if x == l)
            {
                3
            } else {
                0
            }
        }
        Stmt::Eval(Expr::RawCall(..)) => 1,
        _ => 0,
    }
}

/// Strip every IL2CPP null-guard statement that references only the enumerator.
fn strip_enumerator_guards(body: &mut Vec<Stmt>, v: &str) {
    let mut targets: HashMap<String, usize> = HashMap::new();
    count_goto_targets(body, &mut targets);
    let mut i = 0usize;
    while i < body.len() {
        let step = guard_len_at(body, i, v, &targets);
        if step > 0 {
            body.drain(i..i + step);
        } else {
            i += 1;
        }
    }
}

/// A loop-variable name the body does not already use.
fn fresh_loop_var(body: &[Stmt]) -> String {
    let mut names: HashSet<String> = HashSet::new();
    for_each_expr_node(body, &mut |e| {
        if let Expr::Var(v) = e {
            names.insert(v.clone());
        }
    });
    for i in 0..1000u32 {
        let n = if i == 0 { "item".to_string() } else { format!("item{i}") };
        if !names.contains(&n) {
            return n;
        }
    }
    "item".to_string()
}

/// Number of times `name` is ASSIGNED anywhere in the tree.
fn count_writes(stmts: &[Stmt], name: &str) -> usize {
    let mut n = 0usize;
    count_writes_go(stmts, name, &mut n);
    n
}

fn count_writes_go(stmts: &[Stmt], name: &str, n: &mut usize) {
    for s in stmts {
        if matches!(s, Stmt::Assign(t, _, _) if t == name) {
            *n += 1;
        }
        match s {
            Stmt::If { then_body, else_body, .. } => {
                count_writes_go(then_body, name, n);
                count_writes_go(else_body, name, n);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                count_writes_go(body, name, n)
            }
            Stmt::ForEach { body, .. } => count_writes_go(body, name, n),
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    count_writes_go(b, name, n);
                }
                count_writes_go(tail, name, n);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                count_writes_go(try_body, name, n);
                for (_, b) in catches {
                    count_writes_go(b, name, n);
                }
                if let Some(f) = finally_body {
                    count_writes_go(f, name, n);
                }
            }
            _ => {}
        }
    }
}

/// Drop the first top-level `«_» = «v».Current;`.
fn remove_first_assign(body: &mut Vec<Stmt>, v: &str) {
    if let Some(i) = body.iter().position(
        |s| matches!(s, Stmt::Assign(_, None, rhs) if is_current_of(rhs, v)),
    ) {
        body.remove(i);
    }
}

/// Visit every expression node in the tree (each node once, children included).
fn for_each_expr_node<F: FnMut(&Expr)>(stmts: &[Stmt], f: &mut F) {
    for s in stmts {
        match s {
            Stmt::Eval(e) | Stmt::Assign(_, _, e) | Stmt::Throw(e) => walk_expr(e, f),
            Stmt::Return(Some(e)) => walk_expr(e, f),
            Stmt::If { cond, then_body, else_body } => {
                walk_expr(cond, f);
                for_each_expr_node(then_body, f);
                for_each_expr_node(else_body, f);
            }
            Stmt::While { cond, body } => {
                walk_expr(cond, f);
                for_each_expr_node(body, f);
            }
            Stmt::DoWhile { body, cond } => {
                for_each_expr_node(body, f);
                walk_expr(cond, f);
            }
            Stmt::ForEach { body, .. } => for_each_expr_node(body, f),
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    for_each_expr_node(b, f);
                }
                for_each_expr_node(tail, f);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                for_each_expr_node(try_body, f);
                for (_, b) in catches {
                    for_each_expr_node(b, f);
                }
                if let Some(fb) = finally_body {
                    for_each_expr_node(fb, f);
                }
            }
            _ => {}
        }
    }
}

/// Mutable mirror of [`for_each_expr_node`], parents before children.
fn for_each_expr_node_mut<F: FnMut(&mut Expr)>(stmts: &mut [Stmt], f: &mut F) {
    for s in stmts {
        match s {
            Stmt::Eval(e) | Stmt::Assign(_, _, e) | Stmt::Throw(e) => walk_expr_mut(e, f),
            Stmt::Return(Some(e)) => walk_expr_mut(e, f),
            Stmt::If { cond, then_body, else_body } => {
                walk_expr_mut(cond, f);
                for_each_expr_node_mut(then_body, f);
                for_each_expr_node_mut(else_body, f);
            }
            Stmt::While { cond, body } => {
                walk_expr_mut(cond, f);
                for_each_expr_node_mut(body, f);
            }
            Stmt::DoWhile { body, cond } => {
                for_each_expr_node_mut(body, f);
                walk_expr_mut(cond, f);
            }
            Stmt::ForEach { body, .. } => for_each_expr_node_mut(body, f),
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    for_each_expr_node_mut(b, f);
                }
                for_each_expr_node_mut(tail, f);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                for_each_expr_node_mut(try_body, f);
                for (_, b) in catches {
                    for_each_expr_node_mut(b, f);
                }
                if let Some(fb) = finally_body {
                    for_each_expr_node_mut(fb, f);
                }
            }
            _ => {}
        }
    }
}

/// Rewrite the machine's jump out of a loop back into the loop's own `break`,
/// and its jump back to a `while` loop's head into the loop's own `continue`.
///
/// The structurer emits `Stmt::Break`/`Stmt::Continue` when it can see the
/// loop's exit/latch edge, but a loop whose edge the walk found through a
/// `goto` keeps the jump. The two are the same statement whenever the jump's
/// target is the label the loop falls out to (`break` transfers control to
/// exactly the statement after the loop, which is where that label sits) or
/// the label the loop falls INTO from (`continue` in a `while` loop skips the
/// rest of the body and re-tests the condition, which is exactly what a jump
/// to just-before-the-loop does: nothing but labels and comments can sit
/// between that label and the loop, so falling through reaches the test).
///
/// The test is deliberately local and exact:
///   * the target label must be defined at the HEAD of the statements following
///     the loop (comments skipped) for `break` — i.e. `break` really does land
///     on it — or at the head of the label run immediately PRECEDING a `while`
///     loop for `continue`;
///   * that label must NOT be defined anywhere inside the loop body, or `goto`
///     would be a jump WITHIN the loop and neither restatement would leave it;
///   * `break` never inside a `switch` arm, where `break` means "leave the
///     switch" (`continue` is allowed there: it passes through the switch to
///     the enclosing loop, which is what the jump did);
///   * `continue` only for `Stmt::While`. A jump to before a `do … while`
///     re-executes the body from the top instead of testing the condition, a
///     jump to before a `for` re-runs its initializer, and a jump to before a
///     `foreach` re-evaluates its collection — none of those is `continue`.
///     (Measured: the refused `do` class is 10 sites on the 4000-method iOS
///     sample.)
///
/// A `goto` further into the parent list than the loop's immediate exit is NOT
/// folded: the loop exit and the target have real statements between them that
/// the jump skips, so `break` would execute code the machine never ran there.
/// Neither is a `goto` from a `switch` arm to another arm's label: C# spells
/// that `goto case`, which the statement tree cannot express without a new
/// variant and a renderer arm in the structurer — out of scope for this pass.
///
/// Public because it has a SECOND call site: the text pipeline retries it
/// after `drop_unreachable_gotos`. The rewrite needs the label the loop falls
/// out to to be adjacent to the loop, and at [`apply`] time a stale dangling
/// `goto` frequently still sits between the two — the label was emitted by the
/// coverage repair, the goto was not. That sweep removes the stale jump a pass
/// later, which is what makes the idiom reachable at all.
pub fn collapse_goto_to_loop_break(stmts: &mut Vec<Stmt>) {
    let root = BreakCtx {
        exit: HashSet::new(),
        header: HashSet::new(),
        inside: HashSet::new(),
        can_break: false,
        can_continue: false,
    };
    let (nb, nc) = rewrite_loop_breaks(stmts, &root);
    if nb > 0 {
        IDIOMS.goto_to_loop_break.fetch_add(nb, Ordering::Relaxed);
    }
    if nc > 0 {
        IDIOMS
            .goto_to_loop_continue
            .fetch_add(nc, Ordering::Relaxed);
    }
}

/// Where a `break`/`continue` would land, and which labels they must not be
/// confused with.
struct BreakCtx {
    /// Labels the enclosing loop falls out to.
    exit: HashSet<String>,
    /// Labels the enclosing `while` loop falls into from (its head label run).
    /// Empty unless the enclosing loop is a `while` — see the `continue`
    /// restrictions in [`collapse_goto_to_loop_break`].
    header: HashSet<String>,
    /// Every label defined inside the enclosing loop body.
    inside: HashSet<String>,
    /// False where a `break` would not mean "leave the enclosing loop" — the
    /// top level, and any `switch` arm.
    can_break: bool,
    /// False where a `continue` would not mean "next iteration of the
    /// enclosing loop" — the top level, and any loop that is not a `while`.
    /// True inside a `switch` arm: unlike `break`, `continue` passes through.
    can_continue: bool,
}

/// Returns `(breaks, continues)` rewritten in this list and its nested bodies.
fn rewrite_loop_breaks(list: &mut [Stmt], ctx: &BreakCtx) -> (u64, u64) {
    // Exit labels are read from the list, so compute them before mutating.
    let loop_exits: Vec<Option<HashSet<String>>> = (0..list.len())
        .map(|i| {
            (matches!(
                list[i],
                Stmt::While { .. } | Stmt::DoWhile { .. } | Stmt::ForEach { .. }
            ))
            .then(|| labels_after(list, i))
        })
        .collect();
    // Head labels likewise, for the `continue` half: only `while` loops get
    // one (see [`collapse_goto_to_loop_break`]).
    let loop_headers: Vec<Option<HashSet<String>>> = (0..list.len())
        .map(|i| matches!(list[i], Stmt::While { .. }).then(|| labels_before(list, i)))
        .collect();
    let (mut nb, mut nc) = (0u64, 0u64);
    for i in 0..list.len() {
        if let Stmt::Goto(l) = &list[i] {
            if ctx.can_break && ctx.exit.contains(l) && !ctx.inside.contains(l) {
                list[i] = Stmt::Break;
                nb += 1;
                continue;
            }
            if ctx.can_continue && ctx.header.contains(l) && !ctx.inside.contains(l) {
                list[i] = Stmt::Continue;
                nc += 1;
                continue;
            }
        }
        // A loop opens a NEW break/continue scope; every other construct stays
        // inside the enclosing one.
        match &mut list[i] {
            Stmt::While { body, .. } => {
                let exit = loop_exits[i].clone().unwrap_or_default();
                let header = loop_headers[i].clone().unwrap_or_default();
                let child = BreakCtx {
                    exit,
                    header,
                    inside: labels_in(body),
                    can_break: true,
                    can_continue: true,
                };
                let (b, c) = rewrite_loop_breaks(body, &child);
                nb += b;
                nc += c;
            }
            Stmt::DoWhile { body, .. } => {
                let exit = loop_exits[i].clone().unwrap_or_default();
                let child = BreakCtx {
                    exit,
                    header: HashSet::new(),
                    inside: labels_in(body),
                    can_break: true,
                    can_continue: false,
                };
                let (b, c) = rewrite_loop_breaks(body, &child);
                nb += b;
                nc += c;
            }
            Stmt::ForEach { body, .. } => {
                let exit = loop_exits[i].clone().unwrap_or_default();
                let child = BreakCtx {
                    exit,
                    header: HashSet::new(),
                    inside: labels_in(body),
                    can_break: true,
                    can_continue: false,
                };
                let (b, c) = rewrite_loop_breaks(body, &child);
                nb += b;
                nc += c;
            }
            Stmt::If { then_body, else_body, .. } => {
                let (b, c) = rewrite_loop_breaks(then_body, ctx);
                nb += b;
                nc += c;
                let (b, c) = rewrite_loop_breaks(else_body, ctx);
                nb += b;
                nc += c;
            }
            Stmt::Try { try_body, catches, finally_body } => {
                let (b, c) = rewrite_loop_breaks(try_body, ctx);
                nb += b;
                nc += c;
                for (_, b2) in catches.iter_mut() {
                    let (b, c) = rewrite_loop_breaks(b2, ctx);
                    nb += b;
                    nc += c;
                }
                if let Some(f) = finally_body {
                    let (b, c) = rewrite_loop_breaks(f, ctx);
                    nb += b;
                    nc += c;
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                // `break` inside a `case` leaves the SWITCH; `continue` still
                // means the enclosing loop, so it keeps the loop's scope.
                let no_break = BreakCtx {
                    exit: HashSet::new(),
                    header: ctx.header.clone(),
                    inside: HashSet::new(),
                    can_break: false,
                    can_continue: ctx.can_continue,
                };
                for (_, b) in cases.iter_mut() {
                    let (b2, c) = rewrite_loop_breaks(b, &no_break);
                    nb += b2;
                    nc += c;
                }
                let (b, c) = rewrite_loop_breaks(tail, ctx);
                nb += b;
                nc += c;
            }
            _ => {}
        }
    }
    (nb, nc)
}

/// Labels defined at the head of the statements after index `i`.
fn labels_after(stmts: &[Stmt], i: usize) -> HashSet<String> {
    let mut out = HashSet::new();
    for s in stmts.iter().skip(i + 1) {
        match s {
            Stmt::Comment(_) => continue,
            Stmt::Label(l) => {
                out.insert(l.clone());
            }
            _ => break,
        }
    }
    out
}

/// Labels defined by the contiguous label/comment run immediately BEFORE
/// index `i` — the head labels a `continue` restatement lands on. Anything but
/// a label or comment stops the run, so falling through the collected labels
/// reaches the loop with no intervening execution.
fn labels_before(stmts: &[Stmt], i: usize) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut j = i;
    while j > 0 {
        j -= 1;
        match &stmts[j] {
            Stmt::Comment(_) => continue,
            Stmt::Label(l) => {
                out.insert(l.clone());
            }
            _ => break,
        }
    }
    out
}

/// Every label defined anywhere in `stmts`.
fn labels_in(stmts: &[Stmt]) -> HashSet<String> {
    let mut out = HashSet::new();
    collect_labels(stmts, &mut out);
    out
}

fn collect_labels(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Label(l) => {
                out.insert(l.clone());
            }
            Stmt::If { then_body, else_body, .. } => {
                collect_labels(then_body, out);
                collect_labels(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_labels(body, out),
            Stmt::ForEach { body, .. } => collect_labels(body, out),
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    collect_labels(b, out);
                }
                collect_labels(tail, out);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                collect_labels(try_body, out);
                for (_, b) in catches {
                    collect_labels(b, out);
                }
                if let Some(f) = finally_body {
                    collect_labels(f, out);
                }
            }
            _ => {}
        }
    }
}

/// Recurse into every nested statement list of `s` with `f`.
fn recurse_into_bodies<F: FnMut(&mut Vec<Stmt>)>(s: &mut Stmt, f: &mut F) {
    match s {
        Stmt::If { then_body, else_body, .. } => {
            f(then_body);
            f(else_body);
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => f(body),
        Stmt::ForEach { body, .. } => f(body),
        Stmt::Try { try_body, catches, finally_body } => {
            f(try_body);
            for (_, b) in catches.iter_mut() {
                f(b);
            }
            if let Some(b) = finally_body {
                f(b);
            }
        }
        Stmt::Switch { cases, tail, .. } => {
            for (_, b) in cases.iter_mut() {
                f(b);
            }
            f(tail);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Expression traversal
// ---------------------------------------------------------------------------

/// Expression positions that evaluate in TEXT order within one statement list.
/// Loop conditions are deliberately excluded (see `propagate_fields`).
fn for_each_source_expr<F: FnMut(&Expr)>(s: &Stmt, f: &mut F) {
    match s {
        Stmt::Eval(e) => f(e),
        Stmt::Assign(_, _, e) => f(e),
        Stmt::Return(Some(e)) => f(e),
        Stmt::Throw(e) => f(e),
        Stmt::If { cond, .. } => f(cond),
        _ => {}
    }
}

fn for_each_source_expr_mut<F: FnMut(&mut Expr)>(s: &mut Stmt, f: &mut F) {
    match s {
        Stmt::Eval(e) => walk_expr_mut(e, f),
        Stmt::Assign(_, _, e) => walk_expr_mut(e, f),
        Stmt::Return(Some(e)) => walk_expr_mut(e, f),
        Stmt::Throw(e) => walk_expr_mut(e, f),
        Stmt::If { cond, .. } => walk_expr_mut(cond, f),
        _ => {}
    }
}

/// Read-only mirror of `walk_expr_mut`: visit this expression and every
/// sub-expression, parents before children.
fn walk_expr<F: FnMut(&Expr)>(e: &Expr, f: &mut F) {
    f(e);
    match e {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Field(a, _) => walk_expr(a, f),
        Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) => walk_expr(a, f),
        Expr::AddressOf(a) | Expr::Await(a) => walk_expr(a, f),
        Expr::Cond(a, b, c) => {
            walk_expr(a, f);
            walk_expr(b, f);
            walk_expr(c, f);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                walk_expr(r, f);
            }
            for a in args {
                walk_expr(a, f);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                walk_expr(a, f);
            }
        }
        _ => {}
    }
}

/// Visit this expression and every sub-expression, parents before children, so
/// a substitution at the root stops the descent into the replaced subtree.
fn walk_expr_mut<F: FnMut(&mut Expr)>(e: &mut Expr, f: &mut F) {
    f(e);
    match e {
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            walk_expr_mut(a, f);
            walk_expr_mut(b, f);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Field(a, _) => walk_expr_mut(a, f),
        Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) => walk_expr_mut(a, f),
        Expr::AddressOf(a) | Expr::Await(a) => walk_expr_mut(a, f),
        Expr::Cond(a, b, c) => {
            walk_expr_mut(a, f);
            walk_expr_mut(b, f);
            walk_expr_mut(c, f);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                walk_expr_mut(r, f);
            }
            for a in args {
                walk_expr_mut(a, f);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                walk_expr_mut(a, f);
            }
        }
        _ => {}
    }
}
