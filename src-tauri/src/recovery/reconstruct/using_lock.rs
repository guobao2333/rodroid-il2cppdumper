//! `lock (…)` and `using (…)` reconstruction.
//!
//! ## What the corpus actually contains (this is the whole point)
//!
//! Measured on the x64 corpus: `lock (` **0**, `using (` **0**, `try {` 18,
//! `finally` **0**, `Monitor.Enter` **145**.
//!
//! The previous version of this file only understood two shapes that do not
//! occur: `Monitor.Enter(x)` with ONE argument immediately followed by a
//! `try { … } finally { Monitor.Exit(x) }`, and `V = acquire()` followed by
//! `try { … } finally { V.Dispose(); }`. On this corpus `finally` is never
//! emitted at all, so both could never fire — the run report printed nothing.
//!
//! The shape that DOES occur is the flat form of the C# `lock` expansion. C#
//! compiles `lock (x) { body }` to
//!
//! ```text
//! bool lockTaken = false;
//! Monitor.Enter(x, ref lockTaken);
//! try { body } finally { if (lockTaken) Monitor.Exit(x); }
//! ```
//!
//! and when the EH structurer does not build a `finally` node (which is what
//! happens on this corpus), the region is emitted FLAT, with the cleanup
//! INLINED in front of every exit from the protected region. Verbatim from the
//! real dump (`dump.cs` line 7988):
//!
//! ```text
//! Monitor.Enter(*(nint*)(*(nint*)(v28 +152) +136), 0);
//! MSCompatUnicodeTable.FillCJKCore(…);
//! MSCompatUnicodeTable.SetCJKReferences(…);
//! Monitor.Exit(*(nint*)(*(nint*)(v28 +152) +136));
//! return;
//! ```
//!
//! ## The proof that lets this be folded
//!
//! Two independent facts identify the region and license the rewrite:
//!
//!   * **the second argument is the literal `0`.** The `lock` codegen passes a
//!     `lockTaken` out-parameter that it has just set to `false`; a hand-written
//!     `Monitor.Enter(o, ref flag)` passes a live local. A literal `0`/`false`
//!     is therefore positive evidence of `lock`, not of a hand-written monitor;
//!   * **the inlined-finally discipline.** Because the cleanup is INLINED, any
//!     exit from the protected region is preceded by exactly the matching
//!     `Monitor.Exit(<same object>)`. That is checkable, and it is what proves
//!     the native code had a `finally` in the first place. The fold then
//!     requires it BOTH ways:
//!       - every `Monitor.Exit(x)` in the region must be immediately followed by
//!         a statement that leaves the region (a `return`/`throw`, or a
//!         `goto`/`break`/`continue` whose target is outside it), and it is
//!         REMOVED as part of the fold — the `lock` statement performs it;
//!       - every statement in the region that leaves it must be immediately
//!         preceded by such a removed `Monitor.Exit(x)`. Without this direction
//!         the fold would ADD a release on a path where the flat code had none,
//!         i.e. it would change observable behaviour.
//!     A `break`/`continue` only leaves the region at loop depth 0 — inside a
//!     nested loop it belongs to that loop and stays in the region;
//!   * no nested `Monitor.Enter(x)` for the SAME object inside the region (a
//!     re-entrant pair is not one `lock`);
//!   * the pairing is unambiguous: scanning forward from the `Enter`, the next
//!     `Monitor.Enter(x)`/`Monitor.Exit(x)` on the same object must be the
//!     `Exit`. Two locks on the same object in one list
//!     (`Enter; A; Exit; Enter; B; Exit;`) therefore pair correctly;
//!   * a label inside the region that some `goto` OUTSIDE the region targets
//!     blocks the fold: C# does not permit jumping INTO a `lock` statement, so
//!     the label would become unreachable in the output while the `goto` stayed
//!     (the dangling-jump-target class). Labels targeted only from within the
//!     region are fine — they move with it.
//!
//! ## Status as MEASURED, not as intended
//!
//! The fold above did NOT fire on the x64 corpus despite that corpus containing
//! 145 `Monitor.Enter` statements (the run report printed no `using_lock` line
//! at all). Reading the gates cannot tell "the shape never occurs in one list"
//! from "a gate refuses everything", and those two need opposite responses, so
//! the pass is now INSTRUMENTED: it counts the sites it examines and the reason
//! each one was left alone, and `report_refusals` prints both through the run
//! report. Only after that breakdown exists is it responsible to relax a gate —
//! every gate here is sound, so a blind relaxation would be trading a silent
//! pass for a wrong program.
//!
//! The gate I would look at first is [`Refusal::NoPair`]: the fold pairs an
//! `Enter` with the next same-object event **in the same statement list**, and
//! the structurer emits unrecoverable regions as separately-labelled blocks. If
//! the measured breakdown is dominated by `NoPair`, the `Exit` is in a sibling
//! list and the fix is to pair across the two lists that form one region — not
//! to weaken the pairing rule.
//!
//! ## What this file deliberately does NOT do
//!
//! The `try/finally` spellings are kept, because they are the correct shape
//! wherever the structurer DOES build a `Try` node — but on the x64 corpus they
//! are unreachable (`finally` count 0) and must not be counted as working. The
//! `using` fold has no reachable spelling here at all: `Dispose()` appears
//! 2,050 times and almost every one of them is the *declaration*
//! `public void Dispose() {`, not a call, so there is no
//! `V = acquire(); try { … } finally { V.Dispose(); }` to find. There is no
//! second, weaker `using` shape to fold: without the `Dispose` call as the
//! cleanup, `V = acquire(); …; V.Dispose();` is indistinguishable from an
//! ordinary local that happens to be disposed at the end of the method, and
//! claiming a `using` there would invent a scope the native code does not have.

use super::{
    children, children_mut, exprs_eq, is_inlinable, leaves_region, monitor_stmt, monitor_stmt_on,
    stmt_reads_deep, walk_mut, Expr, Stmt,
};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Why sites are refused (the difference between a measurement and a silence)
// ---------------------------------------------------------------------------
//
// This pass reported ZERO rewrites on a corpus that contains 145
// `Monitor.Enter` sites. A bare zero cannot distinguish "the shape never
// occurs" from "a gate refuses everything", and those two have opposite
// consequences: the first means the pass is dead code, the second means one
// gate is wrong. These counters answer which — the run report now prints the
// population of sites AND the reason each one was left alone.

/// Every `Monitor.Enter` statement examined (counted once per statement list).
static ENTER_SITES: AtomicU64 = AtomicU64::new(0);
static FOLDED: AtomicU64 = AtomicU64::new(0);
/// No second same-target monitor event in this list.
static REFUSED_NO_PAIR: AtomicU64 = AtomicU64::new(0);
/// A same-target event precedes the `Enter`, or the alternation is broken.
static REFUSED_ORDER: AtomicU64 = AtomicU64::new(0);
/// Neither the literal `lockTaken` nor an inlined cleanup is present.
static REFUSED_NO_EVIDENCE: AtomicU64 = AtomicU64::new(0);
/// A label in the region is targeted from outside it.
static REFUSED_LABEL: AtomicU64 = AtomicU64::new(0);
/// The inlined-finally discipline does not hold.
static REFUSED_DISCIPLINE: AtomicU64 = AtomicU64::new(0);

/// Why the inline-`lock` fold refused one site.
#[derive(Clone, Copy)]
enum Refusal {
    NoPair,
    Order,
    NoEvidence,
    LabelMovesIn,
    Discipline,
}

impl Refusal {
    fn count(self) {
        let c = match self {
            Refusal::NoPair => &REFUSED_NO_PAIR,
            Refusal::Order => &REFUSED_ORDER,
            Refusal::NoEvidence => &REFUSED_NO_EVIDENCE,
            Refusal::LabelMovesIn => &REFUSED_LABEL,
            Refusal::Discipline => &REFUSED_DISCIPLINE,
        };
        c.fetch_add(1, Ordering::Relaxed);
    }
}

/// One line per non-empty counter, for the run report.
///
/// The refusal breakdown is the whole point: a gate that names no site is a
/// dead pass, and a gate that names every site is a bug.
pub fn report_refusals() -> String {
    let sites = ENTER_SITES.load(Ordering::Relaxed);
    let folded = FOLDED.load(Ordering::Relaxed);
    if sites == 0 {
        return String::new();
    }
    format!(
        "[reconstruct]   using_lock: {sites} Monitor.Enter site(s) examined, {folded} folded; \
         refused — no pairing Exit {}, event order {}, no lock-codegen evidence {}, \
         label reachable from outside {}, inlined-finally discipline {}\n",
        REFUSED_NO_PAIR.load(Ordering::Relaxed),
        REFUSED_ORDER.load(Ordering::Relaxed),
        REFUSED_NO_EVIDENCE.load(Ordering::Relaxed),
        REFUSED_LABEL.load(Ordering::Relaxed),
        REFUSED_DISCIPLINE.load(Ordering::Relaxed),
    )
}

/// Classify the FIRST refusable `Monitor.Enter` site in a list that folded
/// nothing, so its reason reaches the report. Called once per fully-refused
/// list (the fold loop returns immediately afterwards), which keeps the
/// counters a population rather than a retry count.
fn record_first_refusal(list: &[Stmt], global_gotos: &HashSet<String>) {
    for i in 0..list.len() {
        if monitor_stmt(&list[i], "Enter").is_none() {
            continue;
        }
        if let Err(r) = plan_inline_lock_at(list, i, global_gotos) {
            r.count();
        }
        return;
    }
}

/// Run the recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    // Goto targets for the whole body: needed to tell a label a `goto` from
    // OUTSIDE the region reaches (which blocks the fold) from one that only
    // the region itself reaches (which moves with it).
    let mut global_gotos: HashSet<String> = HashSet::new();
    super::collect_goto_targets(stmts, &mut global_gotos);
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += rewrite_list(list, &global_gotos);
    });
    n
}

fn rewrite_list(list: &mut Vec<Stmt>, global_gotos: &HashSet<String>) -> usize {
    // Count this list's `Monitor.Enter` sites ONCE. The fold loop below rescans
    // the same list after every rewrite, so counting inside it would multiply
    // the population by the number of rewrites instead of measuring it.
    for s in list.iter() {
        if monitor_stmt(s, "Enter").is_some() {
            ENTER_SITES.fetch_add(1, Ordering::Relaxed);
        }
    }
    let mut n = 0usize;
    loop {
        if let Some(fold) = plan_inline_lock(list, global_gotos) {
            apply_inline_lock(list, fold);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(fold) = plan_try_lock(list) {
            splice_pair(list, fold.at, fold.replacement);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        if let Some(fold) = plan_using(list) {
            splice_pair(list, fold.at, fold.replacement);
            n += 1;
            FOLDED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        // Nothing left to fold. Record WHY the remaining sites were left alone
        // so the report can never show a bare zero for this pass again.
        record_first_refusal(list, global_gotos);
        return n;
    }
}

// ---------------------------------------------------------------------------
// lock, from the flat inlined-finally shape
// ---------------------------------------------------------------------------

/// A planned `Monitor.Enter`/`Monitor.Exit` fold.
struct InlineLock {
    enter: usize,
    exit: usize,
    target: Expr,
    body: Vec<Stmt>,
}

fn apply_inline_lock(list: &mut Vec<Stmt>, fold: InlineLock) {
    let lock = Stmt::Lock { expr: fold.target, body: fold.body };
    let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
    for (idx, s) in std::mem::take(list).into_iter().enumerate() {
        if idx == fold.enter {
            out.push(lock.clone());
            continue;
        }
        if idx > fold.enter && idx <= fold.exit {
            continue; // consumed by the `lock`
        }
        out.push(s);
    }
    *list = out;
}

fn plan_inline_lock(list: &[Stmt], global_gotos: &HashSet<String>) -> Option<InlineLock> {
    (0..list.len()).find_map(|i| plan_inline_lock_at(list, i, global_gotos).ok())
}

/// Plan the fold at `i`, or report the reason it cannot be done.
fn plan_inline_lock_at(
    list: &[Stmt],
    i: usize,
    global_gotos: &HashSet<String>,
) -> Result<InlineLock, Refusal> {
    let enter = monitor_stmt(&list[i], "Enter").ok_or(Refusal::NoPair)?;
    let target = enter.arg.clone();

    // Every same-object monitor event at THIS level, in order. They must
    // alternate `Enter, Exit, Enter, …` starting at `i`: that is what makes the
    // pairing unambiguous (two locks on the same object in one list pair one
    // each, and a stray second `Exit` refuses instead of being merged).
    let mut events: Vec<usize> = Vec::new();
    for (k, s) in list.iter().enumerate() {
        if monitor_stmt_on(s, "Enter", &target) || monitor_stmt_on(s, "Exit", &target) {
            events.push(k);
        }
    }
    if events.first() != Some(&i) {
        return Err(Refusal::Order);
    }
    for (n, k) in events.iter().enumerate() {
        let is_enter = monitor_stmt_on(&list[*k], "Enter", &target);
        if is_enter != (n % 2 == 0) {
            return Err(Refusal::Order);
        }
    }
    // No second same-target event in THIS list: the `Exit` is in another region
    // (a different statement list), which this single-list rewrite cannot pair.
    let exit = *events.get(1).ok_or(Refusal::NoPair)?;

    let region = &list[i + 1..exit];

    // Positivity: either the `lock` codegen's literal `lockTaken`, or the
    // inlined-finally discipline inside the region, must be present. A bare
    // one-argument `Enter` with a perfectly flat body proves neither, so it is
    // refused rather than guessed.
    let mut region_labels: HashSet<String> = HashSet::new();
    super::collect_labels(region, &mut region_labels);
    let mut region_gotos: HashSet<String> = HashSet::new();
    super::collect_goto_targets(region, &mut region_gotos);

    let has_inlined_exit = region_has_inlined_exit(region, &target);
    if !enter.lit_lock_taken && !has_inlined_exit {
        return Err(Refusal::NoEvidence);
    }

    // A label reached by a `goto` from outside the region cannot move into a
    // `lock` body.
    if region_labels
        .iter()
        .any(|l| global_gotos.contains(l) && !region_gotos.contains(l))
    {
        return Err(Refusal::LabelMovesIn);
    }

    // Vet and strip the inlined cleanups.
    let mut body = region.to_vec();
    if !strip_inlined_exits(&mut body, &target, 0, &region_labels) {
        return Err(Refusal::Discipline);
    }

    Ok(InlineLock { enter: i, exit, target, body })
}

/// Does the region contain the positive evidence of an inlined `finally` — an
/// `Exit(target)` immediately followed by a statement that leaves the region?
fn region_has_inlined_exit(list: &[Stmt], target: &Expr) -> bool {
    for (k, s) in list.iter().enumerate() {
        if monitor_stmt_on(s, "Exit", target) && k + 1 < list.len() && leaves_region(&list[k + 1]) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for kid in kids {
            if region_has_inlined_exit(kid, target) {
                return true;
            }
        }
    }
    false
}

/// Vet the region and remove every inlined `Monitor.Exit(target)` cleanup.
///
/// Returns `false` when the region does not obey the discipline, in which case
/// the caller leaves the flat form untouched:
///   * an `Exit(target)` NOT immediately followed by a statement that leaves the
///     region is not an inlined cleanup;
///   * an `Enter(target)` is a re-entrant pair;
///   * a leaving statement NOT preceded by a removed `Exit(target)` would gain a
///     release from the fold that it does not have today.
fn strip_inlined_exits(
    list: &mut Vec<Stmt>,
    target: &Expr,
    depth: usize,
    region_labels: &HashSet<String>,
) -> bool {
    let mut remove = vec![false; list.len()];
    for (k, s) in list.iter().enumerate() {
        if monitor_stmt_on(s, "Enter", target) {
            return false;
        }
        if monitor_stmt_on(s, "Exit", target) {
            let followed_by_exit = list.get(k + 1).is_some_and(leaves_region);
            if !followed_by_exit {
                return false;
            }
            remove[k] = true;
        }
        // A statement that leaves the region must have its cleanup right in
        // front of it. A `break`/`continue` only leaves at loop depth 0; a
        // `goto` to a label inside the region stays inside it.
        let leaves = match s {
            Stmt::Return(_) | Stmt::Throw(_) => true,
            Stmt::Goto(l) => !region_labels.contains(l),
            Stmt::Break | Stmt::Continue => depth == 0,
            _ => false,
        };
        if leaves {
            let preceded = k > 0 && monitor_stmt_on(&list[k - 1], "Exit", target);
            if !preceded {
                return false;
            }
        }
    }
    if remove.iter().any(|r| *r) {
        let mut idx = 0usize;
        list.retain(|_| {
            let keep = !remove[idx];
            idx += 1;
            keep
        });
    }
    // Recurse into the surviving nested bodies, with the loop depth each of
    // them starts at.
    for s in list.iter_mut() {
        let child_depth = depth + loop_delta(s);
        for kid in children_mut(s) {
            if !strip_inlined_exits(kid, target, child_depth, region_labels) {
                return false;
            }
        }
    }
    true
}

/// How much a statement deepens the loop nesting for its own body.
fn loop_delta(s: &Stmt) -> usize {
    match s {
        Stmt::While { .. }
        | Stmt::DoWhile { .. }
        | Stmt::For { .. }
        | Stmt::ForEach { .. } => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// lock, from a recovered `try/finally` (unreachable on the x64 corpus)
// ---------------------------------------------------------------------------

struct PairFold {
    at: usize,
    replacement: Stmt,
}

/// `Monitor.Enter(x)` then `try { … } finally { Monitor.Exit(x) }` → `lock (x)`.
fn plan_try_lock(list: &[Stmt]) -> Option<PairFold> {
    for i in 0..list.len().saturating_sub(1) {
        let Some(enter) = monitor_stmt(&list[i], "Enter") else {
            continue;
        };
        let target = enter.arg;
        let Stmt::Try { try_body, catches, finally_body } = &list[i + 1] else {
            continue;
        };
        // A lock is a bare try/finally; any catch means this is a real try.
        if !catches.is_empty() {
            continue;
        }
        let Some(fin) = finally_body else { continue };
        let Some(exited) = single_monitor_arg(fin, "Exit") else {
            continue;
        };
        if !exprs_eq(&exited, target) {
            continue;
        }
        return Some(PairFold {
            at: i,
            replacement: Stmt::Lock {
                expr: target.clone(),
                body: try_body.clone(),
            },
        });
    }
    None
}

/// The argument of the one meaningful statement of a `finally` body when that
/// statement is `Monitor.Exit(x)` (comments/labels carry no semantics; a second
/// meaningful statement refuses the match).
fn single_monitor_arg(fin: &[Stmt], member: &str) -> Option<Expr> {
    let only = single_meaningful(fin)?;
    monitor_stmt(only, member).map(|c| c.arg.clone())
}

// ---------------------------------------------------------------------------
// using, from a recovered `try/finally` (unreachable on the x64 corpus)
// ---------------------------------------------------------------------------

/// `V = <acquire>` then `try { … } finally { V.Dispose(); }` →
/// `using (V = acquire)`.
fn plan_using(list: &[Stmt]) -> Option<PairFold> {
    for i in 0..list.len().saturating_sub(1) {
        let Stmt::Assign(var, ty, acquire) = &list[i] else {
            continue;
        };
        if !is_disposable_acquire(acquire) {
            continue;
        }
        let Stmt::Try { try_body, catches, finally_body } = &list[i + 1] else {
            continue;
        };
        if !catches.is_empty() {
            continue;
        }
        let Some(fin) = finally_body else { continue };
        let Some(disposed) = single_dispose_receiver(fin) else {
            continue;
        };
        if disposed != *var {
            continue;
        }
        // The declaration must be renderable in a header (`using (<decl>)`).
        if !is_inlinable(&list[i]) {
            continue;
        }
        // `var` must not be read after the try: a live-out resource is an
        // ordinary local, not a scoped `using`.
        if list[i + 2..].iter().any(|s| stmt_reads_deep(s, var)) {
            continue;
        }
        return Some(PairFold {
            at: i,
            replacement: Stmt::Using {
                decl: Box::new(Stmt::Assign(var.clone(), ty.clone(), acquire.clone())),
                body: try_body.clone(),
            },
        });
    }
    None
}

/// Does `e` look like an acquire that yields an object with a `Dispose`?
/// A call, an allocation or a raw call (optionally behind a cast) qualifies;
/// a plain variable/literal does not.
fn is_disposable_acquire(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } | Expr::NewObj(..) | Expr::RawCall(..) => true,
        Expr::Cast(_, inner) => is_disposable_acquire(inner),
        _ => false,
    }
}

/// The receiver name of `V.Dispose()` when the `finally` body's only meaningful
/// statement is exactly that call with no arguments.
fn single_dispose_receiver(fin: &[Stmt]) -> Option<String> {
    let only = single_meaningful(fin)?;
    let Stmt::Eval(Expr::Call { recv, name, args }) = only else {
        return None;
    };
    if !args.is_empty() {
        return None;
    }
    if !(name == "Dispose" || name.ends_with(".Dispose")) {
        return None;
    }
    match recv.as_deref() {
        Some(Expr::Var(r)) => Some(r.clone()),
        // `V.Dispose()` may also be spelled with the receiver folded into the
        // name; only accept it when there is no separate receiver.
        None => name.strip_suffix(".Dispose").map(str::to_string),
        _ => None,
    }
}

/// The single meaningful (non-comment, non-label) statement of `body`, or `None`
/// when there are zero or more than one.
fn single_meaningful(body: &[Stmt]) -> Option<&Stmt> {
    let mut it = body
        .iter()
        .filter(|s| !matches!(s, Stmt::Comment(_) | Stmt::Label(_)));
    let only = it.next()?;
    if it.next().is_some() {
        return None;
    }
    Some(only)
}

/// Replace the two statements at `at`/`at + 1` with one.
fn splice_pair(list: &mut Vec<Stmt>, at: usize, replacement: Stmt) {
    let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
    for (idx, s) in std::mem::take(list).into_iter().enumerate() {
        if idx == at {
            out.push(replacement.clone());
            continue;
        }
        if idx == at + 1 {
            continue;
        }
        out.push(s);
    }
    *list = out;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::CmpKind;

    fn v(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }
    /// `Monitor.<member>(arg)` / `Monitor.<member>(arg, 0)` as a statement.
    fn monitor(member: &str, arg: Expr, lock_taken: bool) -> Stmt {
        let mut args = vec![arg];
        if lock_taken {
            args.push(Expr::Int(0));
        }
        Stmt::Eval(Expr::Call {
            recv: Some(Box::new(v("Monitor"))),
            name: member.into(),
            args,
        })
    }
    /// `*(nint*)(v28 +152)` — the real, non-trivial target expression.
    fn deref() -> Expr {
        Expr::RawDeref(Box::new(v("v28")), 152, 0)
    }

    /// The refusal REASONS are distinguishable, which is what makes the run
    /// report actionable: an `Exit` that lives in another statement list is a
    /// different defect from a broken inlined-finally discipline, and they need
    /// different fixes.
    #[test]
    fn refusal_reasons_are_distinguished() {
        // No second same-object event in this list: the `Exit` is elsewhere.
        let list = vec![monitor("Enter", v("gate"), true), Stmt::Eval(v("body"))];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::NoPair)
        ));

        // A one-argument Enter with no literal `lockTaken` and no inlined
        // cleanup proves nothing about the source.
        let list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::NoEvidence)
        ));

        // An early exit with no cleanup in front of it means the fold would ADD
        // a release on that path.
        let list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Return(None)],
                else_body: vec![],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert!(matches!(
            plan_inline_lock_at(&list, 0, &HashSet::new()),
            Err(Refusal::Discipline)
        ));
    }

    /// The verbatim real shape folds, and the Enter/Exit pair disappears.
    #[test]
    fn the_real_inline_monitor_pair_becomes_a_lock() {
        let mut list = vec![
            monitor("Enter", deref(), true),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "FillCJKCore".into(),
                args: vec![v("culture")],
            }),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "SetCJKReferences".into(),
                args: vec![v("culture")],
            }),
            monitor("Exit", deref(), false),
            Stmt::Return(None),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1, "the lock must be recognized");
        assert_eq!(list.len(), 2, "Enter+body+Exit collapse into one statement + the return");
        match &list[0] {
            Stmt::Lock { expr, body } => {
                assert!(matches!(expr, Expr::RawDeref(..)), "the real target expression");
                assert_eq!(body.len(), 2, "only the protected body is inside");
            }
            other => panic!("expected Stmt::Lock, got {other:?}"),
        }
        assert!(matches!(&list[1], Stmt::Return(None)), "the return stays outside the lock");
    }

    /// Two sequential locks on the SAME object must pair correctly rather than
    /// have the first swallow the second.
    #[test]
    fn two_locks_on_the_same_object_pair_one_each() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("first")),
            monitor("Exit", v("gate"), false),
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("second")),
            monitor("Exit", v("gate"), false),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 2);
        assert_eq!(list.len(), 2);
        assert!(matches!(&list[0], Stmt::Lock { .. }));
        assert!(matches!(&list[1], Stmt::Lock { .. }));
    }

    /// An inlined cleanup in front of an early `return` is removed and the exit
    /// stays outside the lock.
    #[test]
    fn an_inlined_exit_before_an_early_return_is_removed() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![
                    monitor("Exit", v("gate"), false),
                    Stmt::Return(Some(Expr::Int(1))),
                ],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Return(Some(Expr::Int(2))),
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        match &list[0] {
            Stmt::Lock { body, .. } => {
                let Stmt::If { then_body, .. } = &body[0] else {
                    panic!("expected the guarded early exit inside the lock");
                };
                assert!(
                    !then_body.iter().any(|s| matches!(s, Stmt::Eval(_))),
                    "the inlined Exit was removed"
                );
                assert!(matches!(then_body.last(), Some(Stmt::Return(_))));
            }
            other => panic!("expected a lock, got {other:?}"),
        }
    }

    /// A leaving statement with NO inlined cleanup in front of it would GAIN a
    /// release from the fold — refuse.
    #[test]
    fn an_exit_without_a_cleanup_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![Stmt::Return(Some(Expr::Int(1)))],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0, "the early return has no cleanup in the flat code");
        assert_eq!(list.len(), 4);
    }

    /// A bare `Exit` that is not an inlined cleanup (nothing leaving follows it)
    /// is not the region's close.
    #[test]
    fn a_bare_exit_mid_region_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Eval(v("more")),
            monitor("Exit", v("gate"), false),
        ];
        // The first Exit is followed by a statement that does NOT leave the
        // region, so the discipline does not hold.
        assert_eq!(apply(&mut list), 0);
    }

    /// Exit on a DIFFERENT object is a different monitor — never guessed.
    #[test]
    fn exit_on_a_different_object_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Exit", v("other"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A re-entrant pair on the same object is not one `lock`.
    #[test]
    fn a_nested_same_object_enter_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::Eval(v("body")),
            monitor("Enter", v("gate"), true),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A one-argument `Enter` with no inlined cleanup proves nothing about the
    /// source: a hand-written `Monitor.Enter(x)` looks identical.
    #[test]
    fn a_bare_enter_without_cleanup_evidence_is_refused() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0, "no literal lockTaken and no inlined cleanup");
    }

    /// A one-argument `Enter` WITH an inlined cleanup is proof enough.
    #[test]
    fn a_bare_enter_with_cleanup_evidence_folds() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![monitor("Exit", v("gate"), false), Stmt::Return(None)],
                else_body: vec![],
            },
            Stmt::Eval(v("body")),
            monitor("Exit", v("gate"), false),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A `break` inside a NESTED loop does not leave the region.
    #[test]
    fn a_break_in_a_nested_loop_needs_no_cleanup() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::While {
                cond: Expr::Cmp(CmpKind::Lt, Box::new(v("i")), Box::new(Expr::Int(4))),
                body: vec![Stmt::Break],
            },
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 1);
        match &list[0] {
            Stmt::Lock { body, .. } => assert!(matches!(body[0], Stmt::While { .. })),
            other => panic!("expected a lock, got {other:?}"),
        }
    }

    /// A `goto` out of the region with the inlined cleanup in front of it is
    /// fine; and the label it targets, being outside, is not moved.
    #[test]
    fn a_goto_out_of_the_region_with_a_cleanup_is_fine() {
        let mut list = vec![
            monitor("Enter", v("gate"), true),
            Stmt::If {
                cond: Expr::Bool(true),
                then_body: vec![monitor("Exit", v("gate"), false), Stmt::Goto("L".into())],
                else_body: vec![],
            },
            monitor("Exit", v("gate"), false),
            Stmt::Label("L".into()),
            Stmt::Return(None),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A label INSIDE the region targeted by a `goto` from OUTSIDE cannot move
    /// into a `lock` body.
    #[test]
    fn a_label_targeted_from_outside_blocks_the_fold() {
        let mut list = vec![
            Stmt::Goto("inner".into()),
            monitor("Enter", v("gate"), true),
            Stmt::Label("inner".into()),
            monitor("Exit", v("gate"), false),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// The `try/finally` spelling still works where it occurs.
    #[test]
    fn the_try_finally_spelling_is_still_accepted() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Try {
                try_body: vec![Stmt::Comment("body".into())],
                catches: vec![],
                finally_body: Some(vec![monitor("Exit", v("gate"), false)]),
            },
        ];
        assert_eq!(apply(&mut list), 1);
        assert!(matches!(&list[0], Stmt::Lock { .. }));
    }

    /// A catch clause means this is a real `try`, not a `lock`.
    #[test]
    fn a_catch_clause_refuses_the_try_spelling() {
        let mut list = vec![
            monitor("Enter", v("gate"), false),
            Stmt::Try {
                try_body: vec![],
                catches: vec![("Exception".into(), vec![])],
                finally_body: Some(vec![monitor("Exit", v("gate"), false)]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    #[test]
    fn acquire_then_dispose_is_a_using() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![Stmt::Comment("body".into())],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        let n = apply(&mut list);
        assert_eq!(n, 1);
        assert_eq!(list.len(), 1);
        match &list[0] {
            Stmt::Using { decl, body } => {
                assert!(matches!(decl.as_ref(), Stmt::Assign(n, None, _) if n == "tmp"));
                assert_eq!(body.len(), 1);
            }
            other => panic!("expected Stmt::Using, got {other:?}"),
        }
    }

    #[test]
    fn using_refused_when_the_resource_is_used_after_the_try() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(v("tmp"))),
                name: "Read".into(),
                args: vec![],
            }),
        ];
        assert_eq!(apply(&mut list), 0, "a live-out resource is not a `using`");
        assert_eq!(list.len(), 3);
    }

    #[test]
    fn using_refused_when_dispose_is_not_on_the_acquired_var() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, Expr::NewObj("Reader".into(), vec![])),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("other"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        assert_eq!(apply(&mut list), 0);
    }

    #[test]
    fn a_non_acquire_assignment_is_not_a_using() {
        let mut list = vec![
            Stmt::Assign("tmp".into(), None, v("existing")),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Eval(Expr::Call {
                    recv: Some(Box::new(v("tmp"))),
                    name: "Dispose".into(),
                    args: vec![],
                })]),
            },
        ];
        assert_eq!(apply(&mut list), 0, "a plain copy is not an acquire");
    }
}
