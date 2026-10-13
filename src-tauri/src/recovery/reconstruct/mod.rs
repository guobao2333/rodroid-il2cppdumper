//! High-level C# CONSTRUCT reconstruction over the FINISHED statement tree.
//!
//! ## Why this is a sibling of `idioms`, not part of it
//!
//! `idioms` removes machine noise (dead gotos, unreferenced labels, static-init
//! guards). `reconstruct` does the opposite job: it rebuilds the *source-level
//! construct* the structurer could only emit in its flat machine form — a
//! `for` loop that arrived as `init; while (cond) { …; update; }`, a `lock`
//! that arrived as a bare `Monitor.Enter(x, 0)` / `Monitor.Exit(x)` pair, an
//! iterator that arrived as a `<>1__state` dispatch, a lambda that arrived as a
//! delegate construction. Those shapes are exactly what separates "recovered
//! machine code" from "readable C#".
//!
//! ## The two rules (same as `idioms`, and they are not negotiable)
//!
//! 1. **Never delete an observation.** A recognizer that removes a statement
//!    must prove the statement evaluates nothing and gates nothing.
//! 2. **Never move a write.** Hoisting/reordering is allowed only when nothing
//!    between the old and new position can observe the difference (no call, no
//!    construction, no write to a name the moved expression reads).
//!
//! A failed match is a NO-OP: the flat (correct, lower-level) sequence stays.
//! Accurate-but-ugly always beats plausible-but-wrong.
//!
//! ## Measured reality (x64 PE `Simple_6000_6_0_b1/GameAssembly.dll`, 47,924
//! methods, 2,454,023-line `dump.cs`) — this is what the shapes ACTUALLY look
//! like, and every rule below is written against it:
//!
//! | construct | in the real dump | real shape |
//! |---|---|---|
//! | `for (` | **0** | `init; …; while (cond) { …; update; …; continue; }` |
//! | `lock (` | **0** | `Monitor.Enter(X, 0);` … `Monitor.Exit(X);` INLINE — there is no `try/finally` (145 `Monitor.Enter` sites) |
//! | `using (` | **0** | `try` 18, `finally` **0** — the EH structurer emits no `finally` on this corpus, so the trait shape is unreachable here |
//! | `switch (` | **1** | the structurer emits goto-shaped case bodies, so a `Switch`-keyed recognizer cannot fire |
//! | `new T(` | **0** | an allocation renders as `T..ctor(args)` / `recv..ctor(args)` — an `Expr::Call` whose name ends in `.ctor`, never `Expr::NewObj` |
//!
//! The previous round of recognizers was written against *invented* shapes
//! (one-argument `Monitor.Enter` inside a `try/finally`, a `Switch` on
//! `<>1__state`, an adjacent counter init, `Expr::NewObj` for delegates) and
//! therefore fired **zero times in 47,924 methods** while `idioms` fired
//! hundreds of thousands of times. Each file below now documents, next to its
//! rules, which measured shape it exists for and which part of it provably
//! cannot fire here.

use super::csharp::{Expr, Stmt};
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod array_init;
pub mod base_call;
pub mod closures;
pub mod delegate_ctor;
pub mod for_loop;
pub mod nullability;
pub mod state_machine;
pub mod using_lock;

/// A recognizer: rewrites one statement list, returning the number of rewrites
/// it applied (so a run report can show which recognizer actually fires).
pub type Pass = fn(&mut Vec<Stmt>) -> usize;

/// Ordering is deliberate. Peephole repairs (`nullability`) run first so the
/// structural recognizers see the cleanest conditions; the structural
/// recognizers run before `for_loop`, which consumes the `while` shapes
/// `using_lock`/`closures`/`state_machine` may leave behind.
const PASSES: &[(&str, Pass)] = &[
    ("nullability", nullability::apply),
    ("array_init", array_init::apply),
    ("base_call", base_call::apply),
    ("delegate_ctor", delegate_ctor::apply),
    ("using_lock", using_lock::apply),
    ("closures", closures::apply),
    ("state_machine", state_machine::apply),
    ("for_loop", for_loop::apply),
];

/// Run every reconstruct recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) {
    let verbose = std::env::var("ILR_RECON").is_ok();
    for (name, pass) in PASSES {
        let n = pass(stmts);
        if n > 0 {
            RECON_TOTAL.fetch_add(n as u64, Ordering::Relaxed);
            for (pname, c) in BY_PASS.iter() {
                if pname == name {
                    c.fetch_add(n as u64, Ordering::Relaxed);
                }
            }
            if verbose {
                eprintln!("[recon] {name}: {n} rewrite(s)");
            }
        }
    }
}

/// Per-recognizer rewrite counts, reported with the structural census so a
/// recognizer that silently never fires is visible rather than assumed.
static BY_PASS: [(&str, AtomicU64); 8] = [
    ("nullability", AtomicU64::new(0)),
    ("array_init", AtomicU64::new(0)),
    ("base_call", AtomicU64::new(0)),
    ("delegate_ctor", AtomicU64::new(0)),
    ("using_lock", AtomicU64::new(0)),
    ("closures", AtomicU64::new(0)),
    ("state_machine", AtomicU64::new(0)),
    ("for_loop", AtomicU64::new(0)),
];
static RECON_TOTAL: AtomicU64 = AtomicU64::new(0);

/// The run summary line, or an empty string when nothing fired.
pub fn report() -> String {
    let total = RECON_TOTAL.load(Ordering::Relaxed);
    // `total == 0` no longer means "say nothing": a pass can now have a
    // non-empty explanation (sites examined, and why each was refused) even
    // when nothing fired, and that explanation is exactly what a silent zero
    // was hiding.
    let refusals = using_lock::report_refusals();
    if total == 0 && refusals.is_empty() {
        return String::new();
    }
    let mut s = format!("[reconstruct] {total} construct rewrite(s)\n");
    for (name, c) in BY_PASS.iter() {
        let v = c.load(Ordering::Relaxed);
        if v > 0 {
            s.push_str(&format!("[reconstruct]   {name}: {v}\n"));
        }
    }
    // A recognizer with a ZERO count is the interesting case: it is either
    // provably dead or refusing everything. Those need opposite responses, and
    // a bare zero cannot tell them apart, so the passes that can explain
    // themselves print their site population and refusal reasons here.
    s.push_str(&refusals);
    s
}

// ---------------------------------------------------------------------------
// Shared AST helpers (the recognizers must not each reinvent these)
// ---------------------------------------------------------------------------

/// Post-order walk: recurse into every nested statement list, then hand the
/// list to `f`. Recognizers call this so a construct nested inside an `if`,
/// a loop or a `try` is reached.
pub fn walk_mut<F: FnMut(&mut Vec<Stmt>)>(stmts: &mut Vec<Stmt>, f: &mut F) {
    for s in stmts.iter_mut() {
        for c in children_mut(s) {
            walk_mut(c, f);
        }
    }
    f(stmts);
}

/// Every nested statement list a statement owns, mutably.
pub fn children_mut(s: &mut Stmt) -> Vec<&mut Vec<Stmt>> {
    match s {
        Stmt::If { then_body, else_body, .. } => vec![then_body, else_body],
        Stmt::While { body, .. } => vec![body],
        Stmt::DoWhile { body, .. } => vec![body],
        Stmt::For { init, incr, body, .. } => vec![init, incr, body],
        Stmt::Using { body, .. } => vec![body],
        Stmt::Lock { body, .. } => vec![body],
        Stmt::ForEach { body, .. } => vec![body],
        Stmt::Try { try_body, catches, finally_body } => {
            let mut v: Vec<&mut Vec<Stmt>> = vec![try_body];
            for (_, b) in catches.iter_mut() {
                v.push(b);
            }
            if let Some(fb) = finally_body {
                v.push(fb);
            }
            v
        }
        Stmt::Switch { cases, tail, .. } => {
            let mut v: Vec<&mut Vec<Stmt>> = Vec::new();
            for (_, b) in cases.iter_mut() {
                v.push(b);
            }
            v.push(tail);
            v
        }
        _ => Vec::new(),
    }
}

/// Every nested statement list a statement owns, immutably.
pub fn children<'a>(s: &'a Stmt, out: &mut Vec<&'a Vec<Stmt>>) {
    match s {
        Stmt::If { then_body, else_body, .. } => {
            out.push(then_body);
            out.push(else_body);
        }
        Stmt::While { body, .. }
        | Stmt::DoWhile { body, .. }
        | Stmt::Using { body, .. }
        | Stmt::Lock { body, .. }
        | Stmt::ForEach { body, .. } => out.push(body),
        Stmt::For { init, incr, body, .. } => {
            out.push(init);
            out.push(incr);
            out.push(body);
        }
        Stmt::Try { try_body, catches, finally_body } => {
            out.push(try_body);
            for (_, b) in catches {
                out.push(b);
            }
            if let Some(fb) = finally_body {
                out.push(fb);
            }
        }
        Stmt::Switch { cases, tail, .. } => {
            for (_, b) in cases {
                out.push(b);
            }
            out.push(tail);
        }
        _ => {}
    }
}

/// Does evaluating `e` call anything (a call, a raw call, an allocation, an
/// await)? A write-carrying expression — the test for "nothing observable can
/// happen between these two positions".
pub fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call { recv, args, .. } => {
            recv.as_deref().is_some_and(expr_has_call) || args.iter().any(expr_has_call)
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(expr_has_call)
        }
        Expr::Await(x) => expr_has_call(x),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => expr_has_call(x),
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_has_call(r),
        Expr::Cond(c, a, b) => expr_has_call(c) || expr_has_call(a) || expr_has_call(b),
        Expr::Lambda { body, .. } => expr_has_call(body),
        Expr::Var(_)
        | Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::StaticField(..)
        | Expr::MethodGroup(_) => false,
    }
}

/// Does `e` mention the local `name` anywhere?
///
/// Deliberately OVER-approximate: an `Expr::Field(recv, f)` also reports a
/// mention when the field's own name equals `name`. A false mention can only
/// make a recognizer refuse (safe); a missed mention could make it rewrite
/// (unsafe), so the direction of the approximation is chosen on purpose.
pub fn expr_mentions(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Var(n) => n == name,
        Expr::Call { recv, name: cn, args } => {
            cn == name
                || recv.as_deref().is_some_and(|r| expr_mentions(r, name))
                || args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            args.iter().any(|a| expr_mentions(a, name))
        }
        Expr::Await(x) => expr_mentions(x, name),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_mentions(a, name) || expr_mentions(b, name),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) => {
            expr_mentions(x, name)
        }
        Expr::Field(r, f) => f == name || expr_mentions(r, name),
        Expr::StaticField(_, f) => f == name,
        Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => expr_mentions(r, name),
        Expr::Cond(c, a, b) => {
            expr_mentions(c, name) || expr_mentions(a, name) || expr_mentions(b, name)
        }
        Expr::Lambda { params, body } => {
            !params.iter().any(|p| p.as_str() == name) && expr_mentions(body, name)
        }
        Expr::This
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Double(_)
        | Expr::Bool(_)
        | Expr::Str(_)
        | Expr::Null
        | Expr::Unknown
        | Expr::MissingAbiArg
        | Expr::MethodGroup(_) => false,
    }
}

/// Does this statement read `name`? Shallow: the statement's own expressions,
/// not the nested bodies. Used to prove a value is live-out.
pub fn stmt_reads(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions(e, name),
        Stmt::Assign(_, _, e) => expr_mentions(e, name),
        Stmt::If { cond, .. } => expr_mentions(cond, name),
        Stmt::While { cond, .. } => expr_mentions(cond, name),
        Stmt::DoWhile { cond, .. } => expr_mentions(cond, name),
        Stmt::For { cond, .. } => cond.as_ref().is_some_and(|c| expr_mentions(c, name)),
        Stmt::Lock { expr, .. } => expr_mentions(expr, name),
        Stmt::Using { decl, .. } => stmt_reads(decl, name),
        Stmt::Return(Some(e)) => expr_mentions(e, name),
        _ => false,
    }
}

/// Does this statement write `name` (an assignment target, a `for` header, or a
/// `using` declaration)?
pub fn stmt_writes(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Assign(n, _, _) => n == name,
        _ => false,
    }
}

/// Does this statement read `name`, ANYWHERE in its nested bodies?
pub fn stmt_reads_deep(s: &Stmt, name: &str) -> bool {
    if stmt_reads(s, name) {
        return true;
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| list_reads_deep(k, name))
}

/// Does any statement in `list` read `name`, at any depth?
pub fn list_reads_deep(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_reads_deep(s, name))
}

/// Does this statement read OR write `name`, anywhere in its nested bodies?
///
/// This is the conservative liveness test: an assignment to `name` counts as a
/// mention (the name is live there), a read counts, both at any depth.
pub fn stmt_mentions_deep(s: &Stmt, name: &str) -> bool {
    if stmt_writes(s, name) || stmt_reads(s, name) {
        return true;
    }
    // `Stmt::Switch`'s discriminant is a bare name string, not an expression —
    // a switch on the local being tested is a mention.
    if let Stmt::Switch { disc, .. } = s {
        if disc == name {
            return true;
        }
    }
    // `Stmt::ForEach` names its element local as a string.
    if let Stmt::ForEach { var_name, collection, .. } = s {
        if var_name == name || collection == name {
            return true;
        }
    }
    // `Stmt::For`/`Using` carry statement lists that may assign the name.
    if let Stmt::For { init, incr, .. } = s {
        if list_mentions_deep(init, name) || list_mentions_deep(incr, name) {
            return true;
        }
    }
    if let Stmt::Using { decl, .. } = s {
        if stmt_mentions_deep(decl, name) {
            return true;
        }
    }
    let mut kids: Vec<&Vec<Stmt>> = Vec::new();
    children(s, &mut kids);
    kids.iter().any(|k| list_mentions_deep(k, name))
}

/// Does any statement in `list` read or write `name`, at any depth?
pub fn list_mentions_deep(list: &[Stmt], name: &str) -> bool {
    list.iter().any(|s| stmt_mentions_deep(s, name))
}

/// The `(receiver, name, args)` of a call expression, or `None`.
pub fn split_call(e: &Expr) -> Option<(Option<&Expr>, &str, &[Expr])> {
    match e {
        Expr::Call { recv, name, args } => Some((recv.as_deref(), name.as_str(), args.as_slice())),
        _ => None,
    }
}

/// Is `name` a constructor's name? The renderer spells a construction
/// `Type..ctor(args)` (name `"Type..ctor"`) or, when the declaration is
/// unresolved, `recv..ctor(args)` (name `".ctor"`). Measured on the x64 corpus:
/// **`new T(` appears ZERO times** — every allocation in the real output is a
/// `.ctor` CALL, so `Expr::NewObj` is not the shape to key on.
pub fn is_ctor_name(name: &str) -> bool {
    name == ".ctor" || name.ends_with(".ctor") || name.ends_with("..ctor")
}

/// The type name a construction names, when it is spelled on the callee
/// (`"System.Action..ctor"` → `Some("System.Action")`). `recv..ctor` carries no
/// type here and yields `None` — the receiver's type is not in the tree.
pub fn ctor_type_name(name: &str) -> Option<&str> {
    let t = name.strip_suffix("..ctor").or_else(|| name.strip_suffix(".ctor"))?;
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

/// Structural expression equality, via the renderer (which is total over every
/// `Expr` variant, so this cannot silently accept an unhandled shape).
pub fn exprs_eq(a: &Expr, b: &Expr) -> bool {
    let mut sa = String::new();
    let mut sb = String::new();
    a.render(&mut sa);
    b.render(&mut sb);
    sa == sb
}

/// The rendered text of an expression — the key used wherever a name must be
/// compared textually (`Stmt::Assign` targets and `Stmt::Switch` discriminants
/// are plain strings, not expressions).
pub fn render_text(e: &Expr) -> String {
    let mut s = String::new();
    e.render(&mut s);
    s
}

/// Can this statement be rendered in a `for` header / `using` declaration?
///
/// **This is a data-loss gate, not a style choice.** `Renderer::stmt` for
/// `Stmt::For` builds the header with `filter_map(stmt_inline)`, so ANY init or
/// increment element without an inline form is silently DROPPED from the
/// emitted text (`Stmt::Using` renders `using ()` for the same reason). A
/// recognizer that put, say, a `Stmt::If` into `incr` would delete it. Only
/// `Assign` and `Eval` have an inline form; every recognizer that fills a header
/// must check this first.
pub fn is_inlinable(s: &Stmt) -> bool {
    matches!(s, Stmt::Assign(..) | Stmt::Eval(..))
}

/// Is `s` a `continue` AT THIS LOOP LEVEL? Nested loops own their own
/// `continue`s, so their bodies are not descended into.
pub fn contains_continue(list: &[Stmt]) -> bool {
    list.iter().any(|s| match s {
        Stmt::Continue => true,
        Stmt::If { then_body, else_body, .. } => {
            contains_continue(then_body) || contains_continue(else_body)
        }
        Stmt::Try { try_body, catches, finally_body } => {
            contains_continue(try_body)
                || catches.iter().any(|(_, b)| contains_continue(b))
                || finally_body.as_ref().is_some_and(|f| contains_continue(f))
        }
        Stmt::Switch { cases, tail, .. } => {
            cases.iter().any(|(_, b)| contains_continue(b)) || contains_continue(tail)
        }
        // A nested loop's `continue` belongs to the nested loop.
        Stmt::While { .. }
        | Stmt::DoWhile { .. }
        | Stmt::For { .. }
        | Stmt::ForEach { .. }
        | Stmt::Using { .. }
        | Stmt::Lock { .. } => false,
        _ => false,
    })
}

/// Every label DEFINED anywhere in `list`, at any depth.
pub fn collect_labels(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        if let Stmt::Label(l) = s {
            out.insert(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_labels(k, out);
        }
    }
}

/// Every label TARGETED by a `goto` anywhere in `list`, at any depth.
pub fn collect_goto_targets(list: &[Stmt], out: &mut HashSet<String>) {
    for s in list {
        if let Stmt::Goto(l) = s {
            out.insert(l.clone());
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        children(s, &mut kids);
        for k in kids {
            collect_goto_targets(k, out);
        }
    }
}

/// Is this statement an EXIT from the region it sits in — the statements a
/// compiler must inline a `finally` in front of?
///
/// `return`/`throw` leave the method; `goto`/`break`/`continue` leave the
/// protected region of the statement they sit in (the region is a statement
/// list, so any of them can leave it). This is the positive evidence of an
/// inlined `finally`: measured on the x64 corpus the lock shape is
/// `Monitor.Enter(X, 0); … Monitor.Exit(X); return;` with the `Exit` inlined in
/// front of each early exit, and never a `try/finally`.
pub fn leaves_region(s: &Stmt) -> bool {
    matches!(
        s,
        Stmt::Return(_) | Stmt::Throw(_) | Stmt::Goto(_) | Stmt::Break | Stmt::Continue
    )
}

/// The single argument of a `System.Threading.Monitor.<member>(arg[, 0])`
/// statement's call, or `None`.
///
/// Accepts the two spellings the renderer produces (a dotted name
/// `Monitor.Enter` with no receiver, or the owner as a receiver `Monitor`), and
/// reports whether the second argument — the `lockTaken` out-parameter of the
/// C# `lock` codegen — is present as a literal.
pub fn monitor_call<'a>(e: &'a Expr, member: &'a str) -> Option<MonitorCall<'a>> {
    let (recv, name, args) = split_call(e)?;
    let matches_member = name == member || name.ends_with(&format!(".{member}"));
    if !matches_member {
        return None;
    }
    // The qualifier must say Monitor: a bare `Enter`/`Exit` is far too common a
    // member name to rewrite on its own.
    match name.rsplit_once('.') {
        Some((qual, _)) => {
            if !qual.ends_with("Monitor") {
                return None;
            }
        }
        None => match recv {
            Some(Expr::Var(r)) if r.ends_with("Monitor") => {}
            Some(Expr::StaticField(_, f)) if f.ends_with("Monitor") => {}
            _ => return None,
        },
    }
    let arg = args.first()?;
    // The `lock` codegen's second argument is the `lockTaken` out-parameter
    // initialized to `false` (a literal `0` after recovery). A real
    // `Monitor.Enter(o, ref flag)` carries a live local instead, so a literal
    // here is the positive proof that this is a `lock`.
    let lit_lock_taken = match args.len() {
        1 => false,
        2 => matches!(&args[1], Expr::Int(0) | Expr::Bool(false) | Expr::Null),
        _ => return None,
    };
    Some(MonitorCall { arg, lit_lock_taken, single_arg: args.len() == 1 })
}

/// See [`monitor_call`].
pub struct MonitorCall<'a> {
    /// The object the monitor is entered on / exited from.
    pub arg: &'a Expr,
    /// The second argument was the `lockTaken` literal — the C# `lock` codegen.
    pub lit_lock_taken: bool,
    /// The call had exactly one argument (`Monitor.Enter(x)`).
    pub single_arg: bool,
}

/// Is `s` exactly an `Eval(Monitor.<member>(arg…))` statement?
pub fn monitor_stmt<'a>(s: &'a Stmt, member: &'a str) -> Option<MonitorCall<'a>> {
    match s {
        Stmt::Eval(e) => monitor_call(e, member),
        _ => None,
    }
}

/// `true` when `s` is exactly `Monitor.<member>(...)` on an argument rendering
/// identically to `target`.
pub fn monitor_stmt_on(s: &Stmt, member: &str, target: &Expr) -> bool {
    monitor_stmt(s, member).is_some_and(|c| exprs_eq(c.arg, target))
}

/// Does this statement perform no observable action — no call, no construction,
/// no memory write, no control flow? The weaker half of the "nothing can observe
/// the reordering" test.
pub fn is_pure_local_stmt(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_) => true,
        Stmt::Assign(_, _, e) => !expr_has_call(e),
        _ => false,
    }
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn ctor_names_are_recognized_in_both_spellings() {
        assert!(is_ctor_name(".ctor"));
        assert!(is_ctor_name("System.Action..ctor"));
        assert!(is_ctor_name("List<int>..ctor"));
        assert!(!is_ctor_name("get_Item"));
        assert_eq!(ctor_type_name("System.Action..ctor"), Some("System.Action"));
        assert_eq!(ctor_type_name(".ctor"), None);
    }

    #[test]
    fn monitor_call_needs_a_monitor_qualifier() {
        let qualified = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![Expr::Var("g".into()), Expr::Int(0)],
        };
        let c = monitor_call(&qualified, "Enter").expect("dotted name");
        assert!(c.lit_lock_taken);
        assert!(matches!(c.arg, Expr::Var(n) if n == "g"));

        let bare = Expr::Call {
            recv: None,
            name: "Enter".into(),
            args: vec![Expr::Var("g".into())],
        };
        assert!(monitor_call(&bare, "Enter").is_none(), "bare `Enter` proves nothing");

        let recv_form = Expr::Call {
            recv: Some(Box::new(Expr::Var("Monitor".into()))),
            name: "Exit".into(),
            args: vec![Expr::Var("g".into())],
        };
        let c = monitor_call(&recv_form, "Exit").expect("receiver spelling");
        assert!(c.single_arg);
        assert!(!c.lit_lock_taken);
    }

    #[test]
    fn a_live_lock_taken_flag_is_not_the_lock_codegen() {
        let call = Expr::Call {
            recv: None,
            name: "Monitor.Enter".into(),
            args: vec![Expr::Var("g".into()), Expr::Var("flag".into())],
        };
        let c = monitor_call(&call, "Enter").expect("still a Monitor call");
        assert!(!c.lit_lock_taken, "a live flag means a hand-written Enter");
    }

    #[test]
    fn continue_is_counted_at_this_level_only() {
        let nested = vec![Stmt::While {
            cond: Expr::Bool(true),
            body: vec![Stmt::Continue],
        }];
        assert!(!contains_continue(&nested), "a nested loop owns its own continue");
        let here = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Continue],
            else_body: vec![],
        }];
        assert!(contains_continue(&here));
    }

    #[test]
    fn inline_form_is_only_assign_and_eval() {
        assert!(is_inlinable(&Stmt::Assign("i".into(), None, Expr::Int(0))));
        assert!(is_inlinable(&Stmt::Eval(Expr::Var("f".into()))));
        assert!(!is_inlinable(&Stmt::Comment("x".into())));
        assert!(!is_inlinable(&Stmt::Continue));
    }

    #[test]
    fn goto_targets_are_collected_at_any_depth() {
        let body = vec![Stmt::If {
            cond: Expr::Bool(true),
            then_body: vec![Stmt::Goto("L1".into())],
            else_body: vec![],
        }];
        let mut t = HashSet::new();
        collect_goto_targets(&body, &mut t);
        assert!(t.contains("L1"));
    }
}
