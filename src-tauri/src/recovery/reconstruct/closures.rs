//! Lambda / closure reconstruction.
//!
//! IL2CPP lowers a lambda to a display class (`<>c__DisplayClass…`), captured
//! fields, a generated method and a delegate construction. Where a recognizer
//! can PROVE the shape, it emits the source-level lambda instead:
//! `Expr::Lambda { params, body }`.
//!
//! ## The construction spelling on the real corpus
//!
//! `new T(` appears **zero** times in the real dump: every allocation is a
//! `.ctor` call. A delegate construction is therefore
//!
//! ```text
//! d = System.Action..ctor(this, MethodInfo("<>c.<M>b__1_0"));
//! ```
//!
//! i.e. `Expr::Call { name: "System.Action..ctor", args: [target, method] }` —
//! never `Expr::NewObj`. (The previous version of this file keyed on
//! `Expr::NewObj` and on a `Call` path whose "type" was the whole
//! `"System.Action..ctor"` string, which `is_delegate_type` then rejected as
//! `ctor`; between the two it could not fire at all.) Both spellings are
//! accepted now, and the type is taken from the callee with
//! `ctor_type_name`.
//!
//! ## What this pass can and cannot prove
//!
//! A generated lambda's **body is a different method** (`<>c.<M>b__N_M` is its
//! own definition); the construction site does not contain it. So the only
//! shape that can be folded WITHOUT inventing anything is the one where the
//! body is reachable from the same statement list — a value bound to the
//! generated method's name:
//!
//! ```text
//! <>c.<M>b__1_0 = (x) => x.Value;                     // the body is present
//! d = System.Action`1..ctor(this, MethodInfo("<>c.<M>b__1_0"));
//! ```
//!
//! which folds to `d = (x) => x.Value;`. When the body is not present the
//! construction is left EXACTLY as it was and the pass reports 0 for that site —
//! never a fabricated `() => MethodInfo(…)`. Recovering the cross-method case
//! needs the call graph (the callee's recovered body), which this statement-tree
//! pass deliberately does not have.
//!
//! A display-class instance field (`<>4__this`, captured locals) referenced from
//! inside the body is the closure capture; the field access is kept exactly as
//! written (no capture list is invented) — the value is what matters.
//!
//! ## Measured status: PROVABLY INERT in this pipeline (outcome "b")
//!
//! This pass reported **0** on the x64 corpus, and the reason is a proof, not a
//! missing corpus shape: its input is a `bindings.bodies` entry, which is only
//! ever collected from an `Assign(name, _, Expr::Lambda { … })` **already
//! present in the tree** — and the only code in the entire crate that
//! CONSTRUCTS an `Expr::Lambda` is this file itself (`try_fold` cloning a
//! collected binding) plus its tests. Nothing in the lifter, the structurer or
//! the other passes ever materialises one. So `bodies` is empty on entry, the
//! `bindings.bodies.is_empty()` guard returns 0 before any work, and the pass
//! is a provable no-op on every input this pipeline can produce.
//!
//! The machinery below is kept because it is CORRECT and tested, and because it
//! is exactly what would start firing the moment the missing input exists: a
//! producer of lambda bodies. Two ways to supply it, in increasing cost —
//!   1. a renderer rule that materialises `Expr::Lambda` for a display-class
//!      singleton method whose body is a single expression, which this pass
//!      would then fold at the construction site; or
//!   2. cross-method recovery: resolve `<>c.<M>b__N_M` through the recovered
//!      method table and inline its body here. That needs the call graph, which
//!      a statement-tree pass deliberately does not have.
//!
//! Until one of those exists, do NOT read a zero here as "the shape is rare":
//! it is "the input cannot be constructed", and the counter says nothing about
//! the corpus.
//!
//! ## The arity gate
//!
//! When the delegate type states its arity, the recovered body must have that
//! many parameters. A mismatch means the binding is some other method's body
//! that happens to share a name, so the fold is refused rather than attaching
//! the wrong body.

use super::{ctor_type_name, is_ctor_name, walk_mut, Expr, Stmt};

/// Is `name` a compiler-generated lambda method (`<…>b__N_M`)?
fn is_generated_lambda(name: &str) -> bool {
    let last = name.rsplit('.').next().unwrap_or(name);
    last.starts_with('<') && last.contains("b__")
}

/// Is `ty` plausibly a delegate type name? Deliberately conservative: only the
/// BCL delegate families plus the conventional `*EventHandler`/`*Callback`/
/// `*Delegate` suffixes, so an unrelated construction is never lambdafied.
fn is_delegate_type(ty: &str) -> bool {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    base.starts_with("Action")
        || base.starts_with("Func")
        || base == "Predicate"
        || base == "Comparison"
        || base.ends_with("EventHandler")
        || base.ends_with("Delegate")
        || base.ends_with("Callback")
}

/// Number of type arguments in the first `<…>` of `s` (top-level commas + 1),
/// or `None` when there is no generic argument list.
fn angle_arity(s: &str) -> Option<usize> {
    let open = s.find('<')?;
    let close = s.rfind('>')?;
    if close <= open {
        return None;
    }
    let inner = &s[open + 1..close];
    if inner.trim().is_empty() {
        return Some(0);
    }
    let mut depth = 0i32;
    let mut n = 1usize;
    for c in inner.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => n += 1,
            _ => {}
        }
    }
    Some(n)
}

/// Parameter count of a delegate type: `Action`/`Action<N>`/`Action<…>` give N
/// parameters; `Func`/`Func<N>`/`Func<…>` give N-1 (the last type argument is
/// the return type). `None` when the name does not state an arity.
fn delegate_arity(ty: &str) -> Option<usize> {
    let t = ty.rsplit('.').next().unwrap_or(ty);
    let base = t.split('`').next().unwrap_or(t);
    let angle = angle_arity(t);
    let backtick = t
        .split('`')
        .nth(1)
        .and_then(|d| d.split('[').next())
        .and_then(|d| d.trim().parse::<usize>().ok());
    let args = angle.or(backtick);
    if base.starts_with("Action") {
        Some(args.unwrap_or(0))
    } else if base.starts_with("Func") {
        args.map(|a| a.saturating_sub(1))
    } else {
        None
    }
}

/// The same-list bindings a fold needs.
#[derive(Default)]
struct Bindings {
    /// generated method name → its body, from `Assign(<name>, _, Lambda{..})`.
    bodies: Vec<(String, Expr)>,
    /// local → generated method name, from `Assign(v, _, MethodGroup(name))`.
    aliases: Vec<(String, String)>,
}

impl Bindings {
    fn collect(list: &[Stmt]) -> Self {
        let mut b = Bindings::default();
        for s in list {
            if let Stmt::Assign(name, _, e) = s {
                match e {
                    Expr::Lambda { .. } if is_generated_lambda(name) => {
                        b.bodies.push((name.clone(), e.clone()));
                    }
                    Expr::MethodGroup(m) => b.aliases.push((name.clone(), m.clone())),
                    _ => {}
                }
            }
        }
        b
    }

    /// The generated method name a second argument names, directly or through a
    /// local bound to a `MethodGroup`.
    fn method_name<'a>(&'a self, e: &'a Expr) -> Option<&'a str> {
        match e {
            Expr::MethodGroup(m) => Some(m.as_str()),
            Expr::Var(v) => self
                .aliases
                .iter()
                .find(|(n, _)| n == v)
                .map(|(_, m)| m.as_str()),
            _ => None,
        }
    }
}

/// Fold one statement list.
fn fold_list(list: &mut Vec<Stmt>) -> usize {
    let bindings = Bindings::collect(list);
    if bindings.bodies.is_empty() {
        return 0;
    }
    let mut n = 0usize;
    for s in list.iter_mut() {
        let slot: Option<&mut Expr> = match s {
            Stmt::Assign(_, _, e) => Some(e),
            Stmt::Eval(e) => Some(e),
            Stmt::Return(Some(e)) => Some(e),
            Stmt::Throw(e) => Some(e),
            Stmt::YieldReturn(e) => Some(e),
            Stmt::If { cond, .. }
            | Stmt::While { cond, .. }
            | Stmt::DoWhile { cond, .. } => Some(cond),
            Stmt::For { cond: Some(c), .. } => Some(c),
            _ => None,
        };
        if let Some(e) = slot {
            n += rewrite_expr(e, &bindings);
        }
    }
    n
}

/// Rewrite delegate constructions inside `e`, recursing into children.
fn rewrite_expr(e: &mut Expr, b: &Bindings) -> usize {
    if let Some(lambda) = try_fold(e, b) {
        *e = lambda;
        return 1;
    }
    let mut n = 0usize;
    match e {
        Expr::Bin(_, a, c) | Expr::Cmp(_, a, c) => {
            n += rewrite_expr(a, b);
            n += rewrite_expr(c, b);
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::AddressOf(x) | Expr::Await(x) => {
            n += rewrite_expr(x, b);
        }
        Expr::Field(r, _) | Expr::RawDeref(r, _, _) | Expr::RawAddr(r, _) => {
            n += rewrite_expr(r, b);
        }
        Expr::Cond(c, a, d) => {
            n += rewrite_expr(c, b);
            n += rewrite_expr(a, b);
            n += rewrite_expr(d, b);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                n += rewrite_expr(r, b);
            }
            for a in args {
                n += rewrite_expr(a, b);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                n += rewrite_expr(a, b);
            }
        }
        Expr::Lambda { body, .. } => n += rewrite_expr(body, b),
        _ => {}
    }
    n
}

/// The delegate-construction → lambda fold, applied to ONE expression.
///
/// Fires only when the two-operand delegate shape is present, the construction
/// names a delegate TYPE (so an unrelated `.ctor` call is never touched), the
/// generated method's body is bound in the same list, and the body's parameter
/// count agrees with the delegate type's arity.
fn try_fold(e: &Expr, b: &Bindings) -> Option<Expr> {
    // The type name of the construction, and its argument list.
    let (ty, args): (Option<&str>, &[Expr]) = match e {
        Expr::NewObj(ty, args) => (Some(ty.as_str()), args.as_slice()),
        Expr::Call { name, args, .. } if is_ctor_name(name) => {
            (ctor_type_name(name), args.as_slice())
        }
        _ => return None,
    };
    if args.len() != 2 {
        return None;
    }
    // A `recv..ctor(target, method)` spelling carries no type in the tree, so
    // nothing proves it is a delegate: refuse rather than guess.
    let ty = ty?;
    if !is_delegate_type(ty) {
        return None;
    }
    let m = b.method_name(&args[1])?;
    if !is_generated_lambda(m) {
        return None;
    }
    let (_, binding) = b.bodies.iter().find(|(n, _)| n == m)?;
    let Expr::Lambda { params, .. } = binding else {
        return None;
    };
    if let Some(arity) = delegate_arity(ty) {
        if params.len() != arity {
            return None;
        }
    }
    Some(binding.clone())
}

/// Run the closure recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        n += fold_list(list);
    });
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lambda(params: &[&str], body: Expr) -> Expr {
        Expr::Lambda {
            params: params.iter().map(|p| p.to_string()).collect(),
            body: Box::new(body),
        }
    }
    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    /// `System.Action`1..ctor(this, <method>)` — the real construction spelling.
    fn ctor(method: Expr) -> Expr {
        Expr::Call {
            recv: None,
            name: "System.Action`1..ctor".into(),
            args: vec![Expr::This, method],
        }
    }

    /// The provable shape in the REAL spelling: the generated body is bound in
    /// the same list, then the construction referencing it folds.
    #[test]
    fn ctor_call_with_a_provable_body_folds() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], Expr::Field(Box::new(var("x")), "Value".into()));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body.clone()),
            Stmt::Assign("action".to_string(), None, ctor(Expr::MethodGroup(name.to_string()))),
        ];
        let n = apply(&mut stmts);
        assert_eq!(n, 1, "the construction must fold to the lambda");
        match &stmts[1] {
            Stmt::Assign(_, _, Expr::Lambda { params, .. }) => assert_eq!(params.len(), 1),
            other => panic!("expected a folded lambda, got {other:?}"),
        }
    }

    /// The real IL2CPP indirection: the method cell is loaded into a local and
    /// the construction names that local.
    #[test]
    fn a_method_cell_held_in_a_local_is_resolved() {
        let name = "<>c.<M>b__2_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Assign("v12".to_string(), None, Expr::MethodGroup(name.to_string())),
            Stmt::Assign("d".to_string(), None, ctor(var("v12"))),
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[2], Stmt::Assign(_, _, Expr::Lambda { .. })));
    }

    /// Refusal: with no provable body in the list, the construction is left
    /// EXACTLY as it was — never a fabricated lambda.
    #[test]
    fn ctor_call_without_a_provable_body_is_untouched() {
        let mut stmts = vec![Stmt::Eval(ctor(Expr::MethodGroup(
            "<>c.<M>b__1_0".to_string(),
        )))];
        let before = format!("{:?}", stmts);
        assert_eq!(apply(&mut stmts), 0);
        assert_eq!(before, format!("{:?}", stmts), "the call must be preserved verbatim");
    }

    /// A mismatched binding (wrong arity for the delegate type) is refused even
    /// though the name matches.
    #[test]
    fn arity_mismatch_is_refused() {
        let name = "<>c.<M>b__2_0";
        let body = lambda(&[], var("x")); // 0 params, Action`1 wants 1
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(ctor(Expr::MethodGroup(name.to_string()))),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A construction whose callee carries no type (`v12..ctor(a, b)`) proves
    /// nothing about being a delegate.
    #[test]
    fn a_recv_ctor_without_a_type_is_refused() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::Call {
                recv: Some(Box::new(var("v12"))),
                name: ".ctor".into(),
                args: vec![Expr::This, Expr::MethodGroup(name.to_string())],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A non-delegate construction is never lambdafied.
    #[test]
    fn non_delegate_constructions_are_not_lambdafied() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::Call {
                recv: None,
                name: "System.Collections.Generic.List`1..ctor".into(),
                args: vec![Expr::This, Expr::MethodGroup(name.to_string())],
            }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// The legacy `Expr::NewObj` spelling is still accepted.
    #[test]
    fn the_newobj_spelling_is_still_accepted() {
        let name = "<>c.<M>b__1_0";
        let body = lambda(&["x"], var("x"));
        let mut stmts = vec![
            Stmt::Assign(name.to_string(), None, body),
            Stmt::Eval(Expr::NewObj(
                "System.Action`1".into(),
                vec![Expr::This, Expr::MethodGroup(name.to_string())],
            )),
        ];
        assert_eq!(apply(&mut stmts), 1);
    }

    #[test]
    fn delegate_arity_is_read_from_the_name() {
        assert_eq!(delegate_arity("System.Action"), Some(0));
        assert_eq!(delegate_arity("System.Action`1"), Some(1));
        assert_eq!(delegate_arity("System.Action`2"), Some(2));
        assert_eq!(delegate_arity("System.Func`2"), Some(1));
        assert_eq!(delegate_arity("System.Func`3"), Some(2));
        assert_eq!(delegate_arity("Dictionary<string, List<int>>"), None);
        assert_eq!(angle_arity("Action<Foo, Bar>"), Some(2));
    }
}
