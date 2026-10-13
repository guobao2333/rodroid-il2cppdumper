//! Array initializers + CLR bounds-check folding.
//!
//! ## The two halves, and which one is real
//!
//! ### 1. The bounds-check fold (implemented)
//!
//! The CLR's array bounds check compiles to `if ((uint)i >= (uint)len) throw
//! new IndexOutOfRangeException();`. Where the IR-level passes
//! (`passes::remove_null_check_guards`, `passes::drop_vacuous_cond_branches`)
//! deliberately left one behind, it carries no C#-source semantics and is
//! folded away here.
//!
//! The condition's spelling on the real corpus is NOT `.Length`. Measured
//! output (`dump.cs`) writes the length as a raw deref of the array object:
//!
//! ```text
//! while (v13 < *(nint*)(v12 +24))      // Il2CppArray::max_length at 0x18
//! if (v7 < *(nint*)(this.ranges +24) && …)
//! ```
//!
//! `Il2CppArray` is `{ klass; monitor; bounds; max_length; }`, so
//! `*(nint*)(a + 24)` IS the array's length — and for `List<T>`,
//! `{ …; _items; _size; }` puts `_size` at the same 0x18. Both are length reads,
//! and both must be recognized or the pass can never fire (`.Length` appears
//! nowhere in the real output). The `.Length`/`.Count`/`get_Length` spellings
//! are kept for other architectures/pipelines that do resolve the property.
//!
//! Every gate is kept tight, because the cost of a wrong fold is deleting a real
//! check:
//!   * the arm is `if (cond) { throw <range-exception>; }` with NO else and
//!     exactly ONE statement in the arm (so removing it cannot orphan a
//!     label — a label in the arm would have to stay);
//!   * the thrown value names an out-of-range exception
//!     (`…OutOfRange…`) — a user `throw new Exception()` is never touched;
//!   * `cond` compares against an array-length reference on exactly ONE side,
//!     and the other side yields a local (the index);
//!   * the index is still READ later in the same list, so the access the guard
//!     protects (and the index's own only other use) survives.
//!
//! ### 2. The array initializer — BLOCKED by the renderer, deliberately not
//!    implemented
//!
//! `Expr::NewArr(elem, dims)` renders ONLY `dims[0]` as the length; the other
//! entries render as empty commas (`new int[a]` / `new int[a,,]`). There is no
//! `{ … }` initializer form anywhere in `Expr`. Folding
//! `a[0] = v0; a[1] = v1;` into `NewArr` would therefore *delete the stored
//! values* — a direct violation of rule 1, worse than the ugly-but-correct
//! element-store sequence it replaced.
//!
//! There is a second, independent problem: the real corpus never writes
//! `new T[...]` at all. `dump.cs` shows every allocation as a `.ctor` call
//! (`v75..ctor(0);`), and its array-element stores are raw pointer writes
//! (`*(nint*)(v24 +idx*4 +32) = …`), not indexed assignments to a recovered
//! array local. So even with an initializer form in the AST, there is no
//! element-store shape here to fold.
//!
//! The honest fix is a new AST node (`Expr::ArrayInit { elem_type, items }`)
//! plus a renderer arm; that lives in `csharp.rs`, outside this slice.
//! `newarr_has_no_initializer_form` below PINS the blocker as an executable fact
//! so this decision is revisited the moment the renderer gains the form.

use super::super::csharp::{CmpKind, Expr, Stmt};

pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0;
    super::walk_mut(stmts, &mut |list| {
        n += fold_bounds_guards(list);
    });
    n
}

/// Remove every provable CLR bounds guard from one statement list.
fn fold_bounds_guards(list: &mut Vec<Stmt>) -> usize {
    if list.is_empty() {
        return 0;
    }
    let mut remove = vec![false; list.len()];
    let mut n = 0;
    for i in 0..list.len() {
        if !is_bounds_guard(&list[i]) {
            continue;
        }
        // The index the guard tests must still be observed afterwards: folding
        // the guard must never remove the only reference to an index.
        let Some(idx) = guard_index_name(&list[i]) else {
            continue;
        };
        if !list[i + 1..].iter().any(|s| super::stmt_reads(s, &idx)) {
            continue;
        }
        remove[i] = true;
        n += 1;
    }
    if n > 0 {
        let mut i = 0usize;
        list.retain(|_| {
            let keep = !remove[i];
            i += 1;
            keep
        });
    }
    n
}

/// `if (cond) { throw <out-of-range>; }` with no else and a single statement.
fn is_bounds_guard(s: &Stmt) -> bool {
    let Stmt::If { cond, then_body, else_body } = s else {
        return false;
    };
    if !else_body.is_empty() || then_body.len() != 1 {
        return false;
    }
    let Stmt::Throw(e) = &then_body[0] else {
        return false;
    };
    expr_is_range_exception(e) && expr_references_length(cond)
}

/// The index operand of the length comparison (`i` in `i >= arr.Length`).
fn guard_index_name(s: &Stmt) -> Option<String> {
    let Stmt::If { cond, .. } = s else {
        return None;
    };
    let Expr::Cmp(k, a, b) = cond else {
        return None;
    };
    // Only equality/ordering against a length is the CLR shape; require the
    // length on exactly one side and take the other side's local.
    if !matches!(
        k,
        CmpKind::Eq | CmpKind::Ne | CmpKind::Lt | CmpKind::Gt | CmpKind::Le | CmpKind::Ge
    ) {
        return None;
    }
    let a_len = expr_references_length(a);
    let b_len = expr_references_length(b);
    match (a_len, b_len) {
        // Exactly one side must be the length: both sides being a length is not
        // an index test.
        (false, true) => first_var(a),
        (true, false) => first_var(b),
        _ => None,
    }
}

/// Does the thrown value name an out-of-range exception?
fn expr_is_range_exception(e: &Expr) -> bool {
    match e {
        Expr::NewObj(name, _) => names_range_exception(name),
        Expr::Call { recv, name, .. } => {
            names_range_exception(name) || recv.as_deref().is_some_and(expr_is_range_exception)
        }
        Expr::RawCall(_, _, Some(ty)) => names_range_exception(ty),
        Expr::Cast(_, inner) => expr_is_range_exception(inner),
        Expr::Var(n) => names_range_exception(n),
        _ => false,
    }
}

fn names_range_exception(name: &str) -> bool {
    name.contains("OutOfRange")
}

/// Does `e` reference an array/collection length?
///
/// Includes the **raw IL2CPP spelling**: `*(nint*)(base + 24)` is
/// `Il2CppArray::max_length` (0x18) and `List<T>::_size` (also 0x18) — the only
/// length spelling the real corpus emits.
fn expr_references_length(e: &Expr) -> bool {
    match e {
        Expr::Field(_, f) => matches!(f.as_str(), "Length" | "LongLength" | "Count"),
        Expr::StaticField(_, f) => matches!(f.as_str(), "Length" | "LongLength" | "Count"),
        // The width matters here and 4 is NOT a typo. `Il2CppArray` is
        // `{ klass; monitor; bounds; max_length; }` and `max_length` is an
        // `il2cpp_array_size_t`, which the runtime defines as a `uint32_t` — so
        // the compiler loads it with a 32-bit `mov` before widening for the
        // compare, and the REAL spelling of an array length on x64 is
        // `*(int*)(a + 24)`, not `*(nint*)(a + 24)`. Accepting only 0/8 here
        // refused every 4-byte length read, i.e. most of them, once the x86
        // lifter started carrying access widths. 1 and 2 stay refused: a
        // byte/word read at 0x18 would be a different field.
        Expr::RawDeref(_, 24, w) => *w == 0 || *w == 4 || *w == 8,
        Expr::Call { recv, name, args } => {
            matches!(
                name.as_str(),
                "get_Length" | "get_LongLength" | "get_Count" | "Length" | "Count"
            ) || recv.as_deref().is_some_and(expr_references_length)
                || args.iter().any(expr_references_length)
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            expr_references_length(a) || expr_references_length(b)
        }
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => {
            expr_references_length(x)
        }
        Expr::Cond(c, a, b) => {
            expr_references_length(c) || expr_references_length(a) || expr_references_length(b)
        }
        Expr::RawDeref(x, _, _) | Expr::RawAddr(x, _) | Expr::AddressOf(x) => {
            expr_references_length(x)
        }
        _ => false,
    }
}

fn first_var(e: &Expr) -> Option<String> {
    let mut out = Vec::new();
    collect_vars(e, &mut out);
    out.into_iter().next()
}

/// Collect the local names an expression reads (lambda params shadow).
fn collect_vars(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Var(n) => out.push(n.clone()),
        Expr::Field(r, _)
        | Expr::RawDeref(r, _, _)
        | Expr::RawAddr(r, _)
        | Expr::AddressOf(r) => collect_vars(r, out),
        Expr::Not(x) | Expr::Neg(x) | Expr::Cast(_, x) | Expr::Await(x) => collect_vars(x, out),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Cond(c, a, b) => {
            collect_vars(c, out);
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Call { recv, args, .. } => {
            if let Some(r) = recv {
                collect_vars(r, out);
            }
            for a in args {
                collect_vars(a, out);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                collect_vars(a, out);
            }
        }
        Expr::Lambda { params, body } => {
            let mut inner = Vec::new();
            collect_vars(body, &mut inner);
            for v in inner {
                if !params.iter().any(|p| p == &v) {
                    out.push(v);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(n: &str) -> Expr {
        Expr::Var(n.into())
    }
    fn len_of(arr: &str) -> Expr {
        Expr::Field(Box::new(Expr::Var(arr.into())), "Length".into())
    }
    /// `*(nint*)(arr +24)` — the real IL2CPP length spelling.
    fn raw_len(arr: &str) -> Expr {
        Expr::RawDeref(Box::new(Expr::Var(arr.into())), 24, 0)
    }
    /// `if (v2 >= v0.Length) throw new IndexOutOfRangeException();`
    fn guard() -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
            then_body: vec![Stmt::Throw(Expr::NewObj(
                "System.IndexOutOfRangeException".into(),
                vec![],
            ))],
            else_body: vec![],
        }
    }
    /// The real spelling of the same guard.
    fn raw_guard() -> Stmt {
        Stmt::If {
            cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(raw_len("v0"))),
            then_body: vec![Stmt::Throw(Expr::NewObj(
                "System.IndexOutOfRangeException".into(),
                vec![],
            ))],
            else_body: vec![],
        }
    }

    /// The CLR bounds guard disappears when its index is still used afterwards.
    #[test]
    fn bounds_guard_is_folded() {
        let mut list = vec![guard(), Stmt::Eval(idx("v2"))];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
        assert!(matches!(&list[0], Stmt::Eval(Expr::Var(n)) if n == "v2"));
    }

    /// The REAL spelling — a raw deref at offset 0x18 — must fold too.
    #[test]
    fn the_raw_il2cpp_length_spelling_is_folded() {
        let mut list = vec![raw_guard(), Stmt::Eval(idx("v2"))];
        assert_eq!(apply(&mut list), 1);
        assert_eq!(list.len(), 1);
    }

    /// A raw deref at a DIFFERENT offset is a field, not a length.
    #[test]
    fn a_raw_deref_at_another_offset_is_not_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 16, 0)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// The 32-bit load of `max_length` — the spelling the x64 lifter actually
    /// produces once access widths are carried — is a length read too.
    #[test]
    fn a_32_bit_deref_at_24_is_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 24, 4)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 1);
    }

    /// A byte-width deref at 24 is not an object's `max_length`.
    #[test]
    fn a_byte_width_deref_at_24_is_not_a_length() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(
                    CmpKind::Ge,
                    Box::new(idx("v2")),
                    Box::new(Expr::RawDeref(Box::new(idx("v0")), 24, 1)),
                ),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// A guard with no `.Length` reference is an ordinary branch — refused.
    #[test]
    fn guard_without_length_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(Expr::Int(8))),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
        assert_eq!(list.len(), 2);
    }

    /// A non-range throw is a REAL user throw — refused.
    #[test]
    fn non_range_throw_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![Stmt::Throw(Expr::NewObj("System.Exception".into(), vec![]))],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// Removing the guard must not drop the index's only reference.
    #[test]
    fn guard_whose_index_is_not_used_after_is_refused() {
        let mut list = vec![guard(), Stmt::Eval(idx("v9"))];
        assert_eq!(apply(&mut list), 0);
        assert_eq!(list.len(), 2);
    }

    /// An arm with more than one statement may contain content (and labels) —
    /// folding it could orphan a jump target.
    #[test]
    fn a_multi_statement_arm_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![
                    Stmt::Throw(Expr::NewObj("System.IndexOutOfRangeException".into(), vec![])),
                    Stmt::Label("L".into()),
                ],
                else_body: vec![],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// An `else` arm means the guard has a real second path — refused.
    #[test]
    fn an_else_arm_is_refused() {
        let mut list = vec![
            Stmt::If {
                cond: Expr::Cmp(CmpKind::Ge, Box::new(idx("v2")), Box::new(len_of("v0"))),
                then_body: vec![Stmt::Throw(Expr::NewObj(
                    "System.IndexOutOfRangeException".into(),
                    vec![],
                ))],
                else_body: vec![Stmt::Label("L".into())],
            },
            Stmt::Eval(idx("v2")),
        ];
        assert_eq!(apply(&mut list), 0);
    }

    /// The initializer fold is blocked by the renderer: `NewArr` prints only the
    /// first dimension and has no `{ … }` form, so folding element stores into it
    /// would DELETE them. This test pins that fact; when a renderer arm for an
    /// array initializer exists, replace it with the real fold.
    #[test]
    fn newarr_has_no_initializer_form() {
        let mut out = String::new();
        Expr::NewArr("int".into(), vec![Expr::Int(1), Expr::Int(2), Expr::Int(3)]).render(&mut out);
        assert!(
            out.starts_with("new int[1"),
            "expected the first dim to be the length, got {out:?}"
        );
        assert!(
            !out.contains('{'),
            "renderer unexpectedly grew an initializer form — implement the fold: {out:?}"
        );
    }
}
