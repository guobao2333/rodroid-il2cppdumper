//! `for (init; cond; incr)` reconstruction.
//!
//! ## The shape that ACTUALLY occurs
//!
//! Measured on the x64 corpus (47,924 methods): `for (` appears **zero** times,
//! while `while (` appears 6,758 times. The structurer emits the canonical
//! counter loop as a flat `init; …; while (cond) { …; update; …; continue; }`.
//! Verbatim from the real dump (`dump.cs`, `get_Item` @~line 3010):
//!
//! ```text
//! v11 = 0;
//! v12 = this.attrNames;
//! v13 = 0;                       // the counter's init — NOT adjacent to the loop
//! v14 = name;
//! while (v13 < *(nint*)(v12 +24))
//! {
//!     …
//!     v11 = v11 + 1;
//!     v12 = this.attrNames;
//!     v13 = v11 + 1;             // updated through ANOTHER temp, not `v13 = v13 + 1`
//!     v14 = v18;                 // a live copy AFTER the update
//!     continue;                  // an explicit trailing `continue`
//! }
//! ```
//!
//! Three properties of that shape kill the naive rule ("the statement before
//! the `while` is the init, the body's LAST statement is the increment, and it
//! must mention the counter"): the init is not adjacent, the increment does not
//! mention the counter, and the body ends with `continue`. The previous version
//! of this file required all three and therefore never fired.
//!
//! ## The rewrite, and the proof for each gate
//!
//! `for (init; cond; UPDATE) { pre…, TAIL… }` where the flat form was
//! `init; …; while (cond) { pre…, UPDATE, TAIL… (continue) }`:
//!
//!   * **the counter** is an induction variable of the condition — the direct
//!     local operand of the comparison, or the left operand of arithmetic on it
//!     (`v33 + 1 < n`). A condition whose operands are raw derefs
//!     (`*(nint*)(v37 -1) < *(nint*)(buffer +8)`) names no local and is
//!     refused;
//!   * **the update** is the LAST assignment to the counter in the body. It
//!     need not mention the counter (`v13 = v11 + 1` is the real shape);
//!   * **the tail** (statements after the update, up to the end of the body) is
//!     left in the body, which moves it BEFORE the update. Rule 2 allows that
//!     only when nothing in it can observe the difference: every tail statement
//!     must be free of calls and of control flow, must not read the counter,
//!     and must not write anything the update's right-hand side reads;
//!   * **at most one `continue`**, and only as the very last statement: it is
//!     dropped (falling off the end of a `for` body also runs the increment, so
//!     a trailing `continue` is a no-op there). Any OTHER `continue` at this
//!     level skips the update in the flat form but would run it in a `for`, so
//!     it refuses the fold;
//!   * **the init** is the nearest preceding assignment to the counter, hoisted
//!     from its position `p` into the header. That is a move, so: nothing
//!     between `p` and the loop may read or write the counter, the init's
//!     right-hand side must be call-free and must not read anything those
//!     statements write, and the init must have an inline form (see below);
//!   * **the counter must not be read after the loop.** A `for` header's
//!     declaration is scoped to the loop, so hoisting the init while later
//!     statements still read the counter would change the program's scope;
//!   * **the header must be renderable.** `Renderer::stmt` builds a `for` header
//!     with `filter_map(stmt_inline)`, so an init/increment element without an
//!     inline form is silently DROPPED from the output — a data loss, not a
//!     style problem. Both are `Assign`s here, but the gate is checked anyway.
//!
//! A failed match is a no-op: the correct flat `while` stays.

use super::{
    contains_continue, expr_has_call, expr_mentions, is_inlinable, list_mentions_deep,
    stmt_mentions_deep, walk_mut, Expr, Stmt,
};

/// One planned rewrite: where the `while` is, where its init lives, and the
/// pieces of the `for` header.
struct Fold {
    /// Index of the `while` statement in the list.
    seq: usize,
    /// Index of the counter's init statement (hoisted into the header).
    init_at: usize,
    /// The `for` statement that replaces the `while`.
    for_stmt: Stmt,
}

/// Run the recognizer over one method body.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let mut n = 0usize;
    walk_mut(stmts, &mut |list| {
        // Each fold removes one `while` and one assignment, so this terminates
        // after at most `list.len()` iterations.
        while let Some(fold) = find_fold(list) {
            let mut slot = Some(fold.for_stmt);
            let mut out: Vec<Stmt> = Vec::with_capacity(list.len());
            for (idx, s) in std::mem::take(list).into_iter().enumerate() {
                if idx == fold.init_at {
                    continue; // hoisted into the `for` header
                }
                if idx == fold.seq {
                    out.push(slot.take().expect("the `for` is placed exactly once"));
                    continue;
                }
                out.push(s);
            }
            *list = out;
            n += 1;
        }
    });
    n
}

/// The first foldable counter loop in this list, if any.
fn find_fold(list: &[Stmt]) -> Option<Fold> {
    (0..list.len()).find_map(|i| plan_at(list, i))
}

/// Try to fold the `while` at `i`.
fn plan_at(list: &[Stmt], i: usize) -> Option<Fold> {
    let Stmt::While { cond, body } = &list[i] else {
        return None;
    };
    if body.is_empty() {
        return None;
    }
    let counter = condition_counter(cond)?;

    // The counter must be updated inside the loop body: a `while` whose
    // induction variable is never assigned in its own body is not a `for`.
    let update_at = body
        .iter()
        .rposition(|s| matches!(s, Stmt::Assign(n, _, _) if n == &counter))?;

    // At most one `continue`, and only as the very last statement.
    let mut end = body.len();
    if matches!(body[end - 1], Stmt::Continue) {
        end -= 1;
    }
    if update_at >= end {
        return None;
    }
    if contains_continue(&body[..end]) {
        return None;
    }

    let update = body[update_at].clone();
    let Stmt::Assign(_, _, update_rhs) = &update else {
        unreachable!("update_at is an Assign by construction")
    };
    let tail = &body[update_at + 1..end];

    // The tail stays in the body, so it runs BEFORE the update once folded.
    for t in tail {
        if !super::is_pure_local_stmt(t) {
            // A call, a store, or control flow: reordering it past the update
            // could be observed.
            return None;
        }
        if super::stmt_reads_deep(t, &counter) {
            return None;
        }
    }
    // ...and the update must not read anything the tail writes.
    for t in tail {
        if let Stmt::Assign(w, _, _) = t {
            if expr_mentions(update_rhs, w) {
                return None;
            }
        }
    }

    // The init: the nearest preceding assignment to the counter.
    let init_at = (0..i)
        .rev()
        .find(|&p| matches!(&list[p], Stmt::Assign(n, _, _) if n == &counter))?;
    let Stmt::Assign(_, _, init_rhs) = &list[init_at] else {
        unreachable!("the position was matched as an Assign")
    };
    // Hoisting the init into the header moves it LATER, past everything between
    // its old position and the loop.
    let between = &list[init_at + 1..i];
    if list_mentions_deep(between, &counter) {
        return None;
    }
    if expr_has_call(init_rhs) {
        return None;
    }
    for s in between {
        if let Stmt::Assign(w, _, _) = s {
            if expr_mentions(init_rhs, w) {
                return None;
            }
        }
    }

    // A `for` header's declaration is scoped to the loop, so the counter must
    // not be mentioned after it.
    if list[i + 1..]
        .iter()
        .any(|s| stmt_mentions_deep(s, &counter))
    {
        return None;
    }

    // The renderer DROPS any header element without an inline form.
    if !is_inlinable(&list[init_at]) || !is_inlinable(&update) {
        return None;
    }

    let mut new_body: Vec<Stmt> = body[..update_at].to_vec();
    new_body.extend_from_slice(tail);

    Some(Fold {
        seq: i,
        init_at,
        for_stmt: Stmt::For {
            init: vec![list[init_at].clone()],
            cond: Some(cond.clone()),
            incr: vec![update],
            body: new_body,
        },
    })
}

/// The induction variable of a loop condition: a plain local, or a local
/// reached through arithmetic on it (`v33 + 1 < n`).
///
/// Returns `None` for a condition that names no local — e.g. the real
/// `while (*(nint*)(v37 -1) < *(nint*)(buffer +8))`, whose operands are raw
/// derefs of frame slots and a parameter. Refusing there is correct: a frame
/// slot has no assignment in the body to serve as the update.
fn condition_counter(cond: &Expr) -> Option<String> {
    match cond {
        Expr::Cmp(_, a, b) => induction_var(a).or_else(|| induction_var(b)),
        Expr::Var(n) => Some(n.clone()),
        Expr::Not(inner) => condition_counter(inner),
        Expr::Bin(_, a, _) => induction_var(a),
        _ => None,
    }
}

fn induction_var(e: &Expr) -> Option<String> {
    match e {
        Expr::Var(n) => Some(n.clone()),
        Expr::Bin(_, a, _) => induction_var(a),
        Expr::Cast(_, inner) | Expr::Neg(inner) => induction_var(inner),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recovery::csharp::{BinOp, CmpKind};

    fn var(n: &str) -> Expr {
        Expr::Var(n.to_string())
    }
    fn assign(name: &str, e: Expr) -> Stmt {
        Stmt::Assign(name.to_string(), None, e)
    }
    fn add(a: Expr, b: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(a), Box::new(b))
    }
    /// `*(nint*)(base + addend)` — the real array-length spelling.
    fn deref(base: &str, addend: i64) -> Expr {
        Expr::RawDeref(Box::new(var(base)), addend, 0)
    }
    fn lt(a: Expr, b: Expr) -> Expr {
        Expr::Cmp(CmpKind::Lt, Box::new(a), Box::new(b))
    }

    /// The verbatim real shape from the corpus must fold.
    #[test]
    fn the_real_counter_loop_folds() {
        let mut stmts = vec![
            assign("v11", Expr::Int(0)),
            assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
            assign("v13", Expr::Int(0)),
            assign("v14", var("name")),
            Stmt::While {
                // v13 < *(nint*)(v12 +24)
                cond: lt(var("v13"), deref("v12", 24)),
                body: vec![
                    assign("v25", var("this")),
                    assign("v11", add(var("v11"), Expr::Int(1))),
                    assign("v12", Expr::Field(Box::new(Expr::This), "attrNames".into())),
                    assign("v13", add(var("v11"), Expr::Int(1))),
                    assign("v14", var("v18")),
                    Stmt::Continue,
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        // The `for` sits where the `while` was; the init was hoisted out of
        // position 2, and the statements between it and the loop stay BEFORE
        // the loop.
        assert_eq!(stmts.len(), 4);
        assert!(matches!(&stmts[0], Stmt::Assign(n, _, _) if n == "v11"));
        assert!(matches!(&stmts[1], Stmt::Assign(n, _, _) if n == "v12"));
        assert!(matches!(&stmts[2], Stmt::Assign(n, _, _) if n == "v14"), "the copy between init and loop keeps its place");
        match &stmts[3] {
            Stmt::For { init, cond, incr, body } => {
                assert_eq!(init.len(), 1);
                assert!(matches!(&init[0], Stmt::Assign(n, _, Expr::Int(0)) if n == "v13"));
                assert!(cond.is_some(), "the `while` condition becomes the `for` condition");
                assert_eq!(incr.len(), 1);
                assert!(
                    matches!(&incr[0], Stmt::Assign(n, _, rhs) if n == "v13" && expr_mentions(rhs, "v11")),
                    "the real update `v13 = v11 + 1` is the increment"
                );
                // Six statements went in. The update (`v13 = v11 + 1`) moved
                // into the header and the trailing `continue` was dropped, so
                // the body keeps the four statements that run per iteration,
                // with the tail (`v14 = v18`) still last.
                assert_eq!(body.len(), 4, "the update left the body, the tail stayed");
                assert!(matches!(&body[3], Stmt::Assign(n, _, _) if n == "v14"));
                assert!(
                    !body
                        .iter()
                        .any(|s| matches!(s, Stmt::Assign(n, _, _) if n == "v13")),
                    "no statement left in the body may still write the counter"
                );
                assert!(
                    !body.iter().any(|s| matches!(s, Stmt::Continue)),
                    "the trailing `continue` is dropped (falling off a `for` body runs the incr)"
                );
            }
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// An increment the OLD rule demanded (`v13 = v13 + 1`) must still fold.
    #[test]
    fn a_self_referencing_increment_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::Eval(var("work")),
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        assert!(matches!(&stmts[0], Stmt::For { .. }));
    }

    /// A second `continue` at this level would skip the update in the flat form
    /// but run it in a `for` — refuse.
    #[test]
    fn another_continue_refuses_the_fold() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    Stmt::If {
                        cond: var("c"),
                        then_body: vec![Stmt::Continue],
                        else_body: vec![],
                    },
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
        assert!(matches!(&stmts[1], Stmt::While { .. }));
    }

    /// The counter is read after the loop: hoisting the declaration into the
    /// header would change its scope.
    #[test]
    fn counter_used_after_the_loop_refuses_the_fold() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
            Stmt::Eval(Expr::Call { recv: None, name: "Use".into(), args: vec![var("i")] }),
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A body exactly one assignment long is still a `for` (the update is the
    /// whole body).
    #[test]
    fn a_body_that_is_only_the_update_folds() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(apply(&mut stmts), 1);
        match &stmts[0] {
            Stmt::For { body, .. } => assert!(body.is_empty()),
            other => panic!("expected a `for`, got {other:?}"),
        }
    }

    /// A raw-deref condition names no induction variable.
    #[test]
    fn a_condition_without_a_local_is_refused() {
        let mut stmts = vec![Stmt::While {
            cond: lt(deref("v37", -1), deref("buffer", 8)),
            body: vec![assign("v37", Expr::Int(1))],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// The tail moves before the update, so a tail that READS the counter is
    /// observing the pre-update value — refuse.
    #[test]
    fn a_tail_that_reads_the_counter_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("i", add(var("i"), Expr::Int(1))),
                    assign("seen", var("i")),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Control flow in the tail (a `break`) would change meaning under the move.
    #[test]
    fn a_control_flow_tail_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1))), Stmt::Break],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A statement between the init and the loop that OBSERVES the counter
    /// blocks the hoist.
    #[test]
    fn an_observer_between_init_and_loop_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::Eval(Expr::Call { recv: None, name: "Observe".into(), args: vec![var("i")] }),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![assign("i", add(var("i"), Expr::Int(1)))],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Without an init there is nothing to hoist into the header.
    #[test]
    fn a_loop_without_an_init_is_refused() {
        let mut stmts = vec![Stmt::While {
            cond: lt(var("i"), var("n")),
            body: vec![assign("i", add(var("i"), Expr::Int(1)))],
        }];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// A `while` that never assigns its induction variable is not a `for`.
    #[test]
    fn a_loop_that_never_updates_the_counter_is_refused() {
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![Stmt::Eval(var("f"))],
            },
        ];
        assert_eq!(apply(&mut stmts), 0);
    }

    /// Nested loops each fold independently, and the recognizer runs again on
    /// the rewritten list.
    #[test]
    fn a_nested_pair_folds_twice() {
        let inner = Stmt::While {
            cond: lt(var("j"), var("m")),
            body: vec![assign("j", add(var("j"), Expr::Int(1)))],
        };
        let mut stmts = vec![
            assign("i", Expr::Int(0)),
            Stmt::While {
                cond: lt(var("i"), var("n")),
                body: vec![
                    assign("j", Expr::Int(0)),
                    inner,
                    assign("i", add(var("i"), Expr::Int(1))),
                ],
            },
        ];
        assert_eq!(apply(&mut stmts), 2);
        let outer = stmts
            .iter()
            .find_map(|s| match s {
                Stmt::For { body, .. } => Some(body),
                _ => None,
            })
            .expect("the outer loop folded");
        assert!(outer.iter().any(|s| matches!(s, Stmt::For { .. })));
    }
}
