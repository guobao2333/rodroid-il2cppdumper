//! Iterator / coroutine state-machine reconstruction.
//!
//! ## The shape the structurer actually emits
//!
//! Measured on the x64 corpus: `switch (` appears **once** in 2,454,023 lines,
//! and `switch (this.<>1__state)` **zero** times — the structurer emits
//! goto-shaped case bodies, not switch statements. `<>1__state` itself appears
//! 308 times, nearly all of them FIELD DECLARATIONS of the compiler-generated
//! `＜Name＞d__N` types. So a recognizer keyed on `Stmt::Switch` over
//! `<>1__state` cannot fire, which is exactly what the previous version of this
//! file was.
//!
//! The dispatch that DOES reach this layer is the if-ladder the structurer
//! produces for a jump table it could not rebuild:
//!
//! ```text
//! this.<>1__state = -1;
//! if (this.<>1__state == 0) { … }
//! else if (this.<>1__state == 1) { … }
//! else if (this.<>1__state == 2) { … }
//! ```
//!
//! Both spellings are accepted here, and both must satisfy the SAME proof.
//!
//! ## Measured status: the required input was not observed (outcome "b")
//!
//! This pass reported **0** on the x64 corpus. The `switch` spelling is
//! provably dead there (`switch (` occurs once in 2,454,023 lines, and the one
//! occurrence is a property-member annotation, not a statement), so the whole
//! hope rests on the if-ladder spelling — and the ladder has a hard
//! precondition this pass cannot supply: the dispatch must be the LAST statement
//! of the WHOLE method body, over `<>1__state`, with every arm free of control
//! flow. The 308 `<>1__state` occurrences in that dump are FIELD DECLARATIONS of
//! compiler-generated types (`private int <>1__state; // 0x0`); a `MoveNext`
//! body that reaches this layer as a clean two-arm ladder was not observed at
//! all, and the structurer's own census says why — a jump table it cannot
//! rebuild is emitted as arbitrary `goto`s, not as a ladder.
//!
//! The missing input, stated precisely: a `MoveNext` whose recovered body is
//! exactly a bookkeeping prefix plus one state dispatch. Until such a body can
//! reach this layer, the honest reading of the zero is "the structurer never
//! produces the input", and the provable subset below must not be described as
//! working iterator recovery.
//!
//! ## The proof
//!
//! Deleting the state assignments is what makes this rewrite worth doing, and
//! it is only allowed when the state slot carries no other meaning. That is a
//! property of the whole method, so:
//!
//!   1. **the dispatch must be the whole method body.** The body may carry a
//!      bookkeeping prefix (comments and `<slot> = <int literal>` stores), and
//!      then exactly one statement: the `switch` or the if-ladder. Anything
//!      else — code before OR after the dispatch — is refused, because
//!      flattening the dispatch reorders it;
//!   2. **the slot must be pure bookkeeping.** Anywhere in the whole body, the
//!      slot may appear only as `<slot> = <int literal>` or as a test of the
//!      dispatch itself. One read inside an arm, one store from a non-literal,
//!      one mention in a nested condition: all refuse the whole switch;
//!   3. **every arm must convert.** Each arm may contain a state store, a
//!      `<>2__current` store, a bool-valued return, and otherwise only pure
//!      statements. Any control flow in an arm (`if`/loop/`try`/`switch`/
//!      `goto`/`label`/`break`/`continue`) refuses the whole switch, because
//!      flattening an arm across the switch boundary would move control flow;
//!   4. **at least one value must be yielded** — a state dispatch with no
//!      `yield return` is an ordinary switch that happens to test a `__state`
//!      field, and is left alone.
//!
//! A failed match is a no-op: the correct (lower-level) dispatch stays.

use super::super::csharp::{CmpKind, Expr, Stmt};
use super::render_text;

/// `yield return v;` / `yield break;` an arm reconstructs.
enum YieldKind {
    Return(Expr),
    Break,
}

/// Run the recognizer over one method body.
///
/// Only the TOP level is considered: a `yield` is a method-body-level construct
/// (it is what turns the method into a state machine), so a state dispatch that
/// is merely nested inside another construct is not this method's suspension
/// points and is left alone.
pub fn apply(stmts: &mut Vec<Stmt>) -> usize {
    let Some(seq) = convert_dispatch(stmts) else {
        return 0;
    };
    *stmts = seq;
    1
}

/// Convert the whole-body state dispatch, or `None`.
fn convert_dispatch(list: &[Stmt]) -> Option<Vec<Stmt>> {
    // A bookkeeping prefix: comments and `<slot> = <int literal>` stores.
    let mut head = None;
    for (i, s) in list.iter().enumerate() {
        match s {
            Stmt::Comment(_) => continue,
            Stmt::Assign(n, _, Expr::Int(_)) if is_state_disc(n) => continue,
            Stmt::Switch { disc, .. } if is_state_disc(disc) => {
                head = Some(i);
                break;
            }
            Stmt::If { cond, .. } if state_test(cond).is_some() => {
                head = Some(i);
                break;
            }
            _ => return None,
        }
    }
    // The dispatch must be the LAST statement of the body: anything after it
    // would be reordered by the flattening, and anything before it that is not
    // a comment or a state store is real code.
    let head = head?;
    if head + 1 != list.len() {
        return None;
    }

    let (slot, cases) = dispatch_cases(&list[head])?;
    if cases.len() < 2 {
        return None;
    }
    // 2. the whole-slot purity proof.
    if !slot_is_pure_bookkeeping(list, head, &slot) {
        return None;
    }

    // 3. every arm must convert.
    let mut ordered = cases;
    ordered.sort_by_key(|(v, _)| *v);
    let mut seq: Vec<Stmt> = Vec::new();
    let mut saw_return = false;
    for (_v, body) in &ordered {
        let (residual, kind) = convert_arm(body, &slot)?;
        seq.extend(residual);
        match kind {
            YieldKind::Return(e) => {
                saw_return = true;
                seq.push(Stmt::YieldReturn(e));
            }
            YieldKind::Break => seq.push(Stmt::YieldBreak),
        }
    }
    // 4. without a single `yield return` this is not an iterator.
    if !saw_return {
        return None;
    }
    Some(seq)
}

/// The `(slot, cases)` of a dispatch statement — a `switch`, or an if-ladder
/// whose every test is `slot == <literal>`.
fn dispatch_cases(s: &Stmt) -> Option<(String, Vec<(u64, Vec<Stmt>)>)> {
    match s {
        Stmt::Switch { disc, cases, tail } => {
            if !is_state_disc(disc) {
                return None;
            }
            // Real content in `tail` (a shared exit block / default arm) would
            // be reordered by flattening the cases into a sequence.
            if tail
                .iter()
                .any(|s| !matches!(s, Stmt::Comment(_) | Stmt::Label(_)))
            {
                return None;
            }
            Some((disc.clone(), cases.clone()))
        }
        Stmt::If { cond, then_body, else_body } => {
            let mut slot: Option<String> = None;
            let mut cases: Vec<(u64, Vec<Stmt>)> = Vec::new();
            if !ladder_cases(&mut slot, cond, then_body, else_body, &mut cases) {
                return None;
            }
            Some((slot?, cases))
        }
        _ => None,
    }
}

/// Walk one level of an if-ladder, collecting `(state value, arm)` pairs.
///
/// Every level must test the SAME slot with `==` against a distinct integer
/// literal, and the chain must end in an empty (or comment/label-only) else.
/// A `!=` test is refused: the arm under it is not a single state.
fn ladder_cases(
    slot: &mut Option<String>,
    cond: &Expr,
    then_body: &[Stmt],
    else_body: &[Stmt],
    cases: &mut Vec<(u64, Vec<Stmt>)>,
) -> bool {
    let Some((s2, k)) = state_test(cond) else {
        return false;
    };
    // Every level must test the SAME slot. Checked before the update so the two
    // uses of `s2` never mix a move with a borrow.
    if slot.as_deref().is_some_and(|s| s != s2) {
        return false;
    }
    if slot.is_none() {
        *slot = Some(s2);
    }
    if cases.iter().any(|(v, _)| *v == k) {
        return false;
    }
    cases.push((k, then_body.to_vec()));
    let meaningful: Vec<&Stmt> = else_body
        .iter()
        .filter(|s| !matches!(s, Stmt::Comment(_) | Stmt::Label(_)))
        .collect();
    match meaningful.as_slice() {
        [] => true,
        [Stmt::If { cond, then_body, else_body }] => {
            ladder_cases(slot, cond, then_body, else_body, cases)
        }
        _ => false,
    }
}

/// `slot == <int literal>`, with the slot being a compiler state field.
fn state_test(cond: &Expr) -> Option<(String, u64)> {
    let Expr::Cmp(CmpKind::Eq, a, b) = cond else {
        return None;
    };
    let (slot, k) = match (a.as_ref(), b.as_ref()) {
        (slot, Expr::Int(k)) => (slot, *k),
        (Expr::Int(k), slot) => (slot, *k),
        _ => return None,
    };
    if k < 0 {
        return None;
    }
    let text = render_text(slot);
    if is_state_disc(&text) {
        Some((text, k as u64))
    } else {
        None
    }
}

/// Is this the compiler's iterator/coroutine state slot (`<>1__state`)?
///
/// The mangle marker is required so a user's own `switch (gameState)` can never
/// be mistaken for compiler machinery.
fn is_state_disc(name: &str) -> bool {
    name.contains("__state")
}

/// Is this the field a `yield return` value is parked in (`<>2__current`)?
fn is_current(name: &str) -> bool {
    name.to_ascii_lowercase().contains("current")
}

/// Is this the `MoveNext` return, which carries no yielded value?
fn is_bool_like(e: &Expr) -> bool {
    matches!(e, Expr::Bool(_) | Expr::Int(0) | Expr::Int(1))
}

/// `true` when `hay` contains `needle` as a whole identifier.
fn contains_ident(hay: &str, needle: &str) -> bool {
    let mut from = 0usize;
    while let Some(pos) = hay[from..].find(needle) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = start == 0
            || !hay[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let after_ok = end >= hay.len()
            || !hay[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric() || c == '_');
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// Does `e` mention the slot (as an identifier)?
fn expr_mentions_slot(e: &Expr, slot: &str) -> bool {
    contains_ident(&render_text(e), slot)
}

/// Is `s` the bookkeeping store `<slot> = <int literal>`?
fn is_state_store(s: &Stmt, slot: &str) -> bool {
    matches!(s, Stmt::Assign(n, _, Expr::Int(_)) if n == slot)
}

/// Does `s` touch the slot in any way that is NOT a bookkeeping store?
fn touches_slot(s: &Stmt, slot: &str) -> bool {
    match s {
        Stmt::Eval(e) | Stmt::Throw(e) | Stmt::YieldReturn(e) => expr_mentions_slot(e, slot),
        Stmt::Assign(n, _, e) => n == slot || expr_mentions_slot(e, slot),
        Stmt::If { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::While { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::DoWhile { cond, .. } => expr_mentions_slot(cond, slot),
        Stmt::For { cond, init, incr, .. } => {
            cond.as_ref().is_some_and(|c| expr_mentions_slot(c, slot))
                || list_touches_slot_deep(init, slot)
                || list_touches_slot_deep(incr, slot)
        }
        Stmt::Lock { expr, .. } => expr_mentions_slot(expr, slot),
        Stmt::Using { decl, .. } => touches_slot(decl, slot),
        Stmt::Switch { disc, .. } => disc == slot,
        Stmt::ForEach { var_name, collection, .. } => var_name == slot || collection == slot,
        Stmt::Return(Some(e)) => expr_mentions_slot(e, slot),
        _ => false,
    }
}

/// Does any statement in `list`, or in any nested body, touch the slot in a way
/// that is not a bookkeeping store?
fn list_touches_slot_deep(list: &[Stmt], slot: &str) -> bool {
    for s in list {
        if is_state_store(s, slot) {
            continue;
        }
        if touches_slot(s, slot) {
            return true;
        }
        let mut kids: Vec<&Vec<Stmt>> = Vec::new();
        super::children(s, &mut kids);
        for k in kids {
            if list_touches_slot_deep(k, slot) {
                return true;
            }
        }
    }
    false
}

/// The whole-body purity proof: the slot may only be stored from an integer
/// literal, or tested by the dispatch itself.
fn slot_is_pure_bookkeeping(list: &[Stmt], head: usize, slot: &str) -> bool {
    for (i, s) in list.iter().enumerate() {
        if i == head {
            // The dispatch's own tests ARE the dispatch; its arms are checked
            // for any other mention of the slot.
            if !dispatch_arms_are_pure(s, slot) {
                return false;
            }
            continue;
        }
        if list_touches_slot_deep(std::slice::from_ref(s), slot) {
            return false;
        }
    }
    true
}

/// Are the arms of a dispatch statement pure w.r.t. the slot?
fn dispatch_arms_are_pure(s: &Stmt, slot: &str) -> bool {
    let Some((_slot, cases)) = dispatch_cases(s) else {
        return false;
    };
    cases
        .iter()
        .all(|(_, body)| !list_touches_slot_deep(body, slot))
}

/// Convert one state arm: drop the state assignment and the `MoveNext` return,
/// fold the parked `<>2__current` (or a non-boolean return value) into a
/// `yield`, and keep every other statement in front of it.
///
/// Refuses any arm containing control flow (nested `if`/loop/`try`/`switch`, or
/// a `goto`/`label`/`break`/`continue`) — flattening such an arm across a
/// switch boundary would move control flow, which rule 2 forbids.
fn convert_arm(body: &[Stmt], slot: &str) -> Option<(Vec<Stmt>, YieldKind)> {
    let mut residual: Vec<Stmt> = Vec::new();
    let mut current: Option<Expr> = None;
    let mut ret_value: Option<Expr> = None;
    let mut state: Option<i64> = None;

    for s in body {
        match s {
            Stmt::Assign(n, _, Expr::Int(v)) if n == slot => state = Some(*v),
            Stmt::Assign(n, _, e) if is_current(n) => current = Some(e.clone()),
            Stmt::Return(Some(e)) => {
                if !is_bool_like(e) {
                    ret_value = Some(e.clone());
                }
            }
            Stmt::Return(None) => {}
            Stmt::Break
            | Stmt::Continue
            | Stmt::Goto(_)
            | Stmt::Label(_)
            | Stmt::If { .. }
            | Stmt::While { .. }
            | Stmt::DoWhile { .. }
            | Stmt::For { .. }
            | Stmt::ForEach { .. }
            | Stmt::Try { .. }
            | Stmt::Lock { .. }
            | Stmt::Using { .. }
            | Stmt::Switch { .. } => return None,
            other => residual.push(other.clone()),
        }
    }

    let state = state?;
    let kind = match current.or(ret_value) {
        Some(v) => YieldKind::Return(v),
        // No value parked: a negative next state is the iterator's completion
        // (`yield break` / `MoveNext` returns false). A non-negative next state
        // with no value is a plain transition we do not model.
        None if state < 0 => YieldKind::Break,
        None => return None,
    };
    Some((residual, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATE: &str = "this.<>1__state";
    const CUR: &str = "this.<>2__current";

    fn set_state(v: i64) -> Stmt {
        Stmt::Assign(STATE.to_string(), None, Expr::Int(v))
    }
    fn state_test_expr(v: i64) -> Expr {
        Expr::Cmp(
            CmpKind::Eq,
            Box::new(Expr::Var(STATE.into())),
            Box::new(Expr::Int(v)),
        )
    }
    /// One arm that parks `x` and advances to `1`.
    fn yield_zero_arm() -> Vec<Stmt> {
        vec![
            set_state(1),
            Stmt::Assign(CUR.to_string(), None, Expr::Var("x".into())),
            Stmt::Return(Some(Expr::Bool(true))),
        ]
    }
    fn break_arm() -> Vec<Stmt> {
        vec![set_state(-1), Stmt::Return(Some(Expr::Bool(false)))]
    }

    /// The if-LADDER dispatch — the shape the structurer actually emits — folds
    /// to a yield sequence.
    #[test]
    fn a_state_ladder_becomes_yields() {
        let mut body = vec![
            Stmt::Assign(STATE.to_string(), None, Expr::Int(-1)),
            Stmt::If {
                cond: state_test_expr(0),
                then_body: yield_zero_arm(),
                else_body: vec![Stmt::If {
                    cond: state_test_expr(1),
                    then_body: break_arm(),
                    else_body: vec![],
                }],
            },
        ];
        assert_eq!(apply(&mut body), 1);
        assert_eq!(body.len(), 2, "the bookkeeping store is gone too");
        assert!(matches!(&body[0], Stmt::YieldReturn(e) if matches!(e, Expr::Var(n) if n == "x")));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// The `switch` spelling still folds.
    #[test]
    fn a_state_switch_becomes_yields() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
            tail: vec![],
        }];
        assert_eq!(apply(&mut body), 1);
        assert!(matches!(&body[0], Stmt::YieldReturn(_)));
        assert!(matches!(&body[1], Stmt::YieldBreak));
    }

    /// Code AFTER the dispatch would be reordered by flattening — refuse.
    #[test]
    fn content_after_the_dispatch_refuses() {
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, yield_zero_arm()), (1u64, break_arm())],
                tail: vec![],
            },
            Stmt::Return(Some(Expr::Bool(false))),
        ];
        assert_eq!(apply(&mut body), 0);
        assert!(matches!(&body[0], Stmt::Switch { .. }));
    }

    /// The state is consulted for REAL logic inside an arm: the bookkeeping
    /// store may not be deleted, so nothing is rewritten.
    #[test]
    fn a_real_state_test_inside_an_arm_refuses() {
        let mut arm = yield_zero_arm();
        arm.push(Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var(STATE.into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: vec![],
            else_body: vec![],
        });
        let mut body = vec![
            Stmt::Switch {
                disc: STATE.to_string(),
                cases: vec![(0u64, arm), (1u64, break_arm())],
                tail: vec![],
            },
        ];
        assert_eq!(apply(&mut body), 0);
    }

    /// A non-state discriminant is never touched.
    #[test]
    fn an_ordinary_switch_is_left_alone() {
        let mut body = vec![Stmt::Switch {
            disc: "gameState".to_string(),
            cases: vec![
                (0u64, vec![Stmt::Assign("v".into(), None, Expr::Int(1))]),
                (1u64, vec![Stmt::Assign("v".into(), None, Expr::Int(2))]),
            ],
            tail: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    /// A state dispatch with no `yield return` at all is an ordinary switch.
    #[test]
    fn a_dispatch_without_a_value_is_refused() {
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, break_arm()), (1u64, break_arm())],
            tail: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    /// An arm with control flow would move meaning under the flattening.
    #[test]
    fn an_arm_with_control_flow_is_refused() {
        let mut arm = yield_zero_arm();
        arm.insert(
            0,
            Stmt::If { cond: Expr::Bool(true), then_body: vec![], else_body: vec![] },
        );
        let mut body = vec![Stmt::Switch {
            disc: STATE.to_string(),
            cases: vec![(0u64, arm), (1u64, break_arm())],
            tail: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    /// A ladder that tests a NON-state local is not compiler machinery.
    #[test]
    fn a_ladder_on_a_user_variable_is_refused() {
        let mut body = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Eq,
                Box::new(Expr::Var("mode".into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: yield_zero_arm(),
            else_body: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    /// A `!=` test cannot name a single state, so the ladder is refused.
    #[test]
    fn a_negative_state_test_is_refused() {
        let mut body = vec![Stmt::If {
            cond: Expr::Cmp(
                CmpKind::Ne,
                Box::new(Expr::Var(STATE.into())),
                Box::new(Expr::Int(0)),
            ),
            then_body: yield_zero_arm(),
            else_body: vec![],
        }];
        assert_eq!(apply(&mut body), 0);
    }

    #[test]
    fn identifier_matching_respects_boundaries() {
        assert!(contains_ident("this.<>1__state", "<>1__state"));
        assert!(!contains_ident("this.<>1__stateExtra", "<>1__state"));
        assert!(!contains_ident("v12", "v1"));
    }
}
