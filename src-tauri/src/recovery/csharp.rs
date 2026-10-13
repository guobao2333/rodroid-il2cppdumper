//! Native C# pseudo-code generation from recovered IR.
//!
//! Replaces the Ghidra dependency for managed code: because the recovery
//! pipeline produces typed three-address IR from *managed* semantics, we can
//! rebuild real C# statements instead of approximating asm:
//!
//! 1. **Expression folding** — per-block environments collapse three-address
//!    chains into expression trees (single-use temporaries disappear)
//! 2. **Structured control flow** — natural-loop detection via back edges
//!    (`while` / `do..while`), branch merging for `if`/`else`, `break` /
//!    `continue`; labeled `goto` only for genuinely irreducible regions
//! 3. **Rendering** — C# syntax with proper types from the local table

use super::ir::{
    ArithKind, CompareKind, Conversion, FloatArithKind, Insn, Op, Operand, SelectVariant,
};
use super::locals::LocalTable;
use super::probe_dir;
use super::types::{FieldRef, ManagedType, PrimKind};
use std::collections::{BTreeSet, HashMap, HashSet};


// ---------------------------------------------------------------------------
// Expression trees
// ---------------------------------------------------------------------------

type CallParts = (Option<Box<Expr>>, String, Vec<Expr>);

/// How many chained LOCAL INLINES a single expression may perform before the
/// renderer gives up and prints the local's bare name.
///
/// This is the single largest source of UNDEFINED LOCALS in the output, and it
/// is a renderer budget — not a lifter limit. `inline_local_at` substitutes a
/// local's reaching definition for its name, so a straight-line method whose
/// body is one long value chain renders as one big expression and every link
/// consumes budget. The moment the budget runs out the renderer emits
/// `Expr::Var(name)`, and because the definition was consumed by inlining at
/// some other site, NOTHING in the body ever assigns that name:
///
/// ```text
/// static float JumpTime(float y0, float y1, float apexHeight, float accelY)
///     return (... - Sqrt(-(accelY * Sqrt(v10)) * -(accelY * Sqrt(v10)) ...
/// ```
///
/// where `v10` is `apexHeight * -2f / accelY` — a value the native code really
/// computes, spelled in full at its first use and left dangling at its second.
/// At `8` this fired on 236,556 distinct names across 36,412 of the 90,765
/// arm64 methods.
///
/// The recursion is terminated by DEPTH plus the `cur_insn` cycle guard, not by
/// this number, so a larger budget does not risk a runaway: it only lets a
/// genuine value chain be expressed as an expression instead of as a name that
/// no statement defines. `RAYON` workers get a large explicit stack for the
/// deeper trees (see the worker-stack setup in this module).
const MAX_INLINE_DEPTH: usize = 64;
/// Hard ceiling on how many defs ONE expression may inline.
///
/// [`MAX_INLINE_DEPTH`] bounds the recursion's DEPTH, and
/// [`MethodCtx::inline_in_progress`] breaks def cycles, but neither bounds the
/// total size of the expansion: a def DAG whose nodes each read two locals
/// from the level below (`a = b + c` chains, which real SSA produces) doubles
/// the work per level even when it is perfectly acyclic. This makes the
/// renderer's work per expression O(budget) instead of O(2^depth).
///
/// It is deliberately far above anything a real expression reaches — past it
/// the renderer would be printing a single C# line hundreds of thousands of
/// nodes long — so it only ever fires on the super-linear blow-ups, and what
/// it emits then is exactly what the depth guard already emits at its limit:
/// the variable's NAME, never `__unknown`.
const MAX_INLINE_EXPR_NODES: usize = 100_000;

/// Node ceiling for a SHARED subexpression.
///
/// Inlining is a tree COPY. A single-def local read `r` times whose expansion
/// is `n` nodes therefore contributes `r * n` nodes to the rendered text — and
/// when that local reads ANOTHER such local, the products compound and the
/// expression grows exponentially in the def DAG's depth. Neither guard above
/// stops this: the depth guard bounds nesting, and the node budget only runs
/// out after the line is already megabytes long (it fires, but it fires too
/// late to matter).
///
/// The faithful rendering is the one a human writes: name the shared value ONCE
/// and reference the name. A local whose expansion reaches this many nodes and
/// is read more than once is therefore materialised as a temporary (added to
/// `var_reads`, which is exactly the "must be emitted as an assignment" set)
/// and never re-inlined. The ceiling is deliberately generous — an expression
/// of 64 nodes is already large for real code — so ordinary `x * x` / `a < b &&
/// a < c` sharing is untouched.
///
/// Verified on `XmlTextReader.Read` @0xD5B598: before, ONE 1.34 MB line whose
/// `__unknown == 0 ? … : …` ternary and `in_x8` operand each appeared 20,284 /
/// 40,086 times on a single statement; the expansion now stops at the shared
/// temporaries.
const SHARED_EXPR_NODES: u64 = 64;

/// Maximum `emit_region` nesting depth before a region is marked truncated.
///
/// This is a bound on RECURSION and on the indentation width of the emitted
/// text — not a licence to delete code. Reaching it is always a fidelity loss,
/// so the emitted body carries a comment naming the block that was skipped and
/// [`NEST_CAPS`] counts it. Nesting is what the guard-chain flattening in
/// `emit_if` exists to avoid; see the `FLAT GUARD` note there for the measured
/// failure this cap used to hide.
const MAX_NEST_DEPTH: usize = 32;

/// Process-wide count of nesting-cap truncations (see [`MAX_NEST_DEPTH`]).
/// Global rather than per-thread because the run summary is printed once, after
/// every `RAYON` worker has finished.
static NEST_CAPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Nesting-cap truncations seen so far in this process.
pub(crate) fn nest_cap_count() -> u64 {
    NEST_CAPS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Operand index of the AAPCS64 indirect-result register slot on an `Op::Call`:
/// `[target, dest, x0 … x7, x8]`. The arm64 `BL`/`BLR` arms build the list in
/// that order and `lift::attach_sret_register` fills slot 10 when — and only
/// when — the call site materialized the sret pointer (`add x8, sp, #imm`).
const SRET_OPERAND: usize = 10;

#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    This,
    Int(i64),
    Float(f32),
    Double(f64),
    Bool(bool),
    Str(String),
    Null,
    /// ABI argument slot with no recovered value (see `Operand::MissingAbiArg`).
    /// Distinct from `Unknown`: this is a POSITION fact about the register
    /// dump, not an unresolved value. Rendered as `__abi`, and KEPT in every
    /// argument position — including the trailing ones at a RESOLVED call,
    /// where the slot is a declared parameter (see `call_parts`). Trailing slots
    /// are dropped only where the operand list really is the lifter's raw
    /// register dump, i.e. at a call with no signature to read.
    MissingAbiArg,
    Unknown,
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Cmp(CmpKind, Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    /// Instance field access on receiver expression.
    Field(Box<Expr>, String),
    /// Static field access `Type.field`.
    StaticField(String, String),
    /// Instance call `recv.name(args)`; static uses name only.
    Call {
        recv: Option<Box<Expr>>,
        name: String,
        args: Vec<Expr>,
    },
    /// Ternary: cond ? then_e : else_e
    Cond(Box<Expr>, Box<Expr>, Box<Expr>),
    /// Async await: await <task-expr> (async state-machine folding).
    Await(Box<Expr>),
    /// A reconstructed lambda / closure: `(params) => body`.
    ///
    /// The native code holds a display class (`<>c__DisplayClass…`), a
    /// delegate construction and a generated method; this renders the source-
    /// level form once a recognizer has proved the shape. `body` is an
    /// expression (the common `x => x.Value` / `() => Foo(v)` cases).
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
    },
    /// Degraded-but-present raw pointer deref: *(long*)(base + addend)
    RawDeref(Box<Expr>, i64, u8),
    /// The ADDRESS of a stack slot: `(nint*)(base + addend)`.
    ///
    /// The mirror of `RawDeref` — same base/addend shape, but this is the
    /// pointer, not a load of what it points at. Produced by `ADD Rd, sp, #imm`
    /// (`Rd != sp`), which is how the compiler takes the address of a struct
    /// local so it can pass it by reference (`&enumerator`, `&span`, `out`
    /// arguments). Without it the instruction had no destination at all.
    RawAddr(Box<Expr>, i64),
    /// `&expr` — address of a recovered local (after `sp` slots are named).
    AddressOf(Box<Expr>),
    /// Unresolved direct call to a raw address.
    ///
    /// `ret_type` carries the il2cpp-metadata-derived return type when known
    /// (seeded via `Insn.dest_type`), so raw calls render with their real
    /// return type instead of an untyped `__unknown`. This is the metadata
    /// advantage over Ghidra: even unresolved calls keep their signature.
    RawCall(u64, Vec<Expr>, Option<String /* csharp return type */>),
    NewObj(String, Vec<Expr>),
    /// Array allocation `new T[len]` (metadata-typed element).
    NewArr(String, Vec<Expr>),
    /// Explicit numeric conversion `(T)expr` (widening/narrowing kept visible).
    Cast(String, Box<Expr>),
    /// A method used as a VALUE — the `MethodInfo*` operand of a delegate
    /// creation or a method-pointer store (IL2CPP materializes
    /// `<>c.<...>b__N_M` as a pointer and hands it to a delegate).
    ///
    /// Rendered as `MethodInfo("Type.name")`: an unnamed method can never be
    /// confused for a read of the value it belongs to, and a bare dotted name
    /// would be indistinguishable from a static field access.
    MethodGroup(String),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    /// Short-circuit boolean conjunction (`&&`). Only produced when the IR
    /// combines two boolean compare results (a fused `cmp; ccmp; b.cc`
    /// compound condition); bitwise `&` on a bool is legal C# but reads as
    /// wrong intent, and on two comparisons it is never what the source had.
    AndAlso,
    /// Short-circuit boolean disjunction (`||`) — see `AndAlso`.
    OrElse,
}

#[derive(Debug, Clone, Copy)]
pub enum CmpKind {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
}

impl CmpKind {
    fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Lt => Self::Ge,
            Self::Ge => Self::Lt,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
        }
    }

    fn token(self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Ge => ">=",
        }
    }
}

impl BinOp {
    fn token(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Rem => "%",
            Self::And => "&",
            Self::Or => "|",
            Self::Xor => "^",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::AndAlso => "&&",
            Self::OrElse => "||",
        }
    }
}

impl Expr {
    fn precedence(&self) -> u8 {
        match self {
            // lowest binds loosest. Bitwise ops sit BELOW equality so a
            // masked test renders as `(v1 & 1) != 0` — in C# `==` binds
            // tighter than `&`, and the unparenthesized form is a type error.
            Expr::Bin(BinOp::Or, ..) => 4,
            Expr::Bin(BinOp::Xor, ..) => 5,
            Expr::Bin(BinOp::And, ..) => 6,
            Expr::Cmp(..) => 7,
            // `&&` binds tighter than `||` and both bind looser than `==`,
            // so a compound condition renders `a == 0 || b == 0` unparenthesised.
            Expr::Bin(BinOp::AndAlso, ..) => 3,
            Expr::Bin(BinOp::OrElse, ..) => 2,
            Expr::Bin(BinOp::Shl | BinOp::Shr, ..) => 9,
            Expr::Bin(BinOp::Add | BinOp::Sub, ..) => 10,
            Expr::Bin(BinOp::Mul | BinOp::Div | BinOp::Rem, ..) => 11,
            Expr::Not(_) | Expr::Neg(_) | Self::Cast(..) => 12,
            Expr::RawDeref(..) => 13,
            Expr::RawAddr(..) => 13,
            Expr::AddressOf(_) => 12,
            Expr::RawCall(..) => 15,
            // The conditional operator binds LOOSER than everything else in
            // C# (below `||`), and reporting it as a tight leaf meant no
            // parent ever parenthesised it. `1f - c ? a : b` is
            // `(1f - c) ? a : b` to a C# compiler — the subtraction was
            // swallowed by the condition and the select's payload was
            // silently attached to the wrong side of the enclosing operator,
            // in every expression that used a select as an operand.
            Expr::Cond(..) => 1,
            _ => 15,
        }
    }

    pub(crate) fn render(&self, out: &mut String) {
        match self {
            Self::Var(n) => out.push_str(n),
            Self::This => out.push_str("this"),
            Self::Int(v) => {
                // render large values that are really pointers as hex comments
                if *v > 0xFFFF && *v <= i32::MAX as i64 && *v % 4 == 0 && *v > 0x100000 {
                    out.push_str(&format!("{v:#x}"));
                } else {
                    out.push_str(&v.to_string());
                    if *v > i32::MAX as i64 || *v < i32::MIN as i64 {
                        out.push('L');
                    }
                }
            }
            Self::Float(f) => {
                out.push_str(&format!("{f}f"));
            }
            Self::Double(d) => {
                out.push_str(&format!("{d}"));
                if d.fract() == 0.0 && d.abs() < 1e15 {
                    out.push_str(".0");
                }
            }
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => {
                            out.push_str(&format!("\\u{:04x}", c as u32));
                        }
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Self::Null => out.push_str("null"),
            // Explicit unresolved marker — NEVER render as `default`:
            // "genuinely recovered default" and "we have no idea" are
            // different facts, and false confidence is worse than ugly
            // output in reverse engineering.
            Self::MissingAbiArg => out.push_str("__abi"),
            Self::Unknown => out.push_str("__unknown"),
            Self::MethodGroup(name) => {
                out.push_str("MethodInfo(\"");
                out.push_str(name);
                out.push_str("\")");
            }
            Self::Bin(op, a, b) => {
                let p = self.precedence();
                parenthesize(out, p, a.precedence(), |out| a.render(out));
                out.push(' ');
                out.push_str(op.token());
                out.push(' ');
                parenthesize(out, p, b.precedence(), |out| b.render(out));
            }
            Self::Cmp(op, a, b) => {
                let p = self.precedence();
                parenthesize(out, p, a.precedence(), |out| a.render(out));
                out.push(' ');
                out.push_str(op.token());
                out.push(' ');
                parenthesize(out, p, b.precedence(), |out| b.render(out));
            }
            Self::Not(e) => {
                out.push('!');
                parenthesize(out, self.precedence(), e.precedence(), |out| e.render(out));
            }
            Self::Neg(e) => {
                out.push('-');
                parenthesize(out, self.precedence(), e.precedence(), |out| e.render(out));
            }
            Self::Field(recv, name) => {
                // `.` is a postfix operator: a receiver that is itself a binary
                // expression has to be parenthesised or the field binds to the
                // operator's right operand — `this + 40.LinePos` reads as
                // `this + (40.LinePos)`. `prettify_call` folds a `get_X()` into
                // a Field on ANY receiver, including an address computation, so
                // this is reachable for every property read on a derived
                // receiver.
                parenthesize(out, 15, recv.precedence(), |out| recv.render(out));
                out.push('.');
                out.push_str(name);
            }
            Self::StaticField(ty, name) => {
                out.push_str(ty);
                out.push('.');
                out.push_str(name);
            }
            Self::RawDeref(base, addend, width) => {
                // The ACCESS WIDTH the native instruction performs. A `ldrb`
                // reads ONE byte; rendering it as `*(nint*)` claims an
                // eight-byte access to a one-byte slot, and the same for a
                // `strb` store — which would also clobber the seven following
                // fields. `0` stays `nint` (pointer width / not stated).
                let ty = match width {
                    1 => "byte",
                    2 => "short",
                    4 => "int",
                    _ => "nint",
                };
                out.push_str("*(");
                out.push_str(ty);
                out.push_str("*)(");
                // An ABSOLUTE operand is a single address, so it is rendered as
                // one `0x…` literal rather than `base +off`. The `Int(0)` base
                // is produced in exactly one place — `mem_base_expr` recognising
                // the `abs`/`rip` pseudo-local — so this can never reinterpret a
                // computed pointer, and a base that is a real constant `k` still
                // folds normally through the ` addend` path.
                if matches!(**base, Self::Int(0)) {
                    out.push_str(&format!("{:#x}", *addend as u64));
                } else {
                    base.render(out);
                    if *addend != 0 {
                        out.push_str(&format!(" {addend:+}"));
                    }
                }
                out.push(')');
            }
            Self::RawAddr(base, addend) => {
                out.push_str("(nint*)(");
                base.render(out);
                if *addend != 0 {
                    out.push_str(&format!(" {addend:+}"));
                }
                out.push(')');
            }
            Self::AddressOf(inner) => {
                out.push('&');
                parenthesize(out, 12, inner.precedence(), |out| inner.render(out));
            }
            Self::RawCall(addr, args, ret) => {
                // A call into the IL2CPP runtime itself (the static-init guard
                // and the dispatch trampolines) has no metadata name to find —
                // it is the runtime's own `.text`, not a managed method. Where
                // the call-site shape proves what it does, name it as a runtime
                // call; otherwise fall back to `fn_<address>`.
                let name = super::runtime_helpers::display_name(*addr);
                match ret {
                    // metadata-resolved return type: render a typed cast so
                    // the call site is self-describing
                    Some(t) => {
                        out.push_str(&format!("(({t}){name}("));
                        render_args(args, out);
                        out.push_str("))");
                    }
                    None => {
                        out.push_str(&format!("{name}("));
                        render_args(args, out);
                        out.push(')');
                    }
                }
            }
            Self::Cond(c, a, b) => {
                parenthesize(out, 4, c.precedence(), |out| c.render(out));
                out.push_str(" ? ");
                parenthesize(out, 4, a.precedence(), |out| a.render(out));
                out.push_str(" : ");
                parenthesize(out, 4, b.precedence(), |out| b.render(out));
            }
            Self::Call { recv, name, args } => {
                match recv {
                    Some(r) => {
                        // receiver may need parens if not a simple leaf
                        let p = r.precedence();
                        parenthesize(out, 15, p, |out| r.render(out));
                    }
                    None => out.push_str(name),
                }
                if recv.is_some() {
                    out.push('.');
                    out.push_str(name);
                }
                out.push('(');
                render_args(args, out);
                out.push(')');
            }
            Self::NewObj(ty, args) => {
                out.push_str("new ");
                out.push_str(ty);
                out.push('(');
                render_args(args, out);
                out.push(')');
            }
            Self::NewArr(elem, dims) => {
                out.push_str("new ");
                out.push_str(elem);
                out.push('[');
                // C# arrays: first dimension carries the length, the rest
                // render as empty commas (multi-dim is rare in lifted IR).
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if i == 0 {
                        d.render(out);
                    }
                }
                out.push(']');
            }
            Self::Cast(t, e) => {
                out.push_str(&format!("({t})"));
                parenthesize(out, 14, e.precedence(), |out| e.render(out));
            }
            Self::Await(e) => {
                out.push_str("await ");
                parenthesize(out, 2, e.precedence(), |out| e.render(out));
            }
            Self::Lambda { params, body } => {
                out.push('(');
                out.push_str(&params.join(", "));
                out.push_str(") => ");
                body.render(out);
            }
        }
    }
}

/// Stable numeric identity for a register name (frame-slot keying).
/// GPRs x0–x30 → 0–30, xzr/sp → 31/32, FP v?N → 100+N.
fn slot_index(name: &str) -> Option<u32> {
    if name.starts_with('v') && name.len() >= 3 {
        return name[1..].parse::<u32>().ok().map(|n| 100 + n);
    }
    if name.len() >= 2 && name.starts_with('x') {
        return name[1..].parse::<u32>().ok();
    }
    match name {
        "xzr" => Some(31),
        "sp" => Some(32),
        _ => None,
    }
}

fn render_args(args: &[Expr], out: &mut String) {
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        a.render(out);
    }
}

fn parenthesize(out: &mut String, parent: u8, child: u8, f: impl FnOnce(&mut String)) {
    if child < parent {
        out.push('(');
        f(out);
        out.push(')');
    } else {
        f(out);
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Stmt {
    /// Expression evaluated for side effects (calls).
    Eval(Expr),
    Assign(String, Option<String> /*type decl on first assign*/, Expr),
    If {
        cond: Expr,
        then_body: Vec<Stmt>,
        else_body: Vec<Stmt>,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
    },
    #[allow(dead_code)]
    ForEach {
        var_name: String,
        collection: String,
        body: Vec<Stmt>,
    },
    DoWhile {
        body: Vec<Stmt>,
        cond: Expr,
    },
    /// `for (init; cond; incr) { body }`.
    ///
    /// Reconstructed after the structuring passes from the init / condition /
    /// increment / back-edge shape that reaches the structurer as a flat
    /// `init; while (cond) { body; incr; }`. `init` and `incr` are statement
    /// lists (the counter declaration / update), rendered inline in the header;
    /// `cond` is `None` for an infinite `for (;;)`.
    For {
        init: Vec<Stmt>,
        cond: Option<Expr>,
        incr: Vec<Stmt>,
        body: Vec<Stmt>,
    },
    /// `using (decl) { body }` — resource acquisition + `try/finally { Dispose }`.
    Using {
        decl: Box<Stmt>,
        body: Vec<Stmt>,
    },
    /// `lock (expr) { body }` — `Monitor.Enter`/`Exit`, i.e. a `try/finally`.
    Lock {
        expr: Expr,
        body: Vec<Stmt>,
    },
    /// `yield return expr;` — an iterator/coroutine state machine's suspension
    /// point, recovered from `<>1__state`/`MoveNext` compiler machinery.
    YieldReturn(Expr),
    /// `yield break;` — an iterator's early termination.
    YieldBreak,
    Return(Option<Expr>),
    Throw(Expr),
    Switch {
        disc: String,
        /// (case value, arm body). Rendered INLINE under `case N:`. A case that
        /// shares a block with an earlier case, or whose block was already
        /// emitted before the switch, carries a single `Stmt::Goto` to a label
        /// defined in `tail` instead.
        cases: Vec<(u64, Vec<Stmt>)>,
        /// Labelled landing pads for shared / already-emitted case blocks,
        /// emitted after the switch body.
        tail: Vec<Stmt>,
    },
    Break,
    Continue,
    Goto(String),
    /// Jump to a case section of the lexically enclosing switch.
    ///
    /// Produced only when a goto inside a case arm targets the first label of
    /// another case in that same switch. Other CFG labels remain ordinary
    /// gotos because their scope may extend beyond the switch.
    GotoCase(u64),
    Label(String),
    Comment(String),
    /// Recovered exception region (from the binary's EH tables — exact data).
    /// `try_body` wraps the statements whose native range lies inside the
    /// try region; each catch clause carries its source name from the LSDA
    /// type table (`Il2CppExceptionWrapper` = managed catch shim; `...` =
    /// catch-all; `/* filter */` = exception specification). `finally_body`
    /// is `Some` for cleanup regions.
    Try {
        try_body: Vec<Stmt>,
        catches: Vec<(String, Vec<Stmt>)>,
        finally_body: Option<Vec<Stmt>>,
    },
}

struct Renderer {
    out: String,
    indent: usize,
    /// Set by [`Renderer::else_block`] while it renders an else arm that is
    /// exactly one `if`. The `if` renderer sees the flag and writes
    /// `else if (…)` on the line that just closed the previous block instead
    /// of starting a fresh `if` line — C#'s own spelling for the same program.
    pending_else_if: bool,
}

/// Render a single statement in an INLINE (header) position.
///
/// `for`'s init/increment and `using`'s resource declaration are statements in
/// the AST but must read as expressions in the emitted header (`int i = 0`,
/// `i++`, `var r = Open()`), not as braced lines. Returns `None` for any
/// statement that has no inline form, so a recognizer can never smuggle a block
/// into a header.
fn stmt_inline(s: &Stmt) -> Option<String> {
    match s {
        Stmt::Assign(name, ty, e) => {
            let mut b = String::new();
            e.render(&mut b);
            Some(match ty {
                Some(t) => format!("{t} {name} = {b}"),
                None => format!("{name} = {b}"),
            })
        }
        Stmt::Eval(e) => {
            let mut b = String::new();
            e.render(&mut b);
            Some(b)
        }
        _ => None,
    }
}

impl Renderer {
    fn new() -> Self {
        Self { out: String::new(), indent: 0, pending_else_if: false }
    }

    /// Emit an `if (…)` header, honouring a pending `else`.
    ///
    /// Called from every site that would otherwise write a fresh `if` line, so
    /// the `else if` continuation is decided in ONE place.
    fn if_header(&mut self, text: &str) {
        if self.pending_else_if {
            self.pending_else_if = false;
            self.out.push_str(&format!(" else {text}\n"));
        } else {
            self.line(text);
        }
    }

    /// Emit an `else` clause: `} else if (…) { … }` when the arm is exactly one
    /// `if` (the same program with one brace level fewer — the compiler's test
    /// chains arrive as nested `if`s and rendered as a staircase of `}  else  {
    ///   if (…) {`), otherwise a plain `} else { … }`.
    ///
    /// The newline `block()` just wrote is dropped first so `else` continues on
    /// the closing brace's line. Previously the indent of a fresh line was then
    /// written into that gap, so every `else` in the corpus rendered as
    /// `}    else`.
    fn else_block(&mut self, body: &[Stmt]) {
        if body.len() == 1 && matches!(body[0], Stmt::If { .. }) {
            self.out.truncate(self.out.trim_end().len());
            let mark = self.out.len();
            self.pending_else_if = true;
            self.stmt(&body[0]);
            if self.pending_else_if {
                // The inner `if` took one of the early paths (a folded constant
                // condition, or a label-only arm) and so never consulted the
                // marker. Throw its output away and render the plain form.
                self.pending_else_if = false;
                self.out.truncate(mark);
                self.out.push_str(" else\n");
                self.block(body);
            }
            return;
        }
        self.out.truncate(self.out.trim_end().len());
        self.out.push_str(" else\n");
        self.block(body);
    }

    fn line(&mut self, s: impl AsRef<str>) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    fn stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Eval(e) => {
                let mut buf = String::new();
                e.render(&mut buf);
                self.line(format!("{buf};"));
            }
            Stmt::Assign(name, ty, e) => {
                let mut buf = String::new();
                e.render(&mut buf);
                match ty {
                    Some(t) => self.line(format!("{t} {name} = {buf};")),
                    None => self.line(format!("{name} = {buf};")),
                }
            }
            Stmt::If { cond, then_body, else_body } => {
                // constant-condition hygiene: never emit `if (true)` /
                // `if (false)` / `if (!true)` wrappers — fold to the live
                // arm. `Bool(true)` doubles as the branch-condition fallback
                // for unrecovered guards, so a negated constant
                // (`invert(Bool(true))` = `!true`) is the same fallback — a
                // `Not` around a constant only exists from an arm swap of
                // that fallback (never from real folded logic), so it
                // cancels: the payload arm is what the structurer judged
                // reachable.
                let folded_bool = match cond {
                    Expr::Bool(b) => Some(*b),
                    Expr::Not(inner) => match **inner {
                        Expr::Bool(b) => Some(b),
                        _ => None,
                    },
                    _ => None,
                };
                if std::env::var("ILR_CONDBG").is_ok() {
                    eprintln!("[cond] {:?} then={} else={} fold={:?}", cond, then_body.len(), else_body.len(), folded_bool);
                }
                // an else-arm with nothing renderable (empty/label/comment
                // only) carries no semantics — the renderer drops it here so
                // the C# never shows `else { }` shells.
                let else_renders = else_body
                    .iter()
                    .any(|s| !matches!(s, Stmt::Comment(_) | Stmt::Label(_)));
                if !else_renders {
                    let then_renders = then_body
                        .iter()
                        .any(|s| !matches!(s, Stmt::Comment(_) | Stmt::Label(_)));
                    if !then_renders {
                        // Both arms are label/comment-only. The condition gates
                        // nothing renderable, so the `if` shell goes away — BUT
                        // the arms may DEFINE jump labels that gotos elsewhere
                        // still target. Returning outright deleted those
                        // definitions and emitted `goto L<missing>` (the
                        // dangling-jump-target class). Emit the labels alone.
                        self.labels_only(then_body);
                        self.labels_only(else_body);
                        return;
                    }
                    // constant fallback / real fold: emit the live arm with
                    // no wrapper at all (both this path and the general one
                    // below must fold — a guard that renders `!true` was
                    // hitting the early return and skipping the fold).
                    // Emitted flat (no braces): a folded `if` must not leave a
                    // stray `{ … }` block statement behind.
                    match folded_bool {
                        Some(true) => {
                            for s in then_body {
                                self.stmt(s);
                            }
                            self.labels_only(else_body);
                            return;
                        }
                        Some(false) => {
                            self.labels_only(then_body);
                            self.labels_only(else_body);
                            return;
                        }
                        None => {}
                    }
                    let mut buf = String::new();
                    cond.render(&mut buf);
                    self.if_header(&format!("if ({buf})"));
                    self.block(then_body);
                    // The else arm renders nothing, but it may still define
                    // labels — emitted here so no goto is left dangling.
                    self.labels_only(else_body);
                    return;
                }
                if then_body.is_empty() && else_body.is_empty() {
                    return;
                }
                // Labels are goto targets — they count as CONTENT when they
                // live in the (otherwise empty) arm: `if (c) {} else { L1: … }`
                // must render as `if (!c) { L1: … }`, not `if (c) {} else …`.
                let then_content: Vec<&Stmt> = then_body
                    .iter()
                    .filter(|s| !matches!(s, Stmt::Comment(_)))
                    .collect();
                let else_content: Vec<&Stmt> = else_body
                    .iter()
                    .filter(|s| !matches!(s, Stmt::Comment(_)))
                    .collect();
                match folded_bool {
                    // `true` / `!true` → then arm; `false` / `!false` → else
                    // (flat emission: no stray block wrapper). The deferred arm
                    // is never taken, but it may still DEFINE labels.
                    Some(true) => {
                        for s in then_body {
                            self.stmt(s);
                        }
                        self.labels_only(else_body);
                    }
                    Some(false) => {
                        for s in else_body {
                            self.stmt(s);
                        }
                        self.labels_only(then_body);
                    }
                    None => {
                        let mut buf = String::new();
                        cond.render(&mut buf);
                        if std::env::var("ILR_IFDBG").is_ok() {
                            eprintln!("[rend] cond={buf} then_raw={} else_raw={} then_content={} else_content={}", then_body.len(), else_body.len(), then_content.len(), else_content.len());
                        }
                        if then_content.is_empty() && !else_content.is_empty() {
                            self.if_header(&format!("if (!({buf}))"));
                            self.block(else_body);
                        } else {
                            self.if_header(&format!("if ({buf})"));
                            self.block(then_body);
                            if !else_content.is_empty() {
                                self.else_block(else_body);
                            }
                        }
                    }
                }
            }
            Stmt::While { cond, body } => {
                if matches!(cond, Expr::Bool(false))
                    || matches!(cond, Expr::Not(inner)
                        if matches!(**inner, Expr::Bool(true)))
                {
                    // A `while (false)` body never runs, so it is discarded —
                    // but it may still DEFINE labels that gotos target.
                    self.labels_only(body);
                    return;
                }
                let mut buf = String::new();
                cond.render(&mut buf);
                self.line(format!("while ({buf})"));
                self.block(body);
            }
            Stmt::ForEach { var_name, collection, body } => {
                self.line(format!("foreach (var {var_name} in {collection})"));
                self.block(body);
            }
            Stmt::DoWhile { body, cond } => {
                let mut buf = String::new();
                cond.render(&mut buf);
                self.line("do");
                self.block(body);
                // `while (…);` continues on the closing brace's line. The
                // newline `block()` just wrote is dropped, and the tail is
                // appended WITHOUT `line()`'s indent — otherwise every
                // do-while in the corpus rendered as `}        while (…)`,
                // the same defect `else_block` fixes for `else`.
                self.out.truncate(self.out.trim_end().len());
                self.out.push_str(&format!(" while ({buf});\n"));
            }
            Stmt::For { init, cond, incr, body } => {
                let join = |list: &[Stmt]| -> String {
                    list.iter()
                        .filter_map(stmt_inline)
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let mut buf = String::new();
                if let Some(c) = cond {
                    c.render(&mut buf);
                }
                self.line(format!(
                    "for ({}; {}; {})",
                    join(init),
                    buf,
                    join(incr)
                ));
                self.block(body);
            }
            Stmt::Using { decl, body } => {
                let text = stmt_inline(decl).unwrap_or_default();
                self.line(format!("using ({text})"));
                self.block(body);
            }
            Stmt::Lock { expr, body } => {
                let mut buf = String::new();
                expr.render(&mut buf);
                self.line(format!("lock ({buf})"));
                self.block(body);
            }
            Stmt::YieldReturn(e) => {
                let mut buf = String::new();
                e.render(&mut buf);
                self.line(format!("yield return {buf};"));
            }
            Stmt::YieldBreak => self.line("yield break;"),
            Stmt::Return(e) => match e {
                Some(e) => {
                    let mut buf = String::new();
                    e.render(&mut buf);
                    self.line(format!("return {buf};"));
                }
                None => self.line("return;"),
            },
            Stmt::Throw(e) => {
                let mut buf = String::new();
                e.render(&mut buf);
                self.line(format!("throw {buf};"));
            }
            Stmt::Switch { disc, cases, tail } => {
                self.line(format!("switch ({disc}) {{"));
                self.indent += 1;
                for (v, body) in cases {
                    self.line(format!("case {v}:"));
                    self.indent += 1;
                    for s in body {
                        self.stmt(s);
                    }
                    // A C# switch arm does not fall through: terminate it unless
                    // the body already does (return/throw/goto/break/continue).
                    if !ends_with_terminator(body) {
                        self.line("break;");
                    }
                    self.indent -= 1;
                }
                self.indent -= 1;
                self.line("}");
                for s in tail {
                    self.stmt(s);
                }
            }
            Stmt::Break => self.line("break;"),
            Stmt::Continue => self.line("continue;"),
            Stmt::Goto(l) => self.line(format!("goto {l};")),
            Stmt::GotoCase(v) => self.line(format!("goto case {v};")),
            Stmt::Label(l) => self.line(format!("{l}:")),
            Stmt::Comment(c) => self.line(format!("// {c}")),
            Stmt::Try { try_body, catches, finally_body } => {
                self.line("try");
                self.block(try_body);
                for (name, body) in catches {
                    self.line(format!("catch ({name})"));
                    self.block(body);
                }
                if let Some(f) = finally_body {
                    self.line("finally");
                    self.block(f);
                }
            }
        }
    }

    fn block(&mut self, body: &[Stmt]) {
        // Braces are mandatory: without them only the first statement of a
        // multi-statement branch visually belongs to the `if`/`while`.
        self.line("{");
        self.indent += 1;
        for s in body {
            self.stmt(s);
        }
        self.indent -= 1;
        self.line("}");
    }

    /// Emit just the `Label` statements of a body the renderer is about to
    /// DISCARD.
    ///
    /// Dropping a semantic arm is the renderer's job (`if (c) {}` / `else {}`
    /// shells carry nothing), but that arm can still DEFINE jump labels that
    /// `goto`s elsewhere target. Discarding it outright deleted the definition
    /// from the text while the goto stayed, printing `goto L<missing>` — the
    /// unreachable / malformed-jump-target class. Measured on Assembly-CSharp:
    /// this is the last remaining source of dangling targets (`Toggle` @
    /// 0x19BD7C emitted `goto L1;` with no `L1:` anywhere).
    ///
    /// A bare `Lx:` is a no-op that makes the *following* statement the jump
    /// target, which is exactly the shared-tail merge the label stands for.
    /// Recursive because a discarded arm's nested labels are discarded too.
    fn labels_only(&mut self, body: &[Stmt]) {
        for s in body {
            match s {
                Stmt::Label(_) => self.stmt(s),
                Stmt::If { then_body, else_body, .. } => {
                    self.labels_only(then_body);
                    self.labels_only(else_body);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => self.labels_only(body),
                Stmt::Try { try_body, catches, finally_body } => {
                    self.labels_only(try_body);
                    for (_, b) in catches {
                        self.labels_only(b);
                    }
                    if let Some(f) = finally_body {
                        self.labels_only(f);
                    }
                }
                _ => {}
            }
        }
    }
}


fn short_type(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

fn csharp_type(t: &ManagedType) -> Option<String> {
    Some(match t {
        ManagedType::Primitive(p) => match p {
            PrimKind::Void => return None,
            PrimKind::Boolean => "bool".into(),
            PrimKind::Char => "char".into(),
            PrimKind::I1 => "sbyte".into(),
            PrimKind::U1 => "byte".into(),
            PrimKind::I2 => "short".into(),
            PrimKind::U2 => "ushort".into(),
            PrimKind::I4 => "int".into(),
            PrimKind::U4 => "uint".into(),
            PrimKind::I8 => "long".into(),
            PrimKind::U8 => "ulong".into(),
            PrimKind::R4 => "float".into(),
            PrimKind::R8 => "double".into(),
            PrimKind::I => "nint".into(),
            PrimKind::U => "nuint".into(),
        },
        ManagedType::Opaque(name) => {
            // resolved generic display names (e.g. "List<int>") flow through
            let cleaned = name.replace('`', "");
            let ok = cleaned
                .chars()
                .all(|c| c.is_alphanumeric() || matches!(c, '<' | '>' | '.' | ',' | '[' | ']' | '_' | ' '))
                && !cleaned.is_empty();
            if ok {
                cleaned.trim().to_string()
            } else {
                "object".into()
            }
        }
        ManagedType::String => "string".into(),
        ManagedType::IntPtr => "nint".into(),
        ManagedType::Object => "object".into(),
        ManagedType::Class(nt) => nt.name.clone(),
        ManagedType::SzArray(e) => format!("{}[]", csharp_type(e)?),
        ManagedType::ByRef(_) | ManagedType::Pointer(_) => "void*".into(),
        ManagedType::Unknown => "object".into(),
    })
}

fn bin_of(k: ArithKind) -> Option<BinOp> {
    Some(match k {
        ArithKind::Add => BinOp::Add,
        ArithKind::Sub => BinOp::Sub,
        ArithKind::Mul => BinOp::Mul,
        ArithKind::Div => BinOp::Div,
        ArithKind::Rem => BinOp::Rem,
        ArithKind::And => BinOp::And,
        ArithKind::Or => BinOp::Or,
        ArithKind::Xor => BinOp::Xor,
        ArithKind::Shl => BinOp::Shl,
        ArithKind::Shr => BinOp::Shr,
        // A rotate has no C# operator and is NOT an OR. This used to return
        // `BinOp::Or`, so `ror x0, x1, x2` rendered `x1 | x2` — plausible text,
        // wrong logic, exactly the class the port is trying to eliminate.
        // `None` forces the caller to render the explicit pseudo-op, so the
        // approximation is no longer representable.
        ArithKind::Rol | ArithKind::Ror => return None,
    })
}

fn bin_of_float(k: FloatArithKind) -> BinOp {
    match k {
        FloatArithKind::Add => BinOp::Add,
        FloatArithKind::Sub => BinOp::Sub,
        FloatArithKind::Mul => BinOp::Mul,
        FloatArithKind::Div => BinOp::Div,
        FloatArithKind::Rem => BinOp::Rem,
        FloatArithKind::Neg => BinOp::Sub,
        // unary intrinsics — def_expr routes them before this helper ever
        // runs, so these arms are unreachable (kept only for totality)
        FloatArithKind::Sqrt | FloatArithKind::Abs | FloatArithKind::Floor
        | FloatArithKind::Ceiling | FloatArithKind::Truncate | FloatArithKind::Round => BinOp::Rem,
    }
}

fn cmp_of(k: CompareKind) -> CmpKind {
    match k {
        CompareKind::Equal => CmpKind::Eq,
        CompareKind::NotEqual => CmpKind::Ne,
        CompareKind::LessThan => CmpKind::Lt,
        CompareKind::GreaterThan => CmpKind::Gt,
        CompareKind::LessOrEqual => CmpKind::Le,
        CompareKind::GreaterOrEqual => CmpKind::Ge,
    }
}

fn invert(e: &Expr) -> Expr {
    match e {
        Expr::Cmp(k, a, b) => Expr::Cmp(k.invert(), a.clone(), b.clone()),
        Expr::Not(inner) => (**inner).clone(),
        other => Expr::Not(Box::new(other.clone())),
    }
}


// ---------------------------------------------------------------------------
// Method context: global def-map + lazy expression inlining
// ---------------------------------------------------------------------------

/// After out-of-SSA most locals have exactly ONE definition, so expressions
/// can be resolved lazily through the def chain (Cpp2IL/ILSpy style value
/// inlining). Multi-def locals degrade to ordinary assigned variables.
struct MethodCtx<'a> {
    insns: &'a [Insn],
    table: &'a LocalTable,
    declaring_tdi: i32,
    /// Declaring type's base-class chain (self first). An instance field
    /// declared on ANY ancestor is still `this.field` from this method.
    parent_chain: &'a HashMap<i32, Vec<i32>>,
    /// local version id → defining instruction index (LAST def in linear
    /// order; only meaningful when `multi_def` doesn't contain the version)
    defs: HashMap<u32, usize>,
    multi_def: HashSet<u32>,
    /// ALL defining instruction indices per version (for copy-alias
    /// resolution: phi-elimination leaves identical `v←w` copies at several
    /// merges — when every def of a version copies the same source, the
    /// version is a pure alias of that source regardless of path).
    def_sites: HashMap<u32, Vec<usize>>,
    /// REACHING DEFINITIONS (Ghidra-style local def-use): for each
    /// instruction index, the map `local version → defining instruction
    /// index` valid immediately BEFORE that instruction. Computed by a
    /// block-level GEN/KILL fixpoint over the CFG; uses inside a block see
    /// the in-block def when one dominates the use position, otherwise the
    /// block's IN set. This is what turns `if (v1 == 0)` shells into
    /// `if (this.transform == null)` — the use site links to the exact
    /// def that reaches it instead of the last linear def.
    reaching: Vec<HashMap<u32, usize>>,
    /// Locals genuinely READ as plain variables (a use where no single def
    /// reaches — merge/back-edge join). Every def of such a local must be
    /// materialized as an assignment statement or its value silently
    /// vanishes (empty `while` bodies full of pure arithmetic).
    var_reads: HashSet<u32>,
    /// Every local READ anywhere in the method (destination positions
    /// excluded). Distinct from `var_reads`, which only covers *merge/entry*
    /// reads of multi-def locals: this one answers "is this value consumed at
    /// all", which is what decides whether a call whose result is discarded
    /// must still be emitted as a statement.
    read_locals: HashSet<u32>,
    /// Local version → how many times it is read. See [`compute_read_counts`].
    read_count: HashMap<u32, usize>,
    /// Single-def locals read more than once whose inlined expansion is large
    /// enough that copying it at every use would blow the expression up (see
    /// [`SHARED_EXPR_NODES`]). Such a local is emitted as a named temporary and
    /// referenced, never re-inlined. Also unioned into `var_reads` so the
    /// defining assignment actually reaches the text.
    shared_large: HashSet<u32>,
    /// Call def instruction → the name its result was materialized to
    /// (`dst = callee(…)`). Populated as the walk emits statements, so a use
    /// site references the name only when the binding provably exists. See
    /// [`Self::note_call_bound`].
    bound_calls: std::cell::RefCell<HashMap<usize, String>>,
    /// Call def instruction → the name its result WILL be materialized to.
    ///
    /// Same meaning as `bound_calls`, but computed by a DISCOVERY walk that
    /// runs before the real one, so a use site rendered to the LEFT of its
    /// definition (the walk is not in execution order) can reference the
    /// name the def site is about to create instead of re-inlining the call
    /// expression. `bound_calls` alone only knows about defs already walked,
    /// which is exactly why a call like `Mathf.NextPowerOfTwo` — 2 native
    /// `bl`, 11 rendered call expressions on `SetArraySizes` @0x22a5eb8 —
    /// appeared once per use.
    preseed_bindings: std::cell::RefCell<HashMap<usize, String>>,
    /// insn index → block index (for reaching-def block lookup)
    block_of_insn: Option<Vec<usize>>,
    /// (start, end) per block, parallel to `reaching`
    blocks_range: Vec<(usize, usize)>,
    /// cap on inlining chain depth per expression (prevents blowup on
    /// pathological def chains). See `MAX_INLINE_DEPTH`.
    max_inline_depth: usize,
    /// method returns void: a native `ret`'s x0 is stale noise, never a real
    /// return value, so `return <expr>` must become a bare `return;`.
    ret_is_void: bool,
    /// Render cursor: the instruction whose operands are being resolved
    /// right now. `operand_expr` has no use-site parameter (40+ call sites),
    /// so the statement renderer sets this once per insn and lazy local
    /// resolution reads it — each use links to its REACHING def, which is
    /// how `if (v1 == 0)` shells become `if (this.transform == null)`.
    /// Def-expr inlining inside a def's own value must restore the cursor,
    /// otherwise the def's operands would resolve against the USE site.
    cur_insn: std::cell::Cell<usize>,
    /// Fidelity audit (ILR_AUDIT=path): every instruction index whose VALUE
    /// was consumed by the emitted C# (inlined into an expression via
    /// `def_expr`, or emitted as its own statement via `boundary_stmts`).
    /// Unclaimed non-control instructions are logic the renderer dropped —
    /// counted per method so fidelity is measured, not assumed.
    audit: std::cell::RefCell<std::collections::HashSet<usize>>,
    /// Claim log: insn indices in the ORDER `audit_claim` saw them. Lets the
    /// structurer range-slice the claims produced while building ONE
    /// expression (a branch condition) and revoke them if the statement that
    /// was meant to carry that expression never reaches the output.
    claim_log: std::cell::RefCell<Vec<usize>>,
    /// `ILR_AUDIT_STRICT=1` is the ONLY reader of [`Self::claim_log`] (the
    /// structurer range-slices it to revoke the claims of a dropped `if`
    /// shell). Read once per method instead of per claim so the hot
    /// [`Self::audit_claim`] path can skip the log entirely by default.
    audit_strict: bool,
    /// `ILR_RSSDBG=1` — per-method call-count/depth instrumentation.
    rss_dbg: bool,
    /// Locals whose def is currently being resolved on this stack.
    ///
    /// `inline_local_at` already refuses to inline a def into ITS OWN use
    /// site (`i != use_insn`), but that only catches the direct
    /// self-reference. A LOOP-CARRIED value is a cycle of length ≥ 2 and slips
    /// through: arm64 @0x2495410 has `v35 = v90` (i=127) and
    /// `v90 = select(cond, v88, v35)` (i=144), so resolving `v35` resolves
    /// `v90`, which resolves `v35` again, … The depth guard ends it at 65
    /// levels, but every level branches (the `select` has two value operands),
    /// so the expansion is 2^65 nodes: that one 336-byte method made the
    /// renderer call `def_expr` 22 million times in twelve seconds and grow to
    /// multiple GB before the dump was killed. A def that depends on itself is
    /// a PHI and must be referenced by name, not expanded.
    inline_in_progress: std::cell::RefCell<std::collections::HashSet<u32>>,
    /// Nesting depth of [`MethodCtx::inline_local`]. Zero means the next call
    /// starts a NEW expression, which is when the budget is refilled.
    inline_stack: std::cell::Cell<usize>,
    /// Defs left in the current expression's inlining budget.
    inline_budget: std::cell::Cell<usize>,
    /// Claims made while building an expression that was then DISCARDED —
    /// e.g. a branch condition whose `if` shell folded to nothing. The
    /// fidelity audit subtracts these. Without it `sem_un` reads 0 for
    /// methods that visibly lose whole basic blocks, because claims were
    /// recorded at expression-BUILD time rather than at emission time.
    dropped_claims: std::cell::RefCell<std::collections::HashSet<usize>>,
    /// Call instructions the renderer presented as a MEMBER ACCESS or operator
    /// rather than as `callee(…)` — `obj.get_X()` → `obj.X`, `String.Concat`
    /// → `+`, and the setter counterpart. See [`Self::prettify_call_at`].
    /// Consumed at the end of rendering to ledger a
    /// [`super::verify::LedgerKind::PropertyAccessorCall`] per site that
    /// actually survived into the output (i.e. is in `audit` and not in
    /// `dropped_claims`), so the concordance does not charge a faithfully
    /// presented call as a missing one.
    property_fold_sites: std::cell::RefCell<std::collections::HashSet<usize>>,
}

impl<'a> MethodCtx<'a> {
    /// Is `tdi` the declaring type or one of its ancestors?
    fn in_declaring_chain(&self, tdi: i32) -> bool {
        if tdi == self.declaring_tdi {
            return true;
        }
        self.parent_chain
            .get(&self.declaring_tdi)
            .is_some_and(|chain| chain.contains(&tdi))
    }
}

impl<'a> MethodCtx<'a> {
    fn build(
        insns: &'a [Insn],
        table: &'a LocalTable,
        declaring_tdi: i32,
        parent_chain: &'a HashMap<i32, Vec<i32>>,
        ret_is_void: bool,
    ) -> Self {
        let mut defs: HashMap<u32, usize> = HashMap::new();
        let mut multi_def: HashSet<u32> = HashSet::new();
        let mut def_sites: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, insn) in insns.iter().enumerate() {
            if let Some(Some(d)) = insn.dest().map(|d| d.as_local()) {
                if defs.insert(d, i).is_some() {
                    multi_def.insert(d);
                }
                def_sites.entry(d).or_default().push(i);
            }
        }
        let reaching = compute_reaching_definitions(insns);
        let blocks_range: Vec<(usize, usize)> = reaching
            .blocks
            .iter()
            .map(|b| (b.0, b.1))
            .collect();
        let mut block_of_insn = vec![usize::MAX; insns.len()];
        for (bi, &(s, e)) in blocks_range.iter().enumerate() {
            for i in s..e {
                if i < block_of_insn.len() {
                    block_of_insn[i] = bi;
                }
            }
        }
        // Ghidra's `isAddrForce()` census: the population our IR can express for
        // "this value must live at an address". The proposed fix for the
        // duplicate/never-read bindings was a port of `RulePropagateCopy`'s
        // refusal to propagate such a value, and a rule with no population is a
        // no-op — so the population is a number on the loud path, not a guess.
        for insn in insns {
            for op in &insn.operands {
                count_address_escapes(op);
            }
        }
        let mut var_reads = compute_var_reads(
            insns,
            &multi_def,
            &reaching.gen_kill_in,
            &blocks_range,
            &block_of_insn,
            &def_sites,
        );
        let read_count = compute_read_counts(insns);
        let read_locals: HashSet<u32> = read_count.keys().copied().collect();
        // A shared large value must be MATERIALISED, so it joins `var_reads`
        // (the set every emission predicate keys on) as well as being tracked
        // separately for the inline refusal in `inline_local_inner`.
        let shared_large =
            compute_shared_large(insns, &defs, &multi_def, &read_count, SHARED_EXPR_NODES);
        var_reads.extend(shared_large.iter().copied());
        let mut ctx = Self {
            insns,
            table,
            declaring_tdi,
            parent_chain,
            defs,
            multi_def,
            def_sites,
            reaching: reaching.gen_kill_in,
            var_reads,
            read_locals,
            read_count,
            shared_large,
            bound_calls: std::cell::RefCell::new(HashMap::new()),
            preseed_bindings: std::cell::RefCell::new(HashMap::new()),
            block_of_insn: Some(block_of_insn),
            blocks_range,
            max_inline_depth: MAX_INLINE_DEPTH,
            ret_is_void,
            // MAX = "no use-site context": legacy single/multi-def behavior.
            // Renderers that set the cursor per-insn override it.
            cur_insn: std::cell::Cell::new(usize::MAX),
            audit: std::cell::RefCell::new(std::collections::HashSet::new()),
            claim_log: std::cell::RefCell::new(Vec::new()),
            audit_strict: std::env::var("ILR_AUDIT_STRICT").is_ok(),
            rss_dbg: std::env::var("ILR_RSSDBG").is_ok(),
            inline_in_progress: std::cell::RefCell::new(std::collections::HashSet::new()),
            inline_stack: std::cell::Cell::new(0),
            inline_budget: std::cell::Cell::new(MAX_INLINE_EXPR_NODES),
            dropped_claims: std::cell::RefCell::new(std::collections::HashSet::new()),
            property_fold_sites: std::cell::RefCell::new(std::collections::HashSet::new()),
        };
        // A use site that CANNOT inline its def reads the local by name, so the
        // def must reach the text as an assignment. `compute_var_reads` cannot
        // see those refusals (they depend on the use site and on the def's own
        // expression), so the set is completed here with the exact predicates
        // `inline_local_at` applies. This can only ADD locals, never remove one.
        ctx.var_reads.extend(ctx.refused_inline_reads());
        ctx
    }

    /// Display name for an INTERNAL local id that has no variable-table entry.
    ///
    /// Three families of id reach the output and none may leak as a raw
    /// sentinel:
    ///
    /// * `u32::MAX - 1` — the shared return-value slot;
    /// * `u32::MAX - 2`, `- 3`, … — the lifter's per-Compare/Select scratch
    ///   values, allocated DOWNWARD from the top of the id space;
    /// * `0xF000_0000 + n` — the arm32/x86 lifted temporaries (`temp_local`).
    ///
    /// They used to render as `t4294967294` / `t4026531840` in the middle of a
    /// decompiled body. That is not cosmetic: it reads like an undefined local,
    /// it makes two methods' bodies look different purely because one contains
    /// more compares than the other, and it buries the real `vN` naming. The
    /// mapping is a pure function of the id, so it stays deterministic without
    /// carrying any extra state.
    /// Display name of a FRAME CELL (`slot_origin` `stack_<slot>`), or `None`
    /// when the local is not one.
    ///
    /// `stack_analyzer` names a frame cell `stack_<slot>` and uses that NAME as
    /// the base of the `Memory` operand for every access landing in the cell
    /// (`slot_operand`), so the local stands for the cell's storage and no
    /// statement in the body ever assigns it. Printing `v<k>` therefore showed
    /// a name with no value — 22% of the whole undefined-local census — even
    /// though the verifier had ALREADY decided these are entry values rather
    /// than corruption: its undefined-local oracle skips every `stack_` origin
    /// ("a stack cell read but never written anywhere is an INCOMING STACK
    /// ARGUMENT, not corruption"). The verifier was right and the renderer was
    /// the half left behind; naming the local after the cell it IS closes the
    /// gap without moving any value.
    ///
    /// The slot number is SIGNED (a negative slot is below the frame base) and
    /// a minus sign cannot appear in an identifier, so it is spelled `neg`:
    /// `stack_-255` → `stack_neg255`. Two locals of one cell can share a name
    /// (the same cell read at two widths); that is aliasing of one address, not
    /// two variables, which is what the IR means by them.
    fn stack_cell_name(slot_origin: &str) -> Option<String> {
        let rest = slot_origin.strip_prefix("stack_")?;
        match rest.strip_prefix('-') {
            Some(n) => Some(format!("stack_neg{n}")),
            None => Some(format!("stack_{rest}")),
        }
    }

    fn internal_slot_name(v: u32) -> String {
        if v == u32::MAX - 1 {
            return "ret".into();
        }
        if v >= u32::MAX - 2 {
            return format!("cond{}", u32::MAX - 2 - v);
        }
        if v >= 0xF000_0000 {
            return format!("tmp{}", v - 0xF000_0000);
        }
        // Unclassified id (should not happen): keep it visibly synthetic
        // rather than inventing a `vN` name that could collide with a real one.
        format!("t{v}")
    }

    /// Name used when materializing a def of a var-read local as an
    /// assignment. Redefined parameter/`this` registers are NEW variables
    /// (the C# parameter itself is immutable), so they take their register
    /// slot name (`vN`) rather than the parameter's display name.
    fn def_var_name(&self, v: u32) -> String {
        match self.table.index_of.get(&v) {
            // A frame cell keeps its cell name at its definition too: the
            // operand path (`var_name`) and this one name the SAME local, and a
            // `stack_24 = …` in one place with `stack_24` in the other is only
            // consistent if both spellings agree.
            Some(&li) => match Self::stack_cell_name(&self.table.infos[li].slot_origin) {
                Some(n) => n,
                None => format!("v{li}"),
            },
            None => Self::internal_slot_name(v),
        }
    }

    /// Record that the VALUE of instruction `i` was consumed by the emitted
    /// C#. `audit` is the always-on claim set the fidelity audit reads.
    ///
    /// `claim_log` is appended-to ONLY when `ILR_AUDIT_STRICT=1`, because it
    /// is a `Vec` that is never drained and `def_expr` runs once per operand
    /// resolution — including every re-resolution of a shared def on a method
    /// that has a def cycle. On one such 336-byte method (arm64 @0x2495410)
    /// the log reached 2 GiB of `usize` and the process aborted with
    /// `memory allocation of 2147483648 bytes failed`, taking the whole dump
    /// with it. The strict audit still gets an exact ordered log when asked
    /// for one; normal runs no longer pay for it.
    fn audit_claim(&self, i: usize) {
        self.audit.borrow_mut().insert(i);
        if self.audit_strict {
            self.claim_log.borrow_mut().push(i);
        }
    }

    /// Display name for parameter `pi` — or one float component of an
    /// FP-aggregate parameter (`Vector3 v`, comp 1 → "v.y"). Returns None
    /// when the stored name isn't a safe identifier (caller falls back to
    /// arg{pi}); component field names come from metadata so no guessing.
    fn param_display(&self, pi: usize, comp: Option<usize>) -> Option<String> {
        if pi >= self.table.param_names.len() {
            return None;
        }
        let n = self.table.param_names[pi].clone();
        let ok = !n.is_empty()
            && n.chars().all(|c| c.is_alphanumeric() || c == '_')
            && !n.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true);
        if !ok {
            return None;
        }
        match comp {
            None => Some(n),
            Some(k) => {
                let f = self
                    .table
                    .param_comp_names
                    .get(pi)
                    .and_then(|fs| fs.get(k))
                    .cloned();
                match f {
                    Some(f)
                        if !f.is_empty()
                            && f.chars().all(|c| c.is_alphanumeric() || c == '_') =>
                    {
                        Some(format!("{n}.{f}"))
                    }
                    _ => Some(n),
                }
            }
        }
    }

    fn var_name(&self, v: u32) -> String {
        // internal scratch slots must never leak as variable names
        if v >= u32::MAX - 2 {
            return Self::internal_slot_name(v);
        }
        match self.table.index_of.get(&v) {
            Some(&li) => {
                let info = &self.table.infos[li];
                if info.is_this && !self.multi_def.contains(&v) {
                    return "this".into();
                }
                if info.is_this && self.multi_def.contains(&v) {
                    // clobbered receiver register — neutral name
                    return format!("v{li}");
                }
                if let Some(pi) = info.is_param {
                    if info.param_comp.is_some() && self.multi_def.contains(&v) {
                        // merged/redefined FP register: no longer provably
                        // the parameter component (mirrors the this-guard)
                        return format!("v{li}");
                    }
                    if let Some(n) = self.param_display(pi, info.param_comp) {
                        return n;
                    }
                    return format!("arg{pi}");
                }
                if let Some(n) = Self::stack_cell_name(&info.slot_origin) {
                    return n;
                }
                if let Some(n) = self.entry_register_name(v, info) {
                    return n;
                }
                format!("v{li}")
            }
            None => Self::internal_slot_name(v),
        }
    }

    /// A local that is READ but never DEFINED anywhere in the method holds the
    /// value its origin register had on ENTRY — it is neither a parameter (no
    /// parameter metadata) nor a temporary. Naming it `v6` pretends it is a
    /// source variable and leaves a name that no statement in the body ever
    /// assigns, which reads as decompiler corruption; `in_x2` says exactly
    /// what it is.
    ///
    /// Verified on `CollectWeeklyFacebookGift` @0x47014C (arm64): the failure
    /// tail calls `fn_4a58bac(x0, x1, x2)` with the caller's still-live
    /// registers, and `v4`/`v6` — the entry values of x1/x2, which this method
    /// never writes — are passed straight through.
    /// Is this local a STACK CELL (a `stack_N` frame slot) rather than a value?
    ///
    /// A stack local names MEMORY, so a `Memory` operand based on it reads
    /// through the cell's ADDRESS (`*(T*)(&slot + off)`), not through the
    /// cell's value. The stack analyzer emits exactly that shape for an access
    /// that lands inside a cell (`ldr w19, [sp, #0xc]` reading the 4-byte field
    /// at offset 4 of the 8-byte indirect-result buffer at sp+8), and without
    /// the `&` the same expression would be printed as if the cell's VALUE were
    /// a pointer.
    fn is_stack_local(&self, v: u32) -> bool {
        self.table
            .index_of
            .get(&v)
            .is_some_and(|&li| self.table.infos[li].slot_origin.starts_with("stack_"))
    }

    /// The base expression a `Memory` operand dereferences, with the base's
    /// SPELLING decided in one place.
    ///
    /// Called from [`Self::mem_addr_expr`] only — the single address builder
    /// behind the operand path, the `Move`-definition path AND the store path.
    /// It used to be applied by the first two and forgotten by the third, so
    /// one access rendered two ways: method `0x11b2780` printed
    /// `if (*(byte*)(0x1bb72ea) == 0)` and, three lines below,
    /// `*(byte*)(v0 +29061866) = 1;` — the same address, once as the address it
    /// is and once as a name nothing in the body defines.
    fn mem_base_expr(&self, base: &Operand, base_expr: Expr) -> Expr {
        match base {
            Operand::Local(v) if self.is_stack_local(*v) => Expr::AddressOf(Box::new(base_expr)),
            // An ABSOLUTE address is one constant, not a variable plus an
            // offset. `mem_addr_expr` has already folded index*scale into the
            // expression, so this arm only fires for the un-indexed case, where
            // the whole operand really is the single displacement.
            Operand::Local(v) if self.is_abs_base(*v) => Expr::Int(0),
            Operand::Register(r) if r == "abs" || r == "rip" => Expr::Int(0),
            _ => base_expr,
        }
    }

    /// Is `v` the never-written ABSOLUTE-ADDRESS base pseudo-local?
    ///
    /// `lift_x86` materialises an IP-relative (`[rip+disp]`), `[abs+disp]` or
    /// bare `[disp]` memory operand as `Memory { base: Register("abs"), addend:
    /// disp }`, and for that operand the address IS the displacement — the two
    /// facts `locals::absolute_static_addr` and `resolve::absolute_address`
    /// already rely on to name static storage. Once the pseudo-register became
    /// a local, the renderer printed the LOCAL'S NAME as the base
    /// (`*(byte*)(v0 +29061866)`), which claims a source variable holds a
    /// pointer where the compiler emitted a compile-time address. It was also
    /// the largest single group in the undefined-local census — 70% of the
    /// residue — precisely because nothing in the body ever defines it.
    ///
    /// Only a version with NO definition qualifies. An `abs`/`rip`-origin local
    /// the body DOES write holds a computed pointer, and the addend is then a
    /// real offset from it; folding that to a constant address would invent the
    /// pointer.
    fn is_abs_base(&self, v: u32) -> bool {
        if self.def_sites.get(&v).is_some_and(|s| !s.is_empty()) {
            return false;
        }
        self.table
            .index_of
            .get(&v)
            .is_some_and(|&li| matches!(self.table.infos[li].slot_origin.as_str(), "abs" | "rip"))
    }

    fn entry_register_name(&self, v: u32, info: &super::locals::LocalInfo) -> Option<String> {
        if v >= u32::MAX - 2 {
            return None;
        }
        if self.def_sites.get(&v).is_some_and(|s| !s.is_empty()) {
            return None;
        }
        let o = info.slot_origin.as_str();
        // A read-only version whose origin is a PHYSICAL REGISTER is an incoming
        // ABI value: it arrived from the caller and nothing in this body defines
        // it, so it must be NAMED rather than left as a bare `vN` — a local with
        // no value. Every register family the lifters emit qualifies: integer
        // (`xN`/`wN`, ARM32 `gN`), floating/SIMD at every lane width (`vfN`,
        // `vbN`/`vhN`/`vsN`/`vdN`/`vqN`, ARM32 `sN`/`dN`/`qN`), the ABI return
        // register (`rv`) and the frame pointers (`sp`/`fp`).
        //
        // This used to accept only `xN`/`wN`, which meant every INCOMING FP
        // value — an argument the signature did not cover, a return register,
        // a caller-saved slot — printed as an undefined `vN`. That is what the
        // `undefined locals (v*)` census counts, so the narrow predicate was
        // inflating it by tens of thousands of references.
        //
        // `x0` is included like every other register.
        //
        // It used to be excluded on the theory that `entry_var_name` has
        // already handled the receiver and the first parameter — and it has:
        // BOTH `var_name` and `entry_var_name` test `is_this` and `is_param`
        // BEFORE they reach here, so a local arriving at this point with origin
        // `x0` is neither. What is left is a value the ABI placed in the
        // receiver register that the declared signature does not cover — IL2CPP
        // folds several metadata signatures onto one native body, so the member
        // this RVA is keyed to can declare fewer parameters than the body
        // actually reads — which is exactly the case every OTHER argument
        // register is named for. The scratch reuse the exclusion was guarding
        // against is a local with a DEFINITION, and `def_sites` already refuses
        // those above. Measured on the x64 corpus, the exclusion cost 16,671
        // undefined references over 5,927 methods, and 7,001 of those were a
        // method whose `x0` view had been split across two locals — one
        // incoming value read at two widths, not two variables.
        //
        // A non-register origin (`abs`, `rip`, a scratch slot spelled `vN`) is
        // not an incoming ABI value at all and keeps its `vN` spelling.
        let is_reg = match o.as_bytes().first() {
            Some(b'x') | Some(b'w') | Some(b'g') => {
                o.len() > 1 && o[1..].bytes().all(|c| c.is_ascii_digit())
            }
            Some(b'v') => {
                o.len() > 2
                    && matches!(o.as_bytes()[1], b'f' | b'b' | b'h' | b's' | b'd' | b'q')
                    && o[2..].bytes().all(|c| c.is_ascii_digit())
            }
            Some(b's') if o == "sp" => true,
            // ARM32 FP lane views: `sN`/`dN`/`qN`, no width letter.
            Some(b's') | Some(b'd') | Some(b'q') => {
                o.len() > 1 && o[1..].bytes().all(|c| c.is_ascii_digit())
            }
            _ => o == "rv" || o == "fp",
        };
        if !is_reg {
            return None;
        }
        Some(format!("in_{o}"))
    }

    fn is_multi(&self, v: u32) -> bool {
        self.multi_def.contains(&v)
    }

    /// Does `def_insn` emit a statement that BINDS its result to a name
    /// (`dst = call(…)`), as opposed to a bare `call(…)` eval?
    ///
    /// This mirrors the `Op::Call` / `Op::ObjcMsgSend` arm exactly: the
    /// assignment form is chosen when the destination is MULTI-DEF and the
    /// result is not void (`void_ret` emits `Stmt::Eval`), and `this` never
    /// gets assigned. A use site that re-inlines such a call therefore emits
    /// a SECOND copy of ONE native `bl` — see the `Op::Return` arm.
    /// Is the callee PROVABLY void? Only an explicit `void` return type counts:
    /// an unresolved call has `dest_type == None` and still leaves a value in
    /// x0, so "unknown" must never be treated as "void".
    fn call_result_is_void(insn: &Insn) -> bool {
        matches!(
            insn.dest_type.as_ref(),
            Some(t) if matches!(t, ManagedType::Primitive(PrimKind::Void))
        )
    }

    /// The local a call-family instruction writes its result into.
    fn call_op(insn: &Insn) -> bool {
        matches!(
            insn.op,
            Op::Call | Op::CallVoid | Op::ObjcMsgSend { .. } | Op::IndirectCall
        )
    }

    /// Did the call's OWN site already emit a statement for it?
    ///
    /// `boundary_stmts_inner` emits nothing for a `Op::Call` only when the
    /// result is non-void, single-def, and READ (its value is inlined at the
    /// use site instead). In every other case the call became a statement, and
    /// a use site that inlines the call expression again would emit the SAME
    /// native `bl` twice — the `v73 = fn_4a58bac(…);` / `fn_4a58bac(…);` pair on
    /// `CollectWeeklyFacebookGift` @0x47014C.
    /// ONE NATIVE `bl`, ONE EMITTED STATEMENT.
    ///
    /// The name a call's result must be bound to, when the call site emits the
    /// assignment form `dst = callee(…)` and every use site must reference
    /// `dst` instead of re-expanding the call.
    ///
    /// The rule is about USES, not just definitions. A result read at more than
    /// one site used to be inlined at each of them (the result local was
    /// single-def and the assignment form was reserved for multi-def locals), so
    /// the emitted body repeated the native `bl` once per read. Nothing in the
    /// disassembly calls a function N times because its result is used N times.
    ///
    /// It is read ONCE, too, not "more than once". Gating the binding on
    /// `reads > 1` assumed the single use site always inlines the call — and it
    /// usually does, but not when the inline is refused (the def-cycle guard and
    /// the inlining budget both fall back to rendering the local's NAME). The
    /// body then contained `v24 = v54;` with no statement ever defining `v54`,
    /// while the callee ran natively: measured on 0x25e6058 (`set_usageHints`),
    /// 3 `blr` in the final IR and zero call expressions in the text.
    fn call_dest_binding(&self, insn: &Insn) -> Option<String> {
        if Self::call_result_is_void(insn) {
            return None;
        }
        let d = insn.dest().and_then(|d| d.as_local())?;
        let multi = self.is_multi(d);
        let reads = self.read_count.get(&d).copied().unwrap_or(0);
        // Nothing reads it → the call site emits a bare eval (case (c)), which
        // is a perfectly good statement; there is no value to name.
        if !multi && reads == 0 {
            return None;
        }
        let name = self.var_name(d);
        // `this` is not assignable: the receiver slot was clobbered by the call
        // protocol, so there is no place to bind to.
        if multi {
            return (name != "this").then_some(name);
        }
        // Single-def and read. The name must not double as something already
        // meaningful in the emitted signature: binding a call to `value` or
        // `this` would read as a re-assignment of the parameter/receiver.
        //
        // That is not the same as UNBINDABLE, though — the destination is a
        // real slot, and the value it holds after the call is a new value. So
        // mint the slot name (`v<slot>`, the convention the multi-def clobber
        // guard already uses) instead of falling back to lazy inlining. It
        // keeps the two meanings apart: a use site whose reaching def is the
        // call reads `v54`, a use site whose reaching def is the ENTRY value
        // still reads the parameter name. Only case (b) for internally named
        // slots (no table entry) keeps the inline fallback.
        if name.starts_with('v') && name != "this" {
            return Some(name);
        }
        self.table.index_of.get(&d).copied().map(|li| format!("v{li}"))
    }

    fn call_site_statement_emitted(&self, def_insn: usize) -> bool {
        let insn = &self.insns[def_insn];
        if !Self::call_op(insn) {
            return false;
        }
        if Self::call_result_is_void(insn) || matches!(insn.op, Op::ObjcMsgSend { .. }) {
            return true;
        }
        let Some(d) = insn.dest().and_then(|d| d.as_local()) else {
            return true;
        };
        if self.is_multi(d) {
            return true;
        }
        !self.read_locals.contains(&d)
    }

    /// Record that call site `def_insn` was emitted as `name = callee(…)`.
    ///
    /// A use site may only reference that name once the statement provably
    /// exists in the body: referencing a name no statement defines turns the
    /// lost CALL into a lost VALUE, which is strictly worse (measured: gating
    /// on the predicate alone instead of on materialization moved
    /// dropped-call methods from 882 to 1205). Until the def site has been
    /// walked, uses keep the lazy-inline behaviour they had before.
    fn note_call_bound(&self, def_insn: usize, name: &str) {
        self.bound_calls
            .borrow_mut()
            .insert(def_insn, name.to_string());
    }

    /// The name the emitted statement bound the call's result to, when the
    /// statement was the assignment form (`dst = call(…)`). `None` when the
    /// statement is a bare `call(…)` eval — a void callee, or a result nobody
    /// reads — in which case there is no value to reference.
    fn call_binding_name(&self, def_insn: usize) -> Option<String> {
        let insn = &self.insns[def_insn];
        if !Self::call_op(insn) || Self::call_result_is_void(insn) {
            return None;
        }
        if let Some(n) = self.bound_calls.borrow().get(&def_insn) {
            return Some(n.clone());
        }
        // Not emitted YET in this walk, but the discovery pass proved the def
        // site will be reached — reference the name it is going to bind.
        self.preseed_bindings.borrow().get(&def_insn).cloned()
    }

    /// Has THIS walk already emitted a statement for call site `def_insn`?
    ///
    /// Distinct from [`Self::call_binding_name`], which also answers "yes" for
    /// a pre-seeded def: the def site must still emit its one assignment, so
    /// its duplicate guard may only consult what this walk actually emitted.
    fn call_already_emitted(&self, def_insn: usize) -> bool {
        self.bound_calls.borrow().contains_key(&def_insn)
    }

    /// Canonical source of a pure copy chain.
    ///
    /// After SSA phi-elimination a single value is often copied into the
    /// SAME version at several merge points (`v68 ← v34` at each branch
    /// target). The reaching-definition fixpoint sees two different defs and
    /// reports ambiguity (None), forcing the plain `v68` var — even though
    /// every def copies the same source, so the value is path-independent.
    /// When EVERY def of `v` is `Move [v, Local(w)]` with the same `w`, `v`
    /// is a pure alias of `w`; follow to the root (a version with no defs,
    /// i.e. an entry value, or one defined by a real operation). Cycles are
    /// impossible (copies flow from earlier SSA versions) but guarded anyway.
    fn copy_alias(&self, v: u32) -> Option<u32> {
        let mut seen = std::collections::HashSet::new();
        let mut cur = v;
        loop {
            if !seen.insert(cur) {
                return None;
            }
            // absent key == no defs == entry value: this IS the root
            let sites: &[usize] = self.def_sites.get(&cur).map(|v| v.as_slice()).unwrap_or(&[]);
            if sites.is_empty() {
                return Some(cur);
            }
            // every def must be a copy of the same single source
            let mut src: Option<u32> = None;
            for &di in sites {
                let insn = &self.insns[di];
                if !matches!(insn.op, Op::Move) {
                    return None; // a real (value-producing) def → not an alias
                }
                let Some(Operand::Local(w)) = insn.operands.get(1) else {
                    return None;
                };
                match src {
                    None => src = Some(*w),
                    Some(s) if s == *w => {}
                    _ => return None, // different sources per path → ambiguous
                }
            }
            let w = src?;
            if w == cur {
                return None;
            }
            cur = w;
        }
    }

    /// Resolve a local to its inlined expression tree.
    ///
    /// With no use-site context (loop headers, entry params): single-def
    /// locals inline; multi-def locals stay named vars. When a use-site is
    /// known (`inline_local_at`), the REACHING def decides — a def whose
    /// sole assignment reaches the use inlines even inside a multi-def
    /// local, which is exactly how `v1` shells become real expressions.
    fn inline_local(&self, v: u32, depth: usize) -> Expr {
        // Outermost resolution ⇒ a new expression ⇒ a fresh budget, so one
        // bloated statement can never starve the ones after it.
        if self.inline_stack.get() == 0 {
            self.inline_budget.set(MAX_INLINE_EXPR_NODES);
        }
        self.inline_stack.set(self.inline_stack.get() + 1);
        let e = self.inline_local_inner(v, depth);
        self.inline_stack.set(self.inline_stack.get() - 1);
        e
    }

    fn inline_local_inner(&self, v: u32, depth: usize) -> Expr {
        if depth > self.max_inline_depth {
            return Expr::Var(self.var_name(v));
        }
        let budget = self.inline_budget.get();
        if budget == 0 {
            if self.rss_dbg {
                BUDGET_FIRED.with(|f| f.set(true));
            }
            return Expr::Var(self.var_name(v));
        }
        self.inline_budget.set(budget - 1);
        // A shared value the renderer decided to materialise is REFERENCED,
        // never copied into the expression: copying it is what turns a def DAG
        // into an exponential tree (see [`SHARED_EXPR_NODES`]).
        if self.shared_large.contains(&v) {
            return Expr::Var(self.var_name(v));
        }
        // A def that is already being resolved further up this stack is a PHI
        // (see [`Self::inline_in_progress`]) — expanding it again is what turns
        // a loop-carried value into an unbounded expression. Reference it.
        if !self.inline_in_progress.borrow_mut().insert(v) {
            CYCLE_CUTS.with(|c| c.set(c.get() + 1));
            return Expr::Var(self.var_name(v));
        }
        let e = self.inline_local_guarded(v, depth);
        self.inline_in_progress.borrow_mut().remove(&v);
        e
    }

    /// [`Self::inline_local`] with the cycle guard already entered.
    fn inline_local_guarded(&self, v: u32, depth: usize) -> Expr {
        // Use-site-aware: when the render cursor is inside a def_expr chain
        // (non-MAX sentinel), the REACHING def at that point decides — even
        // for multi-def locals, because a single reaching def is exactly the
        // value the hardware read.
        let use_insn = self.cur_insn.get();
        if use_insn != usize::MAX {
            return self.inline_local_at(v, use_insn, depth);
        }
        if self.is_multi(v) {
            return Expr::Var(self.var_name(v));
        }
        match self.defs.get(&v) {
            Some(&i) => {
                // A mutation of the name it assigns cannot be expanded
                // anywhere but at itself, with or without a use-site cursor.
                // The do-while LATCH condition resolves its compare operands
                // through THIS path (`operand_expr(.., 0)` with no cursor),
                // which is where the corpus kept printing
                // `do { … v = v - 1; } while (v - 1 != 0)` after the
                // use-site-aware refusal stopped it on the branch path.
                if self.def_rewrites_the_name_it_assigns(i)
                    && !self.def_chain_has_call(i)
                {
                    Expr::Var(self.var_name(v))
                } else {
                    self.def_expr(i, depth)
                }
            }
            None => Expr::Var(self.var_name(v)),
        }
    }

    /// Would inlining the value of instruction `i` at `use_insn` move the
    /// expression ACROSS a write of one of its own inputs?
    ///
    /// The expression of a def is only equal to the use's value when every
    /// local it reads still holds the same value at the use. A def that reads
    /// a local its own block writes at or after itself is not portable: the
    /// rotated loop's update `x0 = [x0, #0x28]` is exactly that shape, and
    /// `Exception.GetBaseException`@0x1A6FA74 renders the value assigned to
    /// `x0` — but inlining it at the RETURN (outside the loop) re-read the
    /// ALREADY-ADVANCED pointer and returned `null` instead of the base
    /// exception. Same-block uses keep the expression (the writes between
    /// them are ordered), which is why only cross-block inlining is refused.
    fn def_moved_across_a_write(&self, i: usize, use_insn: usize) -> bool {
        let Some(block_of) = self.block_of_insn.as_deref() else {
            return false;
        };
        let (Some(bi), Some(bu)) = (
            block_of.get(i).copied(),
            block_of.get(use_insn).copied(),
        ) else {
            return false;
        };
        if bi == usize::MAX || bu == usize::MAX || bi == bu {
            return false;
        }
        let Some(&(start, end)) = self.blocks_range.get(bi) else {
            return false;
        };
        let mut locals: Vec<u32> = Vec::new();
        self.collect_def_locals(i, 0, &mut locals);
        locals.iter().any(|w| {
            self.def_sites.get(w).is_some_and(|sites| {
                sites
                    .iter()
                    .any(|&d| d >= i && d >= start && d < end)
            })
        })
    }

    /// Does the definition of `i` read, BY NAME, the storage `i` assigns?
    ///
    /// `v266 = v266 - 1` — a loop-carried induction update — is the shape,
    /// and it is the one case where a def's expression cannot be re-expanded
    /// anywhere but AT the def itself. SSA gives the latch its own version
    /// (`v299 = v15 - 1` in the lifted IR), but `highvar` merges versions onto
    /// ONE table slot, so both spell `v266` and the assignment becomes a
    /// self-mutation of a single name.
    ///
    /// Inlining it at a later use re-reads that name AFTER the assignment has
    /// already replaced it, so the expression applies the update a SECOND
    /// time. Measured on `0x1532130`:
    ///
    /// ```text
    ///   do { … v266 = v266 - 1; } while (v266 - 1 != 0);
    /// ```
    ///
    /// — the test decrements a second time, so the loop body runs ONE
    /// ITERATION SHORT of the native. The NAME is the exact value the use
    /// reads, and [`Self::refused_inline_reads`] then materialises the def so
    /// the assignment is present in the body.
    ///
    /// A pure copy (`Move [Local(a), Local(b)]`) is not a mutation for this
    /// purpose: its expression renders as the source's own name, so refusing
    /// it would add nothing but a `v = v;` self-assignment when the two
    /// versions share a slot.
    fn def_rewrites_the_name_it_assigns(&self, i: usize) -> bool {
        let insn = &self.insns[i];
        let Some(dest) = insn.dest().and_then(|d| d.as_local()) else {
            return false;
        };
        if matches!(insn.op, Op::Move) && matches!(insn.operands.get(1), Some(Operand::Local(_))) {
            return false;
        }
        // ONLY for a def that assigns a NAMED local. The refusal renders
        // `Expr::Var(name)`, so it is meaningless for an origin with no
        // variable-table entry — and refusing those trades a wrong value for a
        // name NOTHING defines, measured as +313 undefined references, every
        // one of them an origin the text never assigns (`rv`, the callee-saved
        // `g6`/`g7`). Such a local has no `index_of` entry, so that is the test.
        //
        // It is deliberately NOT "the name is spelled by some var-read version"
        // (the previous gate). `highvar` maps several versions onto one index,
        // so that question is answered by a SIBLING version: on
        // `OutBounce`@0x14D6330 index 18 carried both an emitted version and one
        // whose def never reaches the text, the gate passed for the second, and
        // the refusal replaced `(t - A) * C2` with the bare name — silently
        // dropping the `* C2` factor. That gate was also read at two different
        // times (before and after `refused_inline_reads` completed `var_reads`),
        // so the same def answered "no" while the read set was built and "yes"
        // while it was rendered.
        //
        // Refusing here is safe because the refusal is what MATERIALISES the
        // def: [`Self::refused_inline_reads`] adds every read local whose def is
        // refused to `var_reads`, so the assignment this name refers to is
        // emitted. Keying on `index_of` alone also makes that pass independent
        // of `var_reads`, so one sweep is its exact fixed point.
        if self.table.index_of.get(&dest).is_none() {
            return false;
        }
        let name = self.def_var_name(dest);
        // TRANSITIVE, through the def's own read chain. The corpus shape
        // reaches its own name through the loop's BACK-EDGE COPY
        // (`v266 = v15 - 1` where `v15` is a copy of `v266`), so the def's
        // direct operand is a DIFFERENT slot and a direct comparison never
        // fires. `collect_def_locals` follows each read into its definition,
        // which is what closes that cycle.
        let mut reads: Vec<u32> = Vec::new();
        self.collect_def_locals(i, 0, &mut reads);
        reads.into_iter().any(|w| {
            // A read that is still an ENTRY value renders as `in_xN`, never as
            // a table name, so it cannot collide with what the def assigns.
            self.def_sites.get(&w).is_some_and(|s| !s.is_empty())
                && self.def_var_name(w) == name
        })
    }

    /// Does the definition chain of instruction `i` include a CALL?
    ///
    /// Such a value must stay inlinable at its use sites: the renderer emits a
    /// call as its own statement only when it binds the result to a name, and
    /// when it does not, the use site's inline is the ONLY place the native
    /// `bl` shows up in the recovered text.
    fn def_chain_has_call(&self, i: usize) -> bool {
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut stack = vec![i];
        let mut guard = 0usize;
        while let Some(cur) = stack.pop() {
            guard += 1;
            if guard > 32 {
                return true;
            }
            if !seen.insert(cur) {
                continue;
            }
            if matches!(
                self.insns[cur].op,
                Op::Call | Op::CallVoid | Op::IndirectCall | Op::NewObj(_) | Op::NewArr(_)
            ) {
                return true;
            }
            for w in self.insns[cur].use_locals() {
                if let Some(&d) = self.defs.get(&w) {
                    stack.push(d);
                }
            }
        }
        false
    }

    /// Locals the expression of instruction `i` depends on, following each
    /// dependency into its own definition (bounded depth and width).
    fn collect_def_locals(&self, i: usize, depth: usize, out: &mut Vec<u32>) {
        if depth > 3 || out.len() > 32 {
            return;
        }
        for w in self.insns[i].use_locals() {
            if !out.contains(&w) {
                out.push(w);
            }
            if let Some(&d) = self.defs.get(&w) {
                if d != i {
                    self.collect_def_locals(d, depth + 1, out);
                }
            }
        }
    }

    /// Use-site-aware variant: `use_insn` is the index of the instruction
    /// whose operand we're rendering. The def that reaches THAT point (not
    /// the last linear def) is what the hardware would read.
    fn inline_local_at(&self, v: u32, use_insn: usize, depth: usize) -> Expr {
        if depth > self.max_inline_depth {
            return Expr::Var(self.var_name(v));
        }
        // map use insn → block, then the reaching map valid at block entry,
        // overridden by any def between block start and the use
        let def_at = self.reaching_def_at(v, use_insn);
        // A def that sits LATER in the same block can only reach this use
        // around the back edge — the block is a loop header/latch and the use
        // reads what this same variable held at the end of the previous
        // iteration. Inlining that def here would emit a self-referential
        // expression: `GetBaseException`@0x1A6FA74 reads `x0` at the top of
        // its own loop, the only reaching def is the loop's own
        // `x0 = [x0, #0x28]`, and resolving it printed the FIELD of the
        // already-advanced variable — `return *(nint*)(v0 +40)` (i.e. `null`)
        // where the base exception belonged. The variable name is the exact
        // and only faithful rendering at such a use.
        if let Some(d) = def_at {
            if d > use_insn
                && self
                    .block_of_insn
                    .as_ref()
                    .is_some_and(|m| m.get(d).copied() == m.get(use_insn).copied())
            {
                return Expr::Var(self.var_name(v));
            }
        }
        // ONE NATIVE `bl`, ONE EMITTED STATEMENT.
        //
        // When the call's own site already materialized its result into a name,
        // a use site must reference that NAME. Inlining the call expression
        // again emitted a second copy of the same native instruction — the
        // `v73 = fn_4a58bac(…); return fn_4a58bac(…);` and the
        // `v23 = … .SetLastWeekFacebookGiftWasCollected(…);` pairs on
        // `CollectWeeklyFacebookGift` @0x47014C.
        if let Some(d) = def_at {
            if let Some(name) = self.call_binding_name(d) {
                return Expr::Var(name);
            }
        }
        match def_at {
            Some(i) if i != use_insn => {
                // A definition whose value expression reads (transitively) the
                // local it defines is a MUTATION: `v0 = *(v0 +40)` — the
                // rotated-loop update `do { inner = inner.InnerException; }
                // while (inner != null)`. The *assignment* is a statement and
                // renders fine, but the expression is only valid AT the
                // assignment: at any later point `v0` already holds the new
                // value, so re-inlining it reads the wrong input. Reference the
                // variable instead — that is the value the use actually reads.
                // Only for CALL-FREE values: inlining a call expression is how a
                // single native `bl` reaches a use site whose binding the
                // renderer declined, so refusing it would silently drop the
                // call from the text.
                // A def whose expression reads, BY NAME, the storage it
                // assigns (`v266 = v266 - 1`) is only valid AT itself: at any
                // later use the name already holds the new value, so
                // re-expanding applies the update a second time — the
                // `do { … v266 = v266 - 1; } while (v266 - 1 != 0)` loop that
                // runs one iteration short of the native. See
                // [`Self::def_rewrites_the_name_it_assigns`].
                // Call-containing chains keep the inline: on several of these
                // use sites it is the only place the native `bl` reaches the
                // text, and the existing `def_moved_across_a_write` refusal
                // below makes the same trade. Measured without the guard:
                // `0xf65610` lost a real `InternedString.op_Equality` and two
                // `get_Item` call sites and its loop test changed meaning.
                if (self.def_rewrites_the_name_it_assigns(i) && !self.def_chain_has_call(i))
                    || (self.def_moved_across_a_write(i, use_insn)
                        && !self.def_chain_has_call(i))
                {
                    Expr::Var(self.var_name(v))
                } else {
                    // Ghidra `RulePropagateCopy` refuses to propagate a value
                    // whose COPY is being KEPT — "Don't propagate if we are
                    // keeping the COPY anyway" (`ruleaction.cc`, the
                    // `isAddrForce()` arm). Adding that arm here was tried, with
                    // `var_reads` standing in for `isAddrForce`, and REVERTED on
                    // the measurement: `var_reads` is not the narrow, data-flow
                    // derived predicate Ghidra's flag is. It holds every local
                    // with a merge or back-edge read, which on these corpora is
                    // most ABI-register locals, so refusing to inline them
                    // materialised 260,660 bindings instead of 79,770 and made
                    // the dump BIGGER (941,563 lines vs 827,664) while leaving
                    // 638 of them unread. Porting the rule without first porting
                    // the predicate is a regression; the predicate is the work.
                    // Until then the dead half of the same invariant is enforced
                    // on the output by [`strip_unread_bindings`].
                    self.def_expr(i, depth)
                }
            }
            // No def reaches → the local still holds its ENTRY value (the
            // parameter). Render THAT — `this` for receivers, the metadata
            // param name for parameters — instead of a register-slotted
            // `vN`. This is what makes `*(v0 +192)` read as `this.<field>`:
            // the receiver register is typically redefined by the first
            // field load, and every pre-redefinition use is the original
            // `this` even though the slot is multi-def.
            None => match self.copy_alias(v) {
                // phi-elim copies: every def of `v` is `v ← w` — the value
                // equals `w` on every path, so render w's own resolution.
                Some(t) if t != v => {
                    let prev = self.cur_insn.get();
                    self.cur_insn.set(use_insn);
                    let e = self.inline_local(t, depth + 1);
                    self.cur_insn.set(prev);
                    e
                }
                _ => Expr::Var(self.entry_var_name(v)),
            },
            _ => Expr::Var(self.var_name(v)),
        }
    }

    /// Name for a local's ENTRY (parameter) value, deliberately ignoring
    /// multi-def state: multi-def describes redefinitions inside the body,
    /// which can never change what arrived as a parameter.
    fn entry_var_name(&self, v: u32) -> String {
        // internal scratch slots must never leak as variable names
        if v >= u32::MAX - 2 {
            return Self::internal_slot_name(v);
        }
        match self.table.index_of.get(&v) {
            Some(&li) => {
                let info = &self.table.infos[li];
                if info.is_this {
                    return "this".into();
                }
                if let Some(pi) = info.is_param {
                    if info.param_comp.is_some() && self.multi_def.contains(&v) {
                        return format!("v{li}");
                    }
                    if let Some(n) = self.param_display(pi, info.param_comp) {
                        return n;
                    }
                    return format!("arg{pi}");
                }
                if let Some(n) = Self::stack_cell_name(&info.slot_origin) {
                    return n;
                }
                if let Some(n) = self.entry_register_name(v, info) {
                    return n;
                }
                format!("v{li}")
            }
            None => Self::internal_slot_name(v),
        }
    }

    /// Chase the reaching definition of `v` at `use_insn` through SSA copy /
    /// ABI-return-joiner `Move` chains to the instruction that ultimately
    /// PRODUCED the value.
    ///
    /// The ABI return register is copied on the way out of a call
    /// (`Move v2 = v1`), so the def that reaches a `ret` is very often the
    /// copy, not the `bl` itself. Asking about the copy's own statement says
    /// nothing about whether the call was emitted.
    fn chase_reaching_def(&self, v: u32, use_insn: usize) -> Option<usize> {
        let mut cur_use = use_insn;
        let mut cur_v = v;
        for _ in 0..8 {
            let d = self.reaching_def_at(cur_v, cur_use)?;
            let insn = &self.insns[d];
            if Self::call_op(insn) {
                return Some(d);
            }
            if !matches!(insn.op, Op::Move) {
                return Some(d);
            }
            match insn.operands.get(1) {
                Some(Operand::Local(src)) if *src != cur_v => {
                    cur_v = *src;
                    cur_use = d;
                }
                _ => return Some(d),
            }
        }
        None
    }

    /// Did a `ret`'s deferred return value already get emitted as a call
    /// statement of its own?
    ///
    /// True when the producing call is a void callee (its own site emitted
    /// `Eval(call)`) or when its result was materialized to a name (its own
    /// site emitted `dst = call(…)`). False means nothing was emitted for that
    /// native `bl` yet, so the `ret` must emit it — otherwise the call vanishes
    /// from the body while the callee still runs natively.
    fn ret_value_already_emitted(&self, v: u32, use_insn: usize) -> bool {
        self.chase_reaching_def(v, use_insn).is_some_and(|cd| {
            Self::call_result_is_void(&self.insns[cd]) || self.call_binding_name(cd).is_some()
        })
    }

    /// The single definition of `v` reaching instruction `use_insn`, if
    /// unambiguous. Returns None when: no def reaches (param/entry), or the
    /// merge of multiple defs reaches (φ point) — the latter keeps the named
    /// var (correct C# semantics, same as Ghidra's "high" confidence).
    fn reaching_def_at(&self, v: u32, use_insn: usize) -> Option<usize> {
        reaching_def_at_free(
            &self.insns,
            &self.reaching,
            &self.blocks_range,
            self.block_of_insn.as_deref(),
            v,
            use_insn,
        )
    }

    /// Every `__unknown` the renderer emits must be attributable to the exact
    /// site that produced it. `ILR_UNK=1` prints that attribution on stderr;
    /// the returned Expr is unchanged. Without this, `__unknown` is a single
    /// opaque string in the output with a dozen possible producers, and every
    /// bug report about it degenerates into guessing which one fired.
    fn unk<T: std::fmt::Debug>(&self, site: &str, detail: T) -> Expr {
        if std::env::var("ILR_UNK").is_ok() {
            eprintln!("[unk] {site} {detail:?}");
        }
        Expr::Unknown
    }

    fn operand_expr(&self, op: &Operand, depth: usize) -> Expr {
        match op {
            Operand::MissingAbiArg => Expr::MissingAbiArg,
            // A METHOD used as a value: the `MethodInfo*` handed to a delegate
            // constructor or stored into a delegate cache field. This had no
            // arm at all, so it fell to the `_` catch-all and rendered
            // `__unknown` — deleting the identity of the delegate target at
            // 44k sites binary-wide. The old pass that resolves such calls
            // (`recover_delegate_invokes`) only reads `Move(local, Method)`
            // operands directly; anything that reaches the renderer still has
            // to print something honest.
            Operand::Method(m) => Expr::MethodGroup(format!(
                "{}.{}",
                short_type(&m.declaring_type_name),
                m.name
            )),
            // A genuine unresolved value. Kept as its own tag so a census can
            // tell "the lifter lost this" apart from "no arm handles this
            // operand kind" — they have completely different fixes.
            Operand::Unknown => self.unk("operand:unknown", ()),
            Operand::Local(v) => self.inline_local(*v, depth + 1),
            Operand::Register(n) => {
                if n == "xzr" {
                    Expr::Int(0)
                } else {
                    self.unk("operand:raw-register", n.clone())
                }
            }
            Operand::Immediate(v) => Expr::Int(*v),
            Operand::FloatLiteral(f) => Expr::Float(*f),
            Operand::DoubleLiteral(d) => Expr::Double(*d),
            Operand::StringLiteral { value, .. } => Expr::Str(value.to_string()),
            // The ADDRESS of something, not a load of it. `Operand::Memory`
            // alone renders as a deref, so the wrapper is the only thing that
            // distinguishes `&local` from `local`. This operand kind existed in
            // the IR but had NO arm here, so it fell to the catch-all and every
            // `ADD Rd, sp, #imm` address — and every address-of emitted by the
            // arm32/x86 lifters — rendered `__unknown`.
            Operand::AddressOf(inner) => match inner.as_ref() {
                Operand::Memory { base, addend, .. } => {
                    Expr::RawAddr(Box::new(self.operand_expr(base, depth + 1)), *addend)
                }
                // The ADDRESS of a variable is not its VALUE, so the local's
                // name is rendered directly and never inlined here. Inlining
                // is value propagation, and applying it inside an `&` is a
                // category error: the prologue zeroes the enumerator slot
                // (`stp xzr, xzr, [sp, #0x18]`), the load-forwarding pass
                // resolves every read of that slot to the constant `0`, and
                // `add x0, sp, #0x18` — the `&enumerator` handed to
                // `MoveNext` — then rendered as `(&0).MoveNext()`.
                Operand::Local(v) => Expr::AddressOf(Box::new(Expr::Var(self.var_name(*v)))),
                _ => Expr::AddressOf(Box::new(self.operand_expr(inner, depth + 1))),
            },
            Operand::Field(f) => self.field_expr(None, f, depth),
            // A field on a DERIVED receiver: `base.field`. The base is rendered
            // through `operand_expr`, so a nested field chain
            // (`this.Motor._controller._state`) round-trips at any depth.
            Operand::InstanceField { base, field } => self.field_expr(
                Some(self.operand_expr(base, depth + 1)),
                field,
                depth,
            ),
            Operand::Memory { base, addend, index, scale, width, .. } => {
                let base_expr = self.mem_addr_expr(base, index.as_deref(), *scale, depth);
                // `ldr xN, [usage-slot]`: the value stored in a member's own
                // metadata usage slot IS that member, so a deref here is a
                // double indirection and reads the first field of a structure
                // the expression had just materialised. `MethodGroup` as a
                // value already means "the MethodInfo for this member", which
                // is what the load produced. Checked on the RESOLVED base:
                // the slot address usually reaches the renderer as a local
                // whose definition is the `Move(dst, Method(…))` that the
                // resolver wrote, and only inlining reveals the member.
                if *addend == 0 && index.is_none() {
                    if let Expr::MethodGroup(name) = &base_expr {
                        return Expr::MethodGroup(name.clone());
                    }
                }
                Expr::RawDeref(Box::new(base_expr), *addend, *width)
            }
            _ => self.unk("operand:unsupported", op),
        }
    }

    /// The ADDRESS expression for `[base + index*scale + addend]`.
    ///
    /// Both deref sites destructured `index` and `scale` and then IGNORED
    /// them, so every indexed access rendered as `*(T*)(base)` — the index
    /// vanished. That silently dropped, among others, `ldr r0, [pc, r0]`
    /// (the GOT/statics idiom, where `r0` holds the displacement loaded from
    /// the literal pool one instruction earlier) and every
    /// `ldr w1, [x0, x1, lsl #2]` array-element access. The address IS a
    /// value, so it has to be built as an expression rather than truncated to
    /// `base + addend`.
    fn mem_addr_expr(
        &self,
        base: &Operand,
        index: Option<&Operand>,
        scale: u8,
        depth: usize,
    ) -> Expr {
        // Fold the BASE alone, before `index*scale` joins it. A stack cell's
        // address is `&cell`, and `&cell + i*4` is that address plus the index;
        // wrapping the SUM in `AddressOf` instead (`&(cell + i*4)`) would take
        // the address of a temporary. The base's spelling therefore belongs to
        // the base, not to the finished address.
        let base_expr = self.mem_base_expr(base, self.operand_expr(base, depth + 1));
        let Some(idx) = index else {
            return base_expr;
        };
        let idx_expr = self.operand_expr(idx, depth + 1);
        let scaled = if scale > 1 {
            Expr::Bin(
                BinOp::Mul,
                Box::new(idx_expr),
                Box::new(Expr::Int(scale as i64)),
            )
        } else {
            idx_expr
        };
        Expr::Bin(BinOp::Add, Box::new(base_expr), Box::new(scaled))
    }

    fn field_expr(&self, recv: Option<Expr>, f: &FieldRef, depth: usize) -> Expr {
        if f.is_static {
            Expr::StaticField(short_type(&f.declaring_type_name).to_string(), f.name.clone())
        } else {
            let recv = recv.unwrap_or_else(|| {
                // Instance field: `this` when declared on the declaring type
                // OR ANY ancestor (base-class fields are this-accessible).
                // Previously only an exact type match resolved — inherited
                // fields rendered `__unknown.field` despite the metadata
                // proving the receiver (the whole parent_chains map went
                // unused by the pseudo-C# generator).
                if self.in_declaring_chain(f.type_def_index) {
                    Expr::This
                } else {
                    self.unk("field:unknown-receiver", f.name.clone())
                }
            });
            let _ = depth;
            Expr::Field(Box::new(recv), f.name.clone())
        }
    }

    /// Frame-slot store→load forwarding for spill round-trips.
    ///
    /// A local defined by `ldr/ldp … [base+off]` whose value was stored by
    /// an earlier `str/stp … [base+off]` renders as the STORED expression
    /// (not `*(T*)(base+off)`). Scan backward from `load_insn`; the nearest
    /// store to the same (base, addend) wins. Scanning stops at any call:
    /// the callee owns the outgoing argument area, so a value stored before
    /// a call is no longer provably in the slot afterwards.
    fn forward_memory_load(
        &self,
        load_insn: usize,
        base: &Operand,
        addend: i64,
        depth: usize,
    ) -> Option<Expr> {
        let base_key: Option<(u8, u64)> = match base {
            Operand::Local(v) => Some((0, *v as u64)),
            Operand::Register(n) => Some((1, slot_index(n)? as u64)),
            _ => None,
        };
        let base_key = base_key?;
        for j in (0..load_insn).rev() {
            let insn = &self.insns[j];
            // calls own the outgoing arg area — never forward across one
            if matches!(
                insn.op,
                Op::Call | Op::CallVoid | Op::IndirectCall | Op::ObjcMsgSend { .. }
            ) {
                return None;
            }
            if !matches!(insn.op, Op::Move) {
                continue;
            }
            let Some(dest) = insn.operands.first() else { continue };
            let Operand::Memory { base: db, addend: da, .. } = dest else { continue };
            let dkey: Option<(u8, u64)> = match db.as_ref() {
                Operand::Local(v) => Some((0, *v as u64)),
                Operand::Register(n) => Some((1, slot_index(n)? as u64)),
                _ => None,
            };
            if dkey == Some(base_key) && *da == addend {
                let src = insn.operands.get(1).unwrap_or(&Operand::Unknown).clone();
                let prev = self.cur_insn.replace(j);
                let e = self.operand_expr(&src, depth + 1);
                self.cur_insn.set(prev);
                return Some(e);
            }
        }
        None
    }

    /// The VALUE produced by the definition at insn `i`.
    ///
    /// Cursor discipline: while the def's operands render, the cursor points
    /// AT the def (`i`), so nested `Local` operands resolve against the defs
    /// that reach the def — not the outer use site. Restored on the way out.
    fn def_expr(&self, i: usize, depth: usize) -> Expr {
        // Always counted (a thread-local `Cell` bump): these bound the
        // renderer's work, and the unit test for the def-cycle guard asserts on
        // them. Only the printing is diagnostic.
        let _calls = DEF_CALLS.with(|c| {
            let v = c.get() + 1;
            c.set(v);
            v
        });
        let _depth = DEF_DEPTH.with(|c| {
            let v = c.get() + 1;
            c.set(v);
            v
        });
        DEF_MAXDEPTH.with(|m| {
            if _depth > m.get() {
                m.set(_depth);
            }
        });
        if self.rss_dbg && _calls % 1_000_000 == 0 {
            eprintln!(
                "[def] calls={_calls} depth={_depth} maxdepth={}",
                DEF_MAXDEPTH.with(|m| m.get())
            );
        }
        // Back-edge re-entry guard (see `DEF_INSNS`). Only for a def that has
        // a NAME to reference — an anonymous store must keep its own rules.
        let dest_local = self.insns[i].dest().and_then(|d| d.as_local());
        if let Some(v) = dest_local {
            if DEF_INSNS.with(|s| s.borrow().contains(&i)) {
                CYCLE_CUTS.with(|c| c.set(c.get() + 1));
                return Expr::Var(self.var_name(v));
            }
            DEF_INSNS.with(|s| s.borrow_mut().push(i));
        }
        self.audit_claim(i);
        let prev = self.cur_insn.replace(i);
        let e = self.def_expr_inner(i, depth);
        self.cur_insn.set(prev);
        if dest_local.is_some() {
            DEF_INSNS.with(|s| {
                s.borrow_mut().pop();
            });
        }
        DEF_DEPTH.with(|c| c.set(c.get().saturating_sub(1)));
        e
    }

    fn def_expr_inner(&self, i: usize, depth: usize) -> Expr {
        let insn = &self.insns[i];
        match &insn.op {
            Op::Move => match insn.operands.get(1) {
                Some(Operand::Field(f)) => {
                    let recv = if f.is_static {
                        None
                    } else if self.in_declaring_chain(f.type_def_index) {
                        Some(Expr::This)
                    } else {
                        Some(Expr::Unknown)
                    };
                    self.field_expr(recv, f, depth)
                }
                // Field load through a derived receiver: `base.field`.
                Some(Operand::InstanceField { base, field }) => {
                    self.field_expr(Some(self.operand_expr(base, depth + 1)), field, depth)
                }
                Some(Operand::Memory { base, addend, index, scale, width, .. }) => {
                    // Frame-slot spill forwarding: the compiler spills live
                    // values (esp. floats, which clobber at every call) to
                    // the stack and reloads them right before the tail-merged
                    // call. Without forwarding, a reload renders as
                    // `*(float*)(sp+0x18)` and the whole expression chain
                    // stops there. If the nearest store to this exact slot
                    // reaches the load with no intervening call, render the
                    // STORED value instead — same frame slot ⇒ same value.
                    if let Some(fwd) = self.forward_memory_load(i, base, *addend, depth) {
                        fwd
                    } else {
                        let base_expr = self.mem_addr_expr(base, index.as_deref(), *scale, depth);
                        // A load from a member's own metadata-usage slot IS
                        // that member (see `operand_expr`) — same collapse, so
                        // the two construction sites cannot drift apart.
                        if *addend == 0 && index.is_none()
                            && matches!(base_expr, Expr::MethodGroup(_))
                        {
                            return base_expr;
                        }
                        Expr::RawDeref(Box::new(base_expr), *addend, *width)
                    }
                }
                Some(other) => self.operand_expr(other, depth),
                None => self.unk("def:move-without-operand", i),
            },
            Op::Arith(kind) => {
                let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                let b = self.operand_expr(insn.operands.get(2).unwrap_or(&Operand::Unknown), depth);
                match bin_of(*kind) {
                    Some(op) => Expr::Bin(op, Box::new(a), Box::new(b)),
                    // Rotates: explicit pseudo-op, never a lying infix token.
                    // `RORV` is how `BitOperations.RotateRight` / hash mixing /
                    // `(x >> n) | (x << (w - n))` reaches the CPU.
                    None => Expr::Call {
                        recv: None,
                        name: if matches!(kind, ArithKind::Rol) {
                            "RotateLeft"
                        } else {
                            "RotateRight"
                        }
                        .into(),
                        args: vec![a, b],
                    },
                }
            }
            Op::FloatArith(kind) => match kind {
                FloatArithKind::Neg => {
                    let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                    Expr::Neg(Box::new(a))
                }
                // scalar sqrt/abs/round: no C# operator — named intrinsic call
                // (same convention as CountLeadingZeros below), never a lie
                FloatArithKind::Sqrt | FloatArithKind::Abs | FloatArithKind::Floor
                | FloatArithKind::Ceiling | FloatArithKind::Truncate | FloatArithKind::Round => {
                    let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                    let name = match kind {
                        FloatArithKind::Sqrt => "Sqrt",
                        FloatArithKind::Abs => "Abs",
                        FloatArithKind::Floor => "Floor",
                        FloatArithKind::Ceiling => "Ceiling",
                        FloatArithKind::Truncate => "Truncate",
                        _ => "Round",
                    };
                    Expr::Call {
                        recv: None,
                        name: name.into(),
                        args: vec![a],
                    }
                }
                k => {
                    let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                    let b = self.operand_expr(insn.operands.get(2).unwrap_or(&Operand::Unknown), depth);
                    Expr::Bin(bin_of_float(*k), Box::new(a), Box::new(b))
                }
            },
            Op::Compare(kind) => {
                let a = self.compare_operand_expr(insn, i, 1, depth);
                let b = self.compare_operand_expr(insn, i, 2, depth);
                Expr::Cmp(cmp_of(*kind), Box::new(a), Box::new(b))
            }
            Op::Not => {
                let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                Expr::Not(Box::new(a))
            }
            Op::Negate => {
                let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                Expr::Neg(Box::new(a))
            }
            Op::NewObj(ty) => {
                let tname = ty.as_ref().map(|t| t.ty.name.clone()).unwrap_or_else(|| "object".into());
                Expr::NewObj(tname, vec![])
            }
            Op::Box(ty) => {
                let tname = ty.as_ref().map(|t| t.ty.name.clone()).unwrap_or_else(|| "object".into());
                Expr::Call { recv: None, name: format!("Box<{tname}>"), args: vec![] }
            }
            Op::LoadLength => {
                let arr = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                Expr::Field(Box::new(arr), "Length".into())
            }
            Op::NewArr(elem) => {
                // [dest, len] — render `new T[len]` with the element type from
                // metadata (was Expr::Unknown, hiding real allocations).
                let len = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                let elem_name = elem
                    .as_ref()
                    .map(|t| csharp_type(t).unwrap_or_else(|| "object".into()))
                    .unwrap_or_else(|| "object".into());
                Expr::NewArr(elem_name, vec![len])
            }
            Op::Convert(c) => {
                // [dest, src] — explicit cast so widening/narrowing stays
                // visible instead of silently becoming Unknown.
                let src = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                let target = match c {
                    Conversion::IntToFloat32 => "float",
                    Conversion::IntToFloat64 | Conversion::Float32ToFloat64 => "double",
                    Conversion::FloatToInt => "int",
                    Conversion::Float64ToFloat32 => "float",
                };
                Expr::Cast(target.to_string(), Box::new(src))
            }
            Op::MulAcc { negate_addend, .. } => {
                // dest = a*b ± c (fused). Standard Mul/Add ops render the
                // same arithmetic; the FMA rounding difference is not
                // representable in pseudo-C# and is not observable here.
                let a = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                let b = self.operand_expr(insn.operands.get(2).unwrap_or(&Operand::Unknown), depth);
                let c = self.operand_expr(insn.operands.get(3).unwrap_or(&Operand::Unknown), depth);
                let mul = Expr::Bin(BinOp::Mul, Box::new(a), Box::new(b));
                Expr::Bin(
                    if *negate_addend { BinOp::Sub } else { BinOp::Add },
                    Box::new(mul),
                    Box::new(c),
                )
            }
            Op::Clz => {
                // count-leading-zeros has no C# equivalent — a clearly named
                // pseudo-intrinsic keeps the operation honest.
                let v = self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), depth);
                Expr::Call {
                    recv: None,
                    name: "CountLeadingZeros".into(),
                    args: vec![v],
                }
            }
            Op::Simd(name) => {
                // ---- opcodes whose semantics ARE fully expressible -----------
                //
                // Naming these as pseudo-intrinsics preserved their operand
                // VALUES but threw the OPERATION away, and they dominate the
                // remaining stub population.
                //
                // `smulh`/`umulh` are how the compiler spells MULTIPLY-HIGH.
                // `umulh xN, xM, #magic` followed by `lsr` (and `lsr`+`shl` for
                // the rounding step) IS integer division by a constant, so
                // leaving them symbolic made every constant-divisor
                // computation — `DateTime`, `TimeSpan`, `decimal`, the BCL
                // numeric formatters — opaque, and the `>> s` / `<< 64-s` pair
                // around one high half read as nonsense. Both operands are
                // 64-bit GPRs by definition of the instruction, so the
                // widening product is unambiguous.
                let exact = match name.as_str() {
                    "SMULH" | "UMULH" => {
                        let unsigned = name == "UMULH";
                        let wide = if unsigned { "UInt128" } else { "Int128" };
                        let narrow = if unsigned { "nuint" } else { "nint" };
                        let widen = |k: usize| {
                            Expr::Cast(
                                wide.to_string(),
                                Box::new(self.operand_expr(
                                    insn.operands.get(k).unwrap_or(&Operand::Unknown),
                                    depth + 1,
                                )),
                            )
                        };
                        let product = Expr::Bin(
                            BinOp::Mul,
                            Box::new(widen(1)),
                            Box::new(widen(2)),
                        );
                        Some(Expr::Cast(
                            narrow.to_string(),
                            Box::new(Expr::Bin(
                                BinOp::Shr,
                                Box::new(product),
                                Box::new(Expr::Int(64)),
                            )),
                        ))
                    }
                    // `movi` broadcasts a constant into every lane. The 64-bit
                    // splat depends on the LANE WIDTH, and the decoder's printed
                    // immediate does not always carry it (`movi d0, #0xffffffff`
                    // is two 32-bit lanes, but a `4h, #0x0101` pattern would be
                    // four 16-bit ones). Rather than emit a plausible-but-wrong
                    // constant, only the WIDTH-INDEPENDENT immediates are
                    // resolved here: ZERO, and the all-ones pattern — which
                    // splat identically under every arrangement. Those are 311
                    // of the 335 `movi` on the reference binary; anything else
                    // keeps the stub below.
                    "MOVI" => match insn.operands.get(1) {
                        Some(Operand::Immediate(0)) => Some(Expr::Int(0)),
                        Some(Operand::Immediate(v)) if *v == 0xFFFF_FFFF || *v == -1 => {
                            Some(Expr::Int(-1))
                        }
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(e) = exact {
                    return e;
                }
                // A vector/rare opcode whose exact semantics are not modelled:
                // the mnemonic over every readable operand, destination
                // defined by the caller. See `Op::Simd`'s doc — this is a
                // missing NAME, never a lost value.
                let args: Vec<Expr> = insn.operands[1..]
                    .iter()
                    .map(|o| self.operand_expr(o, depth + 1))
                    .collect();
                Expr::Call {
                    recv: None,
                    name: format!("__{}", name.to_lowercase()),
                    args,
                }
            }
            Op::SimdLane(op) => {
                // A MODELLED lane operation. The IR carries the lane mapping as
                // DATA (see `SimdOp`), so the render STATES it instead of hiding
                // it behind a bare mnemonic: the picks are in the call's own name
                // (`arrange32[a3,a2,b1,b0]`), and the arguments are the registers
                // they are picked from — `A(n)` is lane n of operand 1, `B(n)` of
                // operand 2.
                //
                // This is deliberately NOT lowered to scalar arithmetic. A 128-bit
                // register holds four 32-bit lanes that this IR keeps as ONE
                // value, so `paddd` is genuinely elementwise: printing it as
                // `a + b` would carry BETWEEN the lanes and compute a different
                // number. Naming the operation is the same honest trade `Op::Simd`
                // makes, with the difference that here the operation is KNOWN
                // (the lane table is checked by the lifter's tests) rather than
                // missing.
                let mut args: Vec<Expr> = vec![self.operand_expr(
                    insn.operands.get(1).unwrap_or(&Operand::Unknown),
                    depth + 1,
                )];
                if op.reads_second_source() {
                    args.push(self.operand_expr(
                        insn.operands.get(2).unwrap_or(&Operand::Unknown),
                        depth + 1,
                    ));
                }
                Expr::Call {
                    recv: None,
                    name: format!("vector_{op}"),
                    args,
                }
            }
            Op::Select(variant) => {
                // `[dst, cond, a, b]`. `cond` is a boolean VALUE — the same
                // `Compare` scratch a branch consumes — and it must resolve the
                // SAME way a branch's does. `operand_expr` inlines a local's
                // reaching definition, but a condition defined in another block
                // has no single reaching def under the strict per-block lookup,
                // so it fell back to the local's NAME — and the local is the
                // lifter's internal scratch slot, whose name is its own id
                // (`t4294967293`). Nothing else can produce a `t4294967…`
                // identifier, so its presence in a body is proof of this leak.
                let cond = self.condition_value_expr(i, 1, depth);
                let a = self.operand_expr(
                    insn.operands.get(2).unwrap_or(&Operand::Unknown),
                    depth + 1,
                );
                let b = self.operand_expr(
                    insn.operands.get(3).unwrap_or(&Operand::Unknown),
                    depth + 1,
                );
                // The variant's transform stays on the false arm, so what the
                // instruction actually computed is visible rather than being
                // flattened into a plain `a : b`.
                let fals = match variant {
                    SelectVariant::Identity => b,
                    SelectVariant::Increment => {
                        Expr::Bin(BinOp::Add, Box::new(b), Box::new(Expr::Int(1)))
                    }
                    SelectVariant::Invert => Expr::Not(Box::new(b)),
                    SelectVariant::Negate => Expr::Neg(Box::new(b)),
                };
                Expr::Cond(Box::new(cond), Box::new(a), Box::new(fals))
            }
            Op::Call => self.call_value_expr(i, depth),
            Op::ObjcMsgSend { selector } => {
                // [dest, recv, sel-args…] — render as the selector-derived
                // method call: `recv.setBool_arg(false, key)` keeps C# valid
                // while carrying the REAL selector name.
                let fname = selector.replace(':', "_");
                let fname = fname.trim_end_matches('_').to_string();
                let recv = insn
                    .operands
                    .get(1)
                    .map(|o| self.operand_expr(o, depth + 1))
                    .map(Box::new);
                let args: Vec<Expr> = insn.operands[2..]
                    .iter()
                    .map(|o| self.operand_expr(o, depth + 1))
                    .collect();
                Expr::Call { recv, name: fname, args }
            }
            _ => self.unk("def:unsupported-op", (i, insn.address, insn.op.clone())),
        }
    }

    /// Resolve operand `k` of instruction `at` as a CONDITION VALUE.
    ///
    /// A condition is a boolean produced by `Op::Compare` — or by an
    /// `And`/`Or` of two of them, the fused `cmp; ccmp; b.cc` shape. There are
    /// exactly two consumers of one: the `CondJump` terminator
    /// (`Structurer::branch_condition`) and `Op::Select`. Both must resolve it
    /// the same way. Branches had this resolution; selects did not, so a select
    /// whose compare was defined in another block — or whose scratch slot the
    /// strict per-block reaching analysis declined to attribute — fell back to
    /// inlining the local's NAME, and that local is the lifter's internal
    /// scratch slot, whose name is its own id. Nothing else can produce a
    /// `t4294967…` identifier, so one appearing anywhere in a body is proof of
    /// exactly this leak (1,396 sites before this).
    ///
    /// `depth` MUST be threaded from the caller and incremented on the way in.
    /// The recursion here is mutually recursive with `def_expr`/`inline_local`,
    /// whose ONLY termination guarantee is that the depth strictly increases
    /// until `max_inline_depth` forces a variable reference. Restarting it at 0
    /// (the obvious-looking `def_expr(def, 0)`) makes a value graph with a
    /// cycle through a select's condition recurse without bound — a stack
    /// overflow, which in Rust aborts the process and loses the whole dump.
    fn condition_value_expr(&self, at: usize, k: usize, depth: usize) -> Expr {
        let Some(op) = self.insns.get(at).and_then(|i| i.operands.get(k)) else {
            return Expr::Unknown;
        };
        let Some(v) = op.as_local() else {
            return self.operand_expr(op, depth + 1);
        };
        if !super::lift::is_scratch(v) {
            return self.operand_expr(op, depth + 1);
        }
        // A scratch slot is written by exactly ONE definition (the lifter hands
        // out a fresh id per def), so an exact backward search is strictly
        // better than any block-relative heuristic.
        let def = (0..at)
            .rev()
            .find(|&j| self.insns[j].dest().and_then(|d| d.as_local()) == Some(v))
            .or_else(|| self.defs.get(&v).copied());
        let Some(d) = def else {
            return self.operand_expr(op, depth + 1);
        };
        // Fused compound condition: `Arith(And|Or)` over two compare results.
        // Render it with the boolean operators so a select's condition reads
        // `a == 0 || b == 0`, not a bitwise `&` of two booleans.
        if let Op::Arith(kind @ (ArithKind::And | ArithKind::Or)) = self.insns[d].op {
            let l = self.insns[d].operands.get(1).and_then(|o| o.as_local());
            let r = self.insns[d].operands.get(2).and_then(|o| o.as_local());
            if let (Some(lv), Some(rv)) = (l, r) {
                let ld = self.defs.get(&lv).copied();
                let rd = self.defs.get(&rv).copied();
                if let (Some(ld), Some(rd)) = (ld, rd) {
                    if matches!(self.insns[ld].op, Op::Compare(_))
                        && matches!(self.insns[rd].op, Op::Compare(_))
                    {
                        let binop = if kind == ArithKind::And {
                            BinOp::AndAlso
                        } else {
                            BinOp::OrElse
                        };
                        return Expr::Bin(
                            binop,
                            Box::new(self.def_expr(ld, depth + 1)),
                            Box::new(self.def_expr(rd, depth + 1)),
                        );
                    }
                }
            }
        }
        if matches!(self.insns[d].op, Op::Compare(_)) {
            return self.def_expr(d, depth + 1);
        }
        self.operand_expr(op, depth + 1)
    }

    /// Operand of a Compare with scratch-slot inlining: when the operand is
    /// the lifter's `_mask` scratch (defined by `And(reg, imm)` for a TBZ/TBNZ
    /// test), inline the defining And so the condition renders as
    /// `(reg & mask) != 0` instead of leaking the `_mask` scratch variable.
    fn compare_operand_expr(&self, insn: &Insn, i: usize, k: usize, depth: usize) -> Expr {
        let op = insn.operands.get(k).unwrap_or(&Operand::Unknown);
        if let Some(v) = op.as_local() {
            if v >= u32::MAX - 2 && depth < 6 {
                // find the defining And writing this scratch id (exact for
                // unique per-site ids; nearest-preceding for legacy ones)
                if let Some(def) = (0..i).rev().find(|&j| {
                    matches!(self.insns[j].op, Op::Arith(ArithKind::And))
                        && self.insns[j]
                            .dest()
                            .and_then(|d| d.as_local())
                            == Some(v)
                }) {
                    let d = &self.insns[def];
                    let a = self.operand_expr(d.operands.get(1).unwrap_or(&Operand::Unknown), depth + 1);
                    let b = self.operand_expr(d.operands.get(2).unwrap_or(&Operand::Unknown), depth + 1);
                    return Expr::Bin(BinOp::And, Box::new(a), Box::new(b));
                }
            }
        }
        self.operand_expr(op, depth)
    }

    /// Build the call expression for a Call insn, slicing args by signature
    /// and attaching the receiver for instance methods.
    fn call_parts(&self, i: usize, depth: usize) -> Option<CallParts> {
        let m = match self.insns[i].operands.first()? {
            Operand::Method(m) => m.clone(),
            _ => return None,
        };
        let ops = &self.insns[i].operands;
        let (recv, arg_start) = if m.sig.is_static {
            (None, 2.min(ops.len()))
        } else {
            let recv = ops
                .get(2)
                .map(|o| self.operand_expr(o, depth + 1))
                .map(Box::new);
            (recv, 3.min(ops.len()))
        };
        let mut args: Vec<Expr> = Vec::with_capacity(m.sig.params.len() + 1);
        for k in 0..m.sig.params.len() {
            let e = ops
                .get(arg_start + k)
                .map(|o| self.operand_expr(o, depth + 1))
                .unwrap_or_else(|| {
                    self.unk(
                        "call:no-operand-for-param",
                        (self.insns[i].address, k, m.sig.params.len(), ops.len(), arg_start, self.insns[i].op.clone()),
                    )
                });
            args.push(e);
        }
        // AAPCS64 INDIRECT RESULT REGISTER (x8).
        //
        // A struct-returning callee does not return through x0: the CALLER
        // passes the address of the destination buffer in x8, and the callee
        // writes the value through it. That pointer is a real argument, and it
        // is not one of the declared parameters — so this loop never read it
        // and the renderer emitted `GetTimeSlotStatus(slot, 0)` where the
        // native passes `&result` in x8 and reloads it afterwards from
        // `[sp,#0xc]`. Only a recovered value survives here; the lifter
        // records the operand at all only for the sret idiom.
        // A signature with enough parameters to reach the slot itself would
        // make `SRET_OPERAND` alias a real parameter — then it is a parameter,
        // not an indirect-result pointer, and must not be appended twice.
        let params_clear_of_sret = arg_start + m.sig.params.len() <= SRET_OPERAND;
        if params_clear_of_sret {
            if let Some(op) = ops.get(SRET_OPERAND) {
                if !matches!(
                    op,
                    Operand::Unknown | Operand::MissingAbiArg | Operand::Register(_)
                ) {
                    args.push(self.operand_expr(op, depth + 1));
                }
            }
        }
        // THERE IS NO TRAILING-ABI-FILLER CLEANUP HERE, and the loop that used
        // to be here was deleting declared parameters.
        //
        // The loop dropped every trailing `MissingAbiArg` on the theory that it
        // was register-dump filler. It cannot be: the argument list above is
        // built by iterating `0..m.sig.params.len()`, so it holds exactly ONE
        // entry per DECLARED parameter (plus, only when recovered, the x8
        // indirect-result pointer) — dump filler has no way in. Every
        // `MissingAbiArg` it removed was therefore a declared parameter whose
        // value the call site never handed over, and removing it shortened the
        // call: `IDeserializationCallback.OnDeserialization(object)` rendered as
        // `OnDeserialization()`, a shape no compiler emits and which reads as a
        // different overload. The operand itself is a POSITION fact about this
        // call's arity, so it stays and renders as `__abi`.
        //
        // Interior slots were always kept, for the same reason. The unresolved-
        // call path below DOES have dump filler (its operand list is the raw
        // register dump) and drops it there, which is where that cleanup
        // belongs.
        let name = if m.sig.is_static {
            format!("{}.{}", short_type(&m.declaring_type_name), m.name)
        } else {
            m.name.clone()
        };
        Some((recv, name, args))
    }

    /// `prettify_call`, plus a record of the fact that the CALL at `i` did not
    /// come out as a call.
    ///
    /// Every caller of `prettify_call` must go through this: `get_X()` folds to
    /// `obj.X`, which puts a real native `bl` in the text as a member access
    /// rather than as `ident(`, and the concordance counts C# calls by scanning
    /// for `ident(`. Without the record the fold is charged to the decompiler
    /// as a dropped call (10 of them on `SwitchToRectTransform` @0xFA08, which
    /// is the method's ENTIRE 10-call deficit). The set is handed to the verify
    /// ledger at the end of rendering, so the checker subtracts exactly the
    /// folds that survived into the output.
    fn prettify_call_at(&self, i: usize, recv: Option<Box<Expr>>, name: String, args: Vec<Expr>) -> Expr {
        let e = self.prettify_call(recv, name, args);
        if !matches!(e, Expr::Call { .. }) {
            self.property_fold_sites.borrow_mut().insert(i);
        }
        e
    }

    fn call_value_expr(&self, i: usize, depth: usize) -> Expr {
        let insn = &self.insns[i];
        match self.call_parts(i, depth) {
            Some((recv, name, args)) => self.prettify_call_at(i, recv, name, args),
            None => match insn.operands.first() {
                Some(Operand::Immediate(a)) => {
                    let args: Vec<Expr> = insn
                        .operands
                        .iter()
                        .skip(2)
                        .filter(|op| matches!(op, Operand::Local(_)))
                        .map(|op| self.operand_expr(op, depth + 1))
                        .collect();
                    // il2cpp metadata advantage: dest_type is seeded from
                    // the resolved method signature even when the callee
                    // name couldn't be resolved, so raw calls keep their
                    // return type instead of rendering untyped.
                    let ret = insn
                        .dest_type
                        .as_ref()
                        .and_then(csharp_type)
                        .filter(|t| t != "void");
                    Expr::RawCall(*a as u64, args, ret)
                }
                // INDIRECT call the resolver could not name: delegate /
                // function-pointer invocation, an interface dispatch, or a
                // vtable slot the type oracle could not key.
                //
                // It MUST still produce a VALUE. Returning `Expr::Unknown`
                // here was the single largest `__unknown` cluster in the dump:
                // the call's RESULT was discarded, and every downstream read of
                // it — the returned object's klass, its fields, the argument of
                // the next call that carries it, the branch that tests it —
                // rendered `__unknown`. That is how a method whose callee was
                // merely unnamed turned into `while (__unknown != 0)` with
                // `*(nint*)(*(nint*)(__unknown) +304)` inside it.
                //
                // The callee expression is preserved as the first argument so
                // the reader can still see WHICH slot/pointer was called
                // (`fnptr_0x... (*(nint*)(*(nint*)(elements) +776), …)`)
                // instead of the whole call vanishing.
                Some(callee) => {
                    let mut args: Vec<Expr> = vec![self.operand_expr(callee, depth + 1)];
                    for op in insn.operands.iter().skip(2) {
                        args.push(self.operand_expr(op, depth + 1));
                    }
                    // Drop the trailing ABI filler (`scrub_abi_call_args`
                    // marks register-dump slots `MissingAbiArg`; `Unknown`
                    // survives where that pass had no signature to scrub
                    // against). Both are POSITION-only facts about the tail,
                    // so trimming them cannot misalign a real argument.
                    while args
                        .last()
                        .is_some_and(|a| matches!(a, Expr::MissingAbiArg | Expr::Unknown))
                        && args.len() > 1
                    {
                        args.pop();
                    }
                    Expr::Call {
                        recv: None,
                        name: format!("fnptr_{:#x}", insn.address),
                        args,
                    }
                }
                None => self.unk("call:no-operands", i),
            },
        }
    }

    /// Recognize common BCL idioms and emit higher-level C#:
    /// - `String.Concat(a, b)` → `a + b`
    /// - `obj.get_X()` → `obj.X`
    /// - static `get_X()` → `X.Prop`
    fn prettify_call(&self, recv: Option<Box<Expr>>, name: String, mut args: Vec<Expr>) -> Expr {
        // string concatenation
        if name.ends_with(".Concat") && args.len() >= 2 {
            let first = args.remove(0);
            let second = args.remove(0);
            let mut e = Expr::Bin(BinOp::Add, Box::new(first), Box::new(second));
            for extra in args.drain(..) {
                e = Expr::Bin(BinOp::Add, Box::new(e), Box::new(extra));
            }
            return e;
        }
        // Property getters: `get_X()` with NO arguments is the property `X`.
        //
        // With arguments it is an INDEXER — `get_Item(int)` is the native
        // `this[int]` — and folding it to a bare `obj.Item` DELETED the index:
        // the reader saw a property read where the machine read `obj[i]`, on
        // every `list[i]` in the binary. Never drop an argument to prettify.
        if let Some(prop) = name.strip_prefix("get_") {
            if !prop.is_empty() && args.is_empty() {
                return match recv {
                    Some(r) => Expr::Field(r, prop.to_string()),
                    None => Expr::Var(prop.to_string()),
                };
            }
        }
        // STATIC getter: `call_parts` names a static method `Type.method`,
        // so the bare `get_` check above never matched and every static
        // property stayed a call — `v117 = CultureInfo.get_InvariantCulture();`
        // instead of `v117 = CultureInfo.InvariantCulture;`.
        if recv.is_none() && args.is_empty() {
            if let Some((ty, prop)) = name.rsplit_once(".get_") {
                if !ty.is_empty() && !prop.is_empty() {
                    return Expr::Var(format!("{ty}.{prop}"));
                }
            }
        }
        Expr::Call { recv, name, args }
    }

    /// Statement form of setter calls: obj.set_X(v) → obj.X = v;
    /// Type.set_X(v) → Type.X = v.
    fn prettify_setter_stmt(
        &self,
        m: &super::types::MethodRef,
        ops: &[Operand],
    ) -> Option<Stmt> {
        let prop = m.name.strip_prefix("set_")?;
        if prop.is_empty() || m.sig.params.len() != 1 {
            return None;
        }
        // Operand layout is uniform across every call op: `[target, retv,
        // this, params…]` (see `call_parts`). Read the value from the
        // PARAMETER slot by that layout — `ops.last()` only coincided with it
        // while no sret register had been appended.
        let arg_index = if m.sig.is_static { 2 } else { 3 };
        let value = ops
            .get(arg_index)
            .map(|o| self.operand_expr(o, 1))
            .unwrap_or_else(|| self.unk("setter:no-value-operand", m.name.clone()));
        if m.sig.is_static {
            // Proper AST assignment: Type.Prop = value;
            return Some(Stmt::Assign(
                format!("{}.{}", short_type(&m.declaring_type_name), prop),
                None,
                value,
            ));
        }
        // The RECEIVER is operand 2 — operand 0 is the method target and
        // operand 1 the return-value slot. Taking `ops.first()` used the
        // method target as the receiver, so every folded instance setter
        // rendered as `GetMethodInfo("T.set_X").X = v`, a receiver that does
        // not exist in the source.
        let recv = ops
            .get(2)
            .map(|o| self.operand_expr(o, 1))
            .unwrap_or_else(|| Expr::Var("this".into()));
        // Proper AST assignment: obj.Prop = value; (rendered via the
        // field-path LHS, not a synthetic call name).
        let mut lhs = String::new();
        recv.render(&mut lhs);
        lhs.push('.');
        lhs.push_str(prop);
        Some(Stmt::Assign(lhs, None, value))
    }

    /// Degraded rendering of an unresolved direct call: keep it VISIBLE as
    /// `fn_<rva>(args…)` instead of dropping the statement entirely.
    fn raw_call_stmt(&self, i: usize) -> Vec<Stmt> {
        let insn = &self.insns[i];
        // An INDIRECT call has no call target: the first operand is the
        // register holding the entry point. Rendering it as `fn_0(…)` named
        // nothing and hid the fact that the call is indirect — every such call
        // in the dump was a `fn_0`. Name it after the CALL SITE instead, which
        // is a real address the reader can look up in the disassembly, and say
        // `fnptr_` so it is never mistaken for a direct call.
        let indirect = !matches!(insn.operands.first(), Some(Operand::Immediate(_)));
        let target = match insn.operands.first() {
            Some(Operand::Immediate(a)) => *a as u64,
            _ => 0,
        };
        // ARGUMENT RECONSTRUCTION FOR A CALL WITH NO SIGNATURE.
        //
        // The operand list is `[target, dest, x0 … x7, x8?]`: registers that
        // hold no recovered value are `MissingAbiArg`/`Unknown`/raw
        // `Register`, and only the TRAILING ones may be dropped — an interior
        // one marks a real gap between two arguments.
        //
        // Three previous rules each deleted a real argument:
        //   * `Operand::Method(_)`/`Type(_)` were skipped outright, which
        //     dropped the pointer argument of the class-init idiom
        //     `adrp + add + bl` — 3 of the 4 `fn_4a58960(&slot)` calls on
        //     `CollectWeeklyFacebookGift` rendered as `fn_4a58960()`;
        //   * the adjacent-render dedup deleted a legitimately repeated
        //     argument (`f(x, x)`);
        //   * a hard 4-argument cap silently truncated every wider call.
        let mut args: Vec<Option<Expr>> = Vec::new();
        for op in insn.operands.iter().skip(2) {
            if matches!(
                op,
                Operand::Register(_) | Operand::MissingAbiArg | Operand::Unknown
            ) {
                args.push(None);
                continue;
            }
            args.push(Some(self.operand_expr(op, 1)));
        }
        while matches!(args.last(), Some(None)) {
            args.pop();
        }
        let args: Vec<Expr> = args
            .into_iter()
            .map(|a| a.unwrap_or(Expr::MissingAbiArg))
            .collect();
        let ret = insn
            .dest_type
            .as_ref()
            .and_then(csharp_type)
            .filter(|t| t != "void");
        let expr = if indirect {
            Expr::Call { recv: None, name: format!("fnptr_{:#x}", insn.address), args }
        } else {
            Expr::RawCall(target, args, ret)
        };
        match insn
            .dest()
            .and_then(|d| d.as_local())
            .filter(|_| matches!(insn.op, Op::Call))
            .filter(|d| self.call_result_is_consumed(*d))
        {
            // Only bind a name when some use site can reference it. The
            // class-init idiom `RuntimeClassInit(&slot)` (`fn_4a58960`) is a
            // pure side effect: the native `bl` has no consumer, so binding its
            // result emitted `v62 = fn_4a58960(0x6049a80);` for a local nothing
            // ever reads. Measured on the 4,000-method sample it was the largest
            // single source of write-only variables — 255 of the 1,235 (1,235
            // before this gate, 980 after). A bare call statement is the
            // faithful rendering: one native `bl`, one statement, no invented
            // variable.
            Some(d) => vec![Stmt::Assign(self.var_name(d), None, expr)],
            None => vec![Stmt::Eval(expr)],
        }
    }

    /// Is the result of the call defining local `d` read anywhere?
    fn call_result_is_consumed(&self, d: u32) -> bool {
        self.is_multi(d)
            || self.var_reads.contains(&d)
            || self.read_count.get(&d).copied().unwrap_or(0) > 0
    }

    /// Emit boundary statements for one instruction (calls, returns, throws,
    /// assignments to multi-def locals and memory stores).
    /// Statement form of one instruction (per-insn entry point from the
    /// structurer). Sets the render cursor so every operand resolved inside
    /// links to ITS reaching definition.
    fn boundary_stmts(&self, i: usize) -> Vec<Stmt> {
        let prev = self.cur_insn.replace(i);
        let out = self.boundary_stmts_inner(i);
        self.cur_insn.set(prev);
        if CALLTRACE.with(|c| c.get()) && Self::call_op(&self.insns[i]) {
            eprintln!(
                "[calltrace] i={} {:#x} {:?} target={:?} stmts={} void={} dest_local={:?} reads={:?}",
                i,
                self.insns[i].address,
                self.insns[i].op,
                self.insns[i].operands.first(),
                out.len(),
                Self::call_result_is_void(&self.insns[i]),
                self.insns[i].dest().and_then(|d| d.as_local()),
                self.insns[i]
                    .dest()
                    .and_then(|d| d.as_local())
                    .map(|d| (self.is_multi(d), self.read_count.get(&d).copied().unwrap_or(0))),
            );
        }
        out
    }

    fn boundary_stmts_inner(&self, i: usize) -> Vec<Stmt> {
        self.audit_claim(i);
        let insn = &self.insns[i];
        match &insn.op {
            Op::CallVoid => {
                if let Some(Operand::Method(m)) = insn.operands.first().cloned() {
                    if let Some(stmt) = self.prettify_setter_stmt(&m, &insn.operands) {
                        self.property_fold_sites.borrow_mut().insert(i);
                        return vec![stmt];
                    }
                }
                match self.call_parts(i, 0) {
                    // Route through `prettify_call`, not around it: a bare
                    // `Expr::Call` here skipped the getter/setter idioms and
                    // was why `get_X()` survived in void position.
                    Some((recv, name, args)) => {
                        vec![Stmt::Eval(self.prettify_call_at(i, recv, name, args))]
                    }
                    None => self.raw_call_stmt(i),
                }
            }
            Op::ObjcMsgSend { selector } => {
                // same void/multi-def policy as Op::Call, with the selector
                // as the method name.
                let fname = selector.replace(':', "_");
                let fname = fname.trim_end_matches('_').to_string();
                let recv = insn
                    .operands
                    .get(1)
                    .map(|o| self.operand_expr(o, 1))
                    .map(Box::new);
                let args: Vec<Expr> = insn.operands[2..]
                    .iter()
                    .map(|o| self.operand_expr(o, 1))
                    .collect();
                let expr = Expr::Call { recv, name: fname, args };
                // The result must be bound whenever a use site will reference
                // the name — multi-def OR read at more than one site (see
                // `call_dest_binding`).
                let bound = self.call_dest_binding(insn);
                if bound.is_none() {
                    return vec![Stmt::Eval(expr)];
                }
                let name = bound.unwrap();
                self.note_call_bound(i, &name);
                vec![Stmt::Assign(name, None, expr)]
            }
            Op::Call => {
                // A setter used for its EFFECT (`obj.set_X(v)`) is an assignment,
                // not a call. The `CallVoid` arm already folds it, but a setter
                // whose return the resolver did not prove void arrives HERE, and
                // here the call was emitted verbatim — so `obj.set_X(v)` stayed a
                // call instead of becoming `obj.X = v`. Only fold when the result
                // is UNUSED (a used setter result must stay an expression).
                if let Some(Operand::Method(m)) = insn.operands.first() {
                    let used = insn
                        .dest()
                        .and_then(|d| d.as_local())
                        .map(|d| self.read_count.get(&d).copied().unwrap_or(0))
                        .unwrap_or(0);
                    if used == 0 {
                        if let Some(stmt) = self.prettify_setter_stmt(m, &insn.operands) {
                            return vec![stmt];
                        }
                    }
                }
                // THREE-WAY RULE, and the middle case is the one that used to
                // lose the call.
                //
                //  (a) bindable result (incl. single-read) → emit
                //      `dst = callee(…)` here and let EVERY use site reference
                //      the name.
                //  (b) no result the C# can name (a param slot, an internal
                //      slot, `this`) but the result IS read → emit nothing here;
                //      the use site inlines the call expression. Emitting an
                //      eval here as well would double the `bl`.
                //  (c) result nobody reads → emit the bare call; the callee ran
                //      natively, so dropping it would delete a real effect.
                //
                // "Proven void" means the CALLEE's signature says void. An
                // unresolved call has `dest_type == None`, and treating that as
                // void made this arm emit a bare `call(…)` even when the result
                // IS read — so the reader inlined the very same call again and
                // the body contained the `bl` twice. Unknown is not void: a
                // call always leaves a value in x0.
                let void_ret = matches!(
                    insn.dest_type.as_ref(),
                    Some(t) if matches!(t, ManagedType::Primitive(PrimKind::Void))
                );
                let dest = insn.dest().and_then(|d| d.as_local());
                let reads = dest
                    .map(|d| self.read_count.get(&d).copied().unwrap_or(0))
                    .unwrap_or(0);
                let bound = self.call_dest_binding(insn);
                if void_ret || bound.is_some() {
                    match self.call_parts(i, 0) {
                        Some((recv, name, args)) => {
                            let expr = self.prettify_call_at(i, recv, name, args);
                            if void_ret {
                                return vec![Stmt::Eval(expr)];
                            }
                            let name = bound.unwrap();
                            // A block can be reached by two structuring paths
                            // (`fold_prelude` then the region walk). The name
                            // may exist only once: a second assignment would
                            // re-run the callee a second time in the text for
                            // ONE native `bl`.
                            if self.call_already_emitted(i) {
                                return vec![];
                            }
                            self.note_call_bound(i, &name);
                            return vec![Stmt::Assign(name, None, expr)];
                        }
                        // Unresolved target: `raw_call_stmt` still emits
                        // `dst = fn_0x…(…)`, so the binding exists and must be
                        // recorded — otherwise every use keeps inlining the call
                        // and one native `bl` renders N times (the exact case
                        // this binding exists to prevent).
                        None => {
                            if let Some(name) = bound.as_ref() {
                                if self.call_already_emitted(i) {
                                    return vec![];
                                }
                                self.note_call_bound(i, name);
                            }
                            return self.raw_call_stmt(i);
                        }
                    }
                }
                if reads == 0 {
                    return match self.call_parts(i, 0) {
                        Some((recv, name, args)) => {
                            vec![Stmt::Eval(self.prettify_call_at(i, recv, name, args))]
                        }
                        None => self.raw_call_stmt(i),
                    };
                }
                // (b): read, but not bindable. The use site expands the call.
                vec![]
            }
            Op::Return => {
                if std::env::var("ILR_RETDBG").is_ok() {
                    if let Some(Operand::Local(v)) = insn.operands.first() {
                        let bi = self.block_of_insn.as_deref().and_then(|m| m.get(i).copied());
                        let brange = bi.and_then(|b| self.blocks_range.get(b).copied());
                        let in_entry = bi.and_then(|b| {
                            self.reaching.get(b).and_then(|m| m.get(v).copied())
                        });
                        eprintln!(
                            "[retdbg] i={} v={} multi={} varreads={} cur={} reach={:?} defsites={:?} blk={:?} range={:?} in={:?}",
                            i,
                            v,
                            self.is_multi(*v),
                            self.var_reads.contains(v),
                            self.cur_insn.get(),
                            self.reaching_def_at(*v, i),
                            self.def_sites.get(v),
                            bi,
                            brange,
                            in_entry
                        );
                    }
                }
                let e = match insn.operands.first() {
                    Some(op @ (Operand::Local(_)
                    | Operand::Immediate(_)
                    | Operand::StringLiteral { .. }
                    | Operand::FloatLiteral(_)
                    | Operand::DoubleLiteral(_))) => self.operand_expr(op, 0),
                    Some(Operand::Field(f)) => {
                        let recv = if self.in_declaring_chain(f.type_def_index) {
                            Some(Expr::This)
                        } else {
                            None
                        };
                        self.field_expr(recv, f, 0)
                    }
                    _ => Expr::Null,
                };
                // A void method's native `ret` still carries x0 (SSA names
                // it `this` when nothing else wrote the register). C# cannot
                // `return <value>` from void. Two cases:
                //  - expr WITH side effects (a tail call like
                //    `fn_4a58bac(...)`) = the real last statement — emit it
                //    as an eval, never drop it;
                //  - a bare value (`this`) = epilogue noise — bare `return;`.
                // ONE NATIVE `bl` IS ONE EMITTED CALL.
                //
                // The value in the returned register is usually produced by a
                // call, and `operand_expr` inlines a local to its definition —
                // so this arm used to re-emit the whole call expression. When
                // the call's own site had already materialized
                // `v73 = fn_4a58bac(v17, v18, v19)` (the multi-def binding
                // path), the rendered body ended with BOTH statements for one
                // native instruction:
                //
                //     L2:
                //     v73 = fn_4a58bac(v17, v18, v19);
                //     fn_4a58bac(v17, v18, v19);      <- the same `bl`, again
                //
                // Verified on `CollectWeeklyFacebookGift` @0x47014C (arm64,
                // iOS): native `bl 0x4a58bac` at 0x470264, one instruction.
                // If the binding exists, reference the NAME instead.
                if let Some(v) = insn.operands.first().and_then(|o| o.as_local()) {
                    if let Some(d) = self.reaching_def_at(v, i) {
                        // A VOID `ret` CARRIES NO VALUE. x0 is just whatever the
                        // last call happened to leave in the register; the
                        // method's own return type is the authority.
                        //
                        // When the definition reaching here is a real
                        // instruction, that instruction was already emitted at
                        // ITS OWN site (a statement, or inlined at its uses) —
                        // so re-emitting it here is a second copy of one native
                        // `bl`. `Dirty()` @0x255a6b0 rendered
                        //     if (m_IsDirty == 0) { this.ClearCache(); }
                        //     this.ClearCache();
                        // for a single `bl` at 0x255a6dc, because the void `ret`
                        // in the epilogue re-inlined it. The old gate missed
                        // this: `call_site_statement_emitted` is false exactly
                        // when the result IS read, and a `ret` reads x0.
                        if self.ret_is_void {
                            // Already emitted at the call's own site → bare
                            // `return;`. NOT already emitted → this deferred
                            // value IS the call, so emit it (`Eval`) rather than
                            // let a real call vanish (measured: gating on
                            // "the reaching def is a call" alone moved
                            // dropped-call methods from 830 to 1006).
                            if self.ret_value_already_emitted(v, i) {
                                return vec![Stmt::Return(None)];
                            }
                            if expr_has_call(&e) {
                                return vec![Stmt::Eval(e)];
                            }
                            return vec![Stmt::Return(None)];
                        }
                        if self.call_site_statement_emitted(d) {
                            if let Some(name) = self.call_binding_name(d) {
                                return vec![Stmt::Return(Some(Expr::Var(name)))];
                            }
                            return vec![Stmt::Return(None)];
                        }
                    }
                }
                if self.ret_is_void && !matches!(e, Expr::Null) {
                    if expr_has_call(&e) {
                        return vec![Stmt::Eval(e)];
                    }
                    return vec![Stmt::Return(None)];
                }
                if std::env::var("ILR_RET").is_ok() {
                    let mut s = String::new();
                    e.render(&mut s);
                    eprintln!("[ret] void={} expr={:?} text={}", self.ret_is_void, e, s);
                }
                vec![Stmt::Return(if matches!(e, Expr::Null) { None } else { Some(e) })]
            }
            Op::Throw => {
                let e = insn
                    .operands
                    .first()
                    .map(|o| self.operand_expr(o, 0))
                    .unwrap_or(Expr::Null);
                vec![Stmt::Throw(e)]
            }
            // Pure value defs of var-read locals: a local that is READ as a
            // plain variable somewhere (merge/back-edge join) must have every
            // def materialized as an assignment, or the value silently
            // vanishes — loop bodies holding only arithmetic rendered empty
            // (`// loop body not recovered`) because no statement was ever
            // emitted for `v = v - 1`-class defs.
            Op::Arith(_) | Op::FloatArith(_) | Op::Not | Op::Negate | Op::Convert(_)
            | Op::MulAcc { .. } | Op::Clz | Op::LoadLength | Op::Select(_) => {
                self.value_def_stmt(i)
            }
            Op::Move => {
                // stores: into multi-def locals, or into memory (field stores)
                let d = insn.operands.first();
                // pure register shuffles (local→local, after inlining single-def
                // sources) carry no managed semantics — prologue/epilogue spill
                // traffic renders as `v1 = v0` shells. Drop them; anything
                // meaningful (fields, calls, constants) uses other operand kinds.
                // EXCEPTION: when the dest is a var-read local the copy IS the
                // value update (`v = v - 1` compiles to `t = v - 1; v = t`)
                // and dropping it orphans the variable — keep it.
                if let Some(Operand::Local(_)) = d {
                    if let Some(src) = insn.operands.get(1) {
                        let is_shuffle = match src {
                            Operand::Local(_) => true,
                            _ => false,
                        };
                        if is_shuffle {
                            // A copy OF A VOID CALL's result has no value to
                            // carry. The call's own statement already ran, and
                            // this copy is the SSA joiner for the ABI return
                            // register — `CollectWeeklyFacebookGift` @0x47014C
                            // ends in a TAIL call to a void method and the
                            // joiner rendered `v23 = …SetLastWeekFacebookGiftWasCollected(…)`,
                            // a second copy of the one native `b`. The method's
                            // own return type decides: for a void method the
                            // native `b` simply discards the value.
                            if self.ret_is_void {
                                if let Some(sv) = src.as_local() {
                                    if self.reaching_def_at(sv, i).is_some_and(|d| {
                                        Self::call_result_is_void(&self.insns[d])
                                            && self.call_site_statement_emitted(d)
                                    }) {
                                        return vec![];
                                    }
                                }
                            }
                            let needs_copy = d
                                .as_ref()
                                .and_then(|o| o.as_local())
                                .map(|v| self.var_reads.contains(&v) && v < u32::MAX - 2)
                                .unwrap_or(false);

                            if !needs_copy {
                                return vec![];
                            }
                        }
                    }
                }
                match d {
                    // A store into a MULTI-def local, or into a local that is
                    // READ as a plain variable. The second half matters for the
                    // loop-carried register of a rotated loop: `x0` there has
                    // exactly ONE static def (`x0 = [x0, #0x28]`) that feeds
                    // itself, so it is single-def, but it IS read as a variable
                    // (`var_reads`). Gating on `is_multi` alone silently dropped
                    // the only statement of such a loop body, and the loop
                    // rendered as an empty `do { } while (…)` — an infinite
                    // loop with the update thrown away.
                    Some(Operand::Local(v))
                        if self.is_multi(*v) || self.var_reads.contains(v) =>
                    {
                        let e = match insn.operands.get(1) {
                            Some(Operand::Field(f)) => {
                                let recv = if f.is_static {
                                    None
                                } else if self.in_declaring_chain(f.type_def_index) {
                                    Some(Expr::This)
                                } else {
                                    None
                                };
                                self.field_expr(recv, f, 0)
                            }
                            Some(other) => self.operand_expr(other, 0),
                            None => Expr::Unknown,
                        };
                        if matches!(e, Expr::Unknown) {
                            return vec![]; // dead default store — pure noise
                        }
                        let name = self.var_name(*v);
                        if name == "this" {
                            return vec![]; // never emit `this = …` — invalid C#
                        }
                        vec![Stmt::Assign(name, None, e)]
                    }
                    // A store into a RESOLVED field: `str x, [this+0x28]` after
                    // the resolver proved the receiver is this type's `this`.
                    //
                    // The resolver rewrites such a store to
                    // `Move(Field(f), value)`, and this match had no arm for a
                    // Field DESTINATION — it fell into the `_ => vec![]` sweep
                    // below and was dropped in silence. Field LOADS
                    // (`Move(local, Field(f))`) were handled, so a method kept
                    // its reads and lost its writes: `XmlAsyncCheckReader..ctor`
                    // @0x200D5F8 kept `this..ctor(); return;` and dropped
                    // `this.coreReader = reader;` and `this.lastTask = …`, with
                    // every CFG block "emitted", no marker, and nothing but the
                    // offset count to show for it. `ConstraintCollection.Clear`
                    // @0x1DD7A20 lost `this._delayLoadingConstraints = 0;` the
                    // same way.
                    Some(Operand::Field(f)) => {
                        let recv = if f.is_static {
                            None
                        } else if self.in_declaring_chain(f.type_def_index) {
                            Some(Expr::This)
                        } else {
                            None
                        };
                        let value = match insn.operands.get(1) {
                            Some(other) => self.operand_expr(other, 0),
                            None => Expr::Unknown,
                        };
                        if matches!(value, Expr::Unknown) {
                            return vec![]; // dead default store — pure noise
                        }
                        let mut lhs = String::new();
                        self.field_expr(recv, f, 0).render(&mut lhs);
                        if lhs.is_empty() {
                            return vec![];
                        }
                        vec![Stmt::Assign(lhs, None, value)]
                    }
                    // Field STORE through a derived receiver: `base.field = v`.
                    Some(Operand::InstanceField { base, field }) => {
                        let value = match insn.operands.get(1) {
                            Some(other) => self.operand_expr(other, 0),
                            None => Expr::Unknown,
                        };
                        if matches!(value, Expr::Unknown) {
                            return vec![]; // dead default store — pure noise
                        }
                        let recv = self.operand_expr(base, 0);
                        let mut lhs = String::new();
                        self.field_expr(Some(recv), field, 0).render(&mut lhs);
                        if lhs.is_empty() {
                            return vec![];
                        }
                        vec![Stmt::Assign(lhs, None, value)]
                    }
                    Some(Operand::Memory { .. }) => {
                        // A store through a pointer.
                        //
                        // This arm used to emit NOTHING unless the base was a
                        // local the resolver had typed as a managed CLASS (and
                        // then only a comment). Every other store — the sret
                        // buffer of a struct-returning method, an `out`/`ref`
                        // target reached through a register, an untyped pointer
                        // write — was dropped, and with it the whole value chain
                        // that fed it. The struct-return idiom is
                        // `str r5, [r4]` / `stm r4, {r5, r6}` with r4 = the
                        // caller's buffer, so `SafeStringMarshal
                        // MarshalString(string)` @0x17005f8 — 46 instructions,
                        // body `result = new SafeStringMarshal(str, false)` —
                        // rendered as `return v11;`.
                        match insn.operands.first() {
                            Some(Operand::Memory { base, addend, index, scale, width, .. }) => {
                                // Typed managed-class store: name the member.
                                let base_ty = base
                                    .as_local()
                                    .and_then(|v| self.table.index_of.get(&v))
                                    .map(|&li| &self.table.infos[li].ty);
                                if let Some(ManagedType::Class(nt)) = base_ty {
                                    if nt.type_def_index >= 0 {
                                        return vec![Stmt::Comment(format!(
                                            "store to {}.{}+{addend:#x}",
                                            nt.name,
                                            ""
                                        ))];
                                    }
                                }
                                // Frame spills are ABI plumbing, not source:
                                // keep dropping stores whose base is `sp`.
                                let base_expr = self.mem_addr_expr(base, index.as_deref(), *scale, 1);
                                let mut btxt = String::new();
                                base_expr.render(&mut btxt);
                                if btxt == "sp" || btxt.starts_with("sp ") || btxt.starts_with("sp+") {
                                    return vec![];
                                }
                                let value =
                                    self.operand_expr(insn.operands.get(1).unwrap_or(&Operand::Unknown), 0);
                                if matches!(value, Expr::Unknown) {
                                    return vec![]; // dead default store — pure noise
                                }
                                let mut lhs = String::new();
                                Expr::RawDeref(Box::new(base_expr), *addend, *width).render(&mut lhs);
                                vec![Stmt::Assign(lhs, None, value)]
                            }
                            _ => vec![],
                        }
                    }
                    _ => vec![],
                }
            }
            _ => vec![],
        }
    }

    /// Locals whose value CANNOT be inlined at one of their use sites, so that
    /// use reads the local by NAME.
    ///
    /// [`compute_var_reads`] decides "must this def become an assignment?" from
    /// the reaching-definitions graph alone, on the assumption that a SINGLE-def
    /// local is inlined into every use. [`Self::inline_local_at`] has four
    /// further refusals that the graph cannot see, and each one falls back to a
    /// bare `vN` — a name NO statement assigns, which is exactly the
    /// `def-dropped` class of the undefined-local census:
    ///
    ///  * no def reaches the use (`def_at == None`) and the local is not a pure
    ///    copy alias, so the render reads it as the entry/`vN` name;
    ///  * the reaching def sits LATER in the use's own block (a back edge), so
    ///    inlining it would emit a self-referential expression;
    ///  * [`Self::def_moved_across_a_write`] — a local the def READS is itself
    ///    rewritten between the def and a cross-block use;
    ///  * [`Self::def_rewrites_the_name_it_assigns`] — the def's own expression
    ///    reads the storage it assigns (`v266 = v266 - 1`, a loop's induction
    ///    update), so re-expanding it at a later use re-applies the update.
    ///
    /// The third one catches loop-carried counters constantly. `0x1532130`
    /// (a static cctor) holds `v15 = φ(init, v299)` and `v299 = v15 - 1` in the
    /// latch; the latch's own back-edge copy is `v15 = v299` at the very next
    /// instruction, so every cross-block read of `v299` — 384 of them, passed
    /// to dozens of `IL2CPP_ClassInit` thunks — printed a bare `v299` with
    /// nothing defining it. Marking the local var-read makes its def a real
    /// statement (`v299 = v266 - 1;`), which is precisely the value those reads
    /// mean; the def's own expression is unchanged.
    ///
    /// A refusal that keeps a CALL expression inlined is deliberately NOT
    /// counted: that inline is the only place the native `bl` reaches the text,
    /// so materialising the def would still render the same call and the local
    /// would stay unassigned anyway.
    fn refused_inline_reads(&self) -> HashSet<u32> {
        let mut out: HashSet<u32> = HashSet::new();
        for (u, insn) in self.insns.iter().enumerate() {
            for v in insn.use_locals() {
                if v >= u32::MAX - 2 || out.contains(&v) {
                    continue;
                }
                match self.reaching_def_at(v, u) {
                    None => {
                        // `inline_local_at` renders a pure alias through the
                        // alias chain, so only a non-alias read needs a name —
                        // and that decision needs no def to reach, so it is the
                        // same answer `compute_var_reads` gives at a merge.
                        if pure_alias_root(self.insns, &self.def_sites, v).is_none() {
                            out.insert(v);
                        }
                    }
                    Some(d) if d == u => {}
                    Some(d) => {
                        let same_block_later = d > u
                            && self
                                .block_of_insn
                                .as_ref()
                                .is_some_and(|m| m.get(d).copied() == m.get(u).copied());
                        if same_block_later {
                            out.insert(v);
                            continue;
                        }
                        if (self.def_rewrites_the_name_it_assigns(d)
                            && !self.def_chain_has_call(d))
                            || (self.def_moved_across_a_write(d, u)
                                && !self.def_chain_has_call(d))
                        {
                            out.insert(v);
                        }
                    }
                }
            }
        }
        out
    }

    /// Assignment statement for a pure value def (`v = <expr>`) of a
    /// var-read local. Emits nothing when the value resolves to Unknown
    /// (dead default store) or the local is never read as a bare variable
    /// (every use inlines the def, so a statement would duplicate it).
    fn value_def_stmt(&self, i: usize) -> Vec<Stmt> {
        let Some(Operand::Local(v)) = self.insns[i].dest() else { return vec![] };
        let v = *v;
        if v >= u32::MAX - 2 || !self.var_reads.contains(&v) {
            return vec![];
        }
        let e = self.def_expr(i, 0);
        if matches!(e, Expr::Unknown) {
            return vec![]; // dead default store — pure noise
        }
        let name = self.def_var_name(v);

        vec![Stmt::Assign(name, None, e)]
    }
}

// ---------------------------------------------------------------------------
// CFG structuring
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct BlockInfo {
    start: usize,
    end: usize,
    succ_taken: Option<usize>,
    succ_fall: Option<usize>,
}

struct StructuredCfg {
    blocks: Vec<BlockInfo>,
    /// instruction index → block index (branch targets in IR are INSN
    /// indices; the structurer works in BLOCK indices — conflating the two
    /// silently dropped else-arms and emitted `goto L<insn>`)
    block_of: Vec<usize>,
    /// FULL successor list per block (n-way, from the CFG). The two-slot
    /// succ_taken/succ_fall shape cannot hold a switch's case targets;
    /// the arm walker uses this to force labels over the arm-reachable
    /// region so no case/join goto dangles.
    full_succ: Vec<Vec<usize>>,
    labels: HashMap<usize, String>,
    /// Every name in `labels`, for collision checks: a block with no assigned
    /// name is auto-named `L<block index>`, and `labels` numbers its names in
    /// CFG-TARGET order — so `L7` can already belong to an unrelated block.
    label_names: HashSet<String>,
    idom: Vec<Option<usize>>,
    /// Immediate post-dominator per block (None for exit / unreachable-to-exit).
    /// The ipdom of a branch header IS its merge point — exact, no step limits.
    ipdom: Vec<Option<usize>>,
}

/// Reaching-definitions analysis over the lifted IR (Cytron-style,
/// block-level GEN/KILL with a worklist fixpoint).
///
/// Returns, per block, the map `local version → def insn index` of the
/// block's IN set. A local's IN map holds a def ONLY when that single def
/// reaches; a merge of conflicting defs (loop header join, branch merge)
/// removes the entry — the use then renders as the named variable, which
/// is semantically correct C#.
struct ReachingResult {
    /// per block: (start, end) insn range
    blocks: Vec<(usize, usize)>,
    /// per block IN set: version → def insn
    gen_kill_in: Vec<HashMap<u32, usize>>,
}

/// Free-standing variant of `MethodCtx::reaching_def_at` (callable while
/// MethodCtx is still under construction): the single definition of `v`
/// reaching `use_insn`, or None when the use is a variable read (entry
/// value / merge of conflicting defs).
fn reaching_def_at_free(
    insns: &[Insn],
    in_sets: &[HashMap<u32, usize>],
    blocks_range: &[(usize, usize)],
    block_of_insn: Option<&[usize]>,
    v: u32,
    use_insn: usize,
) -> Option<usize> {
    let cfg_blocks = block_of_insn?;
    if *cfg_blocks.get(use_insn)? == usize::MAX {
        return None;
    }
    let bi = cfg_blocks[use_insn];
    let Some(&block) = blocks_range.get(bi) else { return None };
    // scan the block up to the use: a def between start and use wins
    let mut def: Option<usize> = in_sets.get(bi).and_then(|m| m.get(&v).copied());
    for i in block.0..block.1.min(use_insn + 1) {
        if let Some(Some(d)) = insns[i].dest().map(|d| d.as_local()) {
            if d == v {
                def = Some(i);
            }
        }
    }
    // the use insn itself defining `v` is a write, not a readable def
    if def == Some(use_insn) {
        return None;
    }
    def
}

/// Locals that are READ as plain variables somewhere — a use point where no
/// single definition provably reaches (a merge/back-edge join). Such reads
/// render as the bare local name, so every def of the local must be
/// materialized as an `name = value;` assignment statement. Without this,
/// pure value defs (loop counters, pointer arithmetic — `Arith`/`Convert`
/// etc.) produced NO statement at all: `boundary_stmts` only emitted
/// calls/moves, so loop bodies holding only arithmetic rendered empty
/// (`// loop body not recovered`) and the arithmetic silently vanished.
/// Pure copy-alias locals (every def `v ← w` with the same `w`) are
/// excluded: merge reads already resolve them through the alias chain, so
/// no statement is needed.
/// Every local used as an INPUT anywhere in the listing — the destination
/// position of each instruction is skipped (a def is not a read).
/// How many times each local is READ (destination positions excluded).
///
/// `read_locals` answers "is this value consumed at all"; this answers "is it
/// consumed MORE THAN ONCE", which is what decides whether a value has to be
/// bound to a name.
///
/// Why a count and not just the set: a call result that a single-def local holds
/// and that is read at N sites was inlined at all N of them, so the body showed
/// the SAME native `bl` N times. That is not a rendering detail — the emitted
/// program contains N calls where the disassembly has one. Measured on
/// 0x2478f0c: 389 call expressions for 218 native `bl`, one inlined
/// `fn_4a589d8(...)` repeated across every guard of the chain.
fn compute_read_counts(insns: &[Insn]) -> HashMap<u32, usize> {
    let mut out: HashMap<u32, usize> = HashMap::new();
    // `use_locals`, not a skip of the destination SLOT: a `Move` whose
    // destination is a `Memory` is a STORE and its base is a read (see
    // [`compute_var_reads`]). A result used ONLY as a store base therefore had
    // `read_count == 0`, so `call_result_is_consumed` was false and the call was
    // emitted as a bare `fn_9000(…);` — with the store then referencing a name
    // nothing assigned, and the SAME native `bl` reached the text twice.
    for insn in insns {
        for r in insn.use_locals() {
            *out.entry(r).or_insert(0) += 1;
        }
    }
    out
}

/// Count operands that take the ADDRESS of a local's storage.
///
/// Two shapes: `AddressOf(Local(v))` (the version addressed directly) and
/// `AddressOf(Memory{ base: Local(v), .. })` (the frame slot whose address is
/// handed to a callee). `AddressOf(Memory{ base: sp, .. })` is the address of a
/// raw stack slot rather than of a local version, and a bare `Memory{ base:
/// Local(v) }` uses v's VALUE as a pointer — neither is an escape of v's
/// storage, so neither is counted.
fn count_address_escapes(op: &Operand) {
    use std::sync::atomic::Ordering::Relaxed;
    match op {
        Operand::AddressOf(inner) => {
            let escapes = match inner.as_ref() {
                Operand::Local(_) => true,
                Operand::Memory { base, .. } => matches!(base.as_ref(), Operand::Local(_)),
                _ => false,
            };
            if escapes {
                super::timing::ADDR_ESCAPES.fetch_add(1, Relaxed);
            }
            count_address_escapes(inner);
        }
        Operand::Memory { base, index, .. } => {
            count_address_escapes(base);
            if let Some(i) = index {
                count_address_escapes(i);
            }
        }
        _ => {}
    }
}

/// Locals that must be MATERIALISED rather than copied into every use: a
/// single-def local read more than once whose inlined expansion reaches
/// `limit` expression nodes. See [`SHARED_EXPR_NODES`] for the failure this
/// prevents.
///
/// The size is the number of nodes inlining would produce, saturating at
/// `limit`. A multi-def local is a leaf (the renderer references it by name
/// already) and a def cycle is a leaf (its own renderer guard does the same),
/// so this metric only ever flags values that would genuinely be copied.
pub(crate) fn compute_shared_large(
    insns: &[Insn],
    defs: &HashMap<u32, usize>,
    multi_def: &HashSet<u32>,
    read_count: &HashMap<u32, usize>,
    limit: u64,
) -> HashSet<u32> {
    #[allow(clippy::too_many_arguments)]
    fn size_of(
        v: u32,
        insns: &[Insn],
        defs: &HashMap<u32, usize>,
        multi_def: &HashSet<u32>,
        memo: &mut HashMap<u32, u64>,
        on_stack: &mut HashSet<u32>,
        limit: u64,
    ) -> u64 {
        if let Some(&s) = memo.get(&v) {
            return s;
        }
        if multi_def.contains(&v) {
            return 1;
        }
        let Some(&i) = defs.get(&v) else { return 1 };
        if !on_stack.insert(v) {
            return 1; // cycle: the renderer references the name
        }
        let dpos = Insn::dest_position(&insns[i].op);
        let mut total: u64 = 1;
        'outer: for (k, op) in insns[i].operands.iter().enumerate() {
            if Some(k) == dpos {
                continue;
            }
            let mut ls: Vec<u32> = Vec::new();
            op.uses(&mut ls);
            for w in ls {
                total = total
                    .saturating_add(size_of(w, insns, defs, multi_def, memo, on_stack, limit));
                if total >= limit {
                    break 'outer;
                }
            }
        }
        on_stack.remove(&v);
        let total = total.min(limit);
        memo.insert(v, total);
        total
    }
    let mut memo: HashMap<u32, u64> = HashMap::new();
    let mut out: HashSet<u32> = HashSet::new();
    for &v in defs.keys() {
        if read_count.get(&v).copied().unwrap_or(0) < 2 {
            continue;
        }
        let s = size_of(v, insns, defs, multi_def, &mut memo, &mut HashSet::new(), limit);
        if s >= limit {
            out.insert(v);
        }
    }
    out
}

fn compute_var_reads(
    insns: &[Insn],
    multi_def: &HashSet<u32>,
    in_sets: &[HashMap<u32, usize>],
    blocks_range: &[(usize, usize)],
    block_of_insn: &[usize],
    def_sites: &HashMap<u32, Vec<usize>>,
) -> HashSet<u32> {
    let mut out: HashSet<u32> = HashSet::new();
    for (i, insn) in insns.iter().enumerate() {
        // `use_locals` — NOT a hand-rolled skip of the destination SLOT. A
        // `Move` whose destination is a `Memory` is a STORE, and its base and
        // index are READS of the value that computes the address. Skipping
        // operand#0 wholesale hid every store base from every read analysis in
        // this file, so a local read only as a store target —
        // `*(nint*)(v193) = …`, fifteen times in `0x12fca60`, and the same
        // shape behind most of the remaining `def-dropped` census — never
        // counted as read, its def was never materialised, and the stores
        // printed a name no statement assigns. `ssa::rename_uses` states the
        // same rule for the same reason ("a MEMORY destination is a STORE: its
        // base and index registers are real reads").
        for v in insn.use_locals() {
            // scratch compare/mask slots render via their defining Compare/And
            // (inlined into the branch condition), never as variables
            if v >= u32::MAX - 2 || out.contains(&v) {
                continue;
            }
            // A local whose defs all lie AFTER this use cannot be inlined here
            // — the renderer resolves a use by searching BACKWARD from it, so a
            // def past it is invisible and the value is loop-carried. Such a
            // local must be materialised even when it has a single def.
            //
            // Without this, `v63 = v30 + 16` in a loop's latch is skipped (its
            // only read is in the header, which precedes it), the assignment is
            // never emitted, the body's single statement disappears and the
            // loop renders as `while (…) { // loop body not recovered }`.
            // Measured: 543 such bodies in Assembly-CSharp alone.
            let defs_all_after_use = def_sites
                .get(&v)
                .is_some_and(|s| !s.is_empty() && s.iter().all(|&d| d > i));
            if defs_all_after_use {
                out.insert(v);
                continue;
            }
            if !multi_def.contains(&v) {
                continue;
            }
            if reaching_def_at_free(insns, in_sets, blocks_range, Some(block_of_insn), v, i).is_some()
            {
                continue; // that use inlines its def — no bare-variable read
            }
            // a merge/entry read: if every def is a copy of one source the
            // renderer already follows the alias — no assignment needed
            if pure_alias_root(insns, def_sites, v).is_some() {
                continue;
            }
            out.insert(v);
        }
    }
    out
}

/// Mirror of `MethodCtx::copy_alias`: Some(root) when every def site of `v`
/// is a `Move [v, Local(w)]` copying the SAME source (phi-elimination
/// copies) — the local is a pure alias and reads resolve through the chain.
fn pure_alias_root(insns: &[Insn], def_sites: &HashMap<u32, Vec<usize>>, v: u32) -> Option<u32> {
    let mut seen = std::collections::HashSet::new();
    let mut cur = v;
    loop {
        if !seen.insert(cur) {
            return None;
        }
        let sites: &[usize] = def_sites.get(&cur).map(|v| v.as_slice()).unwrap_or(&[]);
        if sites.is_empty() {
            return Some(cur);
        }
        let mut src: Option<u32> = None;
        for &di in sites {
            let insn = &insns[di];
            if !matches!(insn.op, Op::Move) {
                return None;
            }
            let Some(Operand::Local(w)) = insn.operands.get(1) else { return None };
            match src {
                None => src = Some(*w),
                Some(s) if s == *w => {}
                _ => return None,
            }
        }
        let w = src?;
        if w == cur {
            return None;
        }
        cur = w;
    }
}

fn compute_reaching_definitions(insns: &[Insn]) -> ReachingResult {
    let cfg = super::ssa::rebuild_cfg(insns);
    let n_blocks = cfg.blocks.len();
    let blocks_range: Vec<(usize, usize)> = cfg
        .blocks
        .iter()
        .map(|b| (b.start, b.end))
        .collect();

    // GEN/OUT per block: the LAST def per version inside the block. A block
    // that defines `v` kills every other def of `v` on exit.
    let mut out_sets: Vec<HashMap<u32, usize>> = vec![HashMap::new(); n_blocks];
    let mut has_def: Vec<HashSet<u32>> = vec![HashSet::new(); n_blocks];
    for (bi, b) in cfg.blocks.iter().enumerate() {
        for i in b.start..b.end {
            if let Some(Some(d)) = insns[i].dest().map(|d| d.as_local()) {
                out_sets[bi].insert(d, i);
                has_def[bi].insert(d);
            }
        }
    }

    // predecessors from successor lists
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n_blocks];
    for (bi, succs) in cfg.successors.iter().enumerate() {
        for &s in succs {
            if !preds[s].contains(&bi) {
                preds[s].push(bi);
            }
        }
    }

    // IN = ∪ preds' OUT, minus versions the block itself defines (those use
    // the in-block def), then GEN overrides. Entry block IN = empty (params
    // come from the local table, not defs).
    //
    // CONFLICT TOMBSTONES: a version with two different defs reaching the
    // same point must be dropped from IN entirely — that is the φ-point case,
    // and the value then renders as the named variable (each def gets an
    // assignment) instead of an arbitrary path's expression being inlined.
    //
    // The previous implementation removed the entry on conflict but left no
    // record, so the NEXT predecessor simply re-inserted its own def and the
    // LAST predecessor won. Every `ret` fed by several predecessors (the
    // compiler's universal early-return/guard shape) therefore resolved to a
    // single path's value: `GetEffectiveSchemaType` @0x1f4e79c returned
    // `GetBuiltInSimpleType(…)` on all five paths where the native returns
    // `[elem+200]`, `[elem+184]`, `0`, `get_Item(…)` and `GetBuiltInSimpleType`
    // respectively. `var_reads` was computed from the same broken IN set, so
    // the shared local was not even materialized as a variable.
    let gen_sets = out_sets.clone();
    let mut in_sets: Vec<HashMap<u32, usize>> = vec![HashMap::new(); n_blocks];
    let mut changed = true;
    let mut iter_budget = 64usize; // hard cap: CFGs are small, this never hits
    while changed && iter_budget > 0 {
        changed = false;
        iter_budget -= 1;
        // OUT is a pure function of IN — recompute it every pass so a pass
        // never meets over the previous pass's stale OUT.
        for bi in 0..n_blocks {
            let mut out = in_sets[bi].clone();
            for (&v, &d) in &gen_sets[bi] {
                out.insert(v, d);
            }
            out_sets[bi] = out;
        }
        for bi in 0..n_blocks {
            let mut new_in: HashMap<u32, usize> = HashMap::new();
            let mut conflicted: HashSet<u32> = HashSet::new();
            for &p in &preds[bi] {
                for (&v, &d) in &out_sets[p] {
                    if conflicted.contains(&v) {
                        continue; // already proven ambiguous — stays dropped
                    }
                    match new_in.get(&v) {
                        Some(&old) if old != d => {
                            new_in.remove(&v);
                            conflicted.insert(v);
                        }
                        Some(_) => {}
                        None => {
                            new_in.insert(v, d);
                        }
                    }
                }
            }
            // kill: versions defined in this block never inherit a def
            for &v in &has_def[bi] {
                new_in.remove(&v);
            }
            if new_in != in_sets[bi] {
                in_sets[bi] = new_in;
                changed = true;
            }
        }
    }

    ReachingResult { blocks: blocks_range, gen_kill_in: in_sets }
}

fn analyze_blocks(insns: &[Insn]) -> StructuredCfg {
    let cfg = super::ssa::rebuild_cfg(insns);
    let mut block_of: Vec<usize> = vec![0; insns.len()];
    for (bi, b) in cfg.blocks.iter().enumerate() {
        for i in b.start..b.end {
            if i < block_of.len() {
                block_of[i] = bi;
            }
        }
    }
    let mut blocks: Vec<BlockInfo> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, b)| BlockInfo {
            start: b.start,
            end: b.end,
            // semantic slots: slot 0 = taken edge, slot 1 = fallthrough.
            // Switch blocks expose their first case target as `taken` (the
            // structurer renders the dispatch itself via per-case labels).
            succ_taken: cfg.successors[bi].get(super::cfg::SUCCESSOR_TAKEN).copied(),
            succ_fall: cfg.successors[bi].get(super::cfg::SUCCESSOR_FALL).copied(),
        })
        .collect();
    let n = blocks.len();
    for b in &mut blocks {
        if b.succ_taken.is_some_and(|t| t >= n) {
            b.succ_taken = None;
        }
        if b.succ_fall.is_some_and(|t| t >= n) {
            b.succ_fall = None;
        }
    }
    // A block whose LAST insn is NOT a conditional branch has no "taken"
    // edge: its single successor is a fallthrough, but the N-way successor
    // list stores it at slot 0 (= taken). The linear walker advances on
    // `succ_fall`, so a terminator-less block (a branch target cutting a
    // straight run, very common right before loop bodies) silently ended
    // the walk there — everything after it (the loop!) was unreachable.
    // Jump/Switch targets are read from the op itself, so their slot does
    // not matter.
    for b in blocks.iter_mut() {
        let term = insns[b.end.saturating_sub(1)].op.clone();
        // A conditional branch normally keeps POSITIONAL slots [taken, fall].
        // But when its TAKEN target left the method, `Cfg::build` resolved it
        // to the one-past-the-end sentinel and the `.flatten()` dropped that
        // edge — so the fallthrough slid into slot 0 and `succ_fall` read as
        // `None`. `emit_if` then had no false arm and emitted an EMPTY body for
        // every `if (…) return …;` guard (the `IsCompilerGenerated` class).
        // Recover the true slots from the terminator: a sentinel target means
        // the single successor IS the fallthrough.
        if let Op::CondJump(t) = term {
            if b.succ_fall.is_none() && t >= insns.len() {
                b.succ_fall = b.succ_taken;
                b.succ_taken = None;
            }
            continue;
        }
        if b.succ_fall.is_none() {
            b.succ_fall = b.succ_taken;
            b.succ_taken = None;
        }
    }

    // labels for non-fallthrough targets
    let mut targets: BTreeSet<usize> = BTreeSet::new();
    for (bi, b) in blocks.iter().enumerate() {
        for succ in [b.succ_taken, b.succ_fall].into_iter().flatten() {
            if bi + 1 != succ {
                targets.insert(succ);
            }
        }
    }
    let labels: HashMap<usize, String> = targets
        .into_iter()
        .enumerate()
        .map(|(k, blk)| (blk, format!("L{k}")))
        .collect();
    let label_names: HashSet<String> = labels.values().cloned().collect();

    // immediate dominators via the shared efficient implementation: reuse
    // DominatorInfo by adapting to its Cfg shape
    let adapter = super::cfg::Cfg {
        blocks: cfg.blocks.clone(),
        successors: cfg.successors.clone(),
    };
    let dom_info = super::dominator::DominatorInfo::compute(&adapter);

    // Immediate post-dominators: dominator analysis on the REVERSED graph.
    // Build a synthetic reversed Cfg and reuse DominatorInfo. Virtual exit
    // node n = one-past-the-last block; every block with no successors (or
    // only out-of-range successors) gets an edge to it.
    let rev_succs: Vec<Vec<usize>> = {
        let mut s: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (bi, succs_of) in cfg.successors.iter().enumerate() {
            // FULL n-way successor set — the old pair-based shape silently
            // truncated 3+ edges to the same target, skewing post-dominators
            // for switch-heavy methods.
            let mut succs: Vec<usize> =
                succs_of.iter().copied().filter(|&t| t < n).collect();
            if succs.is_empty() {
                succs.push(n); // edge to virtual exit
            }
            for &t in &succs {
                s[t].push(bi);
            }
        }
        s
    };
    let rev_cfg = super::cfg::Cfg {
        blocks: (0..=n).map(|i| super::cfg::Block { start: i, end: i + 1 }).collect(),
        successors: rev_succs,
    };
    // ipdom[b] = idom of b in the reversed graph whose ROOT is the virtual
    // exit n. compute_from(n) makes every node that can reach the exit
    // dominated-by-exit in reverse — i.e. post-dominated by its true merge
    // point. (Starting the rev DFS at node 0 left ipdom None everywhere.)
    let rev_dom = super::dominator::DominatorInfo::compute_from(&rev_cfg, n);
    // ipdom[b] = idom of b in the reversed graph (entry of reversed graph is
    // the virtual exit n). Strip the virtual node back to None.
    let mut ipdom: Vec<Option<usize>> = vec![None; n];
    for b in 0..n {
        ipdom[b] = rev_dom.idom.get(b).copied().flatten().filter(|&d| d < n);
    }

    StructuredCfg {
        blocks,
        block_of,
        full_succ: cfg
            .successors
            .iter()
            .map(|s| s.iter().copied().filter(|&t| t < n).collect())
            .collect(),
        labels,
        label_names,
        idom: dom_info.idom,
        ipdom,
    }
}

/// Natural loops: back edges t→h where h dominates t.
fn find_loops(sc: &StructuredCfg) -> HashMap<usize, LoopInfo> {
    let n = sc.blocks.len();
    let succs: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut v = Vec::new();
            if let Some(t) = sc.blocks[i].succ_taken {
                v.push(t);
            }
            if let Some(f) = sc.blocks[i].succ_fall {
                if !v.contains(&f) {
                    v.push(f);
                }
            }
            v
        })
        .collect();

    // Precompute predecessors ONCE (was O(n²) per loop before)
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, ss) in succs.iter().enumerate() {
        for &t in ss {
            preds[t].push(i);
        }
    }

    let mut loops: HashMap<usize, LoopInfo> = HashMap::new();
    for t in 0..n {
        for &h in &succs[t] {
            // does h dominate t? walk t's idom chain
            let mut cur = Some(t);
            let mut dominated = false;
            let mut hops = 0;
            while let Some(c) = cur {
                if c == h {
                    dominated = true;
                    break;
                }
                cur = sc.idom[c];
                hops += 1;
                if hops > n + 1 {
                    break;
                }
            }
            if !dominated {
                continue;
            }
            let entry = loops.entry(h).or_insert_with(|| LoopInfo {
                body: HashSet::new(),
                latches: Vec::new(),
            });
            entry.body.insert(h);
            entry.body.insert(t);
            if !entry.latches.contains(&t) {
                entry.latches.push(t);
            }
            // natural loop body via predecessors (O(edges) not O(n²))
            let mut work = vec![t];
            while let Some(b) = work.pop() {
                if b == h {
                    continue;
                }
                for &p in &preds[b] {
                    if !entry.body.contains(&p) {
                        entry.body.insert(p);
                        work.push(p);
                    }
                }
            }
        }
    }
    loops
}

struct LoopInfo {
    body: HashSet<usize>,
    latches: Vec<usize>,
}

// ---------------------------------------------------------------------------
// EH call-site rows → try regions
// ---------------------------------------------------------------------------

/// Block index whose instruction span contains the native address `addr`.
///
/// Places a landing-pad RVA on the CFG. A pad is normally the first
/// instruction of its block (LLVM emits `landingpad` there), but an otherwise
/// empty pad block can start one instruction earlier, hence the `+4` slack on
/// the last instruction's address — the same rule the EH block map uses.
fn block_containing_addr(sc: &StructuredCfg, insns: &[Insn], addr: u64) -> Option<usize> {
    for (bi, b) in sc.blocks.iter().enumerate() {
        if b.end <= b.start {
            continue;
        }
        let start = insns[b.start].address;
        let last = insns[b.end - 1].address;
        if start <= addr && addr < last.saturating_add(4) {
            return Some(bi);
        }
    }
    None
}

/// Is this block a landing-pad TRAMPOLINE — a pad whose whole body is
/// `nop`/`jmp` (or nothing at all), so control leaves it without doing work?
///
/// LLVM gives every `invoke` its OWN landing pad and then lets the pads fan
/// into one shared cleanup, so the pads of a single `try` typically look like
/// `B17: jmp cleanup  B18: jmp cleanup  …  B21: nop;nop; ─┘`.
fn is_pad_trampoline(sc: &StructuredCfg, insns: &[Insn], b: usize) -> bool {
    let blk = &sc.blocks[b];
    blk.end > blk.start
        && (blk.start..blk.end).all(|i| matches!(&insns[i].op, Op::Nop | Op::Jump(_)))
}

/// Follow a landing pad's trampoline chain to the block where the real handler
/// begins.
fn pad_root(sc: &StructuredCfg, insns: &[Insn], mut b: usize) -> usize {
    for _ in 0..32 {
        if !is_pad_trampoline(sc, insns, b) {
            return b;
        }
        let blk = &sc.blocks[b];
        let next = match (blk.succ_taken, blk.succ_fall) {
            (None, Some(f)) => f,
            (Some(t), None) => t,
            // two edges: a conditional trampoline is the handler itself
            _ => return b,
        };
        if next >= sc.blocks.len() || next == b {
            return b;
        }
        b = next;
    }
    b
}

/// Merge the per-call-site EH rows that describe ONE `try`.
///
/// A `try` holding several throw-capable calls has one call-site table ROW PER
/// CALL: the rows are disjoint address ranges, all naming the same action (same
/// `catches`, same `is_cleanup`), and — the decisive part — their landing pads
/// all fan into the SAME cleanup chain. The rows are the CALL SITES, not the
/// try. Folding each row on its own turned one `try { … } finally { … }` into N
/// sibling `try { call } finally { goto Lpad }` islands that then fenced the
/// structurer's walks and cut the loops around them in half — measured on
/// `GetExistingParameters` @0x12EA65C, where 5 rows fragmented the enumerator
/// loop into 5 detached try blocks.
///
/// Two rows merge when they name the same action, their ranges are disjoint and
/// ordered (so no control flow is reordered), and their pads form ONE
/// contiguous trampoline fan-in onto a single handler. Rows that merely sit near
/// each other but land on different handlers are left alone.
fn merge_eh_regions(
    regions: &[super::exceptions::EhRegion],
    sc: &StructuredCfg,
    insns: &[Insn],
) -> Vec<super::exceptions::EhRegion> {
    if regions.len() < 2 {
        return regions.to_vec();
    }
    let mut out: Vec<super::exceptions::EhRegion> = Vec::with_capacity(regions.len());
    for r in regions {
        if let Some(prev) = out.last_mut() {
            if let Some(merged) = merge_eh_pair(prev, r, sc, insns) {
                *prev = merged;
                continue;
            }
        }
        out.push(r.clone());
    }
    out
}

fn merge_eh_pair(
    prev: &super::exceptions::EhRegion,
    r: &super::exceptions::EhRegion,
    sc: &StructuredCfg,
    insns: &[Insn],
) -> Option<super::exceptions::EhRegion> {
    // Same action, and the ranges must not overlap or reorder control flow.
    if prev.is_cleanup != r.is_cleanup || prev.catches != r.catches || prev.try_end > r.try_start {
        return None;
    }
    let (Some(pb), Some(rb)) = (
        block_containing_addr(sc, insns, prev.landing_pad),
        block_containing_addr(sc, insns, r.landing_pad),
    ) else {
        return None;
    };
    if pb == rb {
        // Two rows, one pad block: unambiguously one try.
        return Some(super::exceptions::EhRegion {
            try_start: prev.try_start,
            try_end: r.try_end,
            landing_pad: prev.landing_pad,
            catches: prev.catches.clone(),
            is_cleanup: prev.is_cleanup,
        });
    }
    let (lo, hi) = (pb.min(rb), pb.max(rb));
    // A pad fan-in is local; a large span means these are two independent tries.
    if hi - lo > 64 {
        return None;
    }
    let root = pad_root(sc, insns, lo);
    for b in lo..=hi {
        if !is_pad_trampoline(sc, insns, b) || pad_root(sc, insns, b) != root {
            return None;
        }
    }
    Some(super::exceptions::EhRegion {
        try_start: prev.try_start,
        try_end: r.try_end,
        landing_pad: prev.landing_pad,
        catches: prev.catches.clone(),
        is_cleanup: prev.is_cleanup,
    })
}

// ---------------------------------------------------------------------------
// Region structurer
// ---------------------------------------------------------------------------

struct Structurer<'a> {
    sc: &'a StructuredCfg,
    insns: &'a [Insn],
    ctx: MethodCtx<'a>,
    deadline: Option<std::time::Instant>,
    /// Blocks already emitted somewhere in the output. A block must never be
    /// emitted twice: when two branches fail to find their merge point, both
    /// arms would otherwise re-emit the whole shared tail — and nested ifs
    /// multiply that into exponential (gigabyte) output. This is the hard
    /// guard that keeps rendering linear-ish for irreducible flow.
    emitted: std::cell::RefCell<HashSet<usize>>,
    /// First-visit order of block emissions. Flat guards snapshot a log index
    /// to prove that labels in their body are private to that fall arm without
    /// cloning the full `emitted` set for every candidate.
    emission_log: std::cell::RefCell<Vec<usize>>,
    /// Total block-visit budget shared by the whole method render.
    remaining: std::cell::Cell<usize>,
    /// Blocks that need a label on first visit even though the CFG does not
    /// mark them as branch targets (a goto was emitted to a stop block that
    /// the walker continues into afterwards).
    forced_labels: std::cell::RefCell<HashSet<usize>>,
    /// Every block the walk ENTERED (insert-only). Coverage, not emission:
    /// `emitted` is removed/re-added around loop bodies, so it cannot answer
    /// "did anything ever look at this block".
    visited: std::cell::RefCell<HashSet<usize>>,
    /// LOOP-FIRST routing map: any block → its enclosing loop header when
    /// that block is a loop body member. The region walker consults this
    /// BEFORE consuming a block linearly; stepping into a loop body without
    /// structuring the loop first made `emit_loop`'s body pass stop
    /// immediately (blocks already emitted / fenced by body_stops), which
    /// produced the 7k `// loop body not recovered` artifacts.
    loop_of_block: HashMap<usize, usize>,
    /// Decoded EH regions for this method (exact try ranges + landing pads
    /// from the binary's exception tables). Empty for most methods.
    eh_regions: &'a [super::exceptions::EhRegion],
    /// block index → try-region ordinal covering it.
    region_of_block: HashMap<usize, usize>,
    /// try-region ordinal → handler (landing pad) block index.
    handler_of: HashMap<usize, usize>,
    /// True while rendering a handler body. Landing pads resume into code
    /// inside the try range; without this flag every such re-entry re-folded
    /// a try, whose handler re-entered again — 50k+ duplicated `try` blocks
    /// per method (mutual recursion capped only by the emission budget).
    in_handler: std::cell::Cell<bool>,
    /// Region ordinal whose BODY is currently being walked. If arms inside
    /// a region re-entered the fold for the SAME region the tail re-rendered
    /// recursively (33k tries in one method). Nested regions map blocks to
    /// the innermost ordinal, so they still fold correctly inside.
    folding: std::cell::Cell<Option<usize>>,
    /// Regions already folded. The LSDA legitimately lists overlapping
    /// call-site rows (wrapper + catch-all as separate entries over the same
    /// range), so `folding` alone cannot stop ALTERNATING ordinals from
    /// re-folding each other — one fold per ordinal, ever.
    folded: std::cell::RefCell<std::collections::HashSet<usize>>,
    /// True for the binding-discovery walk, whose output is thrown away. The
    /// process-wide [`NEST_CAPS`] tally must not count that walk's truncations
    /// — the run summary would then report double the real number.
    discovery: bool,
    /// True while the walk in progress is the BODY pass of a loop.
    ///
    /// A `return (out, Some(next))` from [`Structurer::emit_region`] promises
    /// the caller that the walk resumes at `next`. A guard chain keeps that
    /// promise (the join is a sibling), but `emit_loop` throws the value away
    /// and resumes after the loop at its OWN exit block — so a fall into any
    /// other stop block was silently dropped, and that block was never
    /// emitted at all. Measured: 113 reachable blocks in
    /// `XPathParser.EvaluateDependentExpressions` @0x1DA379C, whose skip block
    /// `B37` is reached only by falling out of the scan loop. Inside a loop
    /// body such a fall is carried explicitly as a `goto` instead.
    in_loop_body: std::cell::Cell<bool>,
    /// The block the enclosing loop body pass resumes at (the loop's primary
    /// exit, i.e. `emit_loop`'s `exit_blk`). Falling into THAT block is the
    /// region's normal exit and needs no `goto`; falling into any other stop
    /// block is a secondary exit and does.
    loop_body_exit: std::cell::Cell<Option<usize>>,
    /// While the BODY of a loop whose HEADER carries interior code (rather
    /// than the loop test) is being walked, this holds that header's block
    /// index. `emit_region` then declines the `loops` lookup for exactly
    /// that block, so the walk may structure the header as an ordinary
    /// block (emitting the null guard / prologue it carries) instead of
    /// re-entering `emit_loop` on it → infinite recursion.
    ///
    /// Root cause it fixes (`SmallXmlParser.OnStartElement` @0x12F45D0): a
    /// natural loop whose header terminates in a hoisted NULL GUARD
    /// (`if (obj != null) goto body; throw;`) rather than the loop test. The
    /// real test sits at the latch. `emit_loop` took the header's guard for
    /// the test, rendered `while (obj == 0)` (the null check, polarity
    /// inverted) and emitted the guard's throw UNCONDITIONALLY inside the
    /// body — an impossible loop that also lost the `do {}` form.
    skip_loop_header: std::cell::Cell<Option<usize>>,
    /// One-shot marker for the FLAT-GUARD chain: the block index the guard
    /// being emitted continues into. See `emit_if` — the constant-depth flat
    /// form is only worth its `goto` when the guard is one of a CHAIN of
    /// guards to the SAME join, and the innermost guard of a chain has a plain
    /// body as its fall-through, so an enclosing frame must be able to say
    /// "stay flat" for it (a plain `succ_taken` test cannot).
    flat_chain_next: std::cell::Cell<Option<usize>>,
    /// Blocks reachable from the entry following only ordinary CFG edges.
    ///
    /// A landing pad is reached by UNWINDING, so it — and the cleanup chain
    /// it runs before rejoining the function — is not in this set, while the
    /// block that chain rejoins IS. `emit_handler` therefore fences its walk
    /// with this set: the handler body stops exactly where cleanup stops and
    /// normal control resumes. Without the fence the walk ran straight
    /// through the cleanup into the REST of the function — every `finally`
    /// body swallowed the method's remaining code (measured on
    /// `GetExistingParameters` @0x12EA65C, whose `finally` contained the
    /// entire DSA-key-extraction half of the method plus its `return`).
    normal_reachable: HashSet<usize>,
}

impl<'a> Structurer<'a> {
    fn fold_linear(&self, bi: usize, out: &mut Vec<Stmt>) -> bool {
        // returns true when the block ends with a boundary handled here
        let b = &self.sc.blocks[bi];
        let mut pushed = false;
        for i in b.start..b.end {
            match &self.insns[i].op {
                Op::Jump(_) | Op::CondJump(_) => break,
                _ => {}
            }
            for s in self.ctx.boundary_stmts(i) {
                out.push(s);
                pushed = true;
            }
        }
        pushed
    }

    /// Boundary statements before a block's terminator (bypasses the
    /// terminator handling of fold_linear).
    fn fold_prelude(&self, bi: usize) -> Vec<Stmt> {
        let b = &self.sc.blocks[bi];
        let mut out = Vec::new();
        for i in b.start..b.end.saturating_sub(1) {
            out.extend(self.ctx.boundary_stmts(i));
        }
        out
    }

    /// Condition of the CondJump terminating block `bi`.
    fn branch_condition(&self, bi: usize) -> Expr {
        let b = &self.sc.blocks[bi];
        let term = b.end - 1;
        if std::env::var("ILR_BCDBG").is_ok() {
            eprintln!("[bc] bi={} term_insn={} term_op={:?}", bi, term, self.insns[term].op);
        }
        let cond_local = self.insns[term]
            .operands
            .first()
            .and_then(|o| o.as_local());
        if let Some(v) = cond_local {
            // Branch tests read the lifter's scratch locals. The arm64 lifter
            // now assigns a UNIQUE id per Compare/And def, so the ordinary
            // def map resolves the condition EXACTLY (the old shared sentinel
            // made every compare multi-def and forced heuristic guessing,
            // which regularly bound the wrong expression to a branch — the
            // contradictory `if (x != 0) { if (x == 0) {` corruption class).
            // The two legacy shared sentinels (arm32/x86 lifters) keep the
            // heuristic and MUST NOT consult the def map: their entry is a
            // last-in-linear guess that is wrong by construction.
            let def = if v >= u32::MAX - 1 {
                self.find_scratch_def(v, term)
            } else {
                self.ctx.defs.get(&v).copied()
            };
            if std::env::var("ILR_BCDBG").is_ok() {
                eprintln!("[bc] v={} def={:?}", v, def);
            }
            if let Some(def) = def {
                if std::env::var("ILR_BCDBG").is_ok() {
                    eprintln!("[bc] def_insn={} op={:?} ops={:?}", def, self.insns[def].op, self.insns[def].operands);
                }
                // Fused compound condition (`cmp; ccmp; b.cc`) — the lifter
                // emits `Arith(And|Or)` over two compare results. Resolve it
                // as a boolean operator so the branch renders the source's
                // `a == 0 || b == 0` instead of nested single-test ifs.
                if v >= u32::MAX - 1 {
                    if let Some(e) = self.compound_condition(def) {
                        return e;
                    }
                }
                if matches!(self.insns[def].op, Op::Compare(_))
                    && (v >= u32::MAX - 1 || !self.ctx.is_multi(v))
                {
                    return self.ctx.def_expr(def, 0);
                }
            }
        }
        Expr::Bool(true)
    }

    /// Resolve a branch condition defined by `Op::Arith(And|Or)` of two
    /// boolean compare results — the fused `cmp; ccmp; b.cc` compound
    /// condition. Returns `None` when either side is not a compare, so the
    /// caller keeps its existing single-compare path (bitwise `And`/`Or` on
    /// non-boolean operands must NOT be rendered as `&&`/`||`).
    fn compound_condition(&self, def: usize) -> Option<Expr> {
        let Op::Arith(kind) = self.insns[def].op else {
            return None;
        };
        let op = match kind {
            ArithKind::And => BinOp::AndAlso,
            ArithKind::Or => BinOp::OrElse,
            _ => return None,
        };
        let lhs_v = self.insns[def].operands.get(1).and_then(|o| o.as_local())?;
        let rhs_v = self.insns[def].operands.get(2).and_then(|o| o.as_local())?;
        let ldef = self.ctx.defs.get(&lhs_v).copied()?;
        let rdef = self.ctx.defs.get(&rhs_v).copied()?;
        if !matches!(self.insns[ldef].op, Op::Compare(_))
            || !matches!(self.insns[rdef].op, Op::Compare(_))
        {
            return None;
        }
        Some(Expr::Bin(
            op,
            Box::new(self.ctx.def_expr(ldef, 0)),
            Box::new(self.ctx.def_expr(rdef, 0)),
        ))
    }

    /// Find the Compare defining scratch slot `v` at/before insn `term`.
    ///
    /// The lifter's scratch slots are re-defined by every compare in the
    /// method, so SSA def lookup is useless for them. The old 8-insn window
    /// missed compares that live in an EARLIER BLOCK (very common: the test
    /// is computed in one block, the branch sits in the next) — which is what
    /// leaked `if (_mask != 0)` into user-visible conditions. Strategy:
    ///
    /// 1. nearest Compare in the current block before `term`;
    /// 2. walk back through predecessor blocks (dominance-ordered, since the
    ///    branch is dominated by its defs) and take the nearest Compare in
    ///    each, up to 4 blocks back;
    /// 3. give up (caller renders a conservative condition).
    fn find_scratch_def(&self, _v: u32, term: usize) -> Option<usize> {
        // 1. same block, nearest preceding Compare. The scan covers the WHOLE
        // block, not a fixed window before the terminator: the compiler puts
        // the compare EARLY in a block and follows it with many loads/moves
        // before the branch (guards that re-load fields on both paths), so a
        // short window missed the block's own compare and stole the idom
        // predecessor's instead — the SAME test rendered twice with inverted
        // polarity (`if (x == 0) { if (x != 0) …`), an impossible path.
        let blk_of = self
            .sc
            .blocks
            .iter()
            .position(|b| term >= b.start && term < b.end);
        if let Some(bi) = blk_of {
            let b = &self.sc.blocks[bi];
            if let Some(d) = (b.start..term)
                .rev()
                .find(|&i| matches!(self.insns[i].op, Op::Compare(_)))
            {
                return Some(d);
            }
        } else if let Some(d) = (term.saturating_sub(8)..term)
            .rev()
            .find(|&i| matches!(self.insns[i].op, Op::Compare(_)))
        {
            return Some(d);
        }
        // 2. predecessors of the block containing `term`, nearest-Compare per
        //    block, deepest first (closest block wins).
        let blk_of = self
            .sc
            .blocks
            .iter()
            .position(|b| term >= b.start && term < b.end)?;
        let mut chain: Vec<usize> = Vec::new();
        let mut cur = Some(blk_of);
        while let Some(c) = cur {
            chain.push(c);
            cur = self.sc.idom[c];
            if chain.len() > 4 {
                break;
            }
        }
        for &bi in chain.iter().skip(1) {
            let b = &self.sc.blocks[bi];
            if let Some(d) = (b.start..b.end)
                .rev()
                .find(|&i| matches!(self.insns[i].op, Op::Compare(_)))
            {
                return Some(d);
            }
        }
        None
    }

    fn emit_region(
        &self,
        cur: usize,
        stops: &HashSet<usize>,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
        depth: usize,
    ) -> (Vec<Stmt>, Option<usize>) {
        let mut out: Vec<Stmt> = Vec::new();
        if cur >= self.sc.blocks.len() {
            return (out, None);
        }
        if depth > MAX_NEST_DEPTH {
            // The cap bounds recursion/output; it must NOT be a silent way to
            // delete code. It used to `return (out, None)` with nothing in
            // `out`, which removed the whole remaining sub-CFG without a trace
            // — measured on 0x2478f0c: 35 of 308 native conditions emitted and
            // no marker, so the loss was invisible to every checker. Mark it,
            // jump to a label for the block that was skipped so control still
            // flows where it belongs, and count it so a run reports how often
            // the cap is doing real work (`NEST_CAPS`).
            if !self.discovery {
                NEST_CAPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            out.push(Stmt::Comment(format!(
                "… nesting depth limit ({MAX_NEST_DEPTH}) reached at L{} — rest of this region omitted",
                label_name(self.sc, cur)
            )));
            return (out, None);
        }
        let mut cur = cur;
        if std::env::var("ILR_EMITDBG").is_ok() && depth == 0 {
            eprintln!("[emit] start cur={} blocks={} stops={:?} eh={:?} region_of_block0={:?}", cur, self.sc.blocks.len(), stops, self.eh_regions.len(), self.region_of_block.get(&0));
            for (bi, b) in self.sc.blocks.iter().enumerate() {
                let ops: Vec<String> = (b.start..b.end)
                    .map(|i| format!("{:?}", self.insns[i].op))
                    .collect();
                eprintln!("[emit] B{}: insns {}..{} succ_t={:?} succ_f={:?} | {}", bi, b.start, b.end, b.succ_taken, b.succ_fall, ops.join(" ; "));
            }
            eprintln!("[emit] total_insns={} block_of58={:?}", self.insns.len(), self.sc.block_of.get(58));
        }

        loop {
            if std::env::var("ILR_IFDBG").is_ok() {
                eprintln!("[visit] cur={} depth={}", cur, depth);
            }
            if std::env::var("ILR_EMITDBG").is_ok()
                && (depth == 0 || std::env::var("ILR_ARMDBG").is_ok())
            {
                eprintln!("[emit] iter cur={} depth={} emitted={}", cur, depth, self.emitted.borrow().len());
            }
            // Wall-clock cut-off — OPT-IN ONLY (`ILR_TIME_LIMIT=<ms>`), never
            // on by default. Where this fires, the body is truncated at
            // whatever point the machine happened to reach, so the same method
            // decompiles differently on a loaded box than on an idle one. The
            // deterministic bound is `self.remaining` (a block-visit count, a
            // pure function of the CFG) checked a few lines below; that is the
            // one that must do the bounding. The two are kept visually
            // distinct in the output so a truncation's CAUSE is readable.
            if let Some(d) = self.deadline {
                if std::time::Instant::now() > d {
                    out.push(Stmt::Comment("… time limit exceeded".into()));
                    return (out, None);
                }
            }
            if cur >= self.sc.blocks.len() || stops.contains(&cur) || out.len() > 50_000 {
                return (out, Some(cur));
            }
            self.visited.borrow_mut().insert(cur);
            // Global per-method emission guard: a block reached from two
            // already-emitted paths (or after budget exhaustion) must not be
            // re-emitted — that duplicated entire shared tails and exploded
            // output size exponentially on nested control flow. Emit a jump
            // instead so control flow stays correct.
            if !self.emitted.borrow_mut().insert(cur) {
                if loops.contains_key(&cur) {
                    out.push(Stmt::Continue);
                } else {
                    out.push(self.goto_stmt_reason(cur, GotoReason::AlreadyEmitted));
                }
                return (out, None);
            }
            self.emission_log.borrow_mut().push(cur);
            let remaining = self.remaining.get();
            if remaining == 0 {
                out.push(Stmt::Comment("… budget exceeded".into()));
                return (out, None);
            }
            self.remaining.set(remaining - 1);

            // EH folding: entering a block covered by a try region → wrap
            // everything from here to the region's last block in
            // `try { … } catch { … }`. The handler body renders from the
            // landing-pad block (already in the CFG, so its klass checks —
            // the REAL managed catch filter — appear inside the catch).
            //
            // Decide this BEFORE emitting the label: a block that both starts
            // a try region and carries a label would otherwise render the
            // label twice — once here and once as the folded body re-walks
            // `cur`. The inner walk owns `cur` (it removes and re-inserts it
            // in `emitted`), so it owns the label too.
            let will_fold = !self.in_handler.get()
                && self.folding.get() != self.region_of_block.get(&cur).copied()
                && self.region_of_block.contains_key(&cur)
                && self.folded.borrow_mut().insert(self.region_of_block[&cur]);

            // First visit of a labelled block: emit its label here so every
            // `goto L…` in the output has a matching definition (previously
            // gotos referenced labels that were never rendered). Skipped when
            // a try region folds here — the folded walk emits it inside.
            if !will_fold
                && (self.sc.labels.contains_key(&cur)
                    || self.forced_labels.borrow().contains(&cur))
            {
                out.push(Stmt::Label(label_name(self.sc, cur)));
            }

            if will_fold {
                let ri = self.region_of_block[&cur];
                let region = &self.eh_regions[ri];
                let handler_blk = self.handler_of.get(&ri).copied();
                // The try body spans every block whose native address lies in
                // the region's range. Walk from `cur`, stopping as soon as the
                // walk leaves the protected range (or control exits). The
                // walker state is saved/restored so the outer region walk can
                // resume after the try statement.
                let mut try_stmts: Vec<Stmt> = Vec::new();
                // The try body must be structured AS CONTROL FLOW — ifs, loops
                // and all — not folded block-by-block. Delegate to
                // emit_region with a stop-set of every block OUTSIDE the
                // region: the walk then freely structures the protected
                // interior and stops exactly at the region boundary. Blocks
                // already emitted (reached earlier by the outer walk through a
                // path that crossed the region) are fenced by the emitted
                // guard; the body still gets whatever remains reachable.
                let mut region_stops: HashSet<usize> = HashSet::new();
                for b in 0..self.sc.blocks.len() {
                    let a = self.insns[self.sc.blocks[b].start].address;
                    // Address-range fence, NOT region_of_block: blocks of a
                    // NESTED region lie inside this region's range too, and
                    // region_of_block maps them to the inner region — fencing
                    // by that map made every outer try body walk stop at its
                    // first nested block and render empty.
                    if !(a >= region.try_start && a < region.try_end) {
                        region_stops.insert(b);
                    }
                }
                // `cur` was already marked by the outer loop; undo so the
                // inner walk owns it (it re-marks with its own budget slot).
                self.emitted.borrow_mut().remove(&cur);
                // Fence re-folding of THIS region while its body walks: arms
                // inside the body target region blocks and would otherwise
                // re-fold the region tail recursively.
                let prev_folding = self.folding.replace(Some(ri));
                let (mut part, _next) =
                    self.emit_region(cur, &region_stops, loops, loop_exits, depth + 1);
                self.folding.set(prev_folding);
                try_stmts.append(&mut part);
                // Handler clauses. The C++ ABI models a managed catch as an
                // action chain: wrapper catch → type filter → catch-all. In
                // C# semantics the whole chain is ONE handler (the managed
                // filter runs inside the landing pad), so dedupe: a run of
                // clauses sharing the same landing pad becomes the wrapper
                // clause plus a single `catch (...)` fallback; `/* filter */`
                // entries collapse into the preceding managed catch.
                let mut catches: Vec<(String, Vec<Stmt>)> = Vec::new();
                let mut finally_body: Option<Vec<Stmt>> = None;
                if region.is_cleanup || region.catches.is_empty() {
                    finally_body = handler_blk
                        .map(|hb| self.emit_handler(hb, loops, loop_exits, stops, depth));
                } else {
                    // The Itanium action chain for ONE managed catch is
                    // [wrapper catch(type) → filter → catch-all fallback].
                    // The fallback is ABI control flow (rethrow when the
                    // filter rejects), NOT a second C# handler — rendering it
                    // produced duplicate `catch` clauses where the second one
                    // only held a goto (the pad was already consumed). Collapse
                    // the chain into a single clause; the pad's klass checks
                    // inside the body are the real filter.
                    let managed = region
                        .catches
                        .iter()
                        .any(|c| c.as_ref().is_some_and(|t| t.contains("managed catch")));
                    if managed {
                        let body = handler_blk
                            .map(|hb| self.emit_handler(hb, loops, loop_exits, stops, depth))
                            .unwrap_or_default();
                        catches.push(("Il2CppExceptionWrapper /* managed catch */".into(), body));
                    } else {
                        let mut seen_names: HashSet<String> = HashSet::new();
                        for c in &region.catches {
                            let Some(t) = c else { continue };
                            if t.contains("/* filter */") {
                                // the type check lives in the pad body; a
                                // separate C# filter clause would be noise
                                continue;
                            }
                            if !seen_names.insert(t.clone()) {
                                continue;
                            }
                            let body = handler_blk
                                .map(|hb| self.emit_handler(hb, loops, loop_exits, stops, depth))
                                .unwrap_or_default();
                            catches.push((t.clone(), body));
                        }
                        if catches.is_empty() {
                            // chain of catch-alls only → one generic clause
                            let body = handler_blk
                                .map(|hb| self.emit_handler(hb, loops, loop_exits, stops, depth))
                                .unwrap_or_default();
                            catches.push(("...".into(), body));
                        }
                    }
                }
                out.push(Stmt::Try {
                    try_body: try_stmts,
                    catches,
                    finally_body,
                });
                // Resume the outer walk at the first block AFTER the try
                // range. Returning None here truncated every method that had
                // code following its try statement.
                let after = (0..self.sc.blocks.len()).find(|&b| {
                    self.insns[self.sc.blocks[b].start].address >= region.try_end
                });
                match after {
                    Some(a) if !self.emitted.borrow().contains(&a) => {
                        cur = a;
                        continue;
                    }
                    _ => return (out, None),
                }
            }

            if let Some(info) = loops.get(&cur) {
                // A loop header whose own branch is interior code (not the
                // loop exit test) is being structured as an ORDINARY block
                // inside its own body walk — see `skip_loop_header`. Decline
                // the loop lookup for exactly that block so the walk falls
                // through to the block's normal prelude/terminator handling
                // rather than recursing into `emit_loop` (infinite).
                if self.skip_loop_header.get() != Some(cur) {
                    let (stmt, next) =
                        self.emit_loop(cur, info, stops, loops, loop_exits, depth);
                    out.extend(stmt);
                    match next {
                        Some(n) => cur = n,
                        None => return (out, None),
                    }
                    continue;
                }
            }

            let b = &self.sc.blocks[cur];
            let last_op = self.insns[b.end - 1].op.clone();

            if let Op::Switch(targets) = last_op.clone() {
                // fold prelude
                for i in b.start..b.end - 1 {
                    out.extend(self.ctx.boundary_stmts(i));
                }
                // Case bodies are reachable ONLY through the table, so the
                // linear walk would never visit them. Render each arm by
                // walking from its target block.
                //
                // Constraints discovered empirically on real jump tables:
                //  1. The enclosing construct's `stops` must NOT fence the
                //     arms: the structurer's two-slot successor model cannot
                //     hold a switch's n targets, so dominance/loop analysis
                //     treats every arm beyond the first two as disconnected
                //     and they land in the enclosing loop's `body_stops` —
                //     seeded from those stops, every arm walk bails at its
                //     own first block and the arms render as dangling
                //     `goto Lx` shells (Evaluate: 37 targets, 0 statements).
                //  2. The arm's OWN block must not sit in the stop-set of its
                //     walk: emit_region's first check is `stops.contains`,
                //     so a claimed-block fence self-fences and yields empty.
                //  3. Arms must not re-enter an already-structured loop: the
                //     wrap loop of a degenerate dispatch is a real natural
                //     loop whose body an arm can reach through backward case
                //     targets; emit_loop CLEARS its body blocks from
                //     `emitted` and re-walks, recursively duplicating the
                //     switch (Evaluate nested 16× before the depth cap). So
                //     fence the arms with every block of any loop whose
                //     header precedes the dispatch: that loop is already
                //     rendered above the switch, and an arm re-entering it is
                //     either a shared dispatch default (nothing to add) or a
                //     `continue` the loop text already implies.
                //  4. Labels: every block the arms can reach (targets, their
                //     bodies, shared tails) must carry a label BEFORE the
                //     walks run, so `goto Lx` produced when a later arm hits
                //     an already-emitted block resolves. Force the whole
                //     arm-reachable region (BFS over the full n-way successor
                //     list, stopped at the loop fence / already-emitted
                //     blocks).
                //
                // When arms cannot render safely (all targets fenced / loop
                // wrapped), emit a bounded comment INSTEAD of a goto shell:
                // a shell whose labels never render is invalid pseudo-C# and
                // worse than an explicit mark that the dispatch is
                // unrecovered.
                // A dispatch INSIDE a loop is part of that loop's body: that
                // loop has NOT been rendered above it — the walker is in the
                // middle of rendering it — so fencing the loop's own body would
                // fence the switch's arms, i.e. exactly the blocks the dispatch
                // exists to reach. (Measured on `DtdParser.GetToken` @0x2055C04:
                // a 14-way table inside the scan loop came out as four empty
                // `shared tail` stubs with 128 reachable blocks never walked.)
                // Only such a loop's HEADER stays fenced, so an arm that jumps
                // back to it renders as a stub — the loop text already implies
                // that `continue`.
                let mut loop_fence: HashSet<usize> = HashSet::new();
                for (h, li) in loops.iter() {
                    if li.body.contains(&cur) {
                        loop_fence.insert(*h);
                        continue;
                    }
                    if self.sc.blocks[*h].start <= b.start {
                        for &m in li.body.iter() {
                            loop_fence.insert(m);
                        }
                        loop_fence.insert(*h);
                    }
                }
                // Force labels over the arm-reachable region so no case/join
                // goto dangles (each arm's blocks and every shared tail).
                let mut arm_region: HashSet<usize> = HashSet::new();
                let mut queue: Vec<usize> = Vec::new();
                for &t in targets.iter() {
                    if let Some(&tb) = self.sc.block_of.get(t) {
                        if tb < self.sc.blocks.len() && !loop_fence.contains(&tb) {
                            arm_region.insert(tb);
                            queue.push(tb);
                        }
                    }
                }
                while let Some(x) = queue.pop() {
                    for &s in self.sc.full_succ.get(x).map(|v| v.as_slice()).unwrap_or(&[]) {
                        if s < self.sc.blocks.len()
                            && !arm_region.contains(&s)
                            && !loop_fence.contains(&s)
                            && !self.emitted.borrow().contains(&s)
                        {
                            arm_region.insert(s);
                            queue.push(s);
                        }
                    }
                }
                for &x in arm_region.iter() {
                    if !self.sc.labels.contains_key(&x) {
                        self.forced_labels.borrow_mut().insert(x);
                    }
                }
                let mut claimed: HashSet<usize> = HashSet::new();
                let mut visited: HashSet<usize> = HashSet::new();
                let mut arm_stmts: Vec<Stmt> = Vec::new();
                let mut walked: usize = 0;
                for &t in targets.iter() {
                    let Some(&tb) = self.sc.block_of.get(t) else { continue };
                    if tb >= self.sc.blocks.len()
                        || loop_fence.contains(&tb)
                        || !claimed.insert(tb)
                        || self.emitted.borrow().contains(&tb)
                    {
                        continue;
                    }
                    visited.insert(tb);
                    let mut stops_arm = claimed.clone();
                    stops_arm.extend(loop_fence.iter().copied());
                    stops_arm.remove(&tb);
                    let (mut s, _) =
                        self.emit_region(tb, &stops_arm, loops, loop_exits, depth + 1);
                    while matches!(s.last(), Some(Stmt::Goto(_))) {
                        s.pop();
                    }
                    walked += s.len();
                    arm_stmts.extend(s);
                }
                if walked == 0 {
                    out.push(Stmt::Comment(format!(
                        "jump-table dispatch over {} cases (arms unrecovered)",
                        targets.len()
                    )));
                    return (out, None);
                }
                // Case targets whose block was never visited (already emitted
                // shared tail / dispatch default) still need a defined label
                // or their `case k: goto Lx` dangles. Emit a stub landing
                // pad: the block's real content was rendered elsewhere (before
                // the switch, as shared code), so a bare terminating label is
                // the honest placeholder.
                let mut unvisited: Vec<usize> = Vec::new();
                for &t in targets.iter() {
                    if let Some(&tb) = self.sc.block_of.get(t) {
                        if tb < self.sc.blocks.len() && !visited.contains(&tb) {
                            unvisited.push(tb);
                        }
                    }
                }
                for tb in unvisited {
                    arm_stmts.push(Stmt::Label(label_name(self.sc, tb)));
                    arm_stmts.push(Stmt::Comment("shared tail / dispatch default".into()));
                    arm_stmts.push(Stmt::Break);
                }
                let mut case_labels: Vec<(u64, String)> = Vec::new();
                for (v, &t) in targets.iter().enumerate() {
                    if let Some(&tb) = self.sc.block_of.get(t) {
                        if tb < self.sc.blocks.len() {
                            case_labels.push((v as u64, label_name(self.sc, tb)));
                        }
                    }
                }
                // The discriminant: a recovered `Op::Switch` stores its case
                // targets but NOT the value that indexes the table, so this
                // used to print a literal `switch (switchValue)` stub — the
                // text no longer named the thing being switched on. The index
                // is still visible as the non-immediate operand of the bounds
                // check immediately before the dispatch (`cmp idx, #N; b.hi
                // default`), which is the LAST compare in this block.
                // The guard sits in the block feeding the dispatch, not in the
                // switch block itself (the table load + `br` are their own
                // block), so scan a short window back across the boundary.
                let disc = (b.start.saturating_sub(12)..b.end)
                    .rev()
                    .find_map(|i| {
                        if !matches!(self.insns[i].op, Op::Compare(_)) {
                            return None;
                        }
                        match (
                            self.insns[i].operands.get(1),
                            self.insns[i].operands.get(2),
                        ) {
                            (Some(Operand::Local(v)), Some(Operand::Immediate(_))) => {
                                Some(self.ctx.var_name(*v))
                            }
                            (Some(Operand::Immediate(_)), Some(Operand::Local(v))) => {
                                Some(self.ctx.var_name(*v))
                            }
                            _ => None,
                        }
                    })
                    .unwrap_or_else(|| "switchValue".into());
                // Split the flat `[Label, body…, Label, body…]` arm list into
                // per-label blocks so each case can be rendered INLINE.
                let mut segs: Vec<(String, Vec<Stmt>)> = Vec::new();
                let mut stray: Vec<Stmt> = Vec::new();
                for s in arm_stmts.into_iter() {
                    match &s {
                        Stmt::Label(l) => segs.push((l.clone(), Vec::new())),
                        _ => match segs.last_mut() {
                            Some((_, b)) => b.push(s),
                            None => stray.push(s),
                        },
                    }
                }
                let mut cases: Vec<(u64, Vec<Stmt>)> = Vec::new();
                let mut used: HashSet<String> = HashSet::new();
                for (v, label) in case_labels {
                    let first = used.insert(label.clone());
                    if first {
                        if let Some((_, body)) = segs.iter().find(|(l, _)| *l == label) {
                            // Keep the label: other gotos may target this arm.
                            let mut b: Vec<Stmt> = Vec::with_capacity(body.len() + 1);
                            b.push(Stmt::Label(label.clone()));
                            b.extend(body.iter().cloned());
                            cases.push((v, b));
                            continue;
                        }
                    }
                    cases.push((v, vec![Stmt::Goto(label)]));
                }
                let mut tail: Vec<Stmt> = stray;
                for (label, body) in &segs {
                    if !used.contains(label) {
                        tail.push(Stmt::Label(label.clone()));
                        tail.extend(body.iter().cloned());
                    }
                }
                out.push(Stmt::Switch { disc, cases, tail });
                return (out, None);
            }

            if let Op::CondJump(taken) = last_op {
                // PRELUDE FIRST. The prelude is the block's own straight-line
                // code and it executes BEFORE the test — on the hardware and,
                // now, in the emitted text. Emitting it after
                // `branch_condition` meant the condition had already inlined a
                // call defined in this same block, and then the prelude emitted
                // `dst = call(…)` for the same native `bl`: one `bl`, two call
                // expressions (the duplicate-call class). It also let a use site
                // reference a binding that had not been materialized yet.
                let stmts_pre = self.fold_prelude(cur);
                out.extend(stmts_pre);
                // Claims produced while building THIS condition, so they can
                // be revoked if the `if` ends up an empty shell.
                let claim_mark = self.ctx.claim_log.borrow().len();
                let cond = self.branch_condition(cur);
                let cond_claims: Vec<usize> =
                    self.ctx.claim_log.borrow()[claim_mark..].to_vec();
                // IR branch targets are INSNS; the structurer works in BLOCKS
                let taken_blk = self
                    .sc
                    .block_of
                    .get(taken)
                    .copied()
                    .unwrap_or(self.sc.blocks.len());
                if taken_blk >= self.sc.blocks.len() {
                    // target past the listing (exit edge): render the
                    // fallthrough arm only, no else, no merge goto
                    let fall = b.succ_fall.filter(|f| *f < self.sc.blocks.len());
                    if let Some(f) = fall {
                        if !stops.contains(&f) {
                            let (arm, _) =
                                self.emit_region(f, stops, loops, loop_exits, depth + 1);
                            out.extend(arm);
                        }
                    }
                    return (out, None);
                }
                let (stmts, cont) = self.emit_if(
                    cur,
                    taken_blk,
                    stops,
                    loops,
                    loop_exits,
                    depth,
                    &cond,
                    &cond_claims,
                );
                out.extend(stmts);
                // `Some(next)` = "the join is a SIBLING of the guard, keep
                // walking there at THIS depth". Guard chains are flat by
                // construction, so a 300-guard chain (the compiler's unrolled
                // early-exit form) costs one stack level, not 300 — which is
                // what used to hit `MAX_NEST_DEPTH` and delete the tail.
                if let Some(next) = cont {
                    if next < self.sc.blocks.len() && !stops.contains(&next) {
                        cur = next;
                        continue;
                    }
                    self.bail_out_of_region(&mut out, next);
                    return (out, Some(next));
                }
                return (out, None);
            }

            self.fold_linear(cur, &mut out);

            match last_op {
                Op::Jump(t) => {
                    let t_blk = self
                        .sc
                        .block_of
                        .get(t)
                        .copied()
                        .unwrap_or(self.sc.blocks.len());
                    if t_blk >= self.sc.blocks.len() {
                        return (out, None); // exit edge
                    }
                    if stops.contains(&t_blk) {
                        // The walker continues into t_blk afterwards; force its
                        // label so this goto has a matching definition.
                        if !loops.contains_key(&t_blk) {
                            self.forced_labels.borrow_mut().insert(t_blk);
                        }
                        out.push(self.jump_stmt(t_blk, loops, loop_exits));
                        let next = stops.contains(&t_blk).then_some(t_blk);
                        return (out, next);
                    }
                    // NOTE: a jump to a loop header that is NOT a stop of this
                    // region must NOT be turned into `continue` — from outside
                    // the loop that jump simply enters the loop. Falling
                    // through to `cur = t_blk` lets the walker reach the
                    // header, where the loop is detected and structured.
                    // (The old `|| loops.contains_key(&t_blk)` condition here
                    // skipped whole loops: it emitted `continue` and returned
                    // without ever rendering the loop body.)
                    // LOOP-FIRST: if the target block belongs to a loop,
                    // route to its header NOW so emit_loop structures the
                    // whole loop before the walk consumes any body block.
                    if let Some(&h) = self.loop_of_block.get(&t_blk) {
                        if !stops.contains(&h) && !self.emitted.borrow().contains(&h) {
                            cur = h;
                            continue;
                        }
                    }
                    if t_blk <= cur {
                        out.push(self.goto_stmt_reason(t_blk, GotoReason::Backward));
                        return (out, None);
                    }
                    cur = t_blk;
                }
                Op::Return | Op::Throw => return (out, None),
                Op::IndirectCall => {
                    // `br Xn` — an indirect JUMP (a `blr` returns and lifts to
                    // `Op::Call`, so this op means the dispatch target is
                    // unknown). It is a CFG terminator, so the walk stops here.
                    //
                    // When the jump-table recognizer fires, this op has already
                    // been rewritten to `Op::Switch`, so reaching the structurer
                    // means the dispatch was NOT recovered. Stopping at the
                    // comment deleted everything that follows — and what
                    // follows is the case bodies, real code the disassembly
                    // has. `0x20d3ee4`: 8 of 64 blocks emitted, 3 of 43 native
                    // conditions, no marker and no checker complaint.
                    //
                    // Continue into the next block instead. It is not reached
                    // by falling through, and a reader must not believe it is —
                    // so jump explicitly to a forced label. The comment above
                    // already states the dispatch is unresolved.
                    out.push(Stmt::Comment(
                        "indirect jump (dispatch not recovered) — following code \
                         is reached only through this jump"
                            .into(),
                    ));
                    let next = cur + 1;
                    if next < self.sc.blocks.len() {
                        out.push(self.goto_stmt_reason(next, GotoReason::Dispatch));
                        if !self.emitted.borrow().contains(&next) {
                            cur = next;
                            continue;
                        }
                    }
                    return (out, None);
                }
                _ => match b.succ_fall {
                    Some(next) => {
                        if stops.contains(&next) {
                            self.bail_out_of_region(&mut out, next);
                            return (out, Some(next));
                        }
                        // LOOP-FIRST: route to the loop header before the
                        // linear walk consumes any body block.
                        if let Some(&h) = self.loop_of_block.get(&next) {
                            if !stops.contains(&h) && !self.emitted.borrow().contains(&h) {
                                cur = h;
                                continue;
                            }
                        }
                        cur = next;
                    }
                    None => return (out, None),
                },
            }
        }
    }

    /// Emit every block the CFG can reach but the walk could not place.
    ///
    /// The walk abandons regions for real structural reasons (a secondary loop
    /// exit, a switch arm fenced by a construct already rendered, a landing pad
    /// shared by clauses) — but "I could not fit this into a structured
    /// statement" must not become silence: the reader would then see a method
    /// that does not contain code the machine runs, and no IR-level check can
    /// see the loss because the block IS in the CFG.
    ///
    /// So each reachable skipped block is emitted as a labelled region, linked
    /// by the `goto`s the walk already emitted for it (a jump to a stop block
    /// always renders one, and `bail_out_of_region` adds the fall-through ones).
    ///
    /// Loops are deliberately NOT re-structured here: the region is reached only
    /// through those gotos, so rendering it FLAT keeps exactly one copy of every
    /// block and cannot duplicate a loop body or a call expression.
    fn coverage_repair(&self, reachable: &[bool]) -> (Vec<Stmt>, usize) {
        let empty_loops: HashMap<usize, LoopInfo> = HashMap::new();
        let mut out: Vec<Stmt> = Vec::new();
        let mut regions = 0usize;
        let mut guard = self.sc.blocks.len() + 8;
        loop {
            let done: HashSet<usize> = self.visited.borrow().clone();
            let next = (0..self.sc.blocks.len())
                .find(|b| reachable.get(*b).copied().unwrap_or(false) && !done.contains(b));
            let Some(b) = next else { break };
            if guard == 0 {
                out.push(Stmt::Comment(
                    "… coverage repair stopped (block budget)".into(),
                ));
                break;
            }
            guard -= 1;
            regions += 1;
            self.forced_labels.borrow_mut().insert(b);
            let (s, _) = self.emit_region(b, &done, &empty_loops, &[], 0);
            out.extend(s);
        }
        if regions > 0 {
            // The marker is placed AFTER the first region's label on purpose:
            // `idioms::dead_code` deletes statements that follow a terminator
            // with no label between them, and the body usually ends in a
            // `return` — so a marker emitted as the very first repair statement
            // was dropped in most methods. After the label it always survives.
            let marker = Stmt::Comment(format!(
                "[recovery] {regions} reachable region(s) the walker could not place in a \
                 structured statement — emitted as labelled regions"
            ));
            let at = out
                .iter()
                .position(|s| matches!(s, Stmt::Label(_)))
                .map(|i| i + 1)
                .unwrap_or(0);
            out.insert(at, marker);
        }
        (out, regions)
    }

    /// Carry a fall-through that LEAVES the region the walk is in.
    ///
    /// `emit_region` reports such a fall as `(out, Some(next))`, a promise that
    /// the caller resumes the walk at `next`. A guard chain keeps that promise
    /// (its join is a sibling at this depth), but a LOOP BODY does not:
    /// `emit_loop` discards the value and resumes after the loop at its own
    /// exit block. So a fall into any other stop block was dropped on the
    /// floor — the exited block was never emitted anywhere. Inside a loop body
    /// the edge is therefore carried explicitly as a `goto`; the label is
    /// forced here and the coverage repair renders its code, so the text keeps
    /// the edge AND the code. Falling into `exit_blk` itself is the loop's
    /// normal exit and needs nothing.
    fn bail_out_of_region(&self, out: &mut Vec<Stmt>, next: usize) {
        if self.in_loop_body.get() && self.loop_body_exit.get() != Some(next) {
            out.push(self.goto_stmt_reason(next, GotoReason::LoopExit));
        }
    }

    /// Emit a goto to a block, ensuring its label will render when (and only
    /// when) the block is first visited. Call sites that skip the block
    /// entirely leave the goto to `drop_unreachable_gotos`.
    /// Emit a `goto`, recording the CAUSE.
    ///
    /// Every structurer site that emits a `goto` must go through this: the
    /// cause is what makes the structural census actionable. "63% of methods
    /// contain a goto" is not a work item, but "58.9% of all gotos come from
    /// the flat-guard lowering" is. `GotoReason::Uncategorised` is a leak
    /// detector — it must stay at 0, and a non-zero count means a call site was
    /// added without saying why.
    fn goto_stmt_reason(&self, blk: usize, reason: GotoReason) -> Stmt {
        // Counted for the structural census — but NOT for the discovery walk,
        // whose statements are thrown away; counting there would double every
        // figure (see `classify_structure`).
        if !self.discovery {
            use std::sync::atomic::Ordering::Relaxed;
            STRUCT_CENSUS.goto_sites.fetch_add(1, Relaxed);
            STRUCT_CENSUS.goto_by_reason[reason.idx()].fetch_add(1, Relaxed);
        }
        self.forced_labels.borrow_mut().insert(blk);
        Stmt::Goto(label_name(self.sc, blk))
    }

    /// Walk a LOOP BODY with the secondary-exit machinery armed.
    ///
    /// The body pass must know which stop block is the loop's primary exit
    /// (`exit_blk`) so that a fall into any OTHER stop block is carried as a
    /// `goto` instead of being dropped (see [`Structurer::bail_out_of_region`]).
    /// Saved/restored around the walk because loops nest and a nested loop's
    /// body replaces its parent's exit for the duration of its own walk.
    #[allow(clippy::too_many_arguments)]
    fn emit_loop_body(
        &self,
        start: usize,
        body_stops: &HashSet<usize>,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
        exit_blk: Option<usize>,
        child_depth: usize,
    ) -> (Vec<Stmt>, Option<usize>) {
        let saved_flag = self.in_loop_body.replace(true);
        let saved_exit = self.loop_body_exit.replace(exit_blk);
        let r = self.emit_region(start, body_stops, loops, loop_exits, child_depth);
        self.in_loop_body.set(saved_flag);
        self.loop_body_exit.set(saved_exit);
        r
    }

    /// Render a catch/finally HANDLER body from the region's landing-pad
    /// block. The landing pad is real CFG code — its klass checks ARE the
    /// managed catch filter — so it gets the same structuring as any block.
    /// The block is marked emitted ONLY if not previously visited; a pad
    /// shared by several regions renders into the first try that reaches it.
    fn emit_handler(
        &self,
        hb: usize,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
        _stops: &HashSet<usize>,
        depth: usize,
    ) -> Vec<Stmt> {
        if hb >= self.sc.blocks.len() {
            return Vec::new();
        }
        if !self.emitted.borrow_mut().insert(hb) {
            // pad already rendered in another clause — reference it
            if std::env::var("ILR_HANDLERDBG").is_ok() {
                eprintln!("[handler] pad B{hb} ALREADY emitted at depth {depth} -> goto");
            }
            return vec![self.goto_stmt_reason(hb, GotoReason::HandlerPad)];
        }
        // UNDO the mark. The insert above is the "has this pad been rendered
        // already?" PROBE, not a claim on the block: the walk below is what
        // must emit it, and `emit_region` refuses to walk a block that is
        // already in `emitted` (its global emit-once guard). Leaving the mark
        // in place made every single handler body compile to `goto L<pad>`
        // aimed at itself — the pad was "already emitted" before anything was
        // emitted — so EVERY catch and finally body in the corpus was silently
        // dropped, leaving `finally { // goto L21; (target not emitted) }`
        // shells. The try-fold path next door already does exactly this
        // (`self.emitted.borrow_mut().remove(&cur);`); the handler path simply
        // never got the same line.
        self.emitted.borrow_mut().remove(&hb);
        // The handler must structure its OWN control flow: the outer walk's
        // `stops` fence the try-region merge points, but a landing pad's
        // branches target resume blocks BEYOND those fences — structuring
        // with the outer stops would drop every arm and render `goto L…`.
        // What fences it instead is WHERE CLEANUP ENDS: a landing pad is
        // entered by unwinding, so neither it nor the cleanup chain it heads
        // is reachable by ordinary control flow, while the block that chain
        // finally falls into IS. Stopping the walk at `normal_reachable`
        // keeps the handler body to the cleanup itself; without it the walk
        // ran straight through into the rest of the method and every
        // `finally` body swallowed the remaining code (and the `return`).
        // `hb` is excluded so a pad that also has an ordinary predecessor
        // still renders its body.
        let mut handler_stops: HashSet<usize> = self.normal_reachable.clone();
        handler_stops.remove(&hb);
        // Re-entry into try-folded code from a pad resume edge must NOT fold
        // another try (pad → resume-in-body → try → pad → …). The handler
        // renders as plain structured code.
        let prev = self.in_handler.replace(true);
        let (mut stmts, _) = self.emit_region(hb, &handler_stops, loops, loop_exits, depth + 1);
        self.in_handler.set(prev);
        if std::env::var("ILR_HANDLERDBG").is_ok() {
            eprintln!("[handler] pad B{hb} at depth {depth} -> {} stmts: {:?}", stmts.len(), stmts);
        }
        if stmts.is_empty() {
            // genuinely empty handler (pure unwinding): keep it explicit
            stmts.push(Stmt::Comment("handler body not recovered".into()));
        }
        stmts
    }

    fn jump_stmt(
        &self,
        to_blk: usize,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
    ) -> Stmt {
        // `continue` and `break` are the INNERMOST loop's statements: they are
        // the faithful rendering only when the target belongs to the loop the
        // body is inside AT THIS POINT (`loop_exits` is outermost-first, so the
        // innermost enclosing loop is last). Emitting `continue` for ANY loop
        // header, and `break` for any loop's exit, put those statements where
        // they do not apply: a `continue;` with no loop around it is not even
        // valid C#, and it jumps somewhere the native branch does not go
        // (measured: 30 methods on the 4,000-method sample, `get_JumpHeight`
        // @0x2DF00 among them). An ordinary `goto` to the target is the branch
        // the disassembly actually has, and a loop header is always a branch
        // target so its label is defined.
        if let Some(&(header, exit)) = loop_exits.last() {
            if loops.contains_key(&header) {
                if to_blk == header {
                    return Stmt::Continue;
                }
                if to_blk == exit {
                    return Stmt::Break;
                }
            }
        }
        self.forced_labels.borrow_mut().insert(to_blk);
        Stmt::Goto(label_name(self.sc, to_blk))
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_if(
        &self,
        blk: usize,
        taken: usize,
        stops: &HashSet<usize>,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
        depth: usize,
        cond: &Expr,
        cond_claims: &[usize],
    ) -> (Vec<Stmt>, Option<usize>) {
        let b = &self.sc.blocks[blk];
        let fall = b.succ_fall.filter(|f| f < &self.sc.blocks.len());

        // SKIP pattern: `if (c) goto JOIN; <body>; JOIN:` — the branch target is
        // reached through the fall arm, so the taken arm is EMPTY and the join
        // is the target itself. Decided here (not in `find_merge`) because it
        // must be suppressed for loop headers: a back-edge target is reachable
        // from the header's arm without being a join.
        let skip_merge = if !loops.contains_key(&taken)
            && !fall.is_some_and(|f| loops.contains_key(&f))
        {
            match fall {
                Some(f) if reachable_block(self.sc, f, taken) => Some(taken),
                Some(f) if reachable_block(self.sc, taken, f) => Some(f),
                _ => None,
            }
        } else {
            None
        };

        let mut merge = skip_merge.or_else(|| find_merge(self.sc, blk, fall, Some(taken)));
        // DETECTION: a merge must be reachable from BOTH arms — that is what
        // "merge" means, and `ipdom[header]` guarantees it. If a chosen merge
        // is reachable from only one arm (a stale answer from the reachability
        // fallback, or a CFG edge the lifter dropped), the text would place
        // the join where only one arm can fall into it, and the other arm
        // would silently execute code it never reaches. Refuse the merge — the
        // arms then carry explicit gotos — and count it as broken control flow.
        if let Some(m) = merge {
            if m < self.sc.blocks.len() {
                let from_taken = reachable_block(self.sc, taken, m);
                let from_fall = fall.is_none_or(|f| reachable_block(self.sc, f, m));
                if !from_taken || !from_fall {
                    STRUCT_CENSUS
                        .merge_not_common
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    merge = None;
                }
            }
        }

        let mut inner_stops: HashSet<usize> = stops.iter().copied().collect();
        if let Some(m) = merge {
            inner_stops.insert(m);
        }

        let terminates = |body: &[Stmt]| {
            body.last().is_some_and(|s| matches!(s, Stmt::Return(_) | Stmt::Throw(_)))
        };

        // FLAT GUARD (`merge == taken`): `if (c) goto JOIN; <fall body>; JOIN:`.
        //
        // The taken edge goes straight to the join, so its arm is always empty
        // (its target is in `inner_stops`) and the fall arm carries the body.
        // The generic path below renders that as `if (!c) { <fall body> }`,
        // which is correct — but it puts the ENTIRE remainder of the region
        // INSIDE this guard, so every following guard nests one level deeper.
        //
        // That staircase is what the compiler's unrolled early-exit form looks
        // like: `cmp #k; b.cc JOIN` repeated N times (all jumping to one shared
        // epilogue — a 32-way zero-fill in a `.cctor`, a long guard chain in a
        // big ToString). N guards then cost N nesting levels, and the renderer's
        // nesting cap truncated the chain — silently deleting the rest of the
        // method. Measured on 0x2478f0c (1708 native insns): the emitted body
        // carried 35 of the 308 native conditions, with no marker anywhere.
        //
        // Emitting the jump FIRST and the fall body as a SIBLING (`c ? goto
        // JOIN : { body }`) says exactly the same thing with constant depth, so
        // a chain of any length renders in full.
        //
        // This is decided BEFORE the arms are emitted, and that ordering is the
        // whole point: the fall arm must be walked as a SIBLING too (see
        // `emit_arm`'s `depth_step`). Walking it at `depth + 1` re-created the
        // staircase one level per guard *inside the arm recursion*, so the
        // chain still reached the nesting cap even though the emitted text was
        // flat — the cap then deleted the arm's tail.
        // A LONE guard is not worth the `goto`. The flat form's own `body` is
        // only this guard's fall arm — the merge is handed back to the walker
        // as a SIBLING (`return (stmts, Some(m))` below) — so nesting it costs
        // exactly ONE level, and the staircase that motivated the flat form is
        // a CHAIN of guards to the SAME join, each nesting the next. Measured
        // on the 4k ARM32 corpus: 17,092 flat-guard chains but only 1,332 fused
        // shells, i.e. ~92% are lone guards, and NO method reaches
        // MAX_NEST_DEPTH. So gate the flat form on the chain actually existing:
        // the fall-through must itself be a guard to the same join, or an
        // enclosing guard must have declared this block its chain's next link
        // (`flat_chain_next`). Otherwise fall to the generic path, which
        // renders `if (!c) { body }` with no goto and no label.
        // …but nesting must never march the walk toward the cap. The generic
        // path emits its arms at `depth + 1`, so a run of lone guards would add
        // one level each; this reserves headroom so the run cannot reach
        // `MAX_NEST_DEPTH`. (It does not remove the ONE arm64 method that
        // already sits at the cap deep in its own body — that region stays
        // marked and coverage-repaired, uncovered insns 6 → 6.)
        const LONE_GUARD_HEADROOM: usize = 12;
        // Chain length: consecutive non-loop `CondJump` blocks to the same
        // join down the fall edge. Generic nesting costs one level per link,
        // so the flat (constant-depth) form is only needed when the chain
        // would otherwise march the walk toward the cap. A 2-guard chain at
        // depth 0 costs 2 levels and leaves zero gotos; making it flat costs
        // 2 gotos to save 2 levels nobody was short of. Measured: ~92% of
        // flat guards are lone/short chains where the fuse never fires (the
        // fall arm begins with assignments, not another guard shell), so the
        // flat form paid a goto and saved nothing.
        let fall_is_guard = fall.is_some_and(|f| {
            self.sc.blocks.get(f).and_then(|b| b.succ_taken) == Some(taken)
        });
        let flat_chain = self.flat_chain_next.get() == Some(blk)
            || depth + LONE_GUARD_HEADROOM >= MAX_NEST_DEPTH
            || fall_is_guard;
        if merge == Some(taken) && flat_chain {
            if std::env::var("ILR_IFDBG").is_ok() {
                eprintln!(
                    "[flat-guard] blk={blk} join={taken} fall={fall:?} fall_in_stops={} cond={cond:?}",
                    fall.map(|f| inner_stops.contains(&f)).unwrap_or(true)
                );
            }
            // Arm the one-shot chain marker for the fall-through block, so a
            // guard that is the NEXT link of this chain stays flat even when
            // its own body is not another guard. Restored after the walk.
            let prev_chain = self.flat_chain_next.replace(fall);
            let emitted_before = self.emission_log.borrow().len();
            let mut else_body = if let Some(f) = fall.filter(|f| !inner_stops.contains(f)) {
                self.emit_arm(f, &inner_stops, loops, loop_exits, depth, 0, Some(taken))
            } else {
                Vec::new()
            };
            let emitted_fall_blocks: HashSet<usize> = self.emission_log.borrow()
                [emitted_before..]
                .iter()
                .copied()
                .collect();
            self.flat_chain_next.set(prev_chain);
            // THE FALSE EDGE MUST BE CARRIED. When the fall arm is a stop of
            // THIS region, `emit_arm` returns nothing for it (it refuses to
            // walk a stop), and the walker then continues at `taken` — so the
            // `c == false` path silently fell into the TAKEN arm instead of
            // leaving the region. That is a dropped control-flow edge, not a
            // cosmetic flattening: it makes the guard unconditional. Measured:
            // 700 of the 1914 empty-fall flat guards on the 4000-method sample
            // are this case (`fall == taken` — a genuinely degenerate branch —
            // is ZERO of them, and the remaining 1214 are blocks whose emitted
            // body is empty and which fall through to `taken` anyway, so they
            // need nothing).
            //
            // The jump is emitted as a SIBLING of the `if`, exactly like the
            // flat-guard's own goto, so the shape stays
            // `if (c) goto T; goto F; T:` at constant depth. If the label is
            // never rendered `drop_unreachable_gotos` removes the goto and the
            // output is what it was before — so this can only add fidelity.
            if else_body.is_empty() {
                if let Some(f) = fall {
                    if f != taken && f < self.sc.blocks.len() && stops.contains(&f) {
                        else_body = vec![self.goto_stmt_reason(f, GotoReason::FlatGuard)];
                        if !self.discovery {
                            STRUCT_CENSUS
                                .flat_fall_edge_kept
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
            }
            // SHORT-CIRCUIT CHAIN COLLAPSE.
            //
            // A chain of guards that all jump to the SAME join is one compound
            // condition, not N branches:
            //
            //     if (c1) goto J; if (c2) goto J; if (c3) goto J;  <body>  J:
            //
            // is exactly `if (c1 || c2 || c3) goto J;` — the taken target is
            // reached when ANY condition holds, and each `ci` is only evaluated
            // when every earlier one was false, which is `||`'s short-circuit
            // rule. This is the compiler's test-chain lowering of
            // `if (a != 0 && b != 0 && …)`, and it is the single largest source
            // of `goto` in the output: 11348 of 19259 goto sites (58.9%) on the
            // 4000-method sample.
            //
            // Fusing them is a strict improvement on every axis — one condition
            // instead of N, one jump instead of N, and the recovered text names
            // the compound predicate the source actually wrote.
            //
            // The shape is precisely: the fall arm's statements BEGIN with a
            // run of `if (ci) { goto <same label>; }` shells, each of which was
            // produced by this same flat-guard arm for the following block (the
            // fall arm was emitted with `depth_step = 0`, so the whole chain
            // lands at ONE level and the shells are plain siblings). Stop at the
            // first statement that is anything else — that is the real body.
            let target_label = label_name(self.sc, taken);
            let mut conds: Vec<Expr> = vec![cond.clone()];
            loop {
                let mergeable = match else_body.first() {
                    Some(Stmt::If { then_body, else_body: e, .. }) => {
                        e.is_empty()
                            && then_body.len() == 1
                            && matches!(&then_body[0], Stmt::Goto(l) if *l == target_label)
                    }
                    _ => false,
                };
                if !mergeable {
                    break;
                }
                let Stmt::If { cond: c2, .. } = else_body.remove(0) else {
                    unreachable!()
                };
                conds.push(c2);
            }
            // Counted for the structural census, but NOT for the discovery
            // walk — the structurer runs twice per method (once to discover
            // call bindings, once for real) and the discovery output is thrown
            // away. Counting there doubles every flat-guard figure, which is
            // exactly the kind of quietly-wrong number this census exists to
            // avoid: without this guard the report claimed 22492 chains and
            // 1914 vacuous guards for 11246 and 957 real ones.
            if !self.discovery {
                use std::sync::atomic::Ordering::Relaxed;
                STRUCT_CENSUS.flat_guard_chains.fetch_add(1, Relaxed);
                if conds.len() > 1 {
                    STRUCT_CENSUS
                        .flat_guard_fused
                        .fetch_add(conds.len() as u64 - 1, Relaxed);
                }
                // A flat guard whose fall arm emitted NOTHING renders as
                // `if (c) goto L; L:` with the label immediately after — both
                // edges reach the same block, so the branch is vacuous.
                if else_body.is_empty() {
                    STRUCT_CENSUS.flat_guard_empty_fall.fetch_add(1, Relaxed);
                    // WHY the fall arm is still empty decides the fix
                    // completely:
                    //  * `fall == taken` — the block has TWO edges to one block,
                    //    so the branch cannot change control flow. (Measured over
                    //    the sample: ZERO occurrences. Real code never has this,
                    //    so a non-zero count here would itself be a defect.)
                    //  * `fall ∈ stops`  — carried explicitly above; 0 now.
                    //  * anything else   — `emit_arm` ran but produced nothing
                    //    because the block's emitted body is empty AND that block
                    //    falls through to `taken`, so falling through is the
                    //    right continuation and no jump is owed.
                    match fall {
                        Some(f) if Some(f) == Some(taken) => {
                            STRUCT_CENSUS.flat_empty_same.fetch_add(1, Relaxed);
                        }
                        Some(f) if inner_stops.contains(&f) => {
                            STRUCT_CENSUS.flat_empty_stop.fetch_add(1, Relaxed);
                        }
                        _ => {
                            STRUCT_CENSUS.flat_empty_other.fetch_add(1, Relaxed);
                        }
                    }
                }
            }
            // Left-associative, so the printed form evaluates left to right
            // (`a || b || c`), which is the order the branches were tested.
            let cond_acc = conds
                .into_iter()
                .reduce(|a, b| Expr::Bin(BinOp::OrElse, Box::new(a), Box::new(b)))
                .unwrap_or_else(|| cond.clone());
            // When the fall body can safely live in an `if` scope, spell the
            // guard as the structured equivalent directly:
            // `if (!(c1 || ...)) { body }`. This preserves short-circuit
            // evaluation and the exact taken/fall edges while avoiding one
            // label jump per guard. Keep the flat goto form when nesting the
            // body would change the scope of a declared local or make a label
            // reachable only conditionally.
            let labels_are_internal = flat_guard_labels_are_internal(
                self.sc,
                &else_body,
                &emitted_fall_blocks,
                blk,
            );
            if let Some(rewritten) = flat_guard_replacement(
                &cond_acc,
                &else_body,
                depth,
                labels_are_internal,
            ) {
                if !self.discovery {
                    STRUCT_CENSUS.flat_guard_structured.fetch_add(
                        1, std::sync::atomic::Ordering::Relaxed,
                    );
                }
                return (vec![rewritten], Some(taken));
            }
            if !else_body.is_empty() && !self.discovery {
                use std::sync::atomic::Ordering::Relaxed;
                STRUCT_CENSUS.flat_guard_scope_refused.fetch_add(1, Relaxed);
                match flat_guard_body_nesting_refusal(&else_body, depth, labels_are_internal) {
                    Some(FlatGuardNestingRefusal::Label) => {
                        STRUCT_CENSUS.flat_guard_refused_label.fetch_add(1, Relaxed);
                    }
                    Some(FlatGuardNestingRefusal::Declaration) => {
                        STRUCT_CENSUS.flat_guard_refused_declaration.fetch_add(1, Relaxed);
                    }
                    Some(FlatGuardNestingRefusal::Depth) => {
                        STRUCT_CENSUS.flat_guard_refused_depth.fetch_add(1, Relaxed);
                    }
                    None => {}
                }
            }
            let mut stmts = Vec::new();
            stmts.push(Stmt::If {
                cond: cond_acc,
                then_body: vec![self.goto_stmt_reason(taken, GotoReason::FlatGuard)],
                else_body: Vec::new(),
            });
            stmts.extend(else_body);
            return (stmts, Some(taken));
        }

        // Polarity: `branch_condition` returns the TAKEN condition (the
        // lifter's `Bcc` → `Compare(cond_to_kind)` + `CondJump`, and
        // `def_expr` renders the Compare un-negated), so the taken arm runs
        // when `cond` is true and the fallthrough arm when it is false.
        // Previously the arms were assigned the other way around — the
        // fallthrough body was emitted under `if (cond)` — which rendered
        // every null-guard inverted (`if (Motor == 0) { body }` where the
        // body runs only when Motor != 0) and produced the impossible
        // `if (x == 0) { if (x != 0) … }` nesting. Swap so `then` = taken.
        let then_body = self.emit_arm(taken, &inner_stops, loops, loop_exits, depth, 1, merge);
        let else_body =
            self.emit_arm(fall.unwrap_or(blk + 1), &inner_stops, loops, loop_exits, depth, 1, merge);

        // SKIP pattern whose BODY arm is fenced: `if (c) goto J; BODY; J:` where
        // BODY is already fenced by an ANCESTOR's merge (the outer `if (a) goto
        // J2; …` shares this block as its join). Both arms then render empty,
        // the `if` is dropped as a pure shell — and every value that fed
        // `cond` goes with it: `get_Item`, the `LDRB [.. +0x130]` pair, the
        // scaled array load. That is the "dropped chain" (verified on
        // GetEffectiveSchemaType @0x1f4e79c: the whole type-hash comparison
        // vanished while its loads stayed in the IR). Keep the branch explicit
        // — the taken edge genuinely jumps to the join, so emit it as a goto
        // and force the join's label. Falling through instead reaches the same
        // block (the body is empty), so the merge continuation below stays
        // valid.
        // `merge == Some(taken)` was handled above, so here `m != taken`: the
        // taken arm flows (transitively) into the fall block. When BOTH arms
        // render empty the `if` would be a pure shell that the printer drops,
        // taking `cond` and every value that fed it with it — so keep the branch
        // explicit as a jump to the join.
        if let Some(m) = skip_merge {
            if then_body.is_empty() && else_body.is_empty() {
                if std::env::var("ILR_IFDBG").is_ok() {
                    eprintln!("[skip-empty] blk={blk} goto={m} cond={cond:?}");
                }
                let stmts = vec![Stmt::If {
                    cond: cond.clone(),
                    then_body: vec![self.goto_stmt_reason(m, GotoReason::SkipEmpty)],
                    else_body: Vec::new(),
                }];
                return (stmts, Some(m));
            }
        }

        // Was the TAKEN block actually walked as this `if`'s arm? That decides
        // the mergeless continuation below: a `goto taken` routes the
        // fall-through to code that was NOT emitted inline, so it is only owed
        // when the taken arm was skipped (a stop/fence). When it WAS walked,
        // control is already there — a post-`if` `goto taken` is a jump to the
        // arm's own first statement. Measured at 0x73BDCC: the coverage region's
        // `if (slotType != 1) { L6: } else { … }` got a trailing `goto L6` whose
        // target was the then-arm's own label, and `branch_repair`'s inversion
        // then left the self-loop `L6: goto L6;`.
        let taken_arm_walked = self.emitted.borrow().contains(&taken);
        let (mut cond_r, mut then_b, mut else_b) = if then_body.is_empty() && !else_body.is_empty() {
            (invert(cond), else_body, Vec::new())
        } else {
            (cond.clone(), then_body, else_body)
        };
        // then-arm terminates and else-arm is empty/noise → control can
        // never reach the else; render `if (c) { … }` without the shell.
        if terminates(&then_b) && else_b.iter().all(is_noise_stmt) {
            else_b.clear();
        }
        // symmetric: else-arm terminates and then-arm is empty → invert the
        // predicate so the payload arm becomes the then-branch.
        if !then_b.is_empty() == false && terminates(&else_b) && !else_b.is_empty() {
            cond_r = invert(&cond_r);
            std::mem::swap(&mut then_b, &mut else_b);
            else_b.clear();
        }
        if std::env::var("ILR_IFDBG").is_ok() {
            eprintln!("[if] blk={} merge={:?} skip={:?} taken={} fall={:?} then={} else={} cond={:?}", blk, merge, skip_merge, taken, fall, then_b.len(), else_b.len(), cond_r);
        }

        // mergeless join: a goto is only needed when control can actually
        let both_terminate =
            !else_b.is_empty() && terminates(&then_b) && terminates(&else_b);

        // An `if` with no content on either side is a pure shell: the renderer
        // drops it, so nothing that went into its CONDITION reaches the
        // output. Revoke those claims — this is the `sem_un: 0` false-clean
        // that hid whole lost basic blocks (the leftover
        // `if (*(nint*)(… .Item +304) >= 0) { }` in the stage dump).
        //
        // OPT-IN (`ILR_AUDIT_STRICT=1`) while the revocation predicate is
        // tuned: on branch-dense methods the empty-arm case fires far more
        // often than the statement really is dropped (arms already emitted
        // via `self.emitted` also yield empty `emit_arm` results), which
        // revokes true claims and turns the false-clean into a false-alarm.
        // Verified to catch a real dropped chain (see the audit docs), but it
        // must not become the default until the predicate distinguishes
        // "shell dropped" from "arm already emitted".
        if std::env::var("ILR_AUDIT_STRICT").is_ok()
            && then_b.is_empty()
            && else_b.is_empty()
        {
            let mut d = self.ctx.dropped_claims.borrow_mut();
            for i in cond_claims {
                d.insert(*i);
            }
        }

        let mut stmts = Vec::new();
        let then_terminated = terminates(&then_b);
        let else_terminated = terminates(&else_b);
        stmts.push(Stmt::If { cond: cond_r, then_body: then_b, else_body: else_b });
        if let Some(m) = merge {
            if std::env::var("ILR_IFDBG").is_ok() {
                eprintln!("[merge] blk={} m={} in_stops={} emitted={}", blk, m, stops.contains(&m), self.emitted.borrow().contains(&m));
            }
            let m_stops = stops.contains(&m);
            let m_emitted = self.emitted.borrow().contains(&m);
            if !m_stops && !m_emitted {
                // The join is the region's CONTINUATION, so it is a SIBLING of
                // this `if`, not a child. Handing it back to the walker (rather
                // than recursing one level deeper into it) is what keeps a long
                // chain of guards at constant nesting depth; recursing here is
                // how a chain reached `MAX_NEST_DEPTH` and lost its tail.
                return (stmts, Some(m));
            }
            if m_emitted && !(then_terminated && else_terminated) {
                // The merge was already emitted by an earlier arm walk —
                // emit an explicit jump so control still flows there.
                stmts.push(self.goto_stmt_reason(m, GotoReason::MergeEmitted));
            }
        } else if !taken_arm_walked
            && !matches!(stmts.last(), Some(Stmt::Goto(_)))
            && !both_terminate
        {
            stmts.push(self.goto_stmt_reason(taken, GotoReason::NoMerge));
        }
        (stmts, None)
    }

    /// Walk one arm of an `if`.
    ///
    /// `depth_step` is how much nesting this arm costs. It is 1 for an arm that
    /// is the guard's BODY (the generic case), and 0 for an arm that is the
    /// guard's SIBLING — the flat-guard case, where the arm is emitted after the
    /// `if (c) goto JOIN;` at the same indent level. Charging 1 there re-created
    /// the nesting staircase inside the arm recursion, so a long guard chain hit
    /// `MAX_NEST_DEPTH` and lost its tail even though the printed text was flat.
    #[allow(clippy::too_many_arguments)]
    fn emit_arm(
        &self,
        start: usize,
        stops: &HashSet<usize>,
        loops: &HashMap<usize, LoopInfo>,
        loop_exits: &[(usize, usize)],
        depth: usize,
        depth_step: usize,
        join: Option<usize>,
    ) -> Vec<Stmt> {
        if stops.contains(&start) || start >= self.sc.blocks.len() {
            return Vec::new();
        }
        // LOOP-FIRST: an arm may start inside a loop body (e.g. the else-arm
        // targets a mid-loop block). Route to the loop header first so
        // emit_loop structures the whole loop; otherwise this walk consumes
        // body blocks and emit_loop later renders an empty body.
        let arm_start = self.loop_of_block.get(&start);
        let effective = match arm_start {
            Some(&h) if !stops.contains(&h) && !self.emitted.borrow().contains(&h) => h,
            _ => start,
        };
        let (mut stmts, next) =
            self.emit_region(effective, stops, loops, loop_exits, depth + depth_step);
        if std::env::var("ILR_IFDBG").is_ok() {
            eprintln!("[arm] start={} effective={} stmts={} next={:?} stops_has_start={}", start, effective, stmts.len(), next, stops.contains(&start));
        }
        // Drop a trailing `goto` ONLY when it targets THIS arm's join. The join
        // is emitted as the sibling immediately after the `if`, so such a goto
        // is a redundant fall-through. A trailing goto to any OTHER block is a
        // REAL edge — very often to a block already emitted under a different
        // parent (a shared merge reached from the other arm) — and must be
        // kept. Popping every trailing goto silently deleted those edges: the
        // `ClampValue` @0x1a8d884 negative-equal arm lost its shared
        // `fn_40c004` block this way, so it fell through into the wrong code.
        if let Some(j) = join {
            let jl = label_name(self.sc, j);
            while matches!(stmts.last(), Some(Stmt::Goto(l)) if *l == jl) {
                stmts.pop();
            }
        }
        stmts
    }

    /// Would the header's own code (everything before its test) produce a
    /// STATEMENT — a call, a store, or a copy that carries a variable's value
    /// forward?
    ///
    /// A plain `[load, compare, CondJump]` header emits nothing: the load is
    /// inlined into the test, so there is nothing to recover and the ordinary
    /// `while (…)` reading is already faithful. A header that DOES carry code
    /// (`i = next` after the test, the prologue slot copies a dynamic dispatch
    /// re-reads) is the case where dropping it loses real statements.
    ///
    /// Mirrors `boundary_stmts`' emission rules but claims nothing, so it is
    /// safe to ask before deciding.
    fn header_prelude_emits_statement(&self, header: usize) -> bool {
        self.header_after_test_range(header).any(|i| {
            let insn = &self.insns[i];
            match &insn.op {
                Op::Call | Op::CallVoid | Op::ObjcMsgSend { .. } => true,
                Op::Move => match (insn.operands.first(), insn.operands.get(1)) {
                    (Some(Operand::Memory { .. }), _) => true,
                    (Some(Operand::Local(v)), Some(Operand::Local(_))) => {
                        self.ctx.var_reads.contains(v)
                    }
                    (Some(Operand::Local(v)), _) => {
                        self.ctx.is_multi(*v) || self.ctx.var_reads.contains(v)
                    }
                    _ => false,
                },
                _ => false,
            }
        })
    }

    /// The insns of `header` that run AFTER its branch test.
    ///
    /// The compare that feeds the branch is rendered BY the test (it is inlined
    /// there), so only what the hardware executes after it is missing from a
    /// `while (test) { … }` body — the test runs first in that form, so the
    /// post-test ops belong at the top of the body. Everything before the
    /// compare is that compare's own input chain.
    fn header_after_test_range(&self, header: usize) -> std::ops::Range<usize> {
        let b = &self.sc.blocks[header];
        let term = b.end.saturating_sub(1);
        let cond_def = self.insns[term]
            .operands
            .first()
            .and_then(|o| o.as_local())
            .and_then(|v| {
                if v >= u32::MAX - 1 {
                    self.find_scratch_def(v, term)
                } else {
                    self.ctx.defs.get(&v).copied()
                }
            });
        let start = match cond_def {
            Some(c) if c >= b.start && c < term => c + 1,
            _ => b.start,
        };
        start..term
    }

    /// The header's post-test statements: the loop-carried updates a rotated
    /// loop keeps at the top (`i = next`, the ABI slot copies a dynamic
    /// dispatch re-reads). Statements BEFORE the test feed the condition, and
    /// the guard inlines those — emitting them as well produced a binding
    /// nothing reads (`v40 = this._pickupRoot;` beside
    /// `if (this._pickupRoot == 0)`), an invented variable.
    fn fold_prelude_after_test(&self, header: usize) -> Vec<Stmt> {
        let mut out = Vec::new();
        for i in self.header_after_test_range(header) {
            out.extend(self.ctx.boundary_stmts(i));
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_loop(
        &self,
        header: usize,
        info: &LoopInfo,
        outer_stops: &HashSet<usize>,
        loops: &HashMap<usize, LoopInfo>,
        outer_exits: &[(usize, usize)],
        depth: usize,
    ) -> (Vec<Stmt>, Option<usize>) {
        let hb = &self.sc.blocks[header];
        let outside = |t: Option<usize>| t.map(|t| !info.body.contains(&t)).unwrap_or(false);
        let exit_on_fall = outside(hb.succ_fall);
        // Does the HEADER's own branch leave the loop? Only a header branch
        // that exits can be the loop's test. When NEITHER header successor
        // exits (its branch keeps both edges inside the body) the header is
        // carrying interior code — a hoisted null guard, a rotated prologue
        // — and the loop's real test sits at a latch. Taking the guard for
        // the test rendered `while (obj == 0)` and made the guard's payload
        // unconditional (`SmallXmlParser.OnStartElement` @0x12F45D0).
        let header_exit = if exit_on_fall {
            hb.succ_fall
        } else if outside(hb.succ_taken) {
            hb.succ_taken
        } else {
            None
        };
        // The exit edge can also live at a LATCH (do-while form). Find the
        // latch branch whose non-header edge leaves the body so `break`,
        // secondary-exit accounting and the do-while test still work.
        let latch_exit = if header_exit.is_some() {
            None
        } else {
            let mut e = None;
            for &l in info.latches.iter() {
                let lb = &self.sc.blocks[l];
                if lb.end <= lb.start
                    || !matches!(self.insns[lb.end - 1].op, Op::CondJump(_))
                {
                    continue;
                }
                if lb.succ_taken == Some(header) && outside(lb.succ_fall) {
                    e = lb.succ_fall;
                    break;
                }
                if lb.succ_fall == Some(header) && outside(lb.succ_taken) {
                    e = lb.succ_taken;
                    break;
                }
            }
            e
        };
        let exit_blk = header_exit.or(latch_exit);

        let mut loop_exits: Vec<(usize, usize)> = outer_exits.to_vec();
        if let Some(e) = exit_blk {
            loop_exits.push((header, e));
        }

        let body_entry = if exit_on_fall {
            hb.succ_taken.or(hb.succ_fall)
        } else {
            hb.succ_fall.or(hb.succ_taken)
        };
        // Degenerate headers (no usable successor) previously fell back to
        // the header itself — which is in body_stops — yielding an empty
        // `while (…) { }`. Fall back to any real body block instead.
        let body_start = match body_entry {
            Some(b) if b < self.sc.blocks.len() && info.body.contains(&b) && b != header => Some(b),
            _ => info.body.iter().copied().filter(|&b| b != header).min(),
        };
        // The global emission guard must not swallow the loop body: when
        // control jumped into the loop interior earlier (goto into the loop),
        // the body entry is already in `emitted` and the guard would render
        // an empty body with a stray goto. Body emission is fenced by
        // body_stops (every non-body block stops the walk), so clearing the
        // loop's own blocks here is safe and keeps the emit-once property.
        {
            let mut em = self.emitted.borrow_mut();
            for &b in info.body.iter() {
                em.remove(&b);
            }
        }

        let mut body_stops: HashSet<usize> = outer_stops.clone();
        if let Some(e) = exit_blk {
            body_stops.insert(e);
        }
        // the header terminates the body pass — without this the region
        // walker re-enters the loop and nests it infinitely
        body_stops.insert(header);
        for bi in 0..self.sc.blocks.len() {
            if !info.body.contains(&bi) {
                body_stops.insert(bi);
            }
        }

        // The header is the loop TEST only when its branch actually exits.
        // A header `CondJump` that keeps both edges inside the body is
        // interior code (see `header_exit`), so the do-while/latch path below
        // must own the condition instead.
        let has_header_test = header_exit.is_some()
            && matches!(self.insns[hb.end - 1].op, Op::CondJump(_));

        if has_header_test {
            let mut cond = self.branch_condition(header);
            if !exit_on_fall {
                cond = invert(&cond); // taken exits → invert for `while`
            }
            // ROTATED loop: the header is its OWN latch — the back edge is the
            // header's own branch back to its first instruction, so the body
            // lives inside the header and the native code runs it BEFORE the
            // test. `while (cond) { body }` is then not merely optimistic, it
            // is a different program: it may skip the body entirely and, even
            // when it does not, every loop-carried register was renamed to its
            // entry version (`while (this._innerException != 0)` with an empty
            // body — an unkillable infinite loop). Only `do { body } while
            // (cond)` keeps the body's "at least once" and its reassignments.
            if info.latches.contains(&header) {
                let body = match body_start {
                    Some(b) => {
                        let (body, _) = self.emit_loop_body(
                            b,
                            &body_stops,
                            loops,
                            &loop_exits,
                            exit_blk,
                            depth + 1,
                        );
                        if body.is_empty() {
                            self.fold_prelude(header)
                        } else {
                            body
                        }
                    }
                    None => self.fold_prelude(header),
                };
                return (vec![Stmt::DoWhile { body, cond }], exit_blk);
            }
            // Single-block loop (body == {header}): the body statements live
            // in the header itself before its CondJump — emit them directly
            // instead of re-walking a region that body_stops would stop
            // immediately (the old `while (…) { }` artifact).
            //
            // The header's POST-TEST statements belong at the top of the body.
            // The hardware runs the header top to bottom every iteration —
            // [pre-test ops] test [post-test ops] then the body — and a
            // `while (test) { … }` re-runs the test first, so emitting the
            // post-test ops as the body's first statements reproduces that order
            // exactly. A rotated loop keeps its induction update there
            // (`i = next` sits after the test): dropping it left the update as
            // `v128 = v49 + 1;` (a local nothing reads) while the test kept
            // reading the entry `v49` — the exact "unused variable plus logic
            // the disassembly does not contain" shape, measured on
            // `get_JumpHeight` @0x2DF00. Pre-test statements are NOT emitted:
            // the test inlines them, and emitting them as well produced a dead
            // binding (`v40 = this._pickupRoot;` beside
            // `if (this._pickupRoot == 0)`).
            let (body, _) = match body_start {
                Some(b) => {
                    let (rest, _) = self.emit_loop_body(
                        b,
                        &body_stops,
                        loops,
                        &loop_exits,
                        exit_blk,
                        depth + 1,
                    );
                    if self.header_prelude_emits_statement(header) {
                        let mut pre = self.fold_prelude_after_test(header);
                        pre.extend(rest);
                        (pre, None::<usize>)
                    } else {
                        (rest, None::<usize>)
                    }
                }
                None => (self.fold_prelude(header), None),
            };
            if std::env::var("ILR_LOOPDBG").is_ok() {
                // Diagnostic only, and deliberately SIDE-EFFECT FREE: do not
                // call `fold_prelude` here, it appends to the claim log.
                eprintln!(
                    "[loop] header=B{} body_start={:?} body_blocks={:?} body_len={} stops={:?}",
                    header,
                    body_start,
                    info.body,
                    body.len(),
                    body_stops
                );
                eprintln!("[loop]   body={:?}", body);
            }
            // Single-block loop (body == {header}): the walk stops at the
            // header itself (body_stops), so emit_region returns nothing even
            // though the header's prelude (before its CondJump) IS the body.
            if body.is_empty() && info.body.len() == 1 {
                let prelude = self.fold_prelude(header);
                if !prelude.is_empty() {
                    let stmts = vec![Stmt::While { cond, body: prelude }];
                    return (stmts, exit_blk);
                }
            }
            let stmts = vec![Stmt::While { cond, body }];
            (stmts, exit_blk)
        } else {
            // The header's OWN code runs first on every iteration. When the
            // header terminates in a CondJump that does NOT leave the loop,
            // that branch is interior code (a hoisted null guard, a rotated
            // prologue), not the loop test — and it must be rendered INSIDE
            // the do-while body. Dropping it made its payload unconditional
            // and inverted the loop text (0x12F45D0). Walk the body from the
            // HEADER, excluding it from the loop lookup for the duration so
            // `emit_region` structures it as an ordinary block; the latch's
            // back edge then re-enters an already-`emitted` header and
            // renders as `continue` (its label is the loop itself), exactly
            // the loop-carried edge.
            let header_interior_branch = header_exit.is_none()
                && info.body.len() > 1
                && matches!(self.insns[hb.end - 1].op, Op::CondJump(_));
            let (body, _) = if header_interior_branch {
                body_stops.remove(&header);
                let prev = self.skip_loop_header.replace(Some(header));
                let r = self.emit_loop_body(
                    header,
                    &body_stops,
                    loops,
                    &loop_exits,
                    exit_blk,
                    depth + 1,
                );
                self.skip_loop_header.set(prev);
                r
            } else {
                match body_start {
                    Some(b) => self.emit_loop_body(
                        b,
                        &body_stops,
                        loops,
                        &loop_exits,
                        exit_blk,
                        depth + 1,
                    ),
                    None => (self.fold_prelude(header), None),
                }
            };
            // A latch that is a single self-jumping insn (e.g. the async
            // state machine's parking branch `B: jump B`) has NO statements
            // in the recovered IR — the interesting work sits in the blocks
            // BEFORE the loop. Calling that "loop body not recovered" is
            // misleading; the loop really is an empty spin.
            let empty_body = body.is_empty();
            // Latch condition: prefer the latch's own conditional jump back
            // to the header — that is the real loop test even when the
            // defining Compare lives a few instructions earlier (outside the
            // latch block). When the latch ends in an unconditional jump
            // (test computed in a predecessor inside the body), walk the
            // latch's in-body predecessors and take their tail test. Only
            // fall back to `true` when no test exists at all.
            let mut c = Expr::Bool(true);
            'latches: for &latch in info.latches.iter() {
                let lb = &self.sc.blocks[latch];
                if lb.end > lb.start {
                    if let Op::CondJump(_) = self.insns[lb.end - 1].op {
                        if lb.succ_taken == Some(header) {
                            c = self.branch_condition(latch);
                            break 'latches;
                        }
                        if lb.succ_fall == Some(header) {
                            c = invert(&self.branch_condition(latch));
                            break 'latches;
                        }
                    }
                    // Latch has no test of its own: scan the latch block for
                    // a Compare feeding a jump elsewhere, else walk in-body
                    // predecessors for their tail Compare/CondJump.
                    for i in lb.start..lb.end {
                        if let Op::Compare(k) = self.insns[i].op {
                            let a = self.ctx.operand_expr(
                                self.insns[i].operands.get(1).unwrap_or(&Operand::Unknown),
                                0,
                            );
                            let bb = self.ctx.operand_expr(
                                self.insns[i].operands.get(2).unwrap_or(&Operand::Unknown),
                                0,
                            );
                            c = Expr::Cmp(cmp_of(k), Box::new(a), Box::new(bb));
                            break 'latches;
                        }
                    }
                }
                // Predecessor walk: the test block usually falls through into
                // the latch. Take the nearest in-body predecessor with a
                // tail CondJump (or a Compare) as the loop condition source.
                let mut preds: Vec<usize> = Vec::new();
                for bi in info.body.iter() {
                    let bb = &self.sc.blocks[*bi];
                    if bb.succ_fall == Some(latch) || bb.succ_taken == Some(latch) {
                        preds.push(*bi);
                    }
                }
                preds.sort_unstable_by_key(|b| std::cmp::Reverse(self.sc.blocks[*b].start));
                for &p in preds.iter() {
                    let pb = &self.sc.blocks[p];
                    if p == header || pb.end <= pb.start {
                        continue;
                    }
                    if let Op::CondJump(_) = self.insns[pb.end - 1].op {
                        c = self.branch_condition(p);
                        break 'latches;
                    }
                    for i in (pb.start..pb.end).rev() {
                        if let Op::Compare(k) = self.insns[i].op {
                            let a = self.ctx.operand_expr(
                                self.insns[i].operands.get(1).unwrap_or(&Operand::Unknown),
                                0,
                            );
                            let bb = self.ctx.operand_expr(
                                self.insns[i].operands.get(2).unwrap_or(&Operand::Unknown),
                                0,
                            );
                            c = Expr::Cmp(cmp_of(k), Box::new(a), Box::new(bb));
                            break 'latches;
                        }
                    }
                }
            }
            let stmts = vec![Stmt::DoWhile {
                body: if empty_body {
                    vec![Stmt::Comment(
                        "spin wait (native self-loop; work precedes loop)".into(),
                    )]
                } else {
                    body
                },
                cond: c,
            }];
            (stmts, exit_blk)
        }
    }
}

/// A flat guard's fall body can move under an `if` only when doing so does
/// not make a declaration's scope conditional or make a landing label
/// conditional. No statement is reordered: the body executes exactly on the
/// short-circuit false path, and all its statements retain their order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlatGuardNestingRefusal {
    Label,
    Declaration,
    Depth,
}

fn flat_guard_replacement(
    cond: &Expr,
    body: &[Stmt],
    depth: usize,
    labels_are_internal: bool,
) -> Option<Stmt> {
    if body.is_empty()
        || flat_guard_body_nesting_refusal(body, depth, labels_are_internal).is_some()
    {
        return None;
    }
    Some(Stmt::If {
        cond: Expr::Not(Box::new(cond.clone())),
        then_body: body.to_vec(),
        else_body: Vec::new(),
    })
}

/// A label may move under this flat guard only if every CFG edge entering its
/// block comes from a block emitted into this same fall arm. The edge set is
/// taken from the complete CFG, so an outside branch, switch arm, or earlier
/// emitted path keeps the guard in goto form.
fn flat_guard_labels_are_internal(
    sc: &StructuredCfg,
    body: &[Stmt],
    fall_blocks: &HashSet<usize>,
    guard_header: usize,
) -> bool {
    let mut labels = HashSet::new();
    collect_all_labels(body, &mut labels);
    // Use the CFG's assigned names directly. Forced labels on blocks that are
    // not in `sc.labels` use `label_name`'s collision-safe auto name.
    let mut block_for_name: HashMap<String, usize> = sc
        .labels
        .iter()
        .map(|(&block, name)| (name.clone(), block))
        .collect();
    for block in 0..sc.blocks.len() {
        block_for_name.entry(label_name(sc, block)).or_insert(block);
    }
    labels.into_iter().all(|name| {
        let Some(&target) = block_for_name.get(&name) else {
            return false;
        };
        if !fall_blocks.contains(&target) {
            return false;
        }
        let mut has_predecessor = false;
        for (source, successors) in sc.full_succ.iter().enumerate() {
            if successors.contains(&target) {
                has_predecessor = true;
                if !fall_blocks.contains(&source) {
                    let is_guard_fall_edge = source == guard_header
                        && sc.blocks.get(guard_header).is_some_and(|guard| {
                            guard.succ_fall == Some(target)
                                && guard.succ_taken != Some(target)
                        });
                    if !is_guard_fall_edge {
                        return false;
                    }
                }
            }
        }
        has_predecessor
    })
}

fn flat_guard_body_nesting_refusal(
    stmts: &[Stmt],
    depth: usize,
    labels_are_internal: bool,
) -> Option<FlatGuardNestingRefusal> {
    fn inspect(
        stmts: &[Stmt],
        labels_are_internal: bool,
    ) -> Result<usize, FlatGuardNestingRefusal> {
        let mut max_child = 0usize;
        for s in stmts {
            let child_depth = match s {
                Stmt::Label(_) if !labels_are_internal => {
                    return Err(FlatGuardNestingRefusal::Label)
                }
                Stmt::Label(_) => 0,
                Stmt::Assign(_, Some(_), _) => {
                    return Err(FlatGuardNestingRefusal::Declaration)
                }
                Stmt::If { then_body, else_body, .. } => {
                    inspect(then_body, labels_are_internal)?
                        .max(inspect(else_body, labels_are_internal)?)
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                    inspect(body, labels_are_internal)?
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    let mut d = inspect(try_body, labels_are_internal)?;
                    for (_, body) in catches {
                        d = d.max(inspect(body, labels_are_internal)?);
                    }
                    if let Some(body) = finally_body {
                        d = d.max(inspect(body, labels_are_internal)?);
                    }
                    d
                }
                Stmt::Switch { cases, tail, .. } => {
                    let mut d = inspect(tail, labels_are_internal)?;
                    for (_, body) in cases {
                        d = d.max(inspect(body, labels_are_internal)?);
                    }
                    d
                }
                Stmt::For { init, incr, body, .. } => {
                    inspect(init, labels_are_internal)?
                        .max(inspect(incr, labels_are_internal)?)
                        .max(inspect(body, labels_are_internal)?)
                }
                Stmt::Using { decl, body } => inspect(std::slice::from_ref(decl), labels_are_internal)?
                    .max(inspect(body, labels_are_internal)?),
                Stmt::Lock { body, .. } | Stmt::ForEach { body, .. } => {
                    inspect(body, labels_are_internal)?
                }
                _ => 0,
            };
            let is_compound = matches!(
                s,
                Stmt::If { .. }
                    | Stmt::While { .. }
                    | Stmt::DoWhile { .. }
                    | Stmt::Try { .. }
                    | Stmt::Switch { .. }
                    | Stmt::For { .. }
                    | Stmt::Using { .. }
                    | Stmt::Lock { .. }
                    | Stmt::ForEach { .. }
            );
            max_child = max_child.max(child_depth + usize::from(is_compound));
        }
        Ok(max_child)
    }
    // The rewrite adds one C# block level. Keep it within the same structural
    // nesting ceiling used by the walker; deeply nested bodies stay flat.
    match inspect(stmts, labels_are_internal) {
        Err(refusal) => Some(refusal),
        Ok(nested) if depth + 1 + nested > MAX_NEST_DEPTH => {
            Some(FlatGuardNestingRefusal::Depth)
        }
        Ok(_) => None,
    }
}

/// Is `to` reachable from `from` in the structured CFG (bounded BFS)?
///
/// Used to recognise the compiler's `SKIP` pattern, which every null guard
/// compiles to:
///
/// ```text
///   if (<null test>) goto JOIN;      ; the branch's TAKEN edge
///   <body>;                          ; the fall arm
///   JOIN:                            ; == the taken target
/// ```
/// One arm's target is reachable *through* the other arm, so that target is
/// the join and the other arm is the only real body.
fn reachable_block(sc: &StructuredCfg, from: usize, to: usize) -> bool {
    if from == to {
        return true;
    }
    let n = sc.blocks.len();
    if from >= n || to >= n {
        return false;
    }
    let mut seen = vec![false; n];
    let mut stack = vec![from];
    let mut steps = 0usize;
    while let Some(x) = stack.pop() {
        if x >= n || seen[x] || steps > 512 {
            continue;
        }
        if x == to {
            return true;
        }
        seen[x] = true;
        steps += 1;
        for s in [sc.blocks[x].succ_taken, sc.blocks[x].succ_fall]
            .into_iter()
            .flatten()
        {
            if s < n {
                stack.push(s);
            }
        }
    }
    false
}

fn find_merge(
    sc: &StructuredCfg,
    hdr: usize,
    fall: Option<usize>,
    taken: Option<usize>,
) -> Option<usize> {
    // Exact merge point via immediate post-dominators: the merge of a branch
    // is the immediate post-dominator of the branch HEADER — the block that
    // owns the `CondJump`, not either arm. Using the FALL arm's ipdom instead
    // (the previous behaviour) is wrong whenever the fall arm's nearest
    // post-dominator is a block the TAKEN arm never reaches: on 0x1a8d884 the
    // inner `if (d16 != ±0.5)` merged at the fall arm's shared `fn_40c004`
    // block, and the taken arm silently fell into code it never executes.
    if hdr < sc.blocks.len() {
        if let Some(m) = sc.ipdom.get(hdr).copied().flatten() {
            if m < sc.blocks.len() {
                STRUCT_CENSUS
                    .merge_via_header
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(m);
            }
        }
    }
    // Fall back to the bounded reachability search only when ipdom is
    // unavailable (unreachable exit, degenerate graph). It stays asymmetric
    // on purpose: `fall` is the arm that is NOT the branch target, so "fall
    // reaches taken" IS the skip pattern (the target is a join reached
    // through the fall-through).
    let (Some(a), Some(b)) = (fall, taken) else {
        STRUCT_CENSUS
            .merge_fail_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    const LIMIT: usize = 128;
    let reach_from = |start: usize| -> Vec<usize> {
        let mut seen = vec![false; sc.blocks.len()];
        let mut order = Vec::new();
        let mut stack = vec![start];
        let mut steps = 0;
        while let Some(x) = stack.pop() {
            if x >= sc.blocks.len() || seen[x] || steps > LIMIT {
                continue;
            }
            seen[x] = true;
            order.push(x);
            steps += 1;
            let blk = &sc.blocks[x];
            for s in [blk.succ_taken, blk.succ_fall].into_iter().flatten() {
                stack.push(s);
            }
        }
        order
    };
    let ra = reach_from(a);
    let rb_set: HashSet<usize> = reach_from(b).into_iter().collect();
    // Skip pattern (see `reachable_block`): the branch target is reachable
    // THROUGH the other arm, so it IS the join. This must be decided before
    // the reachability fallback below, whose `x != a && x != b` filter
    // excluded the arm heads themselves — so it picked the block *after* the
    // join and the structurer then emitted the entire remainder of the method
    // as the taken arm. That is what nested every null-guard body (and the
    // whole rest of `OnPrepare`) inside `if (_cityLoader == 0)`.
    if ra.contains(&b) {
        return Some(b);
    }
    if rb_set.contains(&a) {
        return Some(a);
    }
    let m = ra
        .into_iter()
        .find(|x| x != &a && x != &b && rb_set.contains(x));
    if m.is_none() {
        STRUCT_CENSUS
            .merge_fail_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    m
}

fn st_name(li: usize) -> String {
    format!("v{li}")
}

/// Remove `goto L…;` statements whose label is never defined anywhere in the
/// rendered body (the emission guard can reference blocks that end up
/// elided). Also drops now-empty trailing constructs.
fn drop_unreachable_gotos(stmts: &mut Vec<Stmt>) {
    let mut labels: HashSet<String> = HashSet::new();
    fn collect(stmts: &[Stmt], labels: &mut HashSet<String>) {
        // RECURSE: labels defined inside nested bodies (an if's else-arm, a
        // loop body) are still valid goto targets. The old flat scan only
        // saw the top list, so a `goto L1` pointing at a label inside an
        // if-arm was wrongly stripped and its merge disappeared.
        for s in stmts {
            if let Stmt::Label(l) = s {
                labels.insert(l.clone());
            }
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    collect(then_body, labels);
                    collect(else_body, labels);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect(body, labels),
                Stmt::Try { try_body, catches, finally_body } => {
                    collect(try_body, labels);
                    for (_, b) in catches {
                        collect(b, labels);
                    }
                    if let Some(f) = finally_body {
                        collect(f, labels);
                    }
                }
                _ => {}
            }
        }
    }
    collect(stmts, &mut labels);
    fn strip(stmts: &mut Vec<Stmt>, labels: &HashSet<String>) {
        // ANNOTATE, never delete. `goto L;` whose target the walker never
        // rendered is a real native edge (the emission guard can stop at a
        // block the arms elided), and deleting the statement also deleted the
        // GUARD around it — `if (c) { goto L; }` became an empty `if (c) {}`
        // shell that the sanitizer then dropped, taking a real native
        // condition with it (933 of them on the 4,000-method sample, and the
        // single largest source of `lost-condition` methods). Keeping the text
        // "no dangling goto" invariant is `strip_undefined_gotos`' job: it
        // rewrites exactly this shape into a comment in the rendered text.
        for s in stmts.iter_mut() {
            if let Stmt::Goto(l) = s {
                if !labels.contains(l.as_str()) {
                    *s = Stmt::Comment(format!("goto {l}; (target not emitted)"));
                }
            }
        }
        for s in stmts.iter_mut() {
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    strip(then_body, labels);
                    strip(else_body, labels);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => strip(body, labels),
                // Gotos inside catch/finally bodies are ABI rethrow/cleanup
                // edges to blocks emitted elsewhere — without recursing here
                // they survive every sweep and render as dangling labels.
                Stmt::Try { try_body, catches, finally_body } => {
                    strip(try_body, labels);
                    for (_, b) in catches {
                        strip(b, labels);
                    }
                    if let Some(f) = finally_body {
                        strip(f, labels);
                    }
                }
                _ => {}
            }
        }
    }
    strip(stmts, &labels);
}

/// Final invariant: no `goto Lx;` in the finished text may reference a label the
/// text never defines.
///
/// `drop_unreachable_gotos` sweeps the statement TREE, and it is not enough: the
/// renderer itself discards arms that carry nothing (`if (c) {}` shells) and
/// folds constant conditions, so a label can be present for the tree sweep and
/// still never reach the text. That is how `GetMultipliedType` @0x1F0114 kept
/// printing `goto L3;` with no `L3:` anywhere. Sweeping the finished text makes
/// the invariant hold for every render path, including ones added later.
///
/// The statement is rewritten to a comment rather than dropped so the reader
/// still sees that control flowed somewhere the text could not name — the same
/// rule the rest of the recovery follows: never silently lose an observation.
pub(crate) fn strip_undefined_gotos(out: &mut String) {
    if !out.contains("goto L") {
        return;
    }
    let mut defined: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in out.lines() {
        let t = line.trim();
        if let Some(l) = t.strip_suffix(':') {
            if l.starts_with('L') && l[1..].bytes().all(|b| b.is_ascii_digit()) && l.len() > 1 {
                defined.insert(l.to_string());
            }
        }
    }
    /// `Lx` if `t` is exactly a label token, else `None`.
    fn label_tok(t: &str) -> Option<&str> {
        Some(t.strip_suffix(';')?)
            .filter(|l| l.starts_with('L') && l.len() > 1 && l[1..].bytes().all(|b| b.is_ascii_digit()))
    }
    let mut result = String::with_capacity(out.len());
    for raw in out.split_inclusive('\n') {
        let body = raw.trim_end_matches(['\n', '\r']);
        let lead = &body[..body.len() - body.trim_start().len()];
        let t = body.trim();
        // `goto Lx;` plus the shapes a recovered `switch` renders,
        // `case 3: goto Lx;` and `default: goto Lx;`.
        let (selector, missing) = match t.strip_prefix("goto ") {
            Some(rest) => ("", label_tok(rest)),
            None => match t.split_once(": goto ") {
                Some((head, rest)) => (head, label_tok(rest)),
                None => ("", None),
            },
        };
        if let Some(l) = missing {
            if !defined.contains(l) {
                result.push_str(&format!(
                    "{lead}{selector}{} // goto {l}; (target not emitted)\n",
                    if selector.is_empty() { "" } else { ":" }
                ));
                continue;
            }
        }
        result.push_str(raw);
    }
    *out = result;
}

/// Sweep the finished text: a `v N = …;` binding that the text never reads by
/// NAME is not a binding.
///
/// The value was inlined at every use, so the statement is a DUPLICATE of the
/// expression the reader already sees on the next line, and the reader has to
/// skip it to find the code:
///
/// ```text
/// v17 = *(nint*)(v0 + v11 * 8 +32);
/// v18 = *(nint*)(*(nint*)(v15 +184) +128);
/// v19 = *(nint*)(*(nint*)(v0 + v11 * 8 +32) + *(nint*)(*(nint*)(v15 +184) +128));
/// ```
///
/// Measured: 16,834 of 273,016 bindings in recovered bodies on the v24.5 PE
/// dump (6.2%), 11,923 of 130,728 on the Android corpus (9.1%).
///
/// This is the TEXT-level counterpart of the tree-level dead-store pass, and it
/// is needed for the same reason [`strip_undefined_gotos`] is: whether a use
/// prints the NAME or the inlined expression is a RENDER-time decision (the
/// use-site-aware inline path resolves the reaching def), while the tree sees
/// an `Expr::Var` read in both cases and therefore must keep the binding. The
/// text is the only place where "is this name ever mentioned" is answerable.
///
/// A binding whose right-hand side contains a CALL is rewritten to a bare
/// expression instead of being deleted — the callee still runs.
///
/// The order the statements are emitted in is not execution order, so a
/// binding is kept whenever the name appears ANYWHERE else in the body
/// (including a loop condition or an earlier line reached from a `goto`). Only
/// a name that appears nowhere else at all is unreachable, which is what makes
/// this an invariant rather than a heuristic.
pub(crate) fn strip_unread_bindings(out: &mut String) {
    if !out.contains(" = ") {
        return;
    }
    /// The `vN` of a `vN = …;` binding line, else `None`.
    fn binding_target(t: &str) -> Option<&str> {
        let (name, _) = t.split_once(" = ")?;
        if name.len() > 1 && name.starts_with('v') && name[1..].bytes().all(|b| b.is_ascii_digit()) {
            Some(name)
        } else {
            None
        }
    }
    /// Every `vN` token in `t`.
    fn tokens(t: &str) -> Vec<&str> {
        let bytes = t.as_bytes();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < bytes.len() {
            if bytes[i] == b'v'
                && (i == 0 || !bytes[i - 1].is_ascii_alphanumeric())
                && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit())
            {
                let mut j = i + 1;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if !bytes.get(j).is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_') {
                    out.push(&t[i..j]);
                    i = j;
                    continue;
                }
            }
            i += 1;
        }
        out
    }

    // Reads per name. The TARGET of a binding is a def and never counts, or a
    // name bound twice with no reader would look read by its own other binding.
    let lines: Vec<&str> = out.split_inclusive('\n').collect();
    let bindings: Vec<Option<&str>> = lines
        .iter()
        .map(|l| binding_target(l.trim()))
        .collect();
    let mut reads: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, line) in lines.iter().enumerate() {
        // Only the RIGHT-HAND SIDE counts on a binding line: the target is a
        // def, not a read. Everything after the first ` = ` is kept — including
        // the binding's own name (`v31 = v31 - 1` is loop-carried and the only
        // read that update will ever have) and, just as important, every OTHER
        // name it reads. Counting only the self-read made `v1 = v2` fail to
        // register `v2` as read, and the sweep then deleted the `v2 = …` that
        // defined it.
        let rhs = if bindings[i].is_some() {
            line.trim().split_once(" = ").map(|(_, r)| r).unwrap_or("")
        } else {
            line
        };
        for t in tokens(rhs) {
            *reads.entry(t).or_insert(0) += 1;
        }
    }
    // NOTE: no `reads.is_empty()` early return here. An empty read set is the
    // STRONGEST case this pass applies to (every binding in the body is
    // unused), not a reason to skip work — returning early there left exactly
    // the bodies whose bindings are all orphans untouched.
    let mut result = String::with_capacity(out.len());
    for (i, raw) in lines.iter().enumerate() {
        let Some(name) = bindings[i] else {
            result.push_str(raw);
            continue;
        };
        if reads.get(name).copied().unwrap_or(0) > 0 {
            result.push_str(raw);
            continue;
        }
        let body = raw.trim_end_matches(['\n', '\r']);
        let lead = &body[..body.len() - body.trim_start().len()];
        let rhs = body.trim().split_once(" = ").map(|(_, r)| r).unwrap_or("");
        // keep the EFFECT of a call, drop only the unused binding
        if rhs.contains('(') {
            result.push_str(&format!("{lead}{rhs}\n"));
        }
    }
    *out = result;
}

/// Final sanitize pass:
/// 1. `this = <expr>` is invalid C# (receiver register clobbered by the call
///    protocol) — keep the side effect, drop the assignment target.
/// 2. Empty `while`/`do..while` bodies are mis-recovered loop exits — they
///    carry no information and read as infinite loops, so drop them.
/// 3. Self-assignments `x = x;` are register-coalescing artifacts.
/// 4. Label-only `if` arms (empty then/else with no side effects) are branch
///    scheduling noise; the condition never gates anything observable.
/// 5. `__unknown..ctor(__unknown)` pairs are artifact ctor calls with zero
///    recovered operands — indistinguishable from noise, never real code.
fn sanitize_stmts(stmts: &mut Vec<Stmt>) {
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::Assign(target, _, rhs) if target == "this" => {
                if matches!(rhs, Expr::Call { .. } | Expr::RawCall(..)) {
                    *stmt = Stmt::Eval(rhs.clone());
                } else {
                    *stmt = Stmt::Comment("receiver updated by call".into());
                }
            }
            // x = x — register coalescing artifact. The assignment carries no
            // information (the register was already that value), so the
            // statement is DROPPED rather than left as a `// (coalesced)`
            // marker: on a large binary that marker alone appeared 1,603 times
            // and told the reader nothing the surrounding code did not.
            Stmt::Assign(target, _, Expr::Var(src)) if target == src => {
                *stmt = Stmt::Comment("\u{0}removed".into());
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. }
                if body.is_empty() =>
            {
                // Keep the loop itself (its condition may be meaningful) but
                // mark the empty body instead of rendering a bare infinite
                // loop that looks like a decompiler failure.
                body.push(Stmt::Comment("loop body not recovered".into()));
            }
            _ => {}
        }
        // Structural cleanup that removes the statement:
        // - If with both arms empty/label-only/noise → drop the whole branch
        //   (keep a side-effecting condition as a bare Eval).
        //   SAFETY RULE: only when nothing follows in this list — a later
        //   `goto L…` may target a label INSIDE this branch (labels are
        //   "noise" by text, but deleting the branch would delete the label
        //   definition and every remaining goto would dangle). When later
        //   statements exist, keep the shell.
        // - Artifact ctor calls with no recoverable operands.
        match stmt {
            Stmt::If { cond, then_body, else_body }
                if then_body.iter().all(is_noise_stmt)
                    && else_body.iter().all(is_noise_stmt)
                    // A label inside the branch can be a goto target from
                    // elsewhere in the method — dropping the branch would
                    // orphan that goto. Only label-free shells fold.
                    && !then_body.iter().any(|s| matches!(s, Stmt::Label(_)))
                    && !else_body.iter().any(|s| matches!(s, Stmt::Label(_)))
                    // A load-bearing condition (null-check chain: the call
                    // was inlined into the compare) is REAL recovered logic
                    // — keep the shell, don't fold it away. UNLESS both arms
                    // are noise-only (empty, comments, coalescing markers):
                    // the branch gates nothing visible, so the shell itself
                    // is scheduling noise regardless of the condition's
                    // origin. (`all(is_noise_stmt)` not `is_empty()`: the
                    // cleanup stages leave Comment markers behind that the
                    // final sweep drops — an empty-render shell must still
                    // fold.)
                    && ((!expr_has_load(cond) && !expr_has_call(cond))
                        || (then_body.iter().all(is_noise_stmt)
                            && else_body.iter().all(is_noise_stmt))) =>
            {
                *stmt = Stmt::Comment("\u{0}removed".into());
            }
            Stmt::Eval(e) if is_artifact_ctor(e) => {
                *stmt = Stmt::Comment("\u{0}removed".into());
            }
            _ => {}
        }
        // else-arm with no payload after a terminating then-arm → drop it
        // (`return x; else {}` — the else is unreachable scheduling shape).
        // Applied after the whole-branch fold above, before recursion.
        if let Stmt::If { then_body, else_body, .. } = stmt {
            let then_terminates = then_body
                .last()
                .is_some_and(|s| matches!(s, Stmt::Return(_) | Stmt::Throw(_)));
            if then_terminates
                && !else_body.is_empty()
                && else_body.iter().all(is_noise_stmt)
                && !else_body.iter().any(|s| matches!(s, Stmt::Label(_)))
            {
                else_body.clear();
            }
        }
        match stmt {
            Stmt::If { then_body, else_body, .. } => {
                sanitize_stmts(then_body);
                sanitize_stmts(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => sanitize_stmts(body),
            _ => {}
        }
    }
    // second sweep: drop removed markers
    stmts.retain(|s| !matches!(s, Stmt::Comment(c) if c.starts_with('\u{0}')));
}

/// Final structural repair (Ghidra's last structuring step):
/// 1. `if (c) {} else { B }` → `if (!c) { B }` — an empty then-arm is
///    branch-scheduling shape, not source shape. The source was an inverted
///    test (or `if (!c) B`), so flip the predicate and keep the payload.
/// 2. `if (c) { … goto L; }` where L labels the statement AFTER the if —
///    that's just the natural fall-through rendered as a jump; delete it and
///    the if keeps flowing into its merge like real code.
/// 3. Labels that no remaining goto references → remove (they only existed
///    as merge anchors for gotos that fell through naturally).
/// 4. `goto L;` as the LAST statement of a body where L labels the block
///    that follows the whole construct — the merge-edge got duplicated by
///    both the walk and the structured tail; keep one path.
pub(crate) fn branch_repair(stmts: &mut Vec<Stmt>) {
    // A switch arm is a C# control-flow region. Convert only jumps from one of
    // its case bodies to another case body's entry label; labels elsewhere
    // (including the switch tail and exception clauses) remain ordinary
    // method-level gotos. This is inert unless both endpoints are proven to
    // belong to the same recovered switch.
    lower_switch_case_gotos(stmts);
    // Label/goto resolution must be GLOBAL across the whole body. The
    // structurer routinely emits a `goto Lx` inside one arm and `Lx:` in a
    // completely different (often shallower) list — the merge block of an
    // outer `if`. The old per-list recursion judged a goto dangling against
    // the labels of its OWN nested list, so every cross-level jump was
    // deleted at the innermost level, and with it the enclosing branch that
    // carried it: the whole value chain feeding that branch vanished
    // (the dropped chain in GetEffectiveSchemaType @0x1f4e79c).
    // The same global view decides the guard fold below: pulling a region
    // that holds interior labels under `!c` is sound only when no edge from
    // OUTSIDE that region targets them, and the fold sees one list at a time
    // while the edge may live in any other. Counted once up front, over every
    // `Stmt` variant, before any pass moves or removes a jump.
    let mut global_gotos: HashMap<String, usize> = HashMap::new();
    count_all_gotos(stmts, &mut global_gotos);
    // Examination dedup for the guard-chain fold census below: `branch_repair_local`
    // recurses into `if`/`while` bodies AND `fold_branch_over_join` recurses into
    // them again, so one guard can be examined once per ancestor level. The fold
    // itself is idempotent (a folded guard no longer matches the shape), but the
    // census must count it once — keyed by statement address, never dereferenced.
    let mut fold_seen: HashSet<*const Stmt> = HashSet::new();
    branch_repair_local(stmts, &global_gotos, &mut fold_seen);
    fold_shared_if_suffixes(stmts);
    fold_gotos_to_void_return(stmts);
    fold_gotos_to_break(stmts);
    let mut labels: HashSet<String> = HashSet::new();
    collect_labels(stmts, &mut labels);
    strip_dangling(stmts, &labels);
    let mut referenced: HashSet<String> = HashSet::new();
    collect_gotos(stmts, &mut referenced);
    strip_quiet_labels(stmts, &referenced);
    // Final: drop the removal markers.
    stmts.retain(|s| !matches!(s, Stmt::Comment(c) if c.starts_with('\u{0}')));
}

/// Hoist a label's straight-line suffix out of an `if` when the opposite arm
/// ends by jumping to that label. Keep the label at the new continuation so
/// any other incoming gotos still land at the same code. Neither arm may
/// declare a local, which keeps C# scope intact. Only
/// expression/assignment/return/throw/comment suffix statements are accepted;
/// nested scopes and further control transfers are deliberately refused.
fn fold_shared_if_suffixes(stmts: &mut Vec<Stmt>) {
    fn has_declaration(stmts: &[Stmt]) -> bool {
        stmts.iter().any(|stmt| match stmt {
            Stmt::Assign(_, ty, _) => ty.is_some(),
            Stmt::If { then_body, else_body, .. } => {
                has_declaration(then_body) || has_declaration(else_body)
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::Lock { body, .. } => has_declaration(body),
            Stmt::For { init, incr, body, .. } => {
                has_declaration(init) || has_declaration(incr) || has_declaration(body)
            }
            Stmt::Using { decl, body } => {
                has_declaration(std::slice::from_ref(decl)) || has_declaration(body)
            }
            Stmt::Try { try_body, catches, finally_body } => {
                has_declaration(try_body)
                    || catches.iter().any(|(_, body)| has_declaration(body))
                    || finally_body.as_ref().is_some_and(|body| has_declaration(body))
            }
            Stmt::Switch { cases, tail, .. } => {
                cases.iter().any(|(_, body)| has_declaration(body)) || has_declaration(tail)
            }
            _ => false,
        })
    }

    fn safe_suffix(stmts: &[Stmt]) -> bool {
        stmts.iter().all(|stmt| match stmt {
            Stmt::Eval(_) | Stmt::Return(_) | Stmt::Throw(_) | Stmt::Comment(_) => true,
            Stmt::Assign(_, ty, _) => ty.is_none(),
            _ => false,
        })
    }

    fn unique_label_suffix(arm: &[Stmt], target: &str) -> Option<(usize, Vec<Stmt>)> {
        let positions: Vec<_> = arm
            .iter()
            .enumerate()
            .filter_map(|(i, stmt)| matches!(stmt, Stmt::Label(label) if label == target).then_some(i))
            .collect();
        if positions.len() != 1 {
            return None;
        }
        let label_index = positions[0];
        let suffix = &arm[label_index + 1..];
        safe_suffix(suffix).then(|| (label_index, suffix.to_vec()))
    }

    fn rewrite_list(stmts: &mut Vec<Stmt>, refs: &HashMap<String, usize>) {
        // Find inner candidates first so a successful inner rewrite does not
        // invalidate any of the outer arm slices being examined.
        for stmt in stmts.iter_mut() {
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    rewrite_list(then_body, refs);
                    rewrite_list(else_body, refs);
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. }
                | Stmt::Lock { body, .. } => rewrite_list(body, refs),
                Stmt::For { init, incr, body, .. } => {
                    rewrite_list(init, refs);
                    rewrite_list(incr, refs);
                    rewrite_list(body, refs);
                }
                Stmt::Using { decl, body } => {
                    let _ = decl;
                    rewrite_list(body, refs);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    rewrite_list(try_body, refs);
                    for (_, body) in catches {
                        rewrite_list(body, refs);
                    }
                    if let Some(body) = finally_body {
                        rewrite_list(body, refs);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, body) in cases {
                        rewrite_list(body, refs);
                    }
                    rewrite_list(tail, refs);
                }
                _ => {}
            }
        }

        let mut i = 0;
        while i < stmts.len() {
            let candidate = match &stmts[i] {
                Stmt::If { then_body, else_body, .. } => {
                    STRUCT_CENSUS.shared_if_suffix_attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if has_declaration(then_body) || has_declaration(else_body) {
                        STRUCT_CENSUS.shared_if_suffix_refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        None
                    } else {
                        // The opposite arm must end in the sole goto to a label
                        // in the target arm. Comments after the goto are inert.
                        let inspect = |target_arm: &[Stmt], goto_arm: &[Stmt]| {
                            let (goto_index, label) = goto_arm.iter().enumerate().rev()
                                .find_map(|(n, stmt)| match stmt {
                                    Stmt::Comment(_) => None,
                                    Stmt::Goto(label) => Some((n, label.clone())),
                                    _ => Some((usize::MAX, String::new())),
                                })?;
                            if goto_index == usize::MAX || refs.get(&label).copied().unwrap_or(0) == 0 {
                                return None;
                            }
                            let (label_index, suffix) = unique_label_suffix(target_arm, &label)?;
                            Some((label, label_index, goto_index, suffix))
                        };
                        if let Some((label, label_index, goto_index, suffix)) = inspect(then_body, else_body) {
                            Some((true, label, label_index, goto_index, suffix))
                        } else if let Some((label, label_index, goto_index, suffix)) = inspect(else_body, then_body) {
                            Some((false, label, label_index, goto_index, suffix))
                        } else {
                            STRUCT_CENSUS.shared_if_suffix_refused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                    }
                }
                _ => None,
            };
            if let Some((in_then, label, label_index, goto_index, suffix)) = candidate {
                let Stmt::If { then_body, else_body, .. } = &mut stmts[i] else { unreachable!() };
                let (target, source) = if in_then { (then_body, else_body) } else { (else_body, then_body) };
                target.truncate(label_index);
                source.remove(goto_index);
                // Preserve comments that followed the removed goto in place;
                // they are metadata only and stay associated with that arm.
                let mut continuation = Vec::with_capacity(suffix.len() + 1);
                continuation.push(Stmt::Label(label));
                continuation.extend(suffix);
                stmts.splice(i + 1..i + 1, continuation);
                STRUCT_CENSUS.shared_if_suffix_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    let mut refs = HashMap::new();
    count_all_gotos(stmts, &mut refs);
    rewrite_list(stmts, &refs);
}

/// Replace a jump to a label immediately followed by `return;` with that
/// return. The target executes no intervening statement and reads no local,
/// so moving the terminator to the jump site preserves the selected path and
/// avoids a method-level jump. Keep the label and original return for any
/// fallthrough or remaining references. A return is forbidden in a finally
/// clause, so labels and jump sources inside finally bodies are refused.
fn fold_gotos_to_void_return(stmts: &mut Vec<Stmt>) {
    fn collect(stmts: &[Stmt], in_finally: bool, labels: &mut HashSet<String>) {
        for (i, stmt) in stmts.iter().enumerate() {
            if !in_finally {
                if let Stmt::Label(label) = stmt {
                    if stmts[i + 1..]
                        .iter()
                        .find(|next| !matches!(next, Stmt::Comment(_)))
                        .is_some_and(|next| matches!(next, Stmt::Return(None)))
                    {
                        labels.insert(label.clone());
                    }
                }
            }
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    collect(then_body, in_finally, labels);
                    collect(else_body, in_finally, labels);
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. }
                | Stmt::Lock { body, .. } => collect(body, in_finally, labels),
                Stmt::For { init, incr, body, .. } => {
                    collect(init, in_finally, labels);
                    collect(incr, in_finally, labels);
                    collect(body, in_finally, labels);
                }
                Stmt::Using { decl, body } => {
                    collect(std::slice::from_ref(decl), in_finally, labels);
                    collect(body, in_finally, labels);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    collect(try_body, in_finally, labels);
                    for (_, body) in catches {
                        collect(body, in_finally, labels);
                    }
                    if let Some(body) = finally_body {
                        collect(body, true, labels);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, body) in cases {
                        collect(body, in_finally, labels);
                    }
                    collect(tail, in_finally, labels);
                }
                _ => {}
            }
        }
    }

    fn rewrite(stmts: &mut [Stmt], in_finally: bool, labels: &HashSet<String>) {
        for stmt in stmts {
            if !in_finally {
                if let Stmt::Goto(label) = stmt {
                    if labels.contains(label) {
                        *stmt = Stmt::Return(None);
                        STRUCT_CENSUS
                            .goto_return_folds
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        continue;
                    }
                }
            }
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    rewrite(then_body, in_finally, labels);
                    rewrite(else_body, in_finally, labels);
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. }
                | Stmt::Lock { body, .. } => rewrite(body, in_finally, labels),
                Stmt::For { init, incr, body, .. } => {
                    rewrite(init, in_finally, labels);
                    rewrite(incr, in_finally, labels);
                    rewrite(body, in_finally, labels);
                }
                Stmt::Using { decl, body } => {
                    rewrite(std::slice::from_mut(decl), in_finally, labels);
                    rewrite(body, in_finally, labels);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    rewrite(try_body, in_finally, labels);
                    for (_, body) in catches {
                        rewrite(body, in_finally, labels);
                    }
                    if let Some(body) = finally_body {
                        rewrite(body, true, labels);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, body) in cases {
                        rewrite(body, in_finally, labels);
                    }
                    rewrite(tail, in_finally, labels);
                }
                _ => {}
            }
        }
    }

    let mut labels = HashSet::new();
    collect(stmts, false, &mut labels);
    if !labels.is_empty() {
        rewrite(stmts, false, &labels);
    }
}

/// Replace a goto with `break;` only when its target is the immediate
/// continuation after the nearest enclosing loop or switch. Every breakable
/// construct receives a scope ID, so a nested loop or switch blocks breaking
/// an outer construct. Sources in finally bodies are handled only when the
/// target construct is within that exact same finally body.
fn fold_gotos_to_break(stmts: &mut Vec<Stmt>) {
    type ScopeId = usize;
    type FinallyId = usize;

    fn continuation_label(stmts: &[Stmt], index: usize) -> Option<String> {
        stmts[index + 1..]
            .iter()
            .find(|stmt| !matches!(stmt, Stmt::Comment(_)))
            .and_then(|stmt| match stmt {
                Stmt::Label(label) => Some(label.clone()),
                _ => None,
            })
    }

    fn collect(
        stmts: &[Stmt],
        finally_id: FinallyId,
        next_scope: &mut ScopeId,
        next_finally: &mut FinallyId,
        scopes: &mut HashMap<*const Stmt, (ScopeId, FinallyId)>,
        try_finally: &mut HashMap<*const Stmt, FinallyId>,
        targets: &mut HashMap<String, Option<(ScopeId, FinallyId)>>,
    ) {
        for (index, stmt) in stmts.iter().enumerate() {
            let ptr = stmt as *const Stmt;
            let is_breakable = matches!(
                stmt,
                Stmt::While { .. }
                    | Stmt::DoWhile { .. }
                    | Stmt::For { .. }
                    | Stmt::ForEach { .. }
                    | Stmt::Switch { .. }
            );
            if is_breakable {
                let scope = *next_scope;
                *next_scope += 1;
                scopes.insert(ptr, (scope, finally_id));
                if let Some(label) = continuation_label(stmts, index) {
                    let candidate = (scope, finally_id);
                    match targets.entry(label) {
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(Some(candidate));
                        }
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            if entry.get() != &Some(candidate) {
                                entry.insert(None);
                            }
                        }
                    }
                }
            }

            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    collect(then_body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    collect(else_body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. } => {
                    collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                Stmt::For { init, incr, body, .. } => {
                    collect(init, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    collect(incr, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                Stmt::Lock { body, .. } => {
                    collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                Stmt::Using { decl, body } => {
                    collect(std::slice::from_ref(decl), finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    collect(try_body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    for (_, body) in catches {
                        collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    }
                    if let Some(body) = finally_body {
                        let id = *next_finally;
                        *next_finally += 1;
                        try_finally.insert(ptr, id);
                        collect(body, id, next_scope, next_finally, scopes, try_finally, targets);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, body) in cases {
                        collect(body, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                    }
                    // The tail is rendered after the switch, outside its break scope.
                    collect(tail, finally_id, next_scope, next_finally, scopes, try_finally, targets);
                }
                _ => {}
            }
        }
    }

    fn rewrite(
        stmts: &mut [Stmt],
        nearest_scope: Option<(ScopeId, FinallyId)>,
        finally_id: FinallyId,
        scopes: &HashMap<*const Stmt, (ScopeId, FinallyId)>,
        try_finally: &HashMap<*const Stmt, FinallyId>,
        targets: &HashMap<String, Option<(ScopeId, FinallyId)>>,
    ) {
        for stmt in stmts {
            if let Stmt::Goto(label) = stmt {
                if nearest_scope.is_some()
                    && nearest_scope == targets.get(label).copied().flatten()
                    && nearest_scope.is_some_and(|(_, scope_finally)| scope_finally == finally_id)
                {
                    *stmt = Stmt::Break;
                    STRUCT_CENSUS.goto_break_folds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
            }
            let ptr = stmt as *const Stmt;
            let own_scope = scopes.get(&ptr).copied();
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    rewrite(then_body, nearest_scope, finally_id, scopes, try_finally, targets);
                    rewrite(else_body, nearest_scope, finally_id, scopes, try_finally, targets);
                }
                Stmt::While { body, .. }
                | Stmt::DoWhile { body, .. }
                | Stmt::ForEach { body, .. } => {
                    rewrite(body, own_scope, finally_id, scopes, try_finally, targets);
                }
                Stmt::For { init, incr, body, .. } => {
                    rewrite(init, nearest_scope, finally_id, scopes, try_finally, targets);
                    rewrite(incr, nearest_scope, finally_id, scopes, try_finally, targets);
                    rewrite(body, own_scope, finally_id, scopes, try_finally, targets);
                }
                Stmt::Lock { body, .. } => {
                    rewrite(body, nearest_scope, finally_id, scopes, try_finally, targets);
                }
                Stmt::Using { decl, body } => {
                    rewrite(std::slice::from_mut(decl), nearest_scope, finally_id, scopes, try_finally, targets);
                    rewrite(body, nearest_scope, finally_id, scopes, try_finally, targets);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    rewrite(try_body, nearest_scope, finally_id, scopes, try_finally, targets);
                    for (_, body) in catches {
                        rewrite(body, nearest_scope, finally_id, scopes, try_finally, targets);
                    }
                    if let Some(body) = finally_body {
                        let nested_id = try_finally.get(&ptr).copied().unwrap_or(finally_id);
                        rewrite(body, nearest_scope, nested_id, scopes, try_finally, targets);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, body) in cases {
                        rewrite(body, own_scope, finally_id, scopes, try_finally, targets);
                    }
                    rewrite(tail, nearest_scope, finally_id, scopes, try_finally, targets);
                }
                _ => {}
            }
        }
    }

    let mut next_scope = 1;
    let mut next_finally = 1;
    let mut scopes = HashMap::new();
    let mut try_finally = HashMap::new();
    let mut targets = HashMap::new();
    collect(stmts, 0, &mut next_scope, &mut next_finally, &mut scopes, &mut try_finally, &mut targets);
    rewrite(stmts, None, 0, &scopes, &try_finally, &targets);
}

/// Recursive, per-list half of `branch_repair`: Pass A (invert empty-then
/// ifs) and Pass C (fold gotos whose label is the very next statement).
fn branch_repair_local(
    stmts: &mut Vec<Stmt>,
    global_gotos: &HashMap<String, usize>,
    fold_seen: &mut HashSet<*const Stmt>,
) {
    // Bottom-up so inner constructs settle before outer ones decide which
    // gotos are "last in arm".
    for stmt in stmts.iter_mut() {
        match stmt {
            Stmt::If { then_body, else_body, .. } => {
                branch_repair_local(then_body, global_gotos, fold_seen);
                branch_repair_local(else_body, global_gotos, fold_seen);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                branch_repair_local(body, global_gotos, fold_seen)
            }
            _ => {}
        }
    }
    // Pass A: invert empty-then ifs (repeat while the head keeps flipping —
    // a doubly-inverted pair `if(c){}else{ if(c2){}else{B} }` needs 2 rounds).
    loop {
        let mut flipped = false;
        for stmt in stmts.iter_mut() {
            if let Stmt::If { cond, then_body, else_body } = stmt {
                let then_empty = then_body.iter().all(is_noise_stmt);
                let else_has_payload = !else_body.is_empty() && !else_body.iter().all(is_noise_stmt);
                if then_empty && else_has_payload {
                    *cond = invert(cond);
                    std::mem::swap(then_body, else_body);
                    // the old then-arm (noise) is now the else-arm: an empty
                    // shell after a terminating payload arm is unreachable —
                    // drop it so the output doesn't read `else { }`.
                    if then_body.last().is_some_and(|s| matches!(s, Stmt::Return(_) | Stmt::Throw(_))) {
                        else_body.clear();
                    }
                    flipped = true;
                }
            }
        }
        if !flipped {
            break;
        }
    }
    // Pass C: a goto to a label defined LATER IN THE SAME LIST at the exact
    // next position is a fall-through, not a jump — but only when nothing
    // between the goto and its label can loop back (which the emission guard
    // prevents by construction, so position adjacency is safe here).
    fn drop_redundant_tails(stmts: &mut Vec<Stmt>) {
        for i in 0..stmts.len() {
            if let Stmt::Goto(l) = &stmts[i] {
                // The label must be the very next meaningful statement.
                let next_is_label = stmts[i + 1..]
                    .iter()
                    .find(|s| !matches!(s, Stmt::Comment(_)))
                    .is_some_and(|s| matches!(s, Stmt::Label(n) if n == l));
                if next_is_label {
                    stmts[i] = Stmt::Comment("\u{0}rm-goto".into());
                }
            }
        }
        for s in stmts.iter_mut() {
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    drop_redundant_tails(then_body);
                    drop_redundant_tails(else_body);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => drop_redundant_tails(body),
                _ => {}
            }
        }
    }
    drop_redundant_tails(stmts);
    drop_vacuous_if_gotos(stmts);
    fold_branch_over_join(stmts, global_gotos, fold_seen);
}

/// `if (c) { goto L; } <body> L:` → `if (!c) { <body> }` with `L:` kept.
///
/// This is the STRUCTURER half of the absorbed-target repair. When
/// `cfg::resolve_target` restores a forward edge that lands on an instruction
/// the lifter absorbed (`cmp`/`test` folded into the branch that consumes its
/// flags), the walk has a conditional edge to a block further down the SAME
/// statement list and only one shape to render it in: an `if` whose arm is a
/// bare `goto` to the merge label. That shape reads as broken control flow,
/// but the statements between the `goto` and the label are exactly the
/// fall-through region a guard protects, so the branch inverts and absorbs:
///
///   * the taken path — skip the region, resume at `L` — is what the new
///     `if (!c)` does, with the region as its body;
///   * the fall-through path still runs the region, now guarded;
///   * `L:` STAYS, so every OTHER `goto L` anywhere in the method keeps
///     skipping the region exactly as before. Keeping the join (rather than
///     absorbing it too) is what keeps those jumps working; labels INSIDE the
///     region get the same guarantee from the method-wide `goto` count taken
///     up front (see the last gate below).
///
/// Gates, each one a way the rewrite could change meaning:
///   * the `if` must have a noise-only else arm (otherwise both arms carry
///     payload and the goto is one of two real branches);
///   * its then-arm must be a lone `goto` (labels/comments around it are
///     structurer scaffolding) — a `goto` PLUS code has already run that code
///     on the taken path, and folding it copies the code onto the other path;
///   * the label must be CLOSED by this list further down, with at least one
///     meaningful statement between (the adjacent case is a vacuous branch and
///     belongs to `drop_vacuous_if_gotos`);
///   * no edge from OUTSIDE the region may target ANY label inside it, at any
///     depth — a label nested in an arm or loop body rides along NO more
///     opaquely than a flat one: pulling the region under `!c` makes every
///     landing site inside it conditional, so an outside edge to a nested
///     label would strand exactly like one to a flat label (a nested
///     landing site reached only from inside moves along with its edges
///     intact, like a flat one). Proved by count, not by position: every
///     interior label's method-wide `goto` count (taken before any pass moves
///     a jump) must equal its count from inside the region — so labels reached
///     only from inside, and dead merge anchors nothing reaches, both move
///     along with their edges intact, while any outside edge still blocks
///     (the `v100` guard in 0x2CA7AC keeps its `goto` for exactly this reason:
///     `L14` is landed on from across the method).
///
/// Nested application is what makes it work on the guard chains: the fall-
/// through region is usually an already-structured `if`/`while`, which is one
/// element of the list here and simply moves into the new body.
/// Rewrite a goto to a case entry only when source and destination are in the
/// same recovered switch. The fall-through and every non-case target keep
/// their existing control-flow representation.
fn lower_switch_case_gotos(stmts: &mut [Stmt]) {
    for stmt in stmts {
        match stmt {
            Stmt::Switch { cases, tail, .. } => {
                let mut case_values = HashMap::<String, u64>::new();
                for (value, body) in cases.iter() {
                    if let Some(Stmt::Label(label)) = body.first() {
                        case_values.entry(label.clone()).or_insert(*value);
                    }
                }
                for (_, body) in cases.iter_mut() {
                    lower_gotos_to_case(body, &case_values);
                    lower_switch_case_gotos(body);
                }
                lower_switch_case_gotos(tail);
            }
            Stmt::If { then_body, else_body, .. } => {
                lower_switch_case_gotos(then_body);
                lower_switch_case_gotos(else_body);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::Lock { body, .. } => lower_switch_case_gotos(body),
            Stmt::For { init, incr, body, .. } => {
                lower_switch_case_gotos(init);
                lower_switch_case_gotos(incr);
                lower_switch_case_gotos(body);
            }
            Stmt::Using { decl, body } => {
                lower_switch_case_gotos(std::slice::from_mut(decl));
                lower_switch_case_gotos(body);
            }
            Stmt::Try { try_body, catches, finally_body } => {
                lower_switch_case_gotos(try_body);
                for (_, body) in catches {
                    lower_switch_case_gotos(body);
                }
                if let Some(body) = finally_body {
                    lower_switch_case_gotos(body);
                }
            }
            _ => {}
        }
    }
}

/// Descend within a case body while retaining its enclosing switch's case
/// map. Never carry the map through a nested switch or exception boundary.
fn lower_gotos_to_case(stmts: &mut [Stmt], case_values: &HashMap<String, u64>) {
    for stmt in stmts {
        match stmt {
            Stmt::Goto(label) => {
                use std::sync::atomic::Ordering::Relaxed;
                STRUCT_CENSUS.case_goto_attempts.fetch_add(1, Relaxed);
                if let Some(value) = case_values.get(label).copied() {
                    *stmt = Stmt::GotoCase(value);
                    STRUCT_CENSUS.case_goto_hits.fetch_add(1, Relaxed);
                } else {
                    STRUCT_CENSUS.case_goto_refused.fetch_add(1, Relaxed);
                }
            }
            Stmt::If { then_body, else_body, .. } => {
                lower_gotos_to_case(then_body, case_values);
                lower_gotos_to_case(else_body, case_values);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. }
            | Stmt::Lock { body, .. } => lower_gotos_to_case(body, case_values),
            Stmt::For { init, incr, body, .. } => {
                lower_gotos_to_case(init, case_values);
                lower_gotos_to_case(incr, case_values);
                lower_gotos_to_case(body, case_values);
            }
            Stmt::Using { decl, body } => {
                lower_gotos_to_case(std::slice::from_mut(decl), case_values);
                lower_gotos_to_case(body, case_values);
            }
            Stmt::Try { .. } | Stmt::Switch { .. } => {}
            _ => {}
        }
    }
}

/// Every goto target in the tree, counted at every statement depth.
///
/// Unlike collect_gotos (which walks only the variants the structurer emits
/// before reconstruction), this covers every Stmt variant, so the guard
/// fold's outside-edge proof cannot miss a jump from a for/using/lock body or
/// a switch arm.
fn count_all_gotos(stmts: &[Stmt], out: &mut HashMap<String, usize>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            *out.entry(l.clone()).or_insert(0) += 1;
        }
        match s {
            Stmt::If { then_body, else_body, .. } => {
                count_all_gotos(then_body, out);
                count_all_gotos(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                count_all_gotos(body, out)
            }
            Stmt::For { init, incr, body, .. } => {
                count_all_gotos(init, out);
                count_all_gotos(incr, out);
                count_all_gotos(body, out);
            }
            Stmt::Using { decl, body, .. } => {
                // `decl` is one statement, not a list — walk it as a slice.
                count_all_gotos(std::slice::from_ref(decl), out);
                count_all_gotos(body, out);
            }
            Stmt::Lock { body, .. } | Stmt::ForEach { body, .. } => count_all_gotos(body, out),
            Stmt::Try { try_body, catches, finally_body } => {
                count_all_gotos(try_body, out);
                for (_, b) in catches {
                    count_all_gotos(b, out);
                }
                if let Some(f) = finally_body {
                    count_all_gotos(f, out);
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    count_all_gotos(b, out);
                }
                count_all_gotos(tail, out);
            }
            _ => {}
        }
    }
}

/// Guard-chain fold census: examinations vs folds vs refusal buckets.
///
/// Bumped from [`fold_branch_over_join`] (atomic increments only — inert on
/// well-formed trees) and reported by `verify::VerifyStats::report` as
/// attempts-vs-hits plus refusals, the same shape as every other `[verify]`
/// counter line. `candidates` counts guard-shaped examinations
/// (`if (c) { goto L; }` with a noise-only else arm); each ends in exactly one
/// of `folds` / `refused_outside` / `refused_unclosed`, so
/// `candidates == folds + refused_outside + refused_unclosed` holds per
/// examination. One guard can be examined more than once: an ancestor fold
/// that nests its region under `!c` clones it with a fresh statement address
/// and a later visit examines the clone again. Re-examinations of the SAME
/// statement (the per-ancestor-level double recursion below) are deduped by
/// address in `fold_branch_over_join`; clone recounts are genuine distinct
/// statements and count again.
static GUARD_CHAIN_CANDIDATES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static GUARD_CHAIN_FOLDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GUARD_CHAIN_REFUSED_OUTSIDE: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
static GUARD_CHAIN_REFUSED_UNCLOSED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Guard-chain fold attempts vs hits plus refusal buckets.
/// `(candidates, folds, refused_outside_edge, refused_unclosed)`.
pub fn guard_chain_fold_counts() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        GUARD_CHAIN_CANDIDATES.load(Relaxed),
        GUARD_CHAIN_FOLDS.load(Relaxed),
        GUARD_CHAIN_REFUSED_OUTSIDE.load(Relaxed),
        GUARD_CHAIN_REFUSED_UNCLOSED.load(Relaxed),
    )
}

/// Every label a statement tree defines, at ANY depth.
///
/// Unlike `collect_labels` (which walks only the variants the structurer emits
/// before reconstruction), this covers every `Stmt` variant — `reconstruct::apply`
/// runs before `branch_repair`, so a fold region can hold `for`/`using`/`switch`
/// bodies with nested landing sites, and the outside-edge proof must see them.
fn collect_all_labels(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        if let Stmt::Label(l) = s {
            out.insert(l.clone());
        }
        match s {
            Stmt::If { then_body, else_body, .. } => {
                collect_all_labels(then_body, out);
                collect_all_labels(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                collect_all_labels(body, out)
            }
            Stmt::For { init, incr, body, .. } => {
                collect_all_labels(init, out);
                collect_all_labels(incr, out);
                collect_all_labels(body, out);
            }
            Stmt::Using { decl, body, .. } => {
                collect_all_labels(std::slice::from_ref(decl), out);
                collect_all_labels(body, out);
            }
            Stmt::Lock { body, .. } | Stmt::ForEach { body, .. } => {
                collect_all_labels(body, out)
            }
            Stmt::Try { try_body, catches, finally_body } => {
                collect_all_labels(try_body, out);
                for (_, b) in catches {
                    collect_all_labels(b, out);
                }
                if let Some(f) = finally_body {
                    collect_all_labels(f, out);
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases {
                    collect_all_labels(b, out);
                }
                collect_all_labels(tail, out);
            }
            _ => {}
        }
    }
}

fn fold_branch_over_join(
    stmts: &mut Vec<Stmt>,
    global_gotos: &HashMap<String, usize>,
    fold_seen: &mut HashSet<*const Stmt>,
) {
    // Bottom-up: an inner list's own guards collapse before an outer list
    // decides whether the region between ITS goto and label is foldable.
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                fold_branch_over_join(then_body, global_gotos, fold_seen);
                fold_branch_over_join(else_body, global_gotos, fold_seen);
            }
            Stmt::While { body, .. }
            | Stmt::DoWhile { body, .. }
            | Stmt::ForEach { body, .. } => fold_branch_over_join(body, global_gotos, fold_seen),
            Stmt::For { body, .. } => fold_branch_over_join(body, global_gotos, fold_seen),
            Stmt::Using { body, .. } => fold_branch_over_join(body, global_gotos, fold_seen),
            Stmt::Lock { body, .. } => fold_branch_over_join(body, global_gotos, fold_seen),
            Stmt::Try { try_body, catches, finally_body } => {
                fold_branch_over_join(try_body, global_gotos, fold_seen);
                for (_, b) in catches.iter_mut() {
                    fold_branch_over_join(b, global_gotos, fold_seen);
                }
                if let Some(f) = finally_body {
                    fold_branch_over_join(f, global_gotos, fold_seen);
                }
            }
            Stmt::Switch { cases, tail, .. } => {
                for (_, b) in cases.iter_mut() {
                    fold_branch_over_join(b, global_gotos, fold_seen);
                }
                fold_branch_over_join(tail, global_gotos, fold_seen);
            }
            _ => {}
        }
    }
    // REVERSE order: a chain of guards to one join reads as siblings
    // `[G1, B1, G2, B2, J:]`. Folding G1 first nests the whole tail
    // (including G2) under `!c1`, which moves G2's own join cross-level and
    // blocks it (the documented limitation). Folding G2 first collapses it
    // to `if (!c2) { B2 }`, and G1's region then holds no flat goto — so G1
    // folds too and the chain leaves zero gotos with nesting equal to its
    // length. Each fold keeps the same gates (lone goto, noise-only else,
    // label later in THIS list, no OUTSIDE edge to a label inside — see
    // the gate below), so reversing only folds chains the forward order left
    // half-done; it never folds a shape the forward order refused.
    let mut i = stmts.len();
    while i > 0 {
        i -= 1;
        let Stmt::If { then_body, else_body, .. } = &stmts[i] else {
            continue;
        };
        if !else_body.iter().all(is_noise_stmt) {
            continue;
        }
        // then-arm: labels/comments are scaffolding, the goto is the payload.
        let mut payload = then_body.iter().filter(|s| {
            !matches!(s, Stmt::Comment(_) | Stmt::Label(_))
        });
        let (Some(Stmt::Goto(l)), None) = (payload.next(), payload.next()) else {
            continue;
        };
        let l = l.clone();
        // Census: this guard has the foldable shape (lone `goto`, noise-only
        // else arm). Deduped by statement address — the per-ancestor-level
        // double recursion above re-examines the same statement, and only the
        // first examination counts. (An ancestor fold can later CLONE this
        // guard into a nested region with a fresh address; that clone is a
        // distinct statement and counts again.)
        let count_this = fold_seen.insert(&stmts[i] as *const Stmt);
        if count_this {
            use std::sync::atomic::Ordering::Relaxed;
            GUARD_CHAIN_CANDIDATES.fetch_add(1, Relaxed);
        }
        // Close the jump: the label must be defined later in THIS list.
        // Comments do not count as "between" (no output for them), but a
        // real statement must be there or this is the vacuous case.
        let mut j = i + 1;
        let mut meaningful = 0usize;
        let mut label_at = None;
        while j < stmts.len() {
            match &stmts[j] {
                Stmt::Comment(_) => {}
                Stmt::Label(n) => {
                    meaningful += 1;
                    if *n == l {
                        label_at = Some(j);
                        break;
                    }
                }
                _ => meaningful += 1,
            }
            j += 1;
        }
        // `meaningful < 2`: the label itself plus nothing before it.
        let Some(label_at) = label_at else {
            if count_this {
                use std::sync::atomic::Ordering::Relaxed;
                GUARD_CHAIN_REFUSED_UNCLOSED.fetch_add(1, Relaxed);
            }
            continue;
        };
        if meaningful < 2 {
            if count_this {
                use std::sync::atomic::Ordering::Relaxed;
                GUARD_CHAIN_REFUSED_UNCLOSED.fetch_add(1, Relaxed);
            }
            continue;
        }
        // A label inside the region is a landing site — but only an edge
        // from OUTSIDE the region makes folding unsound. The method-wide
        // counts were taken before any pass moved a jump, so a label whose
        // every `goto` sits inside the region (a self-contained landing site)
        // or which nothing targets at all (a dead merge anchor the quiet-label
        // sweep removes right after) moves along with its edges intact. Any
        // outside edge still blocks, exactly as before.
        //
        // ALL labels at any depth, not just flat ones: pulling the region
        // under `!c` makes every landing site inside it conditional, and a
        // label nested in an arm or loop body ridden along "opaquely" would
        // strand an outside edge just as surely as a flat one.
        {
            let mut inside: HashMap<String, usize> = HashMap::new();
            count_all_gotos(&stmts[i + 1..label_at], &mut inside);
            let mut region_labels: HashSet<String> = HashSet::new();
            collect_all_labels(&stmts[i + 1..label_at], &mut region_labels);
            // Deterministic: the outcome is one boolean, so set order cannot
            // leak into the output.
            let mut blocked = false;
            for n in &region_labels {
                if global_gotos.get(n).copied().unwrap_or(0)
                    > inside.get(n).copied().unwrap_or(0)
                {
                    blocked = true;
                    break;
                }
            }
            if blocked {
                if count_this {
                    use std::sync::atomic::Ordering::Relaxed;
                    GUARD_CHAIN_REFUSED_OUTSIDE.fetch_add(1, Relaxed);
                }
                continue;
            }
        }
        let (new_cond, region): (Expr, Vec<Stmt>) = {
            let Stmt::If { cond, .. } = &stmts[i] else {
                unreachable!()
            };
            (invert(cond), stmts[i + 1..label_at].to_vec())
        };
        stmts[i] = Stmt::If {
            cond: new_cond,
            then_body: region,
            else_body: Vec::new(),
        };
        stmts.drain(i + 1..label_at);
        // Exact: each drain consumes one guard `goto`, and a folded guard no
        // longer matches the shape above — so this fires exactly once per
        // fold, on any visit. (Predicates are inverted, never deleted; the
        // join label is kept by the drain bounds.)
        {
            use std::sync::atomic::Ordering::Relaxed;
            GUARD_CHAIN_FOLDS.fetch_add(1, Relaxed);
        }
        // Reverse order folds the INNER guard first, so the outer guard's
        // region already holds the folded `if (!c2) { … }` when it is judged:
        // a chain of any length collapses fully, leaving zero gotos. The old
        // forward order folded the outer first and stranded the inner
        // cross-level (the limitation noted below no longer applies to
        // same-list chains; it remains true only for genuinely cross-level
        // jumps, which keep their correct `goto`).
    }
}

/// `if (c) { goto L; } L:` — the predicate gates a jump to the statement that
/// executes next anyway, so both arms run identical code and the branch is
/// provably dead.
///
/// This is the same fact `drop_redundant_tails` uses, but wrapped in the branch
/// shell the structurer builds around the fall-through. The common origin is
/// the IL2CPP null check: `cbnz w8, +8; bl <raise_null>` where the `bl` is a
/// runtime helper excised as `DroppedRuntimeCall`, leaving the skipped slot
/// EMPTY — the two arms then land on one address, and what reaches the reader
/// is a condition plus a `goto` to the next line, which reads as corrupted
/// control flow (1,483 of them in the v24.5 PE dump).
///
/// The SHELL is dropped, not automatically the predicate: when the condition
/// contains a call, the call is real recovered work (an inlined null check),
/// so it survives as a bare `Eval`. A pure comparison/load gates nothing and
/// goes away with the arm.
fn drop_vacuous_if_gotos(stmts: &mut Vec<Stmt>) {
    for i in 0..stmts.len() {
        // Only meaningful statements can be "the next thing".
        let next = stmts[i + 1..]
            .iter()
            .find(|s| !matches!(s, Stmt::Comment(_)));
        let Stmt::If { cond, then_body, else_body } = &stmts[i] else {
            continue;
        };
        if !else_body.iter().all(is_noise_stmt) {
            continue;
        }
        // The then-arm must be a lone `goto`, possibly with labels/comments
        // that the structurer dropped around it.
        let mut payload = then_body.iter().filter(|s| !matches!(s, Stmt::Comment(_)));
        let (Some(Stmt::Goto(l)), None) = (payload.next(), payload.next()) else {
            continue;
        };
        let lands_next = matches!(next, Some(Stmt::Label(n)) if n == l);
        if !lands_next {
            continue;
        }
        let replacement = if expr_has_call(cond) {
            Stmt::Eval(cond.clone())
        } else {
            Stmt::Comment("\u{0}rm-vacuous-if".into())
        };
        stmts[i] = replacement;
    }
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                drop_vacuous_if_gotos(then_body);
                drop_vacuous_if_gotos(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => drop_vacuous_if_gotos(body),
            Stmt::Try { try_body, catches, finally_body } => {
                drop_vacuous_if_gotos(try_body);
                for (_, b) in catches.iter_mut() {
                    drop_vacuous_if_gotos(b);
                }
                if let Some(f) = finally_body {
                    drop_vacuous_if_gotos(f);
                }
            }
            _ => {}
        }
    }
}

/// Collect every label defined anywhere in the tree (labels live in arms,
/// loops and handlers).
fn collect_labels(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        if let Stmt::Label(l) = s {
            out.insert(l.clone());
        }
        match s {
            Stmt::If { then_body, else_body, .. } => {
                collect_labels(then_body, out);
                collect_labels(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_labels(body, out),
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

// Pass D: drop gotos pointing at labels the structurer never emitted. Judged
// against the GLOBAL label set (see `branch_repair`).
fn strip_dangling(stmts: &mut Vec<Stmt>, labels: &HashSet<String>) {
    stmts.retain(|s| !matches!(s, Stmt::Goto(l) if !labels.contains(l)));
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                strip_dangling(then_body, labels);
                strip_dangling(else_body, labels);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => strip_dangling(body, labels),
            _ => {}
        }
    }
}

// Pass E: labels with no remaining references are merge anchors for gotos
// that became fall-throughs — remove them.
fn strip_quiet_labels(stmts: &mut Vec<Stmt>, referenced: &HashSet<String>) {
    stmts.retain(|s| !matches!(s, Stmt::Label(l) if !referenced.contains(l)));
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                strip_quiet_labels(then_body, referenced);
                strip_quiet_labels(else_body, referenced);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                strip_quiet_labels(body, referenced);
            }
            _ => {}
        }
    }
}

/// Does this statement list already end control flow? A C# switch arm does not
/// fall through, so an arm whose body ends in `return`/`throw`/`goto` must not
/// also get a `break;`.
fn ends_with_terminator(body: &[Stmt]) -> bool {
    matches!(
        body.last(),
        Some(
            Stmt::Return(_)
                | Stmt::Throw(_)
                | Stmt::Goto(_)
                | Stmt::GotoCase(_)
                | Stmt::Break
                | Stmt::Continue
        )
    )
}

fn collect_gotos(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        if let Stmt::Goto(l) = s {
            out.insert(l.clone());
        }
        // The switch shell's `case k: goto Lx` references live INSIDE
        // Stmt::Switch as data — without counting them, Pass E strips every
        // arm label as "unreferenced" while the case gotos survive, leaving
        // dangling case targets (Evaluate after arm rendering).
        if let Stmt::Switch { cases, tail, .. } = s {
            for (_, body) in cases {
                collect_gotos(body, out);
            }
            collect_gotos(tail, out);
        }
        match s {
            Stmt::If { then_body, else_body, .. } => {
                collect_gotos(then_body, out);
                collect_gotos(else_body, out);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_gotos(body, out),
            Stmt::Try { try_body, catches, finally_body } => {
                collect_gotos(try_body, out);
                for (_, b) in catches {
                    collect_gotos(b, out);
                }
                if let Some(f) = finally_body {
                    collect_gotos(f, out);
                }
            }
            _ => {}
        }
    }
}

/// Statements that carry no recoverable information: labels, comments,
/// coalesced-assignment markers, and empty/comment-only branches.
/// Does this expression contain a load (deref/field/array access)?
/// A condition built from an inlined call's result (null-check pattern) may
/// have no `Call` node left but still represents a REAL memory load — the
/// shell must not be dropped or the visible logic disappears.
/// Does evaluating this expression READ MEMORY?
///
/// A read is not free: it can fault (an IL2CPP class-init guard dereferences the
/// klass pointer). So a branch over a pure register compare is removable, while a
/// branch over `*(nint*)(*(nint*)(0x6054568) + 224)` is not — deleting it would
/// drop a real memory access. Used to decide whether an `if` shell has anything
/// observable left in it after its body was emptied.
fn expr_has_load(e: &Expr) -> bool {
    match e {
        Expr::RawDeref(..) | Expr::Field(..) | Expr::StaticField(..) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_load(a) || expr_has_load(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) => expr_has_load(a),
        Expr::Cond(a, b, c) => expr_has_load(a) || expr_has_load(b) || expr_has_load(c),
        _ => false,
    }
}

fn is_noise_stmt(s: &Stmt) -> bool {
    match s {
        Stmt::Label(_) | Stmt::Comment(_) => true,
        Stmt::Assign(t, _, Expr::Var(src)) if t == src => true,
        Stmt::Eval(e) => is_artifact_ctor(e),
        Stmt::If { then_body, else_body, .. } => {
            then_body.iter().all(is_noise_stmt) && else_body.iter().all(is_noise_stmt)
        }
        _ => false,
    }
}

/// `__unknown..ctor(__unknown)` — an artifact ctor call where neither the
/// receiver nor any argument was recovered. A REAL ctor call always has at
/// least a `this`/`newobj` receiver; `__unknown` on both sides means the
/// recovery pipeline had nothing, so the call is scheduling noise, not code.
fn is_artifact_ctor(e: &Expr) -> bool {
    match e {
        Expr::Call { recv, name, args }
            if name.ends_with(".ctor")
                && matches!(recv.as_deref(), None | Some(Expr::Unknown))
                && args.iter().all(|a| matches!(a, Expr::Unknown | Expr::MissingAbiArg)) =>
            true,
        Expr::NewObj(ty, args)
            if ty == "__unknown"
                && args.iter().all(|a| matches!(a, Expr::Unknown | Expr::MissingAbiArg)) =>
            true,
        _ => false,
    }
}

/// Does this expression contain a call (side effects worth keeping)?
fn expr_has_call(e: &Expr) -> bool {
    match e {
        Expr::Call { .. } | Expr::RawCall(..) | Expr::NewObj(..) | Expr::NewArr(..)
        | Expr::Await(_) => true,
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_has_call(a) || expr_has_call(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) => expr_has_call(a),
        Expr::Cond(a, b, c) => expr_has_call(a) || expr_has_call(b) || expr_has_call(c),
        Expr::Field(base, _) => expr_has_call(base),
        Expr::RawDeref(a, _, _) => expr_has_call(a),
        Expr::RawAddr(a, _) | Expr::AddressOf(a) => expr_has_call(a),
        _ => false,
    }
}

fn label_name(sc: &StructuredCfg, blk: usize) -> String {
    if let Some(n) = sc.labels.get(&blk) {
        return n.clone();
    }
    // A block that is not a branch target has no assigned name, so it is
    // auto-named after its index. That name can ALREADY belong to an unrelated
    // block: `labels` numbers its names in CFG-TARGET order (`L0`, `L1`, …), not
    // by block index. Two `L7:` labels in one method make every `goto L7`
    // ambiguous — the text then cannot be read as the program the machine runs.
    // (Measured on the 4,000-method sample before this check: 585 methods with
    // an auto name colliding with an assigned one.)
    let base = format!("L{blk}");
    if !sc.label_names.contains(&base) {
        return base;
    }
    let mut k = 1usize;
    loop {
        let candidate = format!("L{blk}_{k}");
        if !sc.label_names.contains(&candidate) {
            return candidate;
        }
        k += 1;
    }
}

// ---------------------------------------------------------------------------
// AST rewriting passes (extensible sugar-recognition pipeline)
// ---------------------------------------------------------------------------

/// Fold patterns into higher-level C# constructs. Each rule is independent;
/// a failed match leaves valid lower-level output (graceful degradation).
pub mod rewrites {
    use super::{Expr, Stmt};

    /// Entry point: apply all sugar passes to a statement list recursively.
    ///
    /// SINGLE bottom-up traversal: children are rewritten once, then this
    /// list's folds run. The previous version re-traversed children from both
    /// `rewrite_children` AND `fold_ternary`'s tail — 4^depth work — which
    /// hung (practically forever) on deeply nested control flow. The AST is a
    /// finite tree, so one post-order pass reaches every fold opportunity:
    /// parent-level folds only reshape THIS list and can never unlock folds
    /// inside already-processed children.
    pub(super) fn apply(stmts: &mut Vec<Stmt>) {
        let prof = std::env::var("ILR_PROF").is_ok();
        let t0 = std::time::Instant::now();
        for stmt in stmts.iter_mut() {
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    apply(then_body);
                    apply(else_body);
                }
                Stmt::While { body, .. } => apply(body),
                Stmt::DoWhile { body, .. } => apply(body),
                Stmt::Try { try_body, catches, finally_body } => {
                    apply(try_body);
                    for (_, body) in catches.iter_mut() {
                        apply(body);
                    }
                    if let Some(f) = finally_body {
                        apply(f);
                    }
                }
                _ => {}
            }
        }
        if prof { eprintln!("[rw] children {}us", t0.elapsed().as_micros()); }
        let t1 = std::time::Instant::now();
        fold_bool_assign(stmts);
        if prof { eprintln!("[rw] bool_assign {}us", t1.elapsed().as_micros()); }
        let t2 = std::time::Instant::now();
        fold_ternary(stmts);
        if prof { eprintln!("[rw] ternary {}us", t2.elapsed().as_micros()); }
        let t3 = std::time::Instant::now();
        fold_async_await(stmts);
        if prof { eprintln!("[rw] async {}us", t3.elapsed().as_micros()); }
    }

    /// IL2CPP async state machines: awaiter patterns render as opaque
    /// `IsCompleted` / `GetResult` / `OnCompleted` call sequences. Recognize
    /// the awaiter protocol and collapse it into `await`-style statements:
    ///
    ///   v = X(); if (!v.IsCompleted) { ...OnCompleted/await suspend... } v.GetResult()
    ///     ⇒
    ///   v = await X();
    ///
    /// A failed match leaves the lower-level sequence (graceful degradation).
    fn fold_async_await(stmts: &mut Vec<Stmt>) {
        // `x.IsCompleted == false` (either operand order)
        fn field_false(e: &Expr) -> bool {
            matches!(e, Expr::Cmp(super::CmpKind::Eq, a, b) if is_field(a, "IsCompleted") && matches!(**b, Expr::Bool(false)))
                || matches!(e, Expr::Cmp(super::CmpKind::Eq, a, b) if is_field(b, "IsCompleted") && matches!(**a, Expr::Bool(false)))
        }
        fn is_field(e: &Expr, name: &str) -> bool {
            matches!(e, Expr::Field(_, f) if f == name)
        }
        /// Count statements whose rendered text mentions any marker.
        fn renders_containing(stmts: &[Stmt], markers: &[&str]) -> usize {
            let mut buf = String::new();
            fn walk(stmts: &[Stmt], buf: &mut String) {
                for s in stmts {
                    match s {
                        Stmt::Eval(e) | Stmt::Assign(_, _, e) => e.render(buf),
                        Stmt::If { then_body, else_body, .. } => {
                            walk(then_body, buf);
                            walk(else_body, buf);
                        }
                        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => walk(body, buf),
                        _ => {}
                    }
                }
            }
            walk(stmts, &mut buf);
            markers.iter().filter(|m| buf.contains(*m)).count()
        }
        let mut i = 0;
        while i < stmts.len() {
            // Pattern: Assign(t, Call F(...)); If { !t.IsCompleted, ... } [, t.GetResult() after]
            let Some(Stmt::Assign(task, _, task_expr)) = stmts.get(i) else { i += 1; continue };
            let Expr::Call { name: task_name, .. } = task_expr else { i += 1; continue };
            let Some(Stmt::If { cond, then_body, else_body }) = stmts.get(i + 1) else { i += 1; continue };
            // condition must be a NOT of `<task>.IsCompleted` (or == false)
            let is_completed_test = match cond {
                Expr::Not(inner) => matches!(**inner, Expr::Field(ref r, ref f) if f == "IsCompleted"),
                Expr::Cmp(super::CmpKind::Eq, a, b) => field_false(a) || field_false(b),
                _ => false,
            };
            if !is_completed_test || then_body.is_empty() {
                i += 1;
                continue;
            }
            // the suspend arm must mention OnCompleted/AwaitSuspension
            let suspend = renders_containing(then_body, &["OnCompleted", "AwaitSuspension", "UnsafeOnCompleted"]);
            if suspend == 0 {
                i += 1;
                continue;
            }
            // optional else arm (fast path) + GetResult tail
            let mut consumed = 2;
            if let Some(Stmt::Eval(Expr::Call { name: gr, .. })) = stmts.get(i + 2) {
                if gr == "GetResult" || gr.ends_with(".GetResult") {
                    consumed = 3;
                }
            }
            let _ = else_body;
            let await_stmt = Stmt::Assign(
                task.clone(),
                None,
                Expr::Await(Box::new(task_expr.clone())),
            );
            stmts.splice(i..i + consumed, vec![await_stmt, Stmt::Comment(format!(
                "await {task_name}() — suspend/resume state machine folded"
            ))]);
            i += 2;
        }
    }

    /// if (c) x = true; else x = false;   =>  x = c;
    /// if (c) x = false; else x = true;   =>  x = !c;
    fn fold_bool_assign(stmts: &mut Vec<Stmt>) {
        let mut i = 0;
        while i < stmts.len() {
            let (cond, target) = {
                let Some(Stmt::If { cond, then_body, else_body }) = stmts.get(i) else {
                    i += 1;
                    continue;
                };
                if then_body.len() != 1 || else_body.len() != 1 {
                    i += 1;
                    continue;
                }
                let (
                    Stmt::Assign(t1, _, Expr::Bool(tv)),
                    Stmt::Assign(t2, _, Expr::Bool(fv)),
                ) = (&then_body[0], &else_body[0])
                else {
                    i += 1;
                    continue;
                };
                (cond.clone(), (t1.clone(), *tv, *fv, t2.clone()))
            };
            let (t, tv, fv, t2) = target;
            if t != t2 {
                i += 1;
                continue;
            }
            let expr = if tv && !fv {
                cond
            } else if !tv && fv {
                super::invert(&cond)
            } else {
                i += 1;
                continue;
            };
            // the If is ONE statement — replacing exactly it (the old
            // splice(i..i+3) silently dropped the two statements after it)
            stmts[i] = Stmt::Assign(t, None, expr);
            i += 1;
        }
    }

    /// General ternary: If { then:[Assign(t,va)], else:[Assign(t,vb)] }
    /// ⇒ Assign(t, cond ? va : vb).
    fn fold_ternary(stmts: &mut Vec<Stmt>) {
        for i in 0..stmts.len() {
            let matched = match stmts.get(i) {
                Some(Stmt::If { cond, then_body, else_body }) => {
                    if then_body.len() == 1 && else_body.len() == 1 {
                        match (&then_body[0], &else_body[0]) {
                            (
                                Stmt::Assign(t1, ty, va),
                                Stmt::Assign(t2, _, vb),
                            ) if t1 == t2 => Some((
                                cond.clone(),
                                t1.clone(),
                                ty.clone(),
                                va.clone(),
                                vb.clone(),
                            )),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some((cond, t, ty, va, vb)) = matched {
                stmts[i] = Stmt::Assign(
                    t,
                    ty,
                    Expr::Cond(Box::new(cond), Box::new(va), Box::new(vb)),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AST cleanup passes
// ---------------------------------------------------------------------------

/// Post-structuring cleanup: dead-store elimination.
/// Removes `x = expr;` when x is overwritten before being read.
pub mod cleanup {
    use super::{Expr, Stmt};

    /// SINGLE bottom-up traversal (mirrors rewrites::apply): recurse into
    /// children once, then run the local dead-store pass on this list.
    pub fn apply(stmts: &mut Vec<Stmt>) {
        // Every name READ anywhere in the method body. The flat dead-store scan
        // below only sees ONE statement list, and the emitted pseudo-C is not
        // in execution order — the structurer emits label/goto shells, so a def
        // can sit textually AFTER the read of the same local.
        //
        // Measured on 0x25e6058 (`set_usageHints`): the structurer emitted
        // `v24 = v54;` at line 17 and `v54 = fn_0(…)` at line 20; the
        // forward-only scan saw no read of `v54` after its def, replaced the
        // assignment with a bare `fn_0(…)` eval, and left line 17 reading a
        // local nothing had defined. The call survived, the VALUE did not.
        let mut reads: std::collections::HashSet<String> = std::collections::HashSet::new();
        collect_expr_reads(stmts, &mut reads);
        apply_inner(stmts, false, &reads);
    }

    /// Collect every identifier used as a value in the statement tree.
    fn collect_expr_reads(stmts: &[Stmt], out: &mut std::collections::HashSet<String>) {
        fn expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
            match e {
                Expr::Var(n) => {
                    out.insert(n.clone());
                }
                Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
                    expr(a, out);
                    expr(b, out);
                }
                Expr::Call { recv, args, .. } => {
                    if let Some(r) = recv {
                        expr(r, out);
                    }
                    for a in args {
                        expr(a, out);
                    }
                }
                Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
                    for a in args {
                        expr(a, out);
                    }
                }
                Expr::Field(r, _) | Expr::Not(r) | Expr::Neg(r) | Expr::RawAddr(r, _)
                | Expr::RawDeref(r, _, _) | Expr::AddressOf(r) | Expr::Await(r)
                | Expr::Cast(_, r) => expr(r, out),
                Expr::Cond(c, a, b) => {
                    expr(c, out);
                    expr(a, out);
                    expr(b, out);
                }
                _ => {}
            }
        }
        for s in stmts {
            match s {
                Stmt::Assign(_, _, r) | Stmt::Eval(r) | Stmt::Throw(r) | Stmt::Return(Some(r)) => {
                    expr(r, out)
                }
                Stmt::If { cond, then_body, else_body } => {
                    expr(cond, out);
                    collect_expr_reads(then_body, out);
                    collect_expr_reads(else_body, out);
                }
                Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
                    expr(cond, out);
                    collect_expr_reads(body, out);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    collect_expr_reads(try_body, out);
                    for (_, b) in catches {
                        collect_expr_reads(b, out);
                    }
                    if let Some(f) = finally_body {
                        collect_expr_reads(f, out);
                    }
                }
                _ => {}
            }
        }
    }

    fn apply_inner(stmts: &mut Vec<Stmt>, in_loop: bool, reads: &std::collections::HashSet<String>) {
        let prof = std::env::var("ILR_PROF").is_ok();
        let t0 = std::time::Instant::now();
        for stmt in stmts.iter_mut() {
            match stmt {
                Stmt::If { then_body, else_body, .. } => {
                    apply_inner(then_body, in_loop, reads);
                    apply_inner(else_body, in_loop, reads);
                }
                Stmt::While { body, .. } => apply_inner(body, true, reads),
                Stmt::DoWhile { body, .. } => apply_inner(body, true, reads),
                Stmt::Try { try_body, catches, finally_body } => {
                    // try/catch bodies are exception paths: the flat scan
                    // cannot see the handler's reads either — same rule as
                    // loops applies (a store in `try` may be read by the
                    // catch's variable live-out).
                    apply_inner(try_body, in_loop, reads);
                    for (_, body) in catches.iter_mut() {
                        apply_inner(body, in_loop, reads);
                    }
                    if let Some(f) = finally_body {
                        apply_inner(f, in_loop, reads);
                    }
                }
                _ => {}
            }
        }
        // Loop bodies are OFF LIMITS for dead-store elimination: the flat
        // scan only sees later statements in the same list, but a loop-carried
        // store (`p <<= 1`) is re-read by the back edge / next iteration and
        // often by the loop condition — invisible here. Removing those stores
        // emptied every recovered loop body (`// loop body not recovered`).
        //
        // NOTE: `body_reads` (the whole-method read set) ALREADY subsumes that
        // argument — a loop-carried store names its own target in its
        // right-hand side, so it is never a candidate. Lifting this guard was
        // tried and measured on the v24.5 PE dump: it removed 138 of 273,016
        // assignments and produced 3 more empty loop bodies, so the guard is
        // kept. The 16.8k assignments (6.2%) whose target is never read in the
        // EMITTED TEXT are NOT held back by this guard: their target IS in
        // `body_reads`, because the use site inlines the defining expression
        // instead of printing the name while the binding's own statement is
        // emitted as well. That is an inlining/rendering inconsistency, and the
        // guard is not what stands between the reader and a shorter body.
        if !in_loop {
            dead_stores_flat(stmts, reads);
        }
        if prof { eprintln!("[cl] dead_stores {}us", t0.elapsed().as_micros()); }
    }

    /// Dead-store elimination for ONE statement list (children already
    /// processed): removes `x = expr;` when x is overwritten before read.
    /// Is this assignment target a PLAIN SCALAR — a local the reader can see\n    /// the whole lifetime of?
    ///
    /// Only those are safe to delete as dead stores. A store through a POINTER
    /// (`*(nint*)x4`, `arr[i]`, `this.field`) is observable by the CALLER — the
    /// sret buffer of a struct-returning method, an `out`/`ref` parameter, a
    /// static field, an array element — and this pass only sees one method, so
    /// "nothing in here reads it again" says nothing about whether the store is
    /// live. Deleting those was erasing whole method bodies: the struct-return
    /// idiom is `str r5, [r4]` / `stm r4, {r5, r6}` (r0 = &result on entry,
    /// copied to r4), so `SafeStringMarshal MarshalString(string)` @0x17005f8, a
    /// 46-instruction method whose real body is `result = new
    /// SafeStringMarshal(str, false)`, came out as `return v11;` — the payload
    /// store removed, the return operand left as an undefined local.
    fn is_scalar_target(target: &str) -> bool {
        let t = target.trim();
        if t.is_empty() {
            return false;
        }
        let mut chars = t.chars();
        match chars.next() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => return false,
        }
        // `*`, `[`, `.`, `->`, `(` all mean "through a pointer / a field".
        t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    }

    /// Does evaluating this expression have an effect the caller can see?
    ///
    /// ONLY calls and allocations. A dead binding to a PURE expression can be
    /// deleted outright; a dead binding to a CALL cannot — the callee still
    /// runs. This is what made dead-store elimination eat real work:
    /// `RuntimeMarshal.MarshalString` @0x17005f8 (ARM32) is
    ///   ldrb r0,[flag]; cmp r0,#0; bne join; <3 loads, bl, ldr, mov, strb>; join:
    /// and the `bl` reached the renderer as `v24 = fn_5827e4(…)` — an
    /// assignment whose target nothing reads — so this pass removed it and the
    /// method's only call vanished from the body.
    /// Does this expression do anything observable beyond computing a value?
    ///
    /// `idioms` uses it to decide whether a statement can be reordered past,
    /// so it must be conservative: anything that can WRITE memory (a call, a
    /// construction, an await) counts, and a raw pointer load does not.
    pub(crate) fn expr_has_side_effect(e: &Expr) -> bool {
        match e {
            Expr::Call { .. }
            | Expr::RawCall(..)
            | Expr::NewObj(..)
            | Expr::NewArr(..)
            | Expr::Await(_) => true,
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
                expr_has_side_effect(a) || expr_has_side_effect(b)
            }
            Expr::Not(inner) | Expr::Neg(inner) | Expr::Field(inner, _) => {
                expr_has_side_effect(inner)
            }
            Expr::Cond(c, a, b) => {
                expr_has_side_effect(c) || expr_has_side_effect(a) || expr_has_side_effect(b)
            }
            _ => false,
        }
    }

    fn dead_stores_flat(stmts: &mut Vec<Stmt>, body_reads: &std::collections::HashSet<String>) {
        let mut i = stmts.len();
        while i > 0 {
            i -= 1;
            let (target, rhs) = match &stmts[i] {
                Stmt::Assign(t, _, rhs) if !t.is_empty() && is_scalar_target(t) => {
                    (t.clone(), rhs.clone())
                }
                _ => continue,
            };
            // READ SOMEWHERE IN THE BODY → NOT DEAD. The scan below is
            // forward-only inside ONE statement list, but the emitted text is
            // not in execution order: a def rendered above the read (or in a
            // sibling block) looks unread here. Keeping the binding costs a
            // statement; deleting it costs a value and leaves the reader
            // chasing a name nothing defines.
            // READ ANYWHERE IN THE BODY → NOT DEAD. The forward scan below
            // only sees reads that appear AFTER this statement in the SAME
            // list, but the emitted pseudo-C is not in execution order (the
            // structurer emits label/goto shells), so a def can sit textually
            // above its reader, or in a sibling list. `body_reads` is the
            // whole-method read set gathered once by `apply`.
            if body_reads.contains(&target) {
                continue;
            }
            let mut live = false;
            for j in (i + 1)..stmts.len() {
                match &stmts[j] {
                    Stmt::Assign(t2, _, rhs) => {
                        if t2 == &target { break; }
                        if reads_var(rhs, &target) { live = true; break; }
                    }
                    Stmt::Eval(e) | Stmt::Throw(e) => {
                        if reads_var(e, &target) { live = true; break; }
                    }
                    Stmt::Return(Some(e)) => {
                        if reads_var(e, &target) { live = true; break; }
                    }
                    Stmt::If { cond, .. } | Stmt::While { cond, .. } | Stmt::DoWhile { cond, .. } => {
                        if reads_var(cond, &target) { live = true; break; }
                        live = true;
                        break;
                    }
                    // Goto/Label/Break/Continue: control LEAVES this linear
                    // list here — a store may be read on the jumped-to path
                    // (e.g. the merge block re-reading a temp). Treating the
                    // store as dead truncated every merge that followed a
                    // `goto L…`. Same for anything we can't see through.
                    _ => { live = true; break; }
                }
            }
            if !live {
                // Keep the EFFECT, drop the dead binding.
                if expr_has_side_effect(&rhs) {
                    stmts[i] = Stmt::Eval(rhs);
                } else {
                    stmts.remove(i);
                }
            }
        }
    }

    fn reads_var(e: &Expr, name: &str) -> bool {
        match e {
            Expr::Var(n) => n == name,
            Expr::This => false,
            Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => reads_var(a, name) || reads_var(b, name),
            Expr::Not(inner) | Expr::Neg(inner) => reads_var(inner, name),
            Expr::Field(recv, _) => reads_var(recv, name),
            Expr::Call { recv: Some(r), args, .. } => {
                reads_var(r, name) || args.iter().any(|a| reads_var(a, name))
            }
            Expr::Call { recv: None, args, .. } | Expr::RawCall(_, args, _) => {
                args.iter().any(|a| reads_var(a, name))
            }
            Expr::NewObj(_, args) => args.iter().any(|a| reads_var(a, name)),
            Expr::RawDeref(base, _, _) => reads_var(base, name),
            Expr::RawAddr(base, _) | Expr::AddressOf(base) => reads_var(base, name),
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Impossible-guard flattening
// ---------------------------------------------------------------------------
// Structurer artifacts from init-guard regions whose merge was never found
// render as `if (c) { if (¬c) { X } }` — the SAME static-flag test under both
// polarities on one path, which real code can never take (the flag is set
// once by the guard's init; nothing between the tests can change it). The
// payload X is the method's real body; the init itself was already erased by
// remove_init_guards' "assume initialized" convention (or DCE). When X is
// effect-free (no calls/allocations — nothing that could BE the init), the
// two tests cancel and X is emitted. Effectful payloads are left untouched:
// hoisting an allocation or call would be a semantic change.

fn exprs_eq(a: &Expr, b: &Expr) -> bool {
    use Expr::*;
    match (a, b) {
        (Var(x), Var(y)) => x == y,
        (This, This) | (Null, Null) | (Unknown, Unknown) | (MissingAbiArg, MissingAbiArg) => true,
        (Int(x), Int(y)) => x == y,
        (Float(x), Float(y)) => x == y,
        (Double(x), Double(y)) => x == y,
        (Bool(x), Bool(y)) => x == y,
        (Str(x), Str(y)) => x == y,
        (Bin(o1, x1, y1), Bin(o2, x2, y2)) => {
            std::mem::discriminant(o1) == std::mem::discriminant(o2)
                && exprs_eq(x1, x2)
                && exprs_eq(y1, y2)
        }
        (Cmp(o1, x1, y1), Cmp(o2, x2, y2)) => {
            std::mem::discriminant(o1) == std::mem::discriminant(o2)
                && exprs_eq(x1, x2)
                && exprs_eq(y1, y2)
        }
        (Not(x), Not(y)) | (Neg(x), Neg(y)) | (Await(x), Await(y)) => exprs_eq(x, y),
        (Field(r1, f1), Field(r2, f2)) => f1 == f2 && exprs_eq(r1, r2),
        (StaticField(t1, f1), StaticField(t2, f2)) => t1 == t2 && f1 == f2,
        (RawDeref(b1, o1, w1), RawDeref(b2, o2, w2)) => o1 == o2 && w1 == w2 && exprs_eq(b1, b2),
        (RawAddr(b1, o1), RawAddr(b2, o2)) => o1 == o2 && exprs_eq(b1, b2),
        (AddressOf(x), AddressOf(y)) => exprs_eq(x, y),
        (Cond(c1, x1, y1), Cond(c2, x2, y2)) => {
            exprs_eq(c1, c2) && exprs_eq(x1, x2) && exprs_eq(y1, y2)
        }
        (Cast(t1, x), Cast(t2, y)) => t1 == t2 && exprs_eq(x, y),
        (
            Call { recv: r1, name: n1, args: a1 },
            Call { recv: r2, name: n2, args: a2 },
        ) => n1 == n2 && args_eq(a1, a2) && match (r1, r2) {
            (Some(x), Some(y)) => exprs_eq(x, y),
            (None, None) => true,
            _ => false,
        },
        (NewObj(t1, a1), NewObj(t2, a2)) => t1 == t2 && args_eq(a1, a2),
        (NewArr(t1, a1), NewArr(t2, a2)) => t1 == t2 && args_eq(a1, a2),
        (RawCall(u1, a1, r1), RawCall(u2, a2, r2)) => u1 == u2 && args_eq(a1, a2) && r1 == r2,
        _ => false,
    }
}

fn args_eq(a: &[Expr], b: &[Expr]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| exprs_eq(x, y))
}

/// `b` is the logical negation of `a` (Cmp inverted kinds, or a Not wrapper).
fn expr_is_complement(a: &Expr, b: &Expr) -> bool {
    use Expr::*;
    if let (Cmp(k1, l1, r1), Cmp(k2, l2, r2)) = (a, b) {
        return std::mem::discriminant(k2)
            == std::mem::discriminant(&(*k1).invert())
            && exprs_eq(l1, l2)
            && exprs_eq(r1, r2);
    }
    if let Not(inner) = b {
        return exprs_eq(a, inner);
    }
    if let Not(inner) = a {
        return exprs_eq(b, inner);
    }
    false
}

/// Does evaluating `e` have side effects (calls, allocation, await)?
fn expr_effectful(e: &Expr) -> bool {
    use Expr::*;
    match e {
        Call { recv, args, .. } => {
            recv.as_deref().is_some_and(expr_effectful) || args.iter().any(expr_effectful)
        }
        RawCall(..) | NewObj(..) | NewArr(..) | Await(_) => true,
        Bin(_, x, y) | Cmp(_, x, y) => expr_effectful(x) || expr_effectful(y),
        Not(x) | Neg(x) | Cast(_, x) => expr_effectful(x),
        Field(r, _) | RawDeref(r, _, _) | RawAddr(r, _) | AddressOf(r) => expr_effectful(r),
        Cond(c, x, y) => expr_effectful(c) || expr_effectful(x) || expr_effectful(y),
        // A lambda literal creates a delegate but evaluates nothing in the
        // source (`x => x.Value` is only the method group's shape); the body
        // runs when the delegate is invoked, so the literal itself is
        // side-effect-free and a guard over it can fold.
        Lambda { .. } => false,
        Var(_) | This | Int(_) | Float(_) | Double(_) | Bool(_) | Str(_) | Null | Unknown
        | MissingAbiArg | StaticField(..) | MethodGroup(_) => false,
    }
}

/// Flatten `if (c) { if (¬c) { X } }` artifacts to `X` (see module docs).
/// Bottom-up; fires only for effect-free payloads.
fn fold_impossible_guards(stmts: &mut Vec<Stmt>) {
    // recurse into children first (innermost artifacts fold first)
    for s in stmts.iter_mut() {
        match s {
            Stmt::If { then_body, else_body, .. } => {
                fold_impossible_guards(then_body);
                fold_impossible_guards(else_body);
            }
            Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => fold_impossible_guards(body),
            Stmt::Try { try_body, catches, finally_body } => {
                fold_impossible_guards(try_body);
                for (_, b) in catches.iter_mut() {
                    fold_impossible_guards(b);
                }
                if let Some(f) = finally_body {
                    fold_impossible_guards(f);
                }
            }
            _ => {}
        }
    }
    let mut i = 0;
    while i < stmts.len() {
        let outer = match &stmts[i] {
            Stmt::If { cond, then_body, else_body }
                if then_body.len() == 1 && else_body.is_empty() =>
            {
                (cond.clone(), then_body[0].clone())
            }
            _ => {
                i += 1;
                continue;
            }
        };
        let (c1, inner) = outer;
        let Stmt::If { cond: c2, then_body: x, else_body: e2 } = &inner else {
            i += 1;
            continue;
        };
        if !e2.is_empty() || !expr_is_complement(&c1, c2) {
            i += 1;
            continue;
        }
        let pure = x.iter().all(|s| match s {
            Stmt::Comment(_) | Stmt::Break | Stmt::Continue => true,
            Stmt::Return(None) => true,
            Stmt::Return(Some(e)) => !expr_effectful(e),
            Stmt::Assign(t, _, e) => !t.contains('.') && !t.contains('[') && !expr_effectful(e),
            _ => false,
        });
        if !pure {
            i += 1;
            continue;
        }
        stmts.splice(i..=i, x.clone());
        // do not advance: spliced content may itself complete an artifact
    }
}

/// Facts about a local that an enclosing guard already established.
///
/// `vN` has ONE definition (SSA), so a comparison `vN <op> C` holds or fails
/// for the whole region its guard dominates. Tracking what a guard adds is
/// therefore exact, not a heuristic.
#[derive(Clone, Default)]
struct GuardFacts {
    /// `v == n` established by an enclosing `if (v == n)`.
    eq: HashMap<String, i64>,
    /// `v != n` values established by enclosing guards (and `if (v)` → `v != 0`).
    ne: HashMap<String, HashSet<i64>>,
}

impl GuardFacts {
    fn retire(&mut self, v: &str) {
        self.eq.remove(v);
        self.ne.remove(v);
    }
}

/// The simple identifier an assignment target names, if any.
fn write_var(t: &str) -> Option<String> {
    let mut ch = t.chars();
    let first = ch.next()?;
    if first.is_ascii_alphabetic() && ch.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Some(t.to_string())
    } else {
        None
    }
}

fn cmp_holds(k: CmpKind, a: i64, b: i64) -> bool {
    match k {
        CmpKind::Eq => a == b,
        CmpKind::Ne => a != b,
        CmpKind::Lt => a < b,
        CmpKind::Gt => a > b,
        CmpKind::Le => a <= b,
        CmpKind::Ge => a >= b,
    }
}

/// Is `name` an SSA local (`v12`, `v4294967293`)? Only those may carry a
/// fact: a NAMED operand (`notation`, `in_x25`, `this`) can be the target of
/// the renderer's copy-inlining, so two printed comparisons on the same name
/// need not be two reads of one SSA value.
fn is_ssa_var(name: &str) -> bool {
    let mut ch = name.chars();
    ch.next() == Some('v') && name.len() > 1 && ch.all(|c| c.is_ascii_digit())
}

/// A comparison of the form `var <op> const`, with a flipped operand order
/// normalised so the variable stays on the left.
fn cmp_var_const(e: &Expr) -> Option<(String, CmpKind, i64)> {
    let Expr::Cmp(k, l, r) = e else { return None };
    let (v, k, n) = match (&**l, &**r) {
        (Expr::Var(v), Expr::Int(n)) => (v.clone(), *k, *n),
        (Expr::Int(n), Expr::Var(v)) => (v.clone(), k.invert(), *n),
        _ => return None,
    };
    if !is_ssa_var(&v) {
        return None;
    }
    Some((v, k, n))
}

fn apply_known(v: &str, k: CmpKind, n: i64, f: &GuardFacts) -> Option<bool> {
    if let Some(&m) = f.eq.get(v) {
        return Some(cmp_holds(k, m, n));
    }
    if matches!(k, CmpKind::Eq | CmpKind::Ne) && f.ne.get(v).is_some_and(|s| s.contains(&n)) {
        // `v != n` established: `v == n` is false and `v != n` is true.
        // (Returning false for both deleted the LIVE arm of a nested
        // same-polarity guard — 0x1EB2F30 lost a 32-line copy loop that way
        // once the guard-chain fold nested it under its superset check.)
        return Some(matches!(k, CmpKind::Ne));
    }
    None
}

/// Evaluate a condition under the facts an enclosing guard established.
/// `None` means the condition is not decidable from those facts alone.
fn eval_guard(e: &Expr, f: &GuardFacts) -> Option<bool> {
    eval_guard_at(e, f, 0)
}

fn eval_guard_at(e: &Expr, f: &GuardFacts, depth: usize) -> Option<bool> {
    if depth > 64 {
        return None;
    }
    match e {
        // Deliberately NOT `Bool(_)` and NOT `&&`/`||`: the structurer FUSES a
        // flat guard chain into `if (a || b)` and emits `if (c || true)` when
        // one edge is a carried goto. That literal `true` is a rendering
        // artifact, not the native condition — folding `X || true` as
        // constant-true deleted the (live) fall-through arm of
        // `CreateProvider` @0x1BFAB28. Only a plain comparison decides.
        Expr::Not(inner) => eval_guard_at(inner, f, depth + 1).map(|v| !v),
        Expr::Var(v) if is_ssa_var(v) => apply_known(v, CmpKind::Ne, 0, f),
        Expr::Cmp(..) => {
            let (v, k, n) = cmp_var_const(e)?;
            apply_known(&v, k, n, f)
        }
        _ => None,
    }
}

/// Facts a condition adds when it is TAKEN (`negate = false`) or when its
/// else-arm is entered (`negate = true`).
fn extend_facts(cond: &Expr, base: &GuardFacts, negate: bool) -> GuardFacts {
    let mut f = base.clone();
    let c = if negate { invert(cond) } else { cond.clone() };
    collect_facts(&c, &mut f);
    f
}

fn collect_facts(e: &Expr, f: &mut GuardFacts) {
    collect_facts_at(e, f, 0)
}

fn collect_facts_at(e: &Expr, f: &mut GuardFacts, depth: usize) {
    if depth > 64 {
        return;
    }
    match e {
        Expr::Var(v) if is_ssa_var(v) => {
            f.ne.entry(v.clone()).or_default().insert(0);
        }
        Expr::Cmp(..) => {
            if let Some((v, k, n)) = cmp_var_const(e) {
                match k {
                    CmpKind::Eq => {
                        f.eq.insert(v, n);
                    }
                    CmpKind::Ne => {
                        f.ne.entry(v).or_default().insert(n);
                    }
                    _ => {}
                }
            }
        }
        Expr::Not(inner) => collect_facts_at(&invert(inner), f, depth + 1),
        Expr::Bin(BinOp::AndAlso, a, b) => {
            collect_facts_at(a, f, depth + 1);
            collect_facts_at(b, f, depth + 1);
        }
        _ => {}
    }
}

/// Fold a guard an enclosing guard has already decided.
///
/// IL2CPP emits a redundant managed null-check on paths where the object was
/// already proven non-null (`ldr x0,[…]; cbz x0,…` a second time), and
/// mirroring both produced impossible scopes — `if (v0 != 0) { … if (v0 == 0)
/// goto L17; … }` — which read as a decompiler defect even though each check
/// really is in the bytes. The enclosing guard is the SUPERSET check; once it
/// is taken the nested one is constant, and folding it away leaves a guard a
/// reader can believe.
///
/// An arm is discarded only when no `goto` OUTSIDE it targets a label it
/// defines — deleting a `goto` is always safe, deleting a `label:` is not.
/// `all_refs` is every `goto` target in the method; the dead arm's own refs
/// are subtracted, so a label reachable only from inside the dead arm is
/// dropped with it. Anything ambiguous is left as emitted.
pub(crate) fn fold_decided_guards(stmts: &mut Vec<Stmt>) {
    let mut all_refs: HashSet<String> = HashSet::new();
    collect_gotos(stmts, &mut all_refs);
    fold_decided_guards_inner(stmts, &GuardFacts::default(), 0, &all_refs);
}

fn fold_decided_guards_inner(
    stmts: &mut Vec<Stmt>,
    facts: &GuardFacts,
    depth: usize,
    all_refs: &HashSet<String>,
) {
    // Statement trees can carry very deep flat-guard chains; folding is a
    // nicety, never worth a stack overflow.
    if depth > 256 {
        return;
    }
    let mut facts = facts.clone();
    let mut i = 0;
    while i < stmts.len() {
        if let Stmt::Assign(t, _, _) = &stmts[i] {
            if let Some(v) = write_var(t) {
                facts.retire(&v);
            }
        }
        let decision = match &stmts[i] {
            Stmt::If { cond, .. } => eval_guard(cond, &facts),
            _ => None,
        };
        let Some(decided) = decision else {
            match &mut stmts[i] {
                Stmt::If { cond, then_body, else_body } => {
                    let cond = cond.clone();
                    let t = extend_facts(&cond, &facts, false);
                    let e = extend_facts(&cond, &facts, true);
                    fold_decided_guards_inner(then_body, &t, depth + 1, all_refs);
                    fold_decided_guards_inner(else_body, &e, depth + 1, all_refs);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                    fold_decided_guards_inner(body, &facts, depth + 1, all_refs)
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    fold_decided_guards_inner(try_body, &facts, depth + 1, all_refs);
                    for (_, b) in catches.iter_mut() {
                        fold_decided_guards_inner(b, &facts, depth + 1, all_refs);
                    }
                    if let Some(f) = finally_body {
                        fold_decided_guards_inner(f, &facts, depth + 1, all_refs);
                    }
                }
                _ => {}
            }
            i += 1;
            continue;
        };
        // `decided == true`  → the then-arm is taken, the else-arm is dead.
        // `decided == false` → the else-arm is taken, the then-arm is dead.
        let dead_unsafe = {
            let Stmt::If { then_body, else_body, .. } = &stmts[i] else { unreachable!() };
            let dead_arm: &Vec<Stmt> = if decided { else_body } else { then_body };
            let mut dead_labels = HashSet::new();
            collect_labels(dead_arm, &mut dead_labels);
            if dead_labels.is_empty() {
                false
            } else {
                let mut dead_refs = HashSet::new();
                collect_gotos(dead_arm, &mut dead_refs);
                dead_labels
                    .iter()
                    .any(|l| all_refs.contains(l) && !dead_refs.contains(l))
            }
        };
        let Stmt::If { cond, then_body, else_body } = &mut stmts[i] else { unreachable!() };
        if dead_unsafe {
            // A `goto` outside the dead arm targets a label it defines. Leave
            // the shell rather than orphan that goto, but still fold nested
            // guards so the contradiction inside eventually folds.
            let cond = cond.clone();
            let t = extend_facts(&cond, &facts, false);
            let e = extend_facts(&cond, &facts, true);
            fold_decided_guards_inner(then_body, &t, depth + 1, all_refs);
            fold_decided_guards_inner(else_body, &e, depth + 1, all_refs);
            i += 1;
            continue;
        }
        let nested = extend_facts(cond, &facts, !decided);
        let mut payload = if decided {
            std::mem::take(then_body)
        } else {
            std::mem::take(else_body)
        };
        fold_decided_guards_inner(&mut payload, &nested, depth + 1, all_refs);
        if payload.is_empty() {
            // The whole guard was decided and both arms are gone: drop it.
            stmts.remove(i);
        } else {
            let n = payload.len();
            stmts.splice(i..=i, payload);
            i += n;
        }
    }
}

/// Event-accessor CAS-retry idiom → the single store the C# source was.
///
/// Roslyn + IL2CPP compile `[CompilerGenerated]` event add/remove accessors
/// into a lock-free CAS retry loop: `Delegate.Combine/Remove(field, value)`
/// plus a compare-exchange write-back of the same field through a
/// generic-sharing runtime thunk. The thunk's result register is never
/// modeled, so the loop's real exit test degrades and the whole loop renders
/// as a bogus `while (Delegate.Combine(this.Evt, v) != 0) { }` shell — with
/// the store itself silently missing (the audit's clean-but-dropped family:
/// output looks structured, yet the method's only real operation is gone).
///
/// Single-threaded pseudo-C# cannot express atomics: for the accessor
/// contract the retry loop IS the plain store `this.Evt = Delegate.Combine(
/// this.Evt, value)`, so the whole loop collapses to that assignment.
///
/// Safety gates (a failed match leaves the low-level output untouched):
///  * the method is accessor-named (add_/remove_);
///  * its ENTIRE body is [plain vN copies] + exactly ONE such loop — real
///    logic (custom accessors) breaks the shape and is preserved;
///  * the loop body is CAS-pure (only the combine call, raw runtime thunks
///    and null-guards — never returns/throws/allocations/managed calls);
///  * the Combine/Remove first argument is a real field access on `this` (or
///    a static field) — never a raw local (the corrupted static-accessor
///    variants where both arguments collapsed into one register stay as-is);
///  * the two arguments are structurally distinct (same guard).
pub(crate) fn collapse_event_cas_accessors(stmts: &mut Vec<Stmt>, method_name: &str) {
    let is_accessor = method_name.starts_with("add_") || method_name.starts_with("remove_");
    if !is_accessor {
        return;
    }
    // Top-level single-def map: vN -> its RHS expression. Collects plain
    // copies AND resolved field reads (`v32 = this.Evt`) — with field
    // resolution active the pre-loop `current = this.Evt` load is a field
    // read, not a copy, and the retry loop's Combine arg is that vN.
    let mut defs: HashMap<String, Expr> = HashMap::new();
    let mut while_idx: Option<usize> = None;
    for (i, s) in stmts.iter().enumerate() {
        match s {
            Stmt::While { .. } => {
                if while_idx.is_some() {
                    return; // more than one loop → not the accessor shape
                }
                while_idx = Some(i);
            }
            Stmt::Assign(n, None, Expr::Var(src)) if n.starts_with('v') => {
                defs.insert(n.clone(), Expr::Var(src.clone()));
            }
            Stmt::Assign(n, None, e @ Expr::Field(..)) | Stmt::Assign(n, None, e @ Expr::StaticField(..))
                if n.starts_with('v') =>
            {
                defs.insert(n.clone(), e.clone());
            }
            // Class-init guard + init sequence: every generated accessor opens
            // with `if ((*(byte*)(0x63a1000 +K) & 1) != 0) goto L0;`, the
            // runtime-metadata init call, and the flag store. Rejecting those
            // (`_ => return`) meant the collapse never fired on ANY Unity
            // accessor — 242 `while (Delegate.Combine(...) != 0)` shells
            // stayed in the text with the store they replace still missing.
            //
            // Accepting them cannot lose information: the prefix is KEPT
            // verbatim and only the one CAS loop is rewritten, and the loop
            // itself still has to pass the Combine/Remove + field-receiver +
            // cas-purity gates below. `accessor_init_stmt` admits only
            // runtime/unnameable work (raw helper calls, raw memory stores,
            // copies, bit-test guards) — never a resolved managed call with
            // recoverable semantics, which is what "custom logic" looks like.
            s if accessor_init_stmt(s) => {}
            _ => return, // real logic present → custom accessor, leave alone
        }
    }
    let Some(wi) = while_idx else { return };
    let Stmt::While { cond, body } = &stmts[wi] else { return };
    let Some((op, a0, a1)) = find_delegate_combine_call(cond) else { return };
    if !body.iter().all(stmt_cas_pure) {
        return;
    }
    // Resolve the Combine's first arg to its terminal expression: when the
    // field load was resolved it is `v32`, whose single def is the field
    // read `v32 = this.Evt`. Chase copies then substitute the def. Only
    // chase single-def names — multiply-defined names are real loop-carried
    // phis and must keep their identity.
    let chase_to = |e: &Expr| -> Expr {
        match e {
            Expr::Var(n) if n.starts_with('v') => {
                let mut cur = n.clone();
                for _ in 0..8 {
                    match defs.get(&cur) {
                        Some(Expr::Var(next)) => cur = next.clone(),
                        Some(other) => return other.clone(),
                        None => break,
                    }
                }
                Expr::Var(cur)
            }
            _ => e.clone(),
        }
    };
    let a0 = chase_to(&a0);
    // Field receiver must be `this`/static — never an unresolved local.
    let mut lhs = String::new();
    let field_ok = match &a0 {
        Expr::Field(base, _) => match base.as_ref() {
            Expr::This => true,
            Expr::Var(v) => v == "this",
            _ => false,
        },
        Expr::StaticField(..) => true,
        _ => false,
    };
    if !field_ok {
        return;
    }
    a0.render(&mut lhs);
    // Two genuinely different values (field vs handler) must be present —
    // identical args mean register collapse (corrupted), not a real call.
    if exprs_eq(&a0, &a1) {
        return;
    }
    // Value is loop-invariant; follow a straight-line vN copy chain to its
    // terminal name so the store reads the source, not a dead copy.
    let arg1 = match &a1 {
        Expr::Var(n) if n.starts_with('v') => {
            let mut single: HashMap<&str, &str> = HashMap::new();
            for (k, v) in &defs {
                if let Expr::Var(src) = v {
                    single.insert(k.as_str(), src.as_str());
                }
            }
            Expr::Var(chase_copy_chain(&single, n))
        }
        _ => a1,
    };
    let call = Expr::Call {
        recv: None,
        name: op,
        args: vec![a0, arg1],
    };
    // Keep the loop body's EFFECTFUL statements. The body's compare-exchange
    // thunk and the store it performs are really in the native code, and
    // dropping them with the loop made the collapse cost visible native calls
    // (measured: 52 `while (Delegate.Combine…)` shells collapsed in a
    // 4000-method sample, and `calls visible in C#` fell by 156). Only pure
    // loop bookkeeping — the copies and the null-guard arms that carry no
    // call — is dropped, since those exist only to run the loop that is no
    // longer there.
    let kept: Vec<Stmt> = body.iter().filter_map(cas_body_effectful).collect();
    let mut replacement = kept;
    replacement.push(Stmt::Assign(lhs, None, call));
    stmts.splice(wi..=wi, replacement);
    // Remove the now-dead straight-line copies and pre-loop field reads
    // (`v32 = this.Evt` — the CAS value) whose target no statement
    // references after the loop collapsed.
    sweep_dead_copies(stmts);
}

/// Keep the statements of a collapsed CAS-retry body that carry a real call,
/// dropping pure bookkeeping. Recurses through guards so a compare-exchange
/// hidden behind `if (v44 != 0) { … }` survives with its test intact.
fn cas_body_effectful(s: &Stmt) -> Option<Stmt> {
    match s {
        Stmt::Eval(e) if cleanup::expr_has_side_effect(e) => Some(s.clone()),
        Stmt::Assign(_, _, e) if cleanup::expr_has_side_effect(e) => Some(s.clone()),
        Stmt::If { cond, then_body, else_body } => {
            let t: Vec<Stmt> = then_body.iter().filter_map(cas_body_effectful).collect();
            let e: Vec<Stmt> = else_body.iter().filter_map(cas_body_effectful).collect();
            if t.is_empty() && e.is_empty() {
                None
            } else {
                Some(Stmt::If { cond: cond.clone(), then_body: t, else_body: e })
            }
        }
        _ => None,
    }
}

/// Find a static `Delegate.Combine`/`Delegate.Remove` call inside a branch
/// condition (through any compare/not/negate/deref wrappers).
fn find_delegate_combine_call(e: &Expr) -> Option<(String, Expr, Expr)> {
    match e {
        Expr::Call { recv: None, name, args } => {
            let nm = name.as_str();
            if (nm == "Delegate.Combine" || nm == "Delegate.Remove") && args.len() >= 2 {
                Some((nm.to_string(), args[0].clone(), args[1].clone()))
            } else {
                None
            }
        }
        Expr::Cmp(_, a, b) | Expr::Bin(_, a, b) => {
            find_delegate_combine_call(a).or_else(|| find_delegate_combine_call(b))
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::RawDeref(a, _, _)
        | Expr::RawAddr(a, _) | Expr::AddressOf(a) => {
            find_delegate_combine_call(a)
        }
        _ => None,
    }
}

/// Statements that may appear OUTSIDE the CAS loop of a generated event
/// accessor: the class-init guard/init sequence and the loop's own preamble.
///
/// Deliberately narrow. It admits the class-init `if (<flag> & 1) goto L;`
/// guard, raw runtime/unnameable work (a `fn_...` helper result stored into a
/// local, a raw `*(byte*)(addr) = 1` flag store, a `RawDeref`/`RawAddr`
/// expression), plain copies and comments. A resolved managed call, an
/// allocation, a return or a throw is NOT admitted: that is the shape of a
/// hand-written accessor, which must be left untouched.
fn accessor_init_stmt(s: &Stmt) -> bool {
    match s {
        Stmt::Comment(_) | Stmt::Break | Stmt::Continue | Stmt::Goto(_) | Stmt::Label(_) => true,
        // `if (<bit test>) goto L;` — the metadata-init guard.
        Stmt::If { cond, then_body, else_body } => {
            else_body.is_empty()
                && then_body
                    .iter()
                    .all(|s| matches!(s, Stmt::Goto(_) | Stmt::Label(_) | Stmt::Comment(_)))
                && stmt_init_expr_ok(cond)
        }
        // `v27 = fn_4a58960(0x60429f8, value);` — runtime init call result.
        Stmt::Assign(_, None, e) => stmt_init_expr_ok(e),
        // `*(byte*)(0x63a1000 +2216) = 1;` — flag store to raw memory.
        Stmt::Assign(t, _, _) => t.starts_with("*(") || t.starts_with("(&"),
        Stmt::Eval(e) => stmt_init_expr_ok(e),
        _ => false,
    }
}

/// Expression form admitted inside the accessor init prefix.
fn stmt_init_expr_ok(e: &Expr) -> bool {
    match e {
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Float(_) | Expr::Double(_)
        | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::Unknown | Expr::MissingAbiArg
        | Expr::StaticField(..) | Expr::Field(..) => true,
        Expr::RawCall(..) => true,
        Expr::RawDeref(b, _, _) | Expr::RawAddr(b, _) | Expr::AddressOf(b) => stmt_init_expr_ok(b),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => stmt_init_expr_ok(a) && stmt_init_expr_ok(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) => stmt_init_expr_ok(a),
        _ => false,
    }
}

/// CAS-retry loop bodies may only hold the combine call itself, raw runtime
/// thunks, plain copies and null-guards — never returns, throws,
/// allocations, awaits or resolved managed calls (custom logic would be
/// dropped by the collapse, so anything else vetoes it).
fn stmt_cas_pure(s: &Stmt) -> bool {
    match s {
        Stmt::Eval(e) => expr_cas_pure(e),
        Stmt::Assign(t, _, e) => !t.contains('.') && !t.contains('[') && expr_cas_pure(e),
        Stmt::If { cond, then_body, else_body } => {
            expr_cas_pure(cond)
                && then_body.iter().all(stmt_cas_pure)
                && else_body.iter().all(stmt_cas_pure)
        }
        Stmt::Comment(_) | Stmt::Break | Stmt::Continue => true,
        _ => false,
    }
}

fn expr_cas_pure(e: &Expr) -> bool {
    match e {
        Expr::Call { name, args, .. } => {
            if name == "Delegate.Combine" || name == "Delegate.Remove" {
                args.iter().all(expr_cas_pure)
            } else {
                false // any other managed call = custom logic
            }
        }
        Expr::RawCall(_, args, _) => args.iter().all(expr_cas_pure),
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => expr_cas_pure(a) && expr_cas_pure(b),
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Field(a, _)
        | Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) | Expr::AddressOf(a) => expr_cas_pure(a),
        Expr::Var(_) | Expr::This | Expr::Int(_) | Expr::Float(_) | Expr::Double(_)
        | Expr::Bool(_) | Expr::Str(_) | Expr::Null | Expr::Unknown | Expr::MissingAbiArg
        | Expr::StaticField(..) => true,
        _ => false,
    }
}

/// Follow a straight-line `vX = vY` copy chain to its terminal name
/// (bounded — the CAS value is a loop-invariant argument, so the chain is
/// short or absent).
fn chase_copy_chain(defined: &HashMap<&str, &str>, name: &str) -> String {
    let mut cur = name;
    for _ in 0..8 {
        match defined.get(cur) {
            Some(next) => cur = next,
            None => break,
        }
    }
    cur.to_string()
}

/// Drop top-level `vN = vM` copies and single-read `vN = this.field` loads
/// whose target no statement references (dead after the CAS loop collapse
/// removed their only use).
fn sweep_dead_copies(stmts: &mut Vec<Stmt>) {
    let mut used: HashSet<String> = HashSet::new();
    for s in stmts.iter() {
        collect_used_vars_stmt(s, &mut used);
    }
    stmts.retain(|s| {
        !matches!(s, Stmt::Assign(n, None, Expr::Var(_)) if n.starts_with('v') && !used.contains(n))
            && !matches!(s, Stmt::Assign(n, None, Expr::Field(..)) if n.starts_with('v') && !used.contains(n))
            && !matches!(s, Stmt::Assign(n, None, Expr::StaticField(..)) if n.starts_with('v') && !used.contains(n))
    });
}

/// Collect the variable names an expression READS (a copy's LHS is a def,
/// so it never keeps itself alive — only RHS/condition reads count).
fn collect_used_vars(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Var(n) => {
            out.insert(n.clone());
        }
        Expr::Bin(_, a, b) | Expr::Cmp(_, a, b) => {
            collect_used_vars(a, out);
            collect_used_vars(b, out);
        }
        Expr::Not(a) | Expr::Neg(a) | Expr::Cast(_, a) | Expr::Field(a, _)
        | Expr::RawDeref(a, _, _) | Expr::RawAddr(a, _) | Expr::AddressOf(a) => {
            collect_used_vars(a, out)
        }
        Expr::Cond(a, b, c) => {
            collect_used_vars(a, out);
            collect_used_vars(b, out);
            collect_used_vars(c, out);
        }
        Expr::Call { recv, name: _, args } => {
            if let Some(r) = recv {
                collect_used_vars(r, out);
            }
            for a in args {
                collect_used_vars(a, out);
            }
        }
        Expr::RawCall(_, args, _) | Expr::NewObj(_, args) | Expr::NewArr(_, args) => {
            for a in args {
                collect_used_vars(a, out);
            }
        }
        Expr::Await(a) => collect_used_vars(a, out),
        _ => {}
    }
}

fn collect_used_vars_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Eval(e) => collect_used_vars(e, out),
        Stmt::Assign(_, _, e) => collect_used_vars(e, out), // RHS only
        Stmt::If { cond, then_body, else_body } => {
            collect_used_vars(cond, out);
            for x in then_body.iter().chain(else_body) {
                collect_used_vars_stmt(x, out);
            }
        }
        Stmt::While { cond, body } | Stmt::DoWhile { cond, body } => {
            collect_used_vars(cond, out);
            for x in body {
                collect_used_vars_stmt(x, out);
            }
        }
        Stmt::Return(e) => {
            if let Some(e) = e {
                collect_used_vars(e, out);
            }
        }
        Stmt::Throw(e) => collect_used_vars(e, out),
        Stmt::Try { try_body, catches, finally_body } => {
            for x in try_body {
                collect_used_vars_stmt(x, out);
            }
            for (_, b) in catches {
                for x in b {
                    collect_used_vars_stmt(x, out);
                }
            }
            if let Some(f) = finally_body {
                for x in f {
                    collect_used_vars_stmt(x, out);
                }
            }
        }
        Stmt::ForEach { collection, body, .. } => {
            for x in body {
                collect_used_vars_stmt(x, out);
            }
            let _ = collection;
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Undefined-local attribution (ILR_UNDEFDBG=path)
// ---------------------------------------------------------------------------
// `scripts/audit_dump.py` counts every `vN` a rendered body MENTIONS but never
// ASSIGNS to: a name with no value, which is the largest single defect class
// visible in the text. That count says HOW MANY. It does not say WHY, and the
// fix is different for every cause below, so this hook joins the emitted text
// back to the local table and writes one TSV record per (method, local):
//
//   * `abs`          — the absolute-image base pseudo-register. `*(byte*)(v0 +
//                      29061866)` is a static-field access whose BASE is the
//                      module image rather than a local: the address is real,
//                      only the name is missing.
//   * `stack`        — a frame cell (`slot_origin` `stack_N`). The body stores
//                      through it all method long and never assigns it, because
//                      it names MEMORY, not a value.
//   * `x0`           — the receiver / first-argument register, deliberately
//                      excluded from `entry_register_name` because
//                      `entry_var_name` handles the declared parameter first.
//                      A leftover `x0` origin is a scratch reuse of it.
//   * `this`         — the local IS the receiver, but it was REDEFINED, so it
//                      must not be printed as the immutable C# `this`.
//   * `param-noname` — the local IS a parameter, but its metadata name is not a
//                      safe identifier, so no display name could be built.
//   * `def-dropped`  — the IR DOES define the local, yet no statement that
//                      assigned the name survived into the text. This is a real
//                      lost definiton, not just a missing name.
//   * `no-def`       — nothing defines it and its origin is not a recognised
//                      incoming ABI register: also a real loss.
//
// The counter is a byte-exact mirror of `audit_dump.py`'s (`V_LOCAL`, `V_DEF`,
// `STRING_LITERAL`), including its per-LINE def set, so the per-record `uses`
// column SUMS to that harness's `undefined locals (v*)` total. Without that,
// the attribution could not be checked against the headline number.

/// Remove C# string literals so a literal cannot masquerade as a local name.
/// Mirrors `STRING_LITERAL` in `scripts/audit_dump.py` exactly, including its
/// treatment of an interior `"` in a char literal (`'"'` starts a "string").
fn undef_dbg_strip_literals(line: &str) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(b.len());
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'"' {
            out.push_str("\"\"");
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            // consume the closing quote when there is one
            if i < b.len() {
                i += 1;
            }
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Every `vN` token on one (literal-stripped) line, each flagged as a DEF when
/// the regex `V_DEF` would match it (`vN =` / `vN +=` / `vN <<=`, never `vN ==`
/// and never `vN <=`).
fn undef_dbg_scan_tokens(code: &str) -> Vec<(u32, bool)> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'v' {
            let prev_ok = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let next_ok = j >= b.len() || !(b[j].is_ascii_alphanumeric() || b[j] == b'_');
            if prev_ok && j > i + 1 && next_ok {
                let n: u32 = code[i + 1..j].parse().unwrap_or(u32::MAX);
                let mut k = j;
                while k < b.len() && b[k].is_ascii_whitespace() {
                    k += 1;
                }
                let is_def = if k < b.len() {
                    if b[k] == b'=' {
                        k + 1 >= b.len() || b[k + 1] != b'='
                    } else if matches!(b[k], b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^') {
                        k + 1 < b.len() && b[k + 1] == b'='
                    } else if b[k] == b'<' || b[k] == b'>' {
                        k + 2 < b.len() && b[k + 1] == b[k] && b[k + 2] == b'='
                    } else {
                        false
                    }
                } else {
                    false
                };
                out.push((n, is_def));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// See the block comment above: one TSV record per undefined `vN`, with the
/// reason its origin was not rendered as a name.
fn undef_dbg_records(
    table: &super::locals::LocalTable,
    insns: &[Insn],
    rva: u64,
    body: &str,
) {
    use std::collections::{HashMap, HashSet};
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static F: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
    let Ok(path) = std::env::var("ILR_UNDEFDBG") else { return };
    let file = F
        .get_or_init(|| Mutex::new(std::fs::File::create(&path).expect("ILR_UNDEFDBG file")));

    // Locals the IR itself DEFINES. A local in this set whose name still reads
    // undefined in the text lost its assignment STATEMENT, which is a far more
    // serious defect than a register the renderer declined to name.
    let mut ir_defines: HashSet<usize> = HashSet::new();
    for insn in insns {
        if let Some(pos) = Insn::dest_position(&insn.op) {
            if let Some(Operand::Local(v)) = insn.operands.get(pos) {
                if let Some(&li) = table.index_of.get(v) {
                    ir_defines.insert(li);
                }
            }
        }
    }

    let mut all_defs: HashSet<u32> = HashSet::new();
    let mut uses: HashMap<u32, usize> = HashMap::new();
    for raw in body.lines() {
        let code = undef_dbg_strip_literals(raw);
        let toks = undef_dbg_scan_tokens(&code);
        let line_defs: HashSet<u32> = toks.iter().filter(|(_, d)| *d).map(|(n, _)| *n).collect();
        for (n, d) in &toks {
            if *d {
                all_defs.insert(*n);
            } else if !line_defs.contains(n) {
                *uses.entry(*n).or_insert(0) += 1;
            }
        }
    }
    let mut undef: Vec<(u32, usize)> = uses
        .iter()
        .filter(|(n, _)| !all_defs.contains(n))
        .map(|(n, c)| (*n, *c))
        .collect();
    undef.sort_unstable();
    if undef.is_empty() {
        return;
    }
    let mut out = String::new();
    for (n, c) in undef {
        let li = n as usize;
        let info = table.infos.get(li);
        let origin = info.map(|i| i.slot_origin.as_str()).unwrap_or("?");
        let reason = match info {
            Some(i) if i.is_this => "this",
            Some(i) if i.is_param.is_some() => "param-noname",
            _ if origin == "abs" => "abs",
            _ if origin.starts_with("stack_") => "stack",
            // BEFORE the origin-specific arms: a local the IR DEFINES cannot be
            // the incoming entry value of its register, whatever that register
            // is, so its missing assignment — not its origin — is the reason it
            // reads undefined.
            _ if ir_defines.contains(&li) => "def-dropped",
            _ if origin == "x0" => "x0",
            _ => "no-def",
        };
        out.push_str(&format!("{rva:#x}\t{li}\t{origin}\t{c}\t{reason}\n"));
    }
    if let Ok(mut f) = file.lock() {
        let _ = f.write_all(out.as_bytes());
    }
}

// Fidelity audit (ILR_AUDIT=path)
// ---------------------------------------------------------------------------
// One JSONL line per rendered method: how many native instructions the
// emitted C# ACCOUNTED for vs dropped. Categories:
//   ctl      control ops (branches/returns/phis) — structure, not "logic"
//   cmp      compares (claimed only when a branch consumed them)
//   reg      register-to-local moves (ABI seeding / copies)
//   sem      real semantic ops: loads, stores, arithmetic, FP, calls,
//            conversions — ANY unclaimed one is dropped logic
// `sem_un == 0` means every data instruction of the method is represented
// in the output (the fidelity test the user asked for).

fn audit_op_cat(op: &Op) -> &'static str {
    match op {
        Op::Nop => "nop",
        Op::Compare(_) | Op::Phi | Op::Jump(_) | Op::CondJump(_) | Op::Return | Op::Throw
        | Op::IndirectCall | Op::Switch(_) => "ctl",
        Op::Move => "move",
        Op::Call | Op::CallVoid | Op::NewObj(_) | Op::NewArr(..) | Op::ObjcMsgSend { .. }
        | Op::MulAcc { .. } | Op::ShiftStack(_) | Op::Arith(_) | Op::FloatArith(_)
        | Op::Convert(_) => "sem",
        _ => "sem",
    }
}

/// ILR_AUDITDETAIL=1: print one line per UNCLAIMED semantic instruction
/// (op + operands) to stderr. Used with ILR_ONLY to classify what the
/// renderer drops for a single method without re-dumping the whole corpus.
fn audit_detail_unclaimed(rva: u64, insns: &[Insn], claims: &std::collections::HashSet<usize>) {
    if std::env::var("ILR_AUDITDETAIL").is_err() {
        return;
    }
    for (i, insn) in insns.iter().enumerate() {
        if claims.contains(&i) {
            continue;
        }
        if matches!(insn.op, Op::Nop | Op::Compare(_) | Op::Phi | Op::Jump(_) | Op::CondJump(_)
            | Op::Return | Op::Throw | Op::IndirectCall | Op::Switch(_)) {
            continue;
        }
        // register-copy moves are ABI plumbing; skip for the detail view
        if matches!(insn.op, Op::Move)
            && matches!(insn.operands.get(1), Some(Operand::Register(_))) {
            continue;
        }
        eprintln!("[audit] {:#x} unclaimed: {:?} {:?}", rva, insn.op, insn.operands);
    }
}

fn audit_write_line(rva: u64, name: &str, insns: &[Insn], claims: &std::collections::HashSet<usize>) {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static F: OnceLock<Mutex<std::fs::File>> = OnceLock::new();
    let Ok(path) = std::env::var("ILR_AUDIT") else { return };
    let file = F.get_or_init(|| {
        Mutex::new(std::fs::File::create(&path).expect("ILR_AUDIT file"))
    });
    let mut ctl = 0usize;
    let mut cmp = 0usize;
    let mut reg = 0usize; // copies/seeding (reg->local, local->local): plumbing
    let mut sem = 0usize; // loads, stores, arith, calls, conversions
    let mut sem_un = 0usize;
    let mut reg_un = 0usize;
    let mut cmp_un = 0usize;
    for (i, insn) in insns.iter().enumerate() {
        let claimed = claims.contains(&i);
        if matches!(insn.op, Op::Nop) {
            continue;
        }
        if matches!(insn.op, Op::Compare(_)) {
            cmp += 1;
            if !claimed {
                cmp_un += 1;
            }
            continue;
        }
        match audit_op_cat(&insn.op) {
            "ctl" => ctl += 1,
            "move" => {
                let (src, dst) = (insn.operands.get(1), insn.operands.get(0));
                let src_reg = matches!(src, Some(Operand::Register(_)));
                let src_loc = matches!(src, Some(Operand::Local(_)));
                let dst_loc = matches!(dst, Some(Operand::Local(_)));
                let dst_mem = matches!(dst, Some(Operand::Memory { .. }));
                if src_reg || (src_loc && dst_loc) {
                    // register seeding / local-to-local copies: ABI plumbing
                    reg += 1;
                    if !claimed {
                        reg_un += 1;
                    }
                } else {
                    // real data op: load (mem/imm -> local), store (-> mem)
                    sem += 1;
                    if !claimed {
                        sem_un += 1;
                    }
                }
                // dst_mem kept out of the copy branch: store -> mem is sem
                let _ = dst_mem;
            }
            _ => {
                sem += 1;
                if !claimed {
                    sem_un += 1;
                }
            }
        }
    }
    let name_s = name.replace('\\', "_").replace('"', "'");
    let line = format!(
        "{{\"rva\":\"{rva:#x}\",\"name\":\"{name_s}\",\"total\":{},\"ctl\":{ctl},\"cmp\":{cmp},\"reg\":{reg},\"sem\":{sem},\"sem_un\":{sem_un},\"reg_un\":{reg_un},\"cmp_un\":{cmp_un}}}\n",
        insns.len()
    );
    let _ = file.lock().unwrap().write_all(line.as_bytes());
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

/// Render a finished statement tree to text.
///
/// The production path renders through [`render_method_csharp_full_chains`],
/// which is also where every pass that builds these trees lives. Unit tests
/// that pin a *rendering* rule (rather than a structuring rule) need to hand a
/// tree straight to the renderer, so this exposes exactly that and nothing
/// else.
#[cfg(test)]
pub(crate) fn render_stmts(stmts: &[Stmt]) -> String {
    let mut r = Renderer::new();
    for s in stmts {
        r.stmt(s);
    }
    r.out
}

/// Decompile a recovered method body to C# text.
pub fn render_method_csharp(
    ir: &super::RecoveredIr,
    method_name: &str,
    is_static: bool,
    param_sigs: &[(String, String)],
    ret_type: &ManagedType,
    declaring_tdi: i32,
) -> String {
    render_method_csharp_full(ir, method_name, is_static, param_sigs, ret_type, declaring_tdi)
}

pub fn render_method_csharp_full(
    ir: &super::RecoveredIr,
    method_name: &str,
    is_static: bool,
    param_sigs: &[(String, String)],
    ret_type: &ManagedType,
    declaring_tdi: i32,
) -> String {
    render_method_csharp_bounded(ir, method_name, is_static, param_sigs, ret_type, declaring_tdi, None)
}

/// Same, with an optional wall-clock budget guaranteeing termination on
/// pathological methods.
pub fn render_method_csharp_bounded(
    ir: &super::RecoveredIr,
    method_name: &str,
    is_static: bool,
    param_sigs: &[(String, String)],
    ret_type: &ManagedType,
    declaring_tdi: i32,
    deadline: Option<std::time::Instant>,
) -> String {
    render_method_csharp_full_chains(
        ir,
        method_name,
        is_static,
        param_sigs,
        ret_type,
        declaring_tdi,
        deadline,
        &std::collections::HashMap::new(),
        &[],
    )
}

/// Assemble a fresh [`Structurer`] for one walk.
///
/// The structurer's per-walk state (`visited`, `emitted`, the statement budget)
/// lives in the struct, so a second walk needs a second structurer. The shared
/// inputs are borrowed; the loop/EH maps are cloned (they are small).
#[allow(clippy::too_many_arguments)]
fn build_structurer<'a>(
    sc: &'a StructuredCfg,
    insns: &'a [Insn],
    ctx: MethodCtx<'a>,
    deadline: Option<std::time::Instant>,
    loop_of_block: &HashMap<usize, usize>,
    eh_regions: &'a [super::exceptions::EhRegion],
    region_of_block: &HashMap<usize, usize>,
    handler_of: &HashMap<usize, usize>,
    discovery: bool,
) -> Structurer<'a> {
    // Ordinary control flow reachability from the entry. See
    // `Structurer::normal_reachable` for why the handler walk needs it.
    let normal_reachable: HashSet<usize> = {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = vec![0];
        while let Some(b) = stack.pop() {
            if b >= sc.blocks.len() || !seen.insert(b) {
                continue;
            }
            for s in [sc.blocks[b].succ_taken, sc.blocks[b].succ_fall]
                .into_iter()
                .flatten()
            {
                stack.push(s);
            }
        }
        seen
    };
    Structurer {
        sc,
        insns,
        ctx,
        deadline,
        emitted: std::cell::RefCell::new(HashSet::new()),
        emission_log: std::cell::RefCell::new(Vec::new()),
        remaining: std::cell::Cell::new(100_000),
        forced_labels: std::cell::RefCell::new(HashSet::new()),
        visited: std::cell::RefCell::new(HashSet::new()),
        loop_of_block: loop_of_block.clone(),
        eh_regions,
        region_of_block: region_of_block.clone(),
        handler_of: handler_of.clone(),
        in_handler: std::cell::Cell::new(false),
        folding: std::cell::Cell::new(None),
        folded: std::cell::RefCell::new(std::collections::HashSet::new()),
        discovery,
        in_loop_body: std::cell::Cell::new(false),
        loop_body_exit: std::cell::Cell::new(None),
        skip_loop_header: std::cell::Cell::new(None),
        flat_chain_next: std::cell::Cell::new(None),
        normal_reachable,
    }
}

/// As `render_method_csharp_bounded`, plus the declaring type's base-class
/// chain (`parent_chains[tdi]`, self first) and the binary's decoded EH
/// regions. Inherited instance fields — declared on a base type but accessed
/// from a derived method — resolve to `this.field` instead of
/// `__unknown.field`, which is exactly what the metadata knows and Ghidra
/// cannot. EH regions fold into real `try { } catch { } finally { }`
/// statements: the structurer wraps the statement span of every block whose
/// native address lies inside a try region, and renders the region's
/// landing-pad block as the handler body.
pub fn render_method_csharp_full_chains(
    ir: &super::RecoveredIr,
    method_name: &str,
    is_static: bool,
    param_sigs: &[(String, String)],
    ret_type: &ManagedType,
    declaring_tdi: i32,
    deadline: Option<std::time::Instant>,
    parent_chain: &std::collections::HashMap<i32, Vec<i32>>,
    eh_regions: &[super::exceptions::EhRegion],
) -> String {
    let t0 = std::time::Instant::now();
    let prof = std::env::var("ILR_PROF").is_ok();
    let sc = analyze_blocks(&ir.insns);
    if prof { eprintln!("[render] analyze_blocks {}us n={}", t0.elapsed().as_micros(), ir.insns.len()); }
    let loops = find_loops(&sc);
    if prof { eprintln!("[render] find_loops {}us nloops={}", t0.elapsed().as_micros(), loops.len()); }
    // Attribute the goto-shaped output to its structural cause. Once per
    // method: everything below this point may build a second `Structurer`
    // (the binding-discovery walk), and those walks see the SAME CFG.
    classify_structure(&sc, &loops, &ir.insns);
    // Blocks that belong to THIS method even though no recovered edge from the
    // entry reaches them — the function's out-of-line cold region. Computed
    // ONCE and used by both walks: the discovery walk must cover the same
    // blocks as the real one or a call that lives only in the cold region has
    // no binding and the real walk expands the same `bl` twice.
    let reachable = adjoined_blocks(&sc, &ir.insns);

    // Must agree with the header renderer (line ~3834): unknown ret_type
    // prints as "void" there, so the body must treat it as void too — a
    // native `ret`'s x0 is stale noise in that case, not a return value.
    let ret_is_void = csharp_type(ret_type).map_or(true, |t| t == "void");
    // ILR_CFG=path — dump the structured CFG + loop bodies for offline
    // debugging of structuring failures (no guessing needed).
    if let Ok(cfg_path) = std::env::var("ILR_CFG") {
        let mut dbg = String::new();
        {
            let mut vs: Vec<_> = ir.table.index_of.iter().collect();
            vs.sort();
            for (v, li) in vs {
                dbg.push_str(&format!("  SLOT v{} -> {}\n", v, ir.table.infos[*li].slot_origin));
            }
        }
        for (ii, insn) in ir.insns.iter().enumerate() {
            dbg.push_str(&format!(
                "  i{} @{:#x}: op={:?} operands={:?}\n",
                ii, insn.address, insn.op, insn.operands
            ));
        }
        dbg.push('\n');
        for (bi, b) in sc.blocks.iter().enumerate() {
            dbg.push_str(&format!(
                "B{}: insns {}..{} term={:?} taken={:?} fall={:?} idom={:?} ipdom={:?}\n",
                bi, b.start, b.end,
                ir.insns.get(b.end.saturating_sub(1)).map(|i| &i.op),
                b.succ_taken, b.succ_fall, sc.idom[bi], sc.ipdom[bi],
            ));
        }
        // Sorted by header: `loops` is a HashMap, so the unsorted dump printed
        // the same loops in a different order every run (which made diffing
        // two dumps of one method look like a structuring difference).
        let mut loop_headers: Vec<usize> = loops.keys().copied().collect();
        loop_headers.sort_unstable();
        for h in loop_headers {
            let info = &loops[&h];
            let mut body: Vec<usize> = info.body.iter().copied().collect();
            body.sort_unstable();
            dbg.push_str(&format!("loop header B{} latches={:?} body={:?}\n", h, info.latches, body));
        }
        let cfg_path_named = format!("{}.{}.txt", cfg_path.trim_end_matches(".txt"), method_name);
        let _ = std::fs::write(&cfg_path_named, dbg);
    }
    // LOOP-FIRST routing: block → enclosing loop header. Computed once so
    // the walker can hand ANY body block to emit_loop (at its header)
    // BEFORE consuming it linearly; otherwise the pre-walk marks body
    // blocks as emitted and emit_loop's body pass stops immediately —
    // the root cause of the `// loop body not recovered` artifacts.
    //
    // NESTED loops share body blocks, so this map is a CHOICE, not a union:
    // iterating the `loops` HashMap meant a block inside an inner loop could
    // be routed to the OUTER header depending on per-process hash order —
    // different nesting (or none) from run to run on the same binary. Route
    // to the INNERMOST enclosing loop: sort by body size descending so the
    // smallest (innermost) loop writes last and wins; ties break on the
    // lowest header, i.e. first in listing order.
    let mut loop_order: Vec<usize> = loops.keys().copied().collect();
    loop_order.sort_by(|a, b| {
        loops[b]
            .body
            .len()
            .cmp(&loops[a].body.len())
            .then(b.cmp(a))
    });
    let mut loop_of_block: HashMap<usize, usize> = HashMap::new();
    for h in loop_order {
        for &b in loops[&h].body.iter() {
            loop_of_block.insert(b, h);
        }
    }

    // EH folding maps: for each try region, the set of blocks whose native
    // addresses lie INSIDE the try range (the protected body), and the block
    // containing the landing pad (the handler). Region RVAs are absolute;
    // insn addresses are absolute RVAs too (lifted with base_rva).
    // block index → region ordinal covering it (a block in NESTED regions
    // maps to the innermost/last matching region, matching C# semantics).
    // A `try` with several throw-capable calls arrives as one ROW PER CALL —
    // see `merge_eh_regions`. Fuse the rows back into the try before anything
    // downstream sees them, so the structurer folds one `try` instead of N
    // sibling islands that fence its walks.
    let merged_eh = merge_eh_regions(eh_regions, &sc, &ir.insns);
    let eh_regions: &[super::exceptions::EhRegion] = &merged_eh;
    let mut try_blocks_of: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut handler_of: HashMap<usize, usize> = HashMap::new();
    for (ri, r) in eh_regions.iter().enumerate() {
        let mut tbs = Vec::new();
        for (bi, b) in sc.blocks.iter().enumerate() {
            let blk_start = ir.insns[b.start].address;
            let blk_end = ir.insns[b.end.saturating_sub(1)].address;
            if blk_start >= r.try_start && blk_start < r.try_end {
                tbs.push(bi);
            }
            let _ = blk_end;
        }
        if !tbs.is_empty() {
            try_blocks_of.push((ri, tbs));
        }
        // handler block: first block whose start contains the landing pad
        for (bi, b) in sc.blocks.iter().enumerate() {
            if ir.insns[b.start].address <= r.landing_pad
                && r.landing_pad < ir.insns[b.end.saturating_sub(1)].address.saturating_add(4)
            {
                handler_of.insert(ri, bi);
                break;
            }
        }
    }
    // block → innermost try region covering it (for the walker)
    let mut region_of_block: HashMap<usize, usize> = HashMap::new();
    for (ri, tbs) in &try_blocks_of {
        for &b in tbs {
            region_of_block.insert(b, *ri);
        }
    }

    if std::env::var("ILR_EHREGDBG").is_ok() {
        for (ri, r) in eh_regions.iter().enumerate() {
            eprintln!(
                "[ehreg] R{ri} try=[{:#x},{:#x}) len={:#x} pad={:#x} cleanup={} catches={:?}",
                r.try_start,
                r.try_end,
                r.try_end.saturating_sub(r.try_start),
                r.landing_pad,
                r.is_cleanup,
                r.catches
            );
        }
        for h in 0..sc.blocks.len() {
            if let Some(info) = loops.get(&h) {
                let mut body: Vec<usize> = info.body.iter().copied().collect();
                body.sort_unstable();
                eprintln!("[ehreg] loop hdr=B{h} body={body:?} latches={:?}", info.latches);
            }
        }
        for (bi, b) in sc.blocks.iter().enumerate() {
            let a = ir.insns[b.start].address;
            let e = if b.end > b.start {
                ir.insns[b.end - 1].address
            } else {
                a
            };
            eprintln!(
                "[ehreg] B{bi} insns {}..{} addr=[{:#x},{:#x}] t={:?} f={:?}",
                b.start, b.end, a, e, b.succ_taken, b.succ_fall
            );
        }
        for (ri, hb) in handler_of.iter() {
            eprintln!("[ehreg] handler R{ri} -> B{hb}");
        }
    }

    // DISCOVERY WALK.
    //
    // The structurer walks the CFG, not execution order, so a use site is
    // often rendered BEFORE the call that produces its value. The first walk
    // therefore records exactly which call def sites actually materialize
    // `dst = callee(…)`; the real walk below is then seeded with those
    // bindings, so every use site links to the name instead of re-expanding
    // the call expression (one native `bl` rendered once per use). The
    // discovery output is discarded — only the binding map is kept.
    let t_d = std::time::Instant::now();
    let seeded = {
        let ctx0 = MethodCtx::build(&ir.insns, &ir.table, declaring_tdi, parent_chain, ret_is_void);
        let st0 = build_structurer(
            &sc, &ir.insns, ctx0, deadline, &loop_of_block, eh_regions,
            &region_of_block, &handler_of, true,
        );
        let _ = st0.emit_region(0, &HashSet::new(), &loops, &[], 0);
        // The discovery walk must cover the SAME blocks as the real one, or a
        // call that lives only in a repaired region has no binding: the real
        // walk would then emit `dst = callee(…)` in the region AND re-expand
        // the same `bl` inline at its use site — two call expressions for one
        // native call, the duplicate-call defect. The repair's output is
        // discarded here exactly like the rest of the discovery walk's.
        let _ = st0.coverage_repair(&reachable);
        st0.ctx.bound_calls.borrow().clone()
    };
    if prof {
        eprintln!(
            "[render] discovery {}us seeded={}",
            t_d.elapsed().as_micros(),
            seeded.len()
        );
    }
    let ctx = MethodCtx::build(&ir.insns, &ir.table, declaring_tdi, parent_chain, ret_is_void);
    *ctx.preseed_bindings.borrow_mut() = seeded;
    if prof { eprintln!("[render] method_ctx {}us", t0.elapsed().as_micros()); }
    let st = build_structurer(
        &sc, &ir.insns, ctx, deadline, &loop_of_block, eh_regions,
        &region_of_block, &handler_of, false,
    );

    CALLTRACE.with(|c| c.set(std::env::var("ILR_CALLTRACE").is_ok()));
    let t_e = std::time::Instant::now();
    let (mut body_stmts, _) = st.emit_region(0, &HashSet::new(), &loops, &[], 0);
    CALLTRACE.with(|c| c.set(false));
    if prof { eprintln!("[render] emit_region {}us nstmts={}", t_e.elapsed().as_micros(), body_stmts.len()); }
    // `ILR_DEFDBG=1` (every method) or `=0xRVA` (one): for each version the IR
    // defines, what the renderer decided to do with that def, and why. Answers
    // the `def-dropped` class of the undefined-local attribution directly — a
    // local the IR defines whose assignment never reached the text.
    if let Ok(f) = std::env::var("ILR_DEFDBG") {
        let ok = match f.as_str() {
            "1" => true,
            _ => u64::from_str_radix(f.trim_start_matches("0x"), 16).ok() == Some(ir.base_rva),
        };
        if ok {
            let ctx = &st.ctx;
            let mut d = String::new();
            // Iterate the VERSIONS, not 0..len: `highvar::merge_names` maps
            // several versions onto one table index (the high-variable merge),
            // so the mapping is not injective and the two id spaces differ.
            let mut versions: Vec<u32> = ir.table.index_of.keys().copied().collect();
            versions.sort_unstable();
            for v in versions {
                let li = ir.table.index_of[&v];
                let info = &ir.table.infos[li];
                let sites: &[usize] = ctx.def_sites.get(&v).map(|s| s.as_slice()).unwrap_or(&[]);
                let reads = ctx.read_count.get(&v).copied().unwrap_or(0);
                if reads == 0 && sites.is_empty() {
                    continue;
                }
                d.push_str(&format!(
                    "v{v} idx={li} slot={} reads={reads} var_read={} multi={} sites=[",
                    info.slot_origin,
                    ctx.var_reads.contains(&v),
                    ctx.multi_def.contains(&v),
                ));
                for (k, &i) in sites.iter().enumerate() {
                    if k > 0 {
                        d.push(',');
                    }
                    d.push_str(&format!("{}:{:?}@{:#x}", i, ir.insns[i].op, ir.insns[i].address));
                }
                d.push(']');
                // Use-site audit: what `inline_local_at` would do at each read.
                let mut none = 0usize;
                let mut later = 0usize;
                let mut moved_call = 0usize;
                let mut moved = 0usize;
                let mut ok = 0usize;
                for (u, insn) in ir.insns.iter().enumerate() {
                    if !insn.use_locals().contains(&v) {
                        continue;
                    }
                    match ctx.reaching_def_at(v, u) {
                        None => none += 1,
                        Some(du) => {
                            let same_block_later = du > u
                                && ctx.block_of_insn.as_ref().is_some_and(|m| {
                                    m.get(du).copied() == m.get(u).copied()
                                });
                            if same_block_later {
                                later += 1;
                            } else if ctx.def_moved_across_a_write(du, u) {
                                if ctx.def_chain_has_call(du) {
                                    moved_call += 1;
                                } else {
                                    moved += 1;
                                }
                            } else {
                                ok += 1;
                            }
                        }
                    }
                }
                let alias = ir.table.index_of.get(&v).map(|_| {
                    pure_alias_root(&ir.insns, &ctx.def_sites, v).is_some()
                }).unwrap_or(false);
                d.push_str(&format!(
                    " uses=[none={none} later={later} moved={moved} moved_call={moved_call} ok={ok} alias={alias}]"
                ));
                if let Some(&i) = sites.first() {
                    let emitted = !ctx.value_def_stmt(i).is_empty();
                    d.push_str(&format!(" stmt_emitted={emitted}"));
                    if !emitted && ctx.var_reads.contains(&v) {
                        let e = ctx.def_expr(i, 0);
                        d.push_str(&format!(" defexpr_unknown={}", matches!(e, Expr::Unknown)));
                    }
                }
                d.push('\n');
            }
            let _ = std::fs::write(probe_dir().join(format!("defdbg.{:#x}.txt", ir.base_rva)), d);
        }
    }
    // STRUCTURER COVERAGE. Blocks the walk never entered are code the text does
    // not contain, and no IR-level check can see them: `verify` is satisfied
    // because the block is in the CFG, and the concordance count only knows
    // that FEWER conditions came out. Recording the block count attributes the
    // shortfall to the structurer and names the method.
    //
    // "Never walked" is two very different things, so the split is computed
    // once here and used for BOTH the diagnostic and the repair below:
    //   * SKIPPED — reachable from the entry, so a real loss.
    //   * CFG-UNREACHABLE — no recovered edge leads there at all (an
    //     irreducible entry, or a terminator whose dispatch was not
    //     recovered). Nothing to walk FROM.
    // `reachable` was computed at the top of this function (see
    // [`adjoined_blocks`]): it is the entry-reachable set PLUS the method's own
    // out-of-line cold region.
    if std::env::var("ILR_COV").is_ok() {
        let visited = st.visited.borrow();
        let unvis: Vec<usize> = (0..sc.blocks.len()).filter(|b| !visited.contains(b)).collect();
        if !unvis.is_empty() {
            // Three different things hide behind "unvisited", and the report must
            // not add them up as if they were one:
            //   * SKIPPED — entry-reachable, and the walk could not place it. A
            //     real structuring loss.
            //   * ADOPTED — not entry-reachable, but proven to be THIS method's
            //     out-of-line cold region (see `adjoined_blocks`). The repair
            //     emits it; it is counted here so the provenance is visible.
            //   * CFG-UNREACHABLE — the remainder: no recovered edge in, and
            //     none back out to this method's body either, so ownership
            //     cannot be proven and it is NOT adopted.
            let entry_reach = reachable_blocks_entry(&sc, &ir.insns);
            let skipped: Vec<usize> = unvis.iter().copied().filter(|&b| entry_reach[b]).collect();
            let adopted_uv = unvis.iter().filter(|&&b| !entry_reach[b] && reachable[b]).count();
            let n = skipped.len().min(10);
            let heads: Vec<String> = skipped[..n]
                .iter()
                .map(|&b| {
                    let bb = &sc.blocks[b];
                    format!(
                        "B{b}(insn {} term {:?} t={:?} f={:?})",
                        bb.start, self_op_name(&ir.insns[bb.end - 1].op), bb.succ_taken, bb.succ_fall
                    )
                })
                .collect();
            // HOW BIG IS THE CFG-UNREACHABLE PART, AND IS IT THIS METHOD'S?
            //
            // `cfg_unreachable` counts code the recovered edges do not enter.
            // Two very different things hide behind that number, and the
            // concordance counts the first against the method while the second
            // is not the method's code at all:
            //   * an out-of-line COLD region of the same function — the body
            //     ended in `ret` and the code that follows is the method's own
            //     exception landing pad / cold split, which SHARES the frame
            //     (no prologue) and may branch back INTO the body; and
            //   * a neighbouring function that simply has no metadata RVA of
            //     its own, sitting in the gap between this method's RVA and the
            //     next method's.
            // `after_ret` = the first unreachable instruction is the one right
            // after a reachable `ret`. `backedge` = an unreachable block has an
            // edge to a REACHABLE block, which only the same function can do.
            // `tail_calls` counts the native calls stranded there, so the size
            // of the concord deficit this attribute is measurable.
            let unreach: Vec<usize> = unvis.iter().copied().filter(|&b| !reachable[b]).collect();
            let tail = if unreach.is_empty() {
                String::new()
            } else {
                let first = unreach
                    .iter()
                    .map(|&b| ir.insns[sc.blocks[b].start].address)
                    .min()
                    .unwrap_or(0);
                let after_ret = sc.blocks.iter().enumerate().any(|(b, bb)| {
                    reachable[b]
                        && matches!(ir.insns[bb.end - 1].op, Op::Return)
                        && ir.insns[bb.end - 1].address + 4 == first
                });
                let backedge = unreach.iter().any(|&b| {
                    let bb = &sc.blocks[b];
                    [bb.succ_taken, bb.succ_fall]
                        .iter()
                        .flatten()
                        .any(|&s| reachable[s])
                });
                let tail_calls: usize = unreach
                    .iter()
                    .map(|&b| {
                        let bb = &sc.blocks[b];
                        (bb.start..bb.end)
                            .filter(|&i| matches!(ir.insns[i].op, Op::Call | Op::IndirectCall))
                            .count()
                    })
                    .sum();
                format!(
                    " tail_blocks={} tail_calls={} after_ret={} backedge={}",
                    unreach.len(),
                    tail_calls,
                    after_ret,
                    backedge
                )
            };
            eprintln!(
                "[cov] {:#x} blocks={} unvisited={} skipped_reachable={} adopted={} \
                 cfg_unreachable={}{} first_skipped=[{}]",
                ir.base_rva,
                sc.blocks.len(),
                unvis.len(),
                skipped.len(),
                adopted_uv,
                unvis.len() - skipped.len() - adopted_uv,
                tail,
                heads.join(", ")
            );
        }
    }
    // COVERAGE REPAIR.
    //
    // Every block the CFG can reach must appear in the text. The walk abandons
    // regions for real structural reasons (a secondary loop exit, a switch arm
    // fenced by a construct already rendered, a pad shared by clauses) — but "I
    // could not fit this into a structured statement" must not become silence,
    // because the reader then sees a method that does not contain code the
    // machine runs. So each reachable block the walk skipped is emitted as a
    // labelled region after the body, linked by the `goto`s the walk already
    // emitted for it (including the explicit secondary-exit gotos from
    // `bail_out_of_region`).
    //
    // Loops are deliberately NOT re-structured here: the repair region is
    // reached only through those gotos, so rendering it FLAT keeps exactly one
    // copy of every block and cannot duplicate a loop body.
    let (repaired, repaired_blocks) = st.coverage_repair(&reachable);
    if repaired_blocks > 0 {
        body_stmts.extend(repaired);
        let _ = COVERAGE_REPAIR
            .blocks
            .fetch_add(repaired_blocks as u64, std::sync::atomic::Ordering::Relaxed);
        let _ = COVERAGE_REPAIR
            .methods
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    {
        // Post-repair coverage: what is STILL unwalked is exactly what no
        // recovered edge reaches. The headline the report quotes must describe
        // the text that was produced, not the text before the repair.
        let visited = st.visited.borrow();
        let un = (0..sc.blocks.len()).filter(|b| !visited.contains(b)).count();
        UNEMITTED_BLOCKS.with(|c| c.set(un));
        TOTAL_BLOCKS.with(|c| c.set(sc.blocks.len()));
        if std::env::var("ILR_COV").is_ok() {
            // The post-repair number is the one that must be zero on the
            // reachable side: the `[cov]` line above reports what the walk
            // missed, this reports what the repair could not pick up.
            let left = (0..sc.blocks.len()).filter(|b| !visited.contains(b)).count();
            let reach_left = (0..sc.blocks.len())
                .filter(|b| !visited.contains(b) && reachable[*b])
                .count();
            if left > 0 {
                eprintln!(
                    "[cov-post] {:#x} unvisited={} reachable_left={} repaired_regions={}",
                    ir.base_rva, left, reach_left, repaired_blocks
                );
            }
        }
    }

    let t_rw = std::time::Instant::now();
    // AST rewriting + cleanup passes (order matters)
    rewrites::apply(&mut body_stmts);
    if std::env::var("ILR_STMTS").is_ok() {
        let mut r2 = Renderer::new();
        for s in &body_stmts { r2.stmt(s); }
        let _ = std::fs::write(probe_dir().join("after_rewrites.txt"), r2.out.clone());
    }
    // Pattern catalog: data-driven C#-construct bindings (see docs). Runs
    // after sugar rewrites, before structural cleanup — a failed match
    // leaves the lower-level sequence untouched.
    catalog::apply(&mut body_stmts);
    if std::env::var("ILR_STMTS").is_ok() {
        let mut r2 = Renderer::new();
        for s in &body_stmts { r2.stmt(s); }
        let _ = std::fs::write(probe_dir().join("after_catalog.txt"), r2.out.clone());
    }
    cleanup::apply(&mut body_stmts);
    // STATEMENT-TREE IDIOMS: dead-end blocks / dead code, then structure-field
    // propagation. These reason about C# SHAPES rather than the CFG, so they run
    // last, after every structural pass has settled — see `idioms` module docs
    // for why they cannot live among the IR-level passes and for the two rules
    // (never delete an observation, never move a write) that bound them.
    //
    // `idioms::apply` folds the class-init guards it can see, and the second
    // fold below catches the rest; BOTH erase a native `bl`. The guards standing
    // here, plus those standing at the second call, are the calls about to
    // vanish, and that is what `cci_erased` counts BY CALL TARGET — the fold
    // works on statement trees and the call SITE's address lives only in the IR,
    // so a per-target count taken on both sides is what ties the two together.
    let cci_before = super::idioms::count_class_init_guards(&body_stmts);
    super::idioms::apply(&mut body_stmts);
    // HIGH-LEVEL CONSTRUCT RECONSTRUCTION: rebuild source-level shapes the
    // structurer could only emit flat (`for`, `using`/`lock`, lambdas, iterator
    // state machines, array initializers). Runs after `idioms` has removed the
    // machine noise, so the recognizers see the cleanest tree; a failed match is
    // a no-op that leaves the correct lower-level sequence in place.
    super::reconstruct::apply(&mut body_stmts);
    if std::env::var("ILR_STMTS").is_ok() {
        let mut r2 = Renderer::new();
        for s in &body_stmts { r2.stmt(s); }
        let _ = std::fs::write(probe_dir().join("after_clean2.txt"), r2.out.clone());
    }
    drop_unreachable_gotos(&mut body_stmts);
    // Second loop-exit rewrite, and the one that fires: the sweep above removed
    // the stale dangling `goto` that usually sat between a loop and the label
    // it falls out to (see `idioms::collapse_goto_to_loop_break`).
    // NOTE: no dead-code sweep here. The coverage repair appends its labelled
    // regions as a bare statement run, and a sweep after `drop_unreachable_gotos`
    // deletes them (the terminator that precedes them is now the last statement
    // of the structured body). The converted `break` leaving a dead tail is the
    // rarer cost, and the tail is still rendered.
    super::idioms::collapse_goto_to_loop_break(&mut body_stmts);
    if std::env::var("ILR_STMTS").is_ok() {
        let mut r2 = Renderer::new();
        for s in &body_stmts { r2.stmt(s); }
        let _ = std::fs::write(probe_dir().join("after_dug.txt"), r2.out.clone());
    }
    sanitize_stmts(&mut body_stmts);
    branch_repair(&mut body_stmts);
    // Fold guards an enclosing guard has already decided (`if (v != 0) { …
    // if (v == 0) … }` — IL2CPP's redundant managed null-check mirrored onto
    // a path where it is constant). Runs before the sanitize so the removal
    // markers it leaves are swept with every other marker.
    fold_decided_guards(&mut body_stmts);
    // Second sanitize: branch_repair's inversions expose new empty else-
    // shells (payload arm moved to then); sweep them with the same rules.
    sanitize_stmts(&mut body_stmts);
    // Flatten `if (c) { if (¬c) { X } }` guard artifacts (see module docs).
    fold_impossible_guards(&mut body_stmts);
    // Event add/remove accessors: CAS-retry loop → single field store.
    collapse_event_cas_accessors(&mut body_stmts, method_name);
    // Final dangling-goto sweep: the passes above (label stripping, guard
    // folding, CAS collapse, empty-shell removal) can orphan gotos whose
    // labels were never rendered — arm/handler exit edges to blocks the
    // walker emitted earlier without a label. Removing them leaves valid
    // fall-through pseudo-C#; keeping them emits `goto L<missing>`.
    drop_unreachable_gotos(&mut body_stmts);
    super::idioms::collapse_goto_to_loop_break(&mut body_stmts);
    // MASKED-ASSEMBLY FOLD, second (and effective) call site. The static-init
    // guard reaches `idioms::apply` in the FLAT-GUARD lowering
    // (`if (guard != 0) goto L0; <classinit>; <guard> = 1; L0:`), which the
    // statement shape does not match. `branch_repair` re-inlines that diamond
    // into `if (guard == 0) { <classinit>; <guard> = 1; }` a few passes later,
    // so the fold is retried here — after every structural pass has settled and
    // right before the text is emitted. Idempotent: the first call is a no-op
    // wherever the guard is still flat, this one catches the rest.
    let mut cci_erased = cci_before;
    cci_erased.merge(&super::idioms::count_class_init_guards(&body_stmts));
    super::idioms::collapse_class_init_guards(&mut body_stmts);
    // EXCUSE THE ERASED CALLS. `concord` scores every native call it cannot
    // find in the text as DROPPED, so a deliberate erasure must reach the verify
    // ledger — the same discipline `resolve::key_function_recovery` follows for
    // the `il2cpp_codegen_initialize_*` thunks, under the same kind
    // (`DroppedRuntimeCall`: "il2cpp_codegen_initialize_* bookkeeping call
    // dropped"). Without it the fold is invisible to the fidelity report: when
    // the recognizer started firing on x64 (the `abs` base spelling made the
    // guard test a FIXED address, which is what `idioms::reads_static_word`
    // requires) `calls visible` fell from 100.0% to 98.5% for 12,981 guards,
    // every one a shape-proven runtime guard rather than a managed call.
    //
    // The two sides are joined PER CALL TARGET, and only where the counts AGREE.
    // Each erased guard removed exactly one call, so a target with as many
    // erased guards as call sites in the IR had every one of those sites erased,
    // and each can be excused exactly once. A target that does not match — some
    // site survived into the text — is left alone, because excusing a call that
    // is still in the output would hide a real drop, the one error this
    // accounting must never make. A guard whose arm call was prettified into a
    // member access carries no address at all and therefore excuses nothing.
    if cci_erased.total() > 0 && cci_erased.unmappable == 0 {
        let mut sites: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
        for insn in &ir.insns {
            if !matches!(insn.op, Op::Call | Op::CallVoid) {
                continue;
            }
            if let Some(Operand::Immediate(a)) = insn.operands.first() {
                sites.entry(*a as u64).or_default().push(insn.address);
            }
        }
        for (target, folded) in &cci_erased.by_target {
            let Some(at) = sites.get(target) else { continue };
            if at.len() as u64 != *folded {
                continue;
            }
            for a in at {
                super::verify::ledger_note(*a, super::verify::LedgerKind::DroppedRuntimeCall);
            }
        }
    }
    if std::env::var("ILR_STMTS").is_ok() {
        let mut r2 = Renderer::new();
        for s in &body_stmts { r2.stmt(s); }
        let _ = std::fs::write(probe_dir().join("after_cleanup.txt"), r2.out.clone());
    }
    if prof { eprintln!("[render] rewrites+cleanup {}us", t_rw.elapsed().as_micros()); }
    // Fidelity audit: which native instructions the output accounted for.
    // Claims made while BUILDING an expression that was then discarded are
    // not representations of the output — subtract them before judging.
    let mut claims = st.ctx.audit.borrow().clone();
    for d in st.ctx.dropped_claims.borrow().iter() {
        claims.remove(d);
    }
    // Calls the renderer presented as a member access (`obj.get_X()` → `obj.X`,
    // `String.Concat(a,b)` → `a + b`, `obj.set_X(v)` → `obj.X = v`) leave a real
    // native `bl` in the text that is NOT an `ident(` token. Ledger exactly the
    // ones that survived into the output — a fold whose expression was built
    // and then discarded never claimed its instructions, so intersecting with
    // `claims` keeps a discarded fold from hiding a genuinely dropped call.
    for &i in st.ctx.property_fold_sites.borrow().iter() {
        if claims.contains(&i) {
            super::verify::ledger_note(
                ir.insns[i].address,
                super::verify::LedgerKind::PropertyAccessorCall,
            );
        }
    }
    audit_write_line(ir.base_rva, method_name, &ir.insns, &claims);
    audit_detail_unclaimed(ir.base_rva, &ir.insns, &claims);

    // Typed-local summary line: compact machine-readable context that makes
    // each pseudo block self-contained for human or LLM refactor passes.
    //
    // OFF BY DEFAULT. It is a DIAGNOSTIC, not part of the decompilation: it
    // restates every IR local the method happens to have (`object v0, object
    // v1, … nint v17, long v62`), which on a large binary is tens of thousands
    // of lines of internal slot names — noise that buries the code it is
    // supposed to annotate (the reference dumps carried ~49,000 of these
    // lines, together roughly half the file). `ILR_LOCALS=1` brings it back
    // when a refactor pass wants the slot→type map.
    let with_locals = std::env::var("ILR_LOCALS").is_ok();
    let mut typed: Vec<String> = Vec::new();
    for li in 0..ir.table.infos.len() {
        let info = &ir.table.infos[li];
        if info.is_this || info.is_param.is_some() {
            continue;
        }
        if let Some(t) = csharp_type(&info.ty) {
            let name = st_name(li);
            typed.push(format!("{t} {name}"));
        }
    }

    let mut r = Renderer::new();
    let params = param_sigs
        .iter()
        .map(|(t, n)| format!("{t} {n}"))
        .collect::<Vec<_>>()
        .join(", ");
    let ret = csharp_type(ret_type).unwrap_or_else(|| "void".into());
    let modifier = if is_static { "static " } else { "" };
    r.line(format!("{modifier}{ret} {method_name}({params})"));
    r.line("{");
    r.indent += 1;
    // The oracle's grade for this body, when it was not provably clean. This is
    // the whole point of the annotation: a reader scanning the dump must be
    // able to tell a recovered value from a read of a slot that was never
    // written, WITHOUT a separate verification log. Absent for clean methods
    // (the common case), so a clean dump is not buried in boilerplate.
    if let Some((verdict, reason)) = super::verify::annotation_for(ir.base_rva) {
        r.line(format!(
            "// [recovery: {} — {}]",
            super::verify::verdict_name(verdict),
            reason
        ));
    }
    if with_locals && !typed.is_empty() && !body_stmts.is_empty() {
        r.out.push_str(&format!("    // locals: {}\n", typed.join(", ")));
    }
    if body_stmts.is_empty() {
        // The renderer produced a header and NO statements. Distinct from the
        // three markers above: the IR was recovered, but nothing structured, so
        // the body is an empty shell rather than a refusal.
        FALLBACK
            .empty_body
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        r.line("// body not recoverable");
    } else {
        if std::env::var("ILR_STMTS").is_ok() {
            let mut r2 = Renderer::new();
            for s in &body_stmts { r2.stmt(s); }
            let _ = std::fs::write(probe_dir().join("final_stmts.txt"), r2.out.clone());
        }
        for s in &body_stmts {
            r.stmt(s);
        }
    }
    r.indent -= 1;
    r.line("}");
    strip_undefined_gotos(&mut r.out);
    strip_unread_bindings(&mut r.out);
    undef_dbg_records(&ir.table, &ir.insns, ir.base_rva, &r.out);
    r.out
}


/// Whole-binary native pseudo-C# generation (replaces Ghidra for supported
/// architectures). Parallel over methods; returns RVA → C# text merged into
/// dump.cs by the caller.
pub fn decompile_methods_to_csharp(
    il2cpp: &crate::il2cpp::base::Il2Cpp,
    metadata: &crate::il2cpp::metadata::Metadata,
    executor: &mut crate::executor::Il2CppExecutor,
    _config: &crate::config::Config,
    logger: &mut impl FnMut(&str),
) -> HashMap<u64, String> {
    use rayon::prelude::*;

    // Bytecode refusal, enforced at the ENGINE rather than only at the caller.
    //
    // Every lift here decodes a native instruction at `get_rva(ptr)`. On
    // WebAssembly that address is a linear-memory offset holding wasm bytecode
    // or plain data, and the ARM/ARM32/x86 decoders will still produce
    // *plausible instructions* from almost any bytes — which is how a 50 MB
    // `.wasm` previously came out with 79,285 confident, entirely fictional
    // method bodies (`v0 = in_x7; *(byte*)(v0 - 1362) = ...`). There is no
    // native instruction stream to recover, so returning nothing is the only
    // honest answer. A real wasm backend belongs in a separate lifter over the
    // wasm engine's own decoded instructions.
    if !il2cpp.detect_architecture().has_native_code() {
        logger(&format!(
            "Recovery skipped: {} has no native method bodies to lift",
            il2cpp.detect_architecture()
        ));
        return HashMap::new();
    }

    // WORKER STACK SIZE is a correctness requirement here, not tuning.
    //
    // Rendering an expression is recursive over its operand tree, and a single
    // real method can nest thousands of operations (an unrolled arithmetic
    // chain, or a select whose condition is itself a select). Recovering MORE
    // values therefore makes expression trees DEEPER, and a stack overflow is
    // not a recoverable error in Rust — it aborts the whole process, losing
    // every method in the run. That is how one pathological method could kill
    // a 90,000-method dump. The pool is given an explicit, generous stack so
    // depth is bounded by the machine rather than by the default 2 MiB, and
    // the per-method deadline still caps the TIME a pathological method may
    // consume. Failures only if a pool was already installed, which is fine —
    // that pool is then the caller's deliberate choice.
    // Thread count and per-thread stack are bounded by MEMORY, not by cores.
    //
    // The whole-dump path already holds every rendered body in memory before
    // the files are written, and the stack is sized for the deepest expression
    // trees. 64 MiB × 16 workers is a GIGABYTE of thread stacks on top of that,
    // and on a 16 GB machine with a few GB free the run ends in
    // `memory allocation of N bytes failed` — full dumps of both the ARM32
    // Android and the arm64 iOS binary aborted that way, at unrelated points
    // and with unrelated failing sizes, which is what exhaustion (not one
    // pathological method) looks like.
    //
    // 16 MiB is still 8× the default stack and enough for the expression depth
    // the renderer allows, and capping the pool keeps the worst case bounded
    // instead of scaling it with whatever CPU the machine reports.
    // `ILR_WORKERS=<n>` overrides it for a machine known to have the headroom.
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let workers = std::env::var("ILR_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| cores.clamp(1, 8));
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .stack_size(16 * 1024 * 1024)
        .build_global();

    let mut ctx = super::RecoveryContext::build(executor, metadata, il2cpp);
    ctx.include_assemblies = _config.dump_pseudo_c_include_assemblies.clone();
    // Grade every method unless asked not to. The verifier's per-method hook is
    // `ILR_VERIFY`; setting it here makes the graded output the DEFAULT rather
    // than an opt-in debug mode, which is what turns the census numbers from a
    // log an expert reads into a label on the code everyone reads. SAFETY: set
    // before the pipeline starts and before any worker is spawned, so no thread
    // observes a torn environment (the dumper is a single-process batch tool and
    // does not otherwise mutate its environment at run time).
    let annotate = _config.annotate_recovery;
    if annotate {
        // A dedicated gate, NOT `ILR_VERIFY`: it turns the verifier on but keeps
        // its per-method log off (that channel is `ILR_VERIFY`'s). The label on
        // the method is the product; the log is a debugging aid.
        // SAFETY: single-threaded here — rayon workers are spawned below.
        unsafe { std::env::set_var("ILR_ANNOTATE", "1") };
    }
    // The type filter, on top of the assembly one. Without this the flag did
    // nothing at all: the selection tree offered types and the config stored
    // them, but the signature loop below never consulted them, so a run asked
    // for one class decompiled the first N methods of the *first* assembly
    // instead. Matching uses the same `Namespace.Type` names `dump.cs` prints.
    let include_types = _config.dump_pseudo_c_include_types.clone();

    logger("Native C#: building method signatures...");
    struct Sig {
        name: String,
        is_static: bool,
        params: Vec<(String, String)>, // (csharp type, name)
        /// Managed types of the parameters straight from metadata — these
        /// seed the local table so ABI slots recover their real types.
        param_types: Vec<ManagedType>,
        /// Float-component field names per parameter (HFA value types only,
        /// e.g. Vector3 → ["x", "y", "z"]). Parallel to param_types; empty
        /// vec for non-FP parameters.
        param_comps: Vec<Vec<String>>,
        ret: ManagedType,
        tdi: i32,
    }
    let mut sigs: HashMap<u64, Sig> = HashMap::new();
    for image_def in &metadata.image_defs {
        let image_name = metadata
            .get_string_from_index(image_def.name_index)
            .unwrap_or_default();
        let type_end = image_def.type_start as usize + image_def.type_count as usize;
        for tdi in image_def.type_start as usize..type_end {
            let Some(td) = metadata.type_defs.get(tdi) else { continue };
            let m_end = td.method_start as usize + td.method_count as usize;
            // Union: whole selected DLLs plus any specifically checked types.
            let picker = crate::output::pseudo_c::picker_type_name(metadata, td);
            let dump_name =
                executor.get_type_def_name(td, tdi, metadata, il2cpp, true, false);
            if !crate::output::pseudo_c::method_body_filter_allows_names(
                &image_name,
                &picker,
                &dump_name,
                &ctx.include_assemblies,
                &include_types,
            ) {
                continue;
            }
            for mi in td.method_start as usize..m_end {
                let Some(md) = metadata.method_defs.get(mi) else { continue };
                let ptr = il2cpp.get_method_pointer(&image_name, md);
                if ptr == 0 {
                    continue;
                }
                let rva = il2cpp.get_rva(ptr);
                if rva == 0 || sigs.contains_key(&rva) {
                    continue;
                }
                let name = metadata
                    .get_string_from_index(md.name_index as i32)
                    .unwrap_or_default();
                let params = (0..md.parameter_count as usize)
                    .filter_map(|j| {
                        let pidx = md.parameter_start as usize + j;
                        let pd = metadata.parameter_defs.get(pidx)?;
                        let pname = metadata
                            .get_string_from_index(pd.name_index)
                            .unwrap_or_else(|_| format!("arg{j}"));
                        let pty = il2cpp
                            .types
                            .get(pd.type_index as usize)
                            .map(|t| {
                                super::managed_type_of_gn(
                                    t,
                                    metadata,
                                    il2cpp,
                                    0,
                                    Some(&ctx.generic_names),
                                )
                            })
                            .map(|t| csharp_type(&t).unwrap_or_else(|| "object".into()))
                            .unwrap_or_else(|| "object".into());
                        Some((pty, pname))
                    })
                    .collect();
                // Exact managed parameter types from metadata (same loop,
                // kept instead of being discarded): they seed the ABI slots
                // in the local table, so `this`/receiver recovery, casts and
                // call-argument typing all see the real declared types.
                let param_types: Vec<ManagedType> = (0..md.parameter_count as usize)
                    .filter_map(|j| {
                        let pidx = md.parameter_start as usize + j;
                        let pd = metadata.parameter_defs.get(pidx)?;
                        il2cpp
                            .types
                            .get(pd.type_index as usize)
                            .map(|t| {
                                super::managed_type_of_gn(
                                    t,
                                    metadata,
                                    il2cpp,
                                    0,
                                    Some(&ctx.generic_names),
                                )
                            })
                    })
                    .collect();
                // HFA float components per parameter (Vector3 → x/y/z):
                // the ABI passes such structs in consecutive FP registers
                // and locals::create binds those registers to the parameter
                // so call args render `rootPosition.x` instead of `v26`.
                let param_comps: Vec<Vec<String>> = param_types
                    .iter()
                    .map(|pt| match pt {
                        ManagedType::Class(nt)
                            if nt.type_def_index >= 0
                                && nt.type_def_index < metadata.type_defs.len() as i32 =>
                        {
                            super::hfa_float_components(
                                nt.type_def_index,
                                metadata,
                                il2cpp,
                            )
                        }
                        _ => Vec::new(),
                    })
                    .collect();
                let ret = il2cpp
                    .types
                    .get(md.return_type as usize)
                    .map(|t| {
                        super::managed_type_of_gn(
                            t,
                            metadata,
                            il2cpp,
                            0,
                            Some(&ctx.generic_names),
                        )
                    })
                    .unwrap_or(ManagedType::Unknown);
                sigs.insert(
                    rva,
                    Sig {
                        name,
                        is_static: (md.flags & 0x0010) != 0,
                        params,
                        param_types,
                        param_comps,
                        ret,
                        tdi: tdi as i32,
                    },
                );
            }
        }
    }

    let sorted = il2cpp.build_sorted_method_addresses();
    logger(&format!("Native C#: decompiling {} methods...", sigs.len()));

    // Precompute try/catch regions for every method from the binary's EH
    // tables (Mach-O __unwind_info/__gcc_except_tab, ELF .eh_frame). Each
    // entry renders as `try { … } catch (T)` headers — exact data, not
    // control-flow guesses.
    //
    // Catch-type names come from C++ RTTI: IL2CPP compiles managed throws to
    // C++ `throw Il2CppExceptionWrapper(...)`, so ttype entries hold
    // `type_info*` for that wrapper (Apple ttypeEncoding 0x9B = pcrel|indirect:
    // the decoded value is the address OF the type_info pointer slot).
    // Deref the slot once → type_info; the name lives at +8 (libc++abi
    // `__class_type_info`: vptr@0, `__type_name`@8) in Itanium-mangled
    // `<len><name>` form. The actual managed catch-type check is emitted by
    // the compiler INSIDE the landing pad, where our IR recovery already
    // sees it as a klass check — this is exactly how the runtime matches
    // exceptions, and metadata cannot know it better.
    let class_name_of_slot = |slot_va: u64| -> Option<String> {
        let ptr_at = |va: u64| -> Option<u64> {
            let rva = il2cpp.get_rva(va);
            let off = il2cpp.map_vatr(rva.wrapping_add(il2cpp.image_base)).ok()?;
            if il2cpp.is_32bit {
                il2cpp.stream.peek_u32_at(off).ok().map(|p| p as u64)
            } else {
                il2cpp.stream.peek_u64_at(off).ok()
            }
        };
        let read_cstr = |va: u64| -> Option<String> {
            if va == 0 {
                return None;
            }
            let rva = il2cpp.get_rva(va);
            let off = il2cpp.map_vatr(rva.wrapping_add(il2cpp.image_base)).ok()?;
            il2cpp.stream.peek_string_to_null_at(off).ok()
        };
        let ti = ptr_at(slot_va)?;
        if ti == 0 {
            return None;
        }
        // libc++abi type_info: vptr (0), __type_name char* (8).
        let mangled = read_cstr(ptr_at(ti.wrapping_add(8))?)?;
        // demangle the simple `<len><name>` producer form: N…E / nested names
        // stay as-is (rare here); a leading digit run is a name length.
        let digits = mangled.chars().take_while(|c| c.is_ascii_digit()).count();
        let name = if digits > 0 && digits < mangled.len() {
            mangled[digits..].to_string()
        } else {
            mangled
        };
        if name == "Il2CppExceptionWrapper" {
            // every managed catch matches the wrapper; the managed type
            // filter is inside the landing pad body
            Some("Il2CppExceptionWrapper /* managed catch */".to_string())
        } else {
            Some(name)
        }
    };
    let eh_regions: HashMap<u64, Vec<super::exceptions::EhRegion>> = if il2cpp.eh_tables.is_empty() {
        HashMap::new()
    } else {
        sigs
            .par_iter()
            .filter_map(|(rva, _)| {
                let size = il2cpp.get_method_body_size(*rva, &sorted);
                if size == 0 {
                    return None;
                }
                let regs = il2cpp
                    .eh_tables
                    .regions_for(*rva, size as u64, &|slot_va| class_name_of_slot(slot_va));
                // The LSDA lists the wrapper row AND the catch-all row over
                // the same try range with the same pad — one managed catch.
                // Duplicate rows would fold the same try twice.
                let mut seen = std::collections::HashSet::new();
                let regs: Vec<super::exceptions::EhRegion> = regs
                    .into_iter()
                    .filter(|r| {
                        seen.insert((r.try_start, r.try_end, r.landing_pad, r.is_cleanup))
                    })
                    .collect();
                if regs.is_empty() {
                    None
                } else {
                    Some((*rva, regs))
                }
            })
            .collect()
    };
    if !eh_regions.is_empty() {
        logger(&format!(
            "Native C#: recovered {} try/catch regions from EH tables",
            eh_regions.values().map(|v| v.len()).sum::<usize>()
        ));
    }

    // Every method WITH native bytes gets a body: recovered C# on success,
    // an explicit failure marker otherwise — so dump.cs never mixes
    // mysterious `{ }` with pseudo-code for the same class.
    let mut method_list: Vec<(u64, &Sig)> =
        sigs.iter().map(|(r, s)| (*r, s)).collect();
    method_list.sort_by_key(|(rva, _)| *rva);
    // `ILR_DET_LIMIT=<n>`: keep ~n methods STRIDED across the RVA-sorted list
    // rather than the first n. The first n would all sit in one assembly's
    // address range and miss whole method-shape families; a stride spreads the
    // sample over the binary without being random (a random sample would
    // itself differ between the two gate runs and defeat the comparison).
    if let Some(n) = det_limit() {
        if n > 0 && method_list.len() > n {
            let stride = method_list.len().div_ceil(n);
            let mut kept: Vec<(u64, &Sig)> = Vec::with_capacity(n + 1);
            for (i, item) in method_list.iter().enumerate() {
                if i % stride == 0 {
                    kept.push(*item);
                }
            }
            method_list = kept;
        }
    }
    let results: HashMap<u64, String> = method_list.par_iter()
        .filter_map(|(rva, sig)| {
            // The largest body this pass will decompile.
            //
            // It was 6 KB, which SILENTLY skipped every method bigger than
            // that — 79 methods in the ARM32 build, 76 in the arm64 one, and the
            // biggest functions in a binary are exactly the ones a reader wants
            // (the state machines, the big switch dispatchers, the serializer
            // helpers). The cap is now the same 0x4000 ceiling
            // `Il2Cpp::get_method_body_size` reports with, so a skip can no
            // longer be caused by the text pass being stricter than the size
            // probe the message quotes.
            const MAX_TEXT_BODY_BYTES: usize = 0x4000;
            let size = il2cpp.get_method_body_size(*rva, &sorted);
            // ILR_ONLY=0x<rva> — decompile just one method (debugging)
            if let Ok(only) = std::env::var("ILR_ONLY") {
                if !format!("{:#x}", rva).eq_ignore_ascii_case(&only) {
                    return None;
                }
            }
            if size == 0 {
                return None; // genuinely no native body (abstract/stub) → { }
            }
            // Denominator for the fallback census: a method with real bytes that
            // this pass was asked to turn into C#.
            FALLBACK.attempted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if size > MAX_TEXT_BODY_BYTES.min(ctx.max_method_bytes) {
                FALLBACK.too_large.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some((
                    *rva,
                    format!(
                        "        // [recovery skipped: method too large ({size} bytes)]"
                    ),
                ));
            }
            let Some(bytes) = il2cpp.read_bytes_at_rva(*rva, size) else {
                FALLBACK
                    .bytes_unreadable
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some((*rva, "        // [recovery failed: body bytes unreadable]".into()));
            };
            let declaring = if sig.tdi >= 0 {
                metadata.type_defs.get(sig.tdi as usize).and_then(|td| {
                    Some(std::sync::Arc::new(super::types::NamedType {
                        namespace: metadata
                            .get_string_from_index(td.namespace_index)
                            .unwrap_or_default(),
                        name: metadata.get_string_from_index(td.name_index).ok()?,
                        assembly: None,
                        type_def_index: sig.tdi,
                    }))
                })
            } else {
                None
            };
            // Metadata-typed parameters (was `Unknown` for every param —
            // the exact types were already computed for the display signature
            // and then thrown away).
            let param_types: Vec<ManagedType> = sig.param_types.clone();
            // AAPCS32 passes value-type arguments as one or more core words.
            // Keep their native size beside the signature so the local seeder
            // does not shift later parameters into the wrong registers.
            let arm32_param_words: Vec<usize> = sig.param_types.iter().map(|ty| {
                let ManagedType::Class(nt) = ty else { return 1 };
                super::arm32_value_type_size(nt.type_def_index, metadata, il2cpp)
                    .map(|size| size.saturating_add(3) / 4)
                    .unwrap_or(1)
            }).collect();
            // ARM32 composite results ride in a caller-allocated buffer passed
            // in `r0`, which pushes `this` to `r1` and every parameter up one
            // register. Decided here (metadata + size table are available) and
            // applied in `locals::create`.
            let ret_hidden_buffer = ctx.is_arm32
                && super::arm32_ret_uses_hidden_buffer(&sig.ret, metadata, il2cpp);
            let seed = super::locals::SeedInfo {
                param_types,
                ret_type: sig.ret.clone(),
                is_static: sig.is_static,
                declaring,
                param_names: sig.params.iter().map(|(_, n)| n.clone()).collect(),
                param_comp_names: sig.param_comps.clone(),
                arm32_param_words,
                is_arm32: ctx.is_arm32,
                ret_hidden_buffer,
            };
            if std::env::var("ILR_SIG").is_ok() {
                eprintln!("[sig] {:#x} static={} params={:?} ptypes={:?} ret={:?}", rva, sig.is_static, sig.params, sig.param_types, sig.ret);
            }
            let trace = std::env::var("ILR_TRACE").is_ok();
            let rssdbg = std::env::var("ILR_RSSDBG").is_ok();
            let concord_on = std::env::var("ILR_CONCORD").is_ok();
            if trace {
                eprintln!("[m+] {:#x} {}B", rva, size);
            }
            if rssdbg {
                rss_probe("start", *rva, size);
            }
            let deadline = time_limit_ms()
                .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms));
            super::verify::clear_graded();
            super::verify::set_tv_source("cs");
            let result = match ctx.recover_ir_deadline(&bytes, *rva, &seed, metadata, il2cpp, deadline) {
                Some(ir) => {
                    // `ILR_VERIFY=1` inside the pipeline can only run when
                    // `recover_ir_deadline` reaches its verification stage. A
                    // method that returns early (deadline, IR cap) never gets
                    // graded, so stamp it here instead of leaving it silently
                    // unlabelled — an ungraded body must not masquerade as a
                    // proven-clean one.
                    if annotate && !super::verify::graded_this_method() {
                        super::verify::annotate_method(
                            *rva,
                            super::verify::Verdict::Suspect,
                            "not verified (pipeline exited early)".into(),
                        );
                    }
                    let t = std::time::Instant::now();
                    if std::env::var("ILR_STAGES").is_ok() {
                        eprintln!("[stage] {:#x} render", rva);
                    }
                    // EH regions come from the binary's exception tables and
                    // fold into real `try { } catch { }` statements inside the
                    // structurer (exact ranges — no control-flow guessing).
                    let regions = eh_regions.get(rva).map(|v| v.as_slice()).unwrap_or(&[]);
                    let text = render_method_csharp_full_chains(
                        &ir,
                        &sig.name,
                        sig.is_static,
                        &sig.params,
                        &sig.ret,
                        sig.tdi,
                        render_deadline(),
                        &ctx.parent_chains,
                        regions,
                    );
                    if let Ok(want) = std::env::var("ILR_CALLDBG") {
                        if want == "all" || want.eq_ignore_ascii_case(&format!("{:#x}", rva)) {
                            let calls = ir
                                .insns
                                .iter()
                                .filter(|k| matches!(k.op, super::ir::Op::Call | super::ir::Op::CallVoid))
                                .count();
                            eprintln!("[calldbg] {rva:#x} final-ir insns={} calls={calls}", ir.insns.len());
                        }
                    }
                    if let Ok(dump) = std::env::var("ILR_DUMP") {
                        let _ = std::fs::write(&dump, format!("// method {:#x} size={}\n{}\n", rva, text.len(), text));
                    }
                    super::tm_add(&super::timing::RENDER, t);
                    // C#-vs-disassembly concordance: read the EMITTED text back
                    // against the native bytes, so a loss the printer itself
                    // introduced is measured rather than assumed absent.
                    if concord_on {
                        let arch = if ctx.arch_is_arm64 {
                            super::concord::Arch::Arm64
                        } else if ctx.is_arm32 {
                            super::concord::Arch::Arm32
                        } else {
                            super::concord::Arch::X64
                        };
                        super::concord::check_and_log(&text, &bytes, *rva, arch);
                    }
                    if rssdbg {
                        let calls = DEF_CALLS.with(|c| c.replace(0));
                        let maxd = DEF_MAXDEPTH.with(|c| c.replace(0));
                        let fired = BUDGET_FIRED.with(|f| f.replace(false));
                        let cuts = CYCLE_CUTS.with(|c| c.replace(0));
                        eprintln!(
                            "[def] {rva:#x} calls={calls} maxdepth={maxd} \
                             budget_fired={fired} cycle_cuts={cuts}"
                        );
                        rss_probe("end", *rva, size);
                    }
                    if trace {
                        eprintln!("[m-] {:#x} rendered", rva);
                    }
                    Some((*rva, text))
                }
                None => {
                    if trace {
                        eprintln!("[m-] {:#x} failed", rva);
                    }
                    FALLBACK
                        .unsupported_pattern
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    Some((
                        *rva,
                        format!(
                            "        // [recovery failed: {} bytes @ {:#x} — unsupported pattern]",
                            size, rva
                        ),
                    ))
                }
            };
            result
        })
        .collect();

    if std::env::var("ILR_TIMING").is_ok() {
        eprintln!("{}", super::timing::report());
    }
    if std::env::var("ILR_CONCORD").is_ok() {
        let report = super::concord::report();
        if !report.is_empty() {
            eprint!("{report}");
        }
    }
    if std::env::var("ILR_VERIFY").is_ok() {
        eprintln!("{}", super::verify::VerifyStats::report());
    }
    if std::env::var("ILR_VERIFY").is_ok() || std::env::var("ILR_CONFORM").is_ok() {
        let conform = super::conform::report();
        if !conform.is_empty() {
            eprintln!("{conform}");
        }
    }
    if std::env::var("ILR_MEM").is_ok() {
        let total: usize = results.values().map(|s| s.len()).sum();
        let max: usize = results.values().map(|s| s.len()).max().unwrap_or(0);
        let mut lens: Vec<usize> = results.values().map(|s| s.len()).collect();
        lens.sort_unstable();
        let med = lens.get(lens.len() / 2).copied().unwrap_or(0);
        let p90 = lens.get(lens.len() * 9 / 10).copied().unwrap_or(0);
        eprintln!(
            "[mem] results={} entries total_bytes={} avg={} med={} p90={} max={}",
            results.len(),
            total,
            if results.is_empty() { 0 } else { total / results.len() },
            med,
            p90,
            max
        );
        let mut top: Vec<(u64, usize)> = results.iter().map(|(r, s)| (*r, s.len())).collect();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        for (r, l) in top.iter().take(10) {
            eprintln!("[mem] top {:#x} {}B", r, l);
        }
    }
    if std::env::var("ILR_LIFT_CENSUS").is_ok() {
        eprintln!("{}", super::lift::lift_census_summary());
    }
    if let Ok(path) = std::env::var("ILR_LIFT_CENSUS_FILE") {
        let summary = super::lift::lift_census_summary();
        match std::fs::write(&path, super::lift::lift_census_report()) {
            Ok(()) => eprintln!("{summary} -> {path}"),
            Err(e) => eprintln!("[lift] census write FAILED: {e}"),
        }
    }
    if let Ok(path) = std::env::var("ILR_DET_MANIFEST") {
        det_write_manifest(&path, &results);
    }
    results
}

std::thread_local! {
    /// `ILR_RSSDBG` diagnostics: nesting depth of `def_expr`, its high-water
    /// mark, and the total call count. A bounded depth with a huge call count
    /// means a re-resolution blow-up rather than unbounded recursion.
    static DEF_DEPTH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static DEF_MAXDEPTH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static DEF_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Did [`MAX_INLINE_EXPR_NODES`] cut any expression in this method short?
    static BUDGET_FIRED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Loop-carried defs referenced by name instead of expanded.
    static CYCLE_CUTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// Instruction indices whose `def_expr` is on the stack right now. A
    /// re-entry means that def was reached through a loop BACK EDGE and would
    /// otherwise re-expand its own arithmetic (`v63 = v30 + 16` read back
    /// through the back edge as `v63 = v30 + 16 + 16`). Reference the
    /// destination name instead — a bounded, loop-carried value.
    static DEF_INSNS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    /// CFG blocks the structured walk never entered, and the method's total
    /// block count. Set by [`render_method_csharp_full_chains`], read by the
    /// concordance driver on the same thread.
    ///
    /// This is the number the IR-level oracle cannot see. `verify` checks that
    /// every native instruction reached the IR; it says nothing about whether
    /// the WALK emitted it. When the structurer stops (an indirect jump whose
    /// dispatch was not recovered is a CFG terminator, so everything past it is
    /// unreachable), the text silently loses whole regions while `verify`
    /// reports the method clean. Measured on 0x20d3ee4: 8 of 64 blocks emitted,
    /// 3 of 43 native conditions, and no checker said a word.
    static UNEMITTED_BLOCKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TOTAL_BLOCKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// `ILR_CALLTRACE` — print one line per `boundary_stmts` visit of a call
    /// instruction together with what it emitted. Diagnosis only.
    static CALLTRACE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run totals for the coverage repair (see [`Structurer::coverage_repair`]).
///
/// Blocks the structurer could not place in a structured statement but the CFG
/// can reach, plus the number of methods that needed them. Global rather than
/// thread-local because rendering runs on a worker pool and the report is
/// printed from a single thread.
pub struct CoverageRepairCounters {
    pub blocks: std::sync::atomic::AtomicU64,
    pub methods: std::sync::atomic::AtomicU64,
}

pub static COVERAGE_REPAIR: CoverageRepairCounters = CoverageRepairCounters {
    blocks: std::sync::atomic::AtomicU64::new(0),
    methods: std::sync::atomic::AtomicU64::new(0),
};

/// `(regions repaired, methods repaired)` for the run report.
pub fn coverage_repair_totals() -> (u64, u64) {
    (
        COVERAGE_REPAIR
            .blocks
            .load(std::sync::atomic::Ordering::Relaxed),
        COVERAGE_REPAIR
            .methods
            .load(std::sync::atomic::Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// Structural census: WHY the structurer has to fall back to `goto`
// ---------------------------------------------------------------------------
//
// A `goto` in the output is a symptom, not a diagnosis. It says the structurer
// met a control-flow SHAPE it could not nest, and there are only a handful of
// such shapes. Counting the shapes is what makes the work that removes gotos
// aimable at the one that actually causes them, instead of rewriting the
// structurer and hoping.
//
// The shapes, from the CFG rather than from the rendered text:
//
//  * **irreducible edge** — an edge `t → h` where `h` does NOT dominate `t`
//    yet still closes a cycle (`h` reaches `t`). `find_loops` accepts a back
//    edge only when `h` dominates `t`, because a natural loop needs the single
//    header that domination gives it. Every skipped edge that nevertheless
//    closes a cycle belongs to a region entered at more than one node, which
//    no nesting can express — the only faithful rendering left is a label and
//    a jump. This is the textbook cause of goto-shaped output.
//  * **multiply-latched loop** — more than one back edge into the same
//    header. Nesting is still possible, but the exit/continue structure has
//    several entries.
//  * **no reconverging if** — `find_merge` found no join for the two arms, so
//    there is no point to nest them under.
//  * **switch / unrecovered dispatch** — fan-out the structurer can only
//    express as a jump: a jump table it DID recover (already rewritten to
//    `Op::Switch`) or an `IndirectCall` whose table it did NOT.
/// Why a particular `goto` was emitted. Each variant is one CALL SITE in the
/// structurer, so the counts partition the `goto_sites` total and can be read
/// as "N gotos exist because of THIS shape". Without the split the census only
/// said how many methods contain a goto, which does not identify what to fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GotoReason {
    /// `emit_region`: the block was already emitted on another path, so control
    /// is sent back to it instead of duplicating the body.
    AlreadyEmitted,
    /// A branch whose target sits at or before the current block and is not a
    /// region stop: a cycle edge the loop finder did not claim as a natural
    /// loop (an irreducible region entered at more than one node).
    Backward,
    /// `br Xn` (an indirect JUMP) whose jump table was never recovered — the
    /// L1 case: the case bodies are real code but the dispatch is missing, so
    /// control must jump to them explicitly.
    Dispatch,
    /// A catch/finally landing pad already rendered by another clause.
    HandlerPad,
    /// The flat-guard lowering `if (c) goto JOIN; <fall body>` — a DELIBERATE
    /// constant-depth goto, not a structuring failure. It exists so a chain of
    /// N guards costs O(1) nesting instead of O(N).
    FlatGuard,
    /// The skip pattern whose body arm is already fenced by an ancestor merge,
    /// so both arms render empty and the branch would otherwise be dropped
    /// along with its condition.
    SkipEmpty,
    /// The join block was already emitted by an earlier arm walk.
    MergeEmitted,
    /// No reconverging join was found at all (`find_merge` returned None), so
    /// there is no point to nest under and control falls to a jump.
    NoMerge,
    /// A call site that has not been classified yet (should stay at 0).
    Uncategorised,
    /// A fall-through that leaves the loop body at a SECONDARY exit — any exit
    /// that is not the one `emit_loop` resumes at. The walker cannot continue
    /// into a region stop, so the edge is carried as an explicit `goto` whose
    /// label the coverage repair renders. Before this, the edge was dropped and
    /// the exited block was never emitted at all.
    LoopExit,
}

impl GotoReason {
    const COUNT: usize = 10;
    fn idx(self) -> usize {
        match self {
            GotoReason::AlreadyEmitted => 0,
            GotoReason::Backward => 1,
            GotoReason::Dispatch => 2,
            GotoReason::HandlerPad => 3,
            GotoReason::FlatGuard => 4,
            GotoReason::SkipEmpty => 5,
            GotoReason::MergeEmitted => 6,
            GotoReason::NoMerge => 7,
            GotoReason::Uncategorised => 8,
            GotoReason::LoopExit => 9,
        }
    }
    fn label(self) -> &'static str {
        match self {
            GotoReason::AlreadyEmitted => "block already emitted on another path",
            GotoReason::Backward => "back edge not claimed as a natural loop (irreducible)",
            GotoReason::Dispatch => "unrecovered indirect dispatch (L1)",
            GotoReason::HandlerPad => "catch/finally landing pad shared by clauses",
            GotoReason::FlatGuard => "flat guard (deliberate, constant depth)",
            GotoReason::SkipEmpty => "skip pattern with a fenced body arm",
            GotoReason::MergeEmitted => "join already emitted by an earlier arm",
            GotoReason::NoMerge => "no reconverging join (find_merge returned None)",
            GotoReason::Uncategorised => "unclassified call site",
            GotoReason::LoopExit => "fall out of a loop body at a secondary exit",
        }
    }
}

pub struct StructCensus {
    pub methods: std::sync::atomic::AtomicU64,
    pub blocks: std::sync::atomic::AtomicU64,
    pub irreducible_methods: std::sync::atomic::AtomicU64,
    pub irreducible_edges: std::sync::atomic::AtomicU64,
    pub loops: std::sync::atomic::AtomicU64,
    pub multi_latch_loops: std::sync::atomic::AtomicU64,
    pub multi_latch_methods: std::sync::atomic::AtomicU64,
    /// `find_merge` returned None: an `if` with no reconverging join.
    pub merge_fail_total: std::sync::atomic::AtomicU64,
    /// `find_merge` answered from the branch HEADER's immediate post-dominator
    /// (the exact merge). Answers from the reachability fallback are the rest.
    pub merge_via_header: std::sync::atomic::AtomicU64,
    /// A chosen merge was NOT reachable from both arms — a broken control-flow
    /// edge. The merge is refused so the arms carry explicit gotos instead.
    pub merge_not_common: std::sync::atomic::AtomicU64,
    pub switch_ops: std::sync::atomic::AtomicU64,
    pub switch_targets: std::sync::atomic::AtomicU64,
    pub dispatch_lost_methods: std::sync::atomic::AtomicU64,
    pub dispatch_lost_ops: std::sync::atomic::AtomicU64,
    /// `goto` statements the structurer ASKED for (creation sites, real walk
    /// only). The final text keeps fewer: `drop_unreachable_gotos` removes the
    /// ones whose label is never defined, so this is an upper bound.
    pub goto_sites: std::sync::atomic::AtomicU64,
    /// Gotos inspected inside switch arms, split into case-target rewrites and
    /// refused non-case destinations. These counters do not affect output.
    pub case_goto_attempts: std::sync::atomic::AtomicU64,
    pub case_goto_hits: std::sync::atomic::AtomicU64,
    pub case_goto_refused: std::sync::atomic::AtomicU64,
    pub goto_return_folds: std::sync::atomic::AtomicU64,
    pub goto_break_folds: std::sync::atomic::AtomicU64,
    pub shared_if_suffix_attempts: std::sync::atomic::AtomicU64,
    pub shared_if_suffix_hits: std::sync::atomic::AtomicU64,
    pub shared_if_suffix_refused: std::sync::atomic::AtomicU64,
    /// `goto_sites` split by the call site that produced them. This is the
    /// number that says WHICH shape to fix: the largest bucket is the leverage.
    pub goto_by_reason: [std::sync::atomic::AtomicU64; GotoReason::COUNT],
    /// Flat-guard chains seen (each is one collapsible `if (c1 || c2 || …)`
    /// candidate).
    pub flat_guard_chains: std::sync::atomic::AtomicU64,
    /// Guard shells fused away by that collapse (sum of `chain_len - 1`): the
    /// number of `goto`s and `if`s the collapse removed.
    pub flat_guard_fused: std::sync::atomic::AtomicU64,
    /// Flat guard bodies safely rewritten as an `if` without a label jump.
    pub flat_guard_structured: std::sync::atomic::AtomicU64,
    /// Nonempty flat guard bodies refused because labels, declarations, or
    /// structural nesting make an `if` scope unsafe.
    pub flat_guard_scope_refused: std::sync::atomic::AtomicU64,
    pub flat_guard_refused_label: std::sync::atomic::AtomicU64,
    pub flat_guard_refused_declaration: std::sync::atomic::AtomicU64,
    pub flat_guard_refused_depth: std::sync::atomic::AtomicU64,
    /// Flat guards whose fall arm produced no statements AFTER the false-edge
    /// preservation above (so: a degenerate branch, or a block whose emitted
    /// body is empty and which falls through to `taken` anyway).
    pub flat_guard_empty_fall: std::sync::atomic::AtomicU64,
    /// Flat guards whose `c == false` edge left the region and was therefore
    /// carried explicitly as `goto <fall>`. Each of these was a DROPPED EDGE
    /// before the fix — the guard had become unconditional.
    pub flat_fall_edge_kept: std::sync::atomic::AtomicU64,
    /// …because both edges really do go to one block (a degenerate branch).
    pub flat_empty_same: std::sync::atomic::AtomicU64,
    /// …because the false edge leaves the region (`fall` is a stop) and is then
    /// NOT carried anywhere: a genuinely dropped edge.
    pub flat_empty_stop: std::sync::atomic::AtomicU64,
    /// …for some other reason (`emit_arm` produced no statements).
    pub flat_empty_other: std::sync::atomic::AtomicU64,
}

pub static STRUCT_CENSUS: StructCensus = StructCensus {
    methods: std::sync::atomic::AtomicU64::new(0),
    blocks: std::sync::atomic::AtomicU64::new(0),
    irreducible_methods: std::sync::atomic::AtomicU64::new(0),
    irreducible_edges: std::sync::atomic::AtomicU64::new(0),
    loops: std::sync::atomic::AtomicU64::new(0),
    multi_latch_loops: std::sync::atomic::AtomicU64::new(0),
    multi_latch_methods: std::sync::atomic::AtomicU64::new(0),
    merge_fail_total: std::sync::atomic::AtomicU64::new(0),
    merge_via_header: std::sync::atomic::AtomicU64::new(0),
    merge_not_common: std::sync::atomic::AtomicU64::new(0),
    switch_ops: std::sync::atomic::AtomicU64::new(0),
    switch_targets: std::sync::atomic::AtomicU64::new(0),
    dispatch_lost_methods: std::sync::atomic::AtomicU64::new(0),
    dispatch_lost_ops: std::sync::atomic::AtomicU64::new(0),
    goto_sites: std::sync::atomic::AtomicU64::new(0),
    case_goto_attempts: std::sync::atomic::AtomicU64::new(0),
    case_goto_hits: std::sync::atomic::AtomicU64::new(0),
    case_goto_refused: std::sync::atomic::AtomicU64::new(0),
    goto_return_folds: std::sync::atomic::AtomicU64::new(0),
    goto_break_folds: std::sync::atomic::AtomicU64::new(0),
    shared_if_suffix_attempts: std::sync::atomic::AtomicU64::new(0),
    shared_if_suffix_hits: std::sync::atomic::AtomicU64::new(0),
    shared_if_suffix_refused: std::sync::atomic::AtomicU64::new(0),
    flat_guard_chains: std::sync::atomic::AtomicU64::new(0),
    flat_guard_fused: std::sync::atomic::AtomicU64::new(0),
    flat_guard_structured: std::sync::atomic::AtomicU64::new(0),
    flat_guard_scope_refused: std::sync::atomic::AtomicU64::new(0),
    flat_guard_refused_label: std::sync::atomic::AtomicU64::new(0),
    flat_guard_refused_declaration: std::sync::atomic::AtomicU64::new(0),
    flat_guard_refused_depth: std::sync::atomic::AtomicU64::new(0),
    flat_guard_empty_fall: std::sync::atomic::AtomicU64::new(0),
    flat_fall_edge_kept: std::sync::atomic::AtomicU64::new(0),
    flat_empty_same: std::sync::atomic::AtomicU64::new(0),
    flat_empty_stop: std::sync::atomic::AtomicU64::new(0),
    flat_empty_other: std::sync::atomic::AtomicU64::new(0),
    goto_by_reason: [
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
        std::sync::atomic::AtomicU64::new(0),
    ],
};

/// Bounded iterative reachability: can `from` reach `to` over `succs`?
fn cfg_reaches(succs: &[Vec<usize>], from: usize, to: usize) -> bool {
    if from == to {
        return true;
    }
    let mut seen = vec![false; succs.len()];
    let mut stack = vec![from];
    seen[from] = true;
    while let Some(n) = stack.pop() {
        for &s in &succs[n] {
            if s == to {
                return true;
            }
            if s < seen.len() && !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    false
}

/// Classify one method's structure and fold it into the census. Called once per
/// rendered method, from the render entry (NOT per walk: the binding-discovery
/// walk builds a second `Structurer` over the same CFG, so counting there would
/// double every number).
fn classify_structure(sc: &StructuredCfg, loops: &HashMap<usize, LoopInfo>, insns: &[Insn]) {
    use std::sync::atomic::Ordering;
    let m = &STRUCT_CENSUS;
    m.methods.fetch_add(1, Ordering::Relaxed);
    m.blocks.fetch_add(sc.blocks.len() as u64, Ordering::Relaxed);

    // Irreducible edges: non-dominated edges that still close a cycle.
    let n = sc.blocks.len();
    let mut irre = 0u64;
    for t in 0..n {
        for &h in &sc.full_succ[t] {
            if h >= n || h == t {
                continue;
            }
            // does h dominate t? (walk t's idom chain)
            let mut cur = Some(t);
            let mut dominated = false;
            let mut hops = 0usize;
            while let Some(c) = cur {
                if c == h {
                    dominated = true;
                    break;
                }
                cur = sc.idom[c];
                hops += 1;
                if hops > n + 1 {
                    break;
                }
            }
            if dominated {
                continue;
            }
            // Not dominated: irreducible only if it is part of a CYCLE, i.e.
            // the nominal header can get back to this block.
            if cfg_reaches(&sc.full_succ, h, t) {
                irre += 1;
            }
        }
    }
    if irre > 0 {
        m.irreducible_methods.fetch_add(1, Ordering::Relaxed);
        m.irreducible_edges.fetch_add(irre, Ordering::Relaxed);
    }

    m.loops.fetch_add(loops.len() as u64, Ordering::Relaxed);
    let multi = loops.values().filter(|l| l.latches.len() > 1).count() as u64;
    if multi > 0 {
        m.multi_latch_loops.fetch_add(multi, Ordering::Relaxed);
        m.multi_latch_methods.fetch_add(1, Ordering::Relaxed);
    }

    let mut sw_ops = 0u64;
    let mut sw_targets = 0u64;
    let mut disp = 0u64;
    for i in insns {
        match &i.op {
            Op::Switch(targets) => {
                sw_ops += 1;
                sw_targets += targets.len() as u64;
            }
            // At render time a recovered jump table has already become
            // `Op::Switch`, so any remaining `IndirectCall` is a dispatch whose
            // table was NOT recovered (the L1 case).
            Op::IndirectCall => disp += 1,
            _ => {}
        }
    }
    m.switch_ops.fetch_add(sw_ops, Ordering::Relaxed);
    m.switch_targets.fetch_add(sw_targets, Ordering::Relaxed);
    if disp > 0 {
        m.dispatch_lost_methods.fetch_add(1, Ordering::Relaxed);
        m.dispatch_lost_ops.fetch_add(disp, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Fallback census: how many methods are a MARKER rather than code
// ---------------------------------------------------------------------------
//
// The per-method verdicts classify each rendered body by what is WRONG with it
// (an unnamed value, a missing region, an unresolved dispatch). That leaves a
// prior question unanswered: how many bodies are not translations at all? Those
// are the four marker strings below, and they have very different causes and
// very different fixes — a size cap, an unreadable body, an unsupported
// pattern, or a body that recovered but structured to nothing. Counting them
// together as "declined" hid which one to work on.
pub struct FallbackCensus {
    /// Methods considered at all (so the counts below have a denominator).
    pub attempted: std::sync::atomic::AtomicU64,
    /// Bigger than the text pass's ceiling: skipped on purpose, not a failure —
    /// the IR pipeline may still have decompiled it.
    pub too_large: std::sync::atomic::AtomicU64,
    /// The method's bytes could not be read from the image.
    pub bytes_unreadable: std::sync::atomic::AtomicU64,
    /// Recovery returned no IR: the lifter met a shape it does not model.
    pub unsupported_pattern: std::sync::atomic::AtomicU64,
    /// IR recovered, but the structurer emitted no statements — an empty shell.
    pub empty_body: std::sync::atomic::AtomicU64,
}

pub static FALLBACK: FallbackCensus = FallbackCensus {
    attempted: std::sync::atomic::AtomicU64::new(0),
    too_large: std::sync::atomic::AtomicU64::new(0),
    bytes_unreadable: std::sync::atomic::AtomicU64::new(0),
    unsupported_pattern: std::sync::atomic::AtomicU64::new(0),
    empty_body: std::sync::atomic::AtomicU64::new(0),
};

/// One line per fallback reason, with the share of attempted methods.
pub fn fallback_census_report() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    let attempted = FALLBACK.attempted.load(Relaxed);
    if attempted == 0 {
        return String::new();
    }
    let g = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed);
    let pct = |n: u64| 100.0 * n as f64 / attempted as f64;
    let too_large = g(&FALLBACK.too_large);
    let unreadable = g(&FALLBACK.bytes_unreadable);
    let unsupported = g(&FALLBACK.unsupported_pattern);
    let empty = g(&FALLBACK.empty_body);
    let total = too_large + unreadable + unsupported + empty;
    let mut s = String::new();
    s.push_str(&format!(
        "[fallback] bodies that are a MARKER, not a translation: {total} of {attempted} attempted ({:.1}%)\n",
        pct(total)
    ));
    s.push_str(&format!(
        "[fallback]   too large for the text pass ({too_large}, {:.1}%) — skipped on purpose; the IR pipeline may still have handled it\n",
        pct(too_large)
    ));
    s.push_str(&format!(
        "[fallback]   body bytes unreadable ({unreadable}, {:.1}%) — image/offset problem, not a decompiler problem\n",
        pct(unreadable)
    ));
    s.push_str(&format!(
        "[fallback]   unsupported pattern ({unsupported}, {:.1}%) — NO IR: a lifter gap (this is the one that is a real decompiler bug)\n",
        pct(unsupported)
    ));
    s.push_str(&format!(
        "[fallback]   recovered but structured to nothing ({empty}, {:.1}%) — an empty body shell, a structurer gap\n",
        pct(empty)
    ));
    s
}

/// Human-readable structural census. Printed next to the fidelity verdicts, so
/// "64% of methods contain a goto" can be read together with the shapes that
/// produced them.
pub fn struct_census_report() -> String {
    use std::sync::atomic::Ordering;
    let g = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
    let m = &STRUCT_CENSUS;
    let n = g(&m.methods);
    if n == 0 {
        return String::new();
    }
    let pct = |num: u64, den: u64| {
        if den == 0 {
            0.0
        } else {
            100.0 * num as f64 / den as f64
        }
    };
    let mut s = String::new();
    s.push_str(&format!(
        "[struct] shapes that force a `goto` ({n} methods, {} blocks):\n",
        g(&m.blocks)
    ));
    s.push_str(&format!(
        "[struct]   irreducible edges (a cycle no single header dominates): {} in {} methods ({:.1}%)\n",
        g(&m.irreducible_edges),
        g(&m.irreducible_methods),
        pct(g(&m.irreducible_methods), n)
    ));
    s.push_str(&format!(
        "[struct]   natural loops: {}; multiply-latched ({} loops in {} methods)\n",
        g(&m.loops),
        g(&m.multi_latch_loops),
        g(&m.multi_latch_methods)
    ));
    s.push_str(&format!(
        "[struct]   if with no reconverging join (find_merge gave up): {} sites; \
         join from the branch HEADER's ipdom: {}; REFUSED (join not reachable from both arms): {}\n",
        g(&m.merge_fail_total),
        g(&m.merge_via_header),
        g(&m.merge_not_common)
    ));
    s.push_str(&format!(
        "[struct]   switches: {} recovered tables, {} case targets | unrecovered dispatch: {} ops in {} methods ({:.1}%)\n",
        g(&m.switch_ops),
        g(&m.switch_targets),
        g(&m.dispatch_lost_ops),
        g(&m.dispatch_lost_methods),
        pct(g(&m.dispatch_lost_methods), n)
    ));
    s.push_str(&format!(
        "[struct]   `goto` statements requested by the walk: {} (upper bound; the final text keeps fewer)\n",
        g(&m.goto_sites)
    ));
    // The split that makes the total actionable: which CALL SITE produced each
    // goto. Sorted biggest-first by hand (the counts are few and the order is
    // part of the report's meaning — the top line is where the leverage is).
    let mut reasons: Vec<(GotoReason, u64)> = [
        GotoReason::NoMerge,
        GotoReason::Backward,
        GotoReason::AlreadyEmitted,
        GotoReason::Dispatch,
        GotoReason::MergeEmitted,
        GotoReason::SkipEmpty,
        GotoReason::FlatGuard,
        GotoReason::HandlerPad,
        GotoReason::Uncategorised,
        GotoReason::LoopExit,
    ]
    .into_iter()
    .map(|r| (r, g(&m.goto_by_reason[r.idx()])))
    .collect();
    s.push_str(&format!(
        "[struct]   gotos folded to void returns: {}\n",
        g(&m.goto_return_folds)
    ));
    s.push_str(&format!(
        "[struct]   gotos folded to break: {}\n",
        g(&m.goto_break_folds)
    ));
    s.push_str(&format!(
        "[struct]   shared-if suffix folds: {}/{}; refused={}\n",
        g(&m.shared_if_suffix_hits),
        g(&m.shared_if_suffix_attempts),
        g(&m.shared_if_suffix_refused)
    ));
    s.push_str(&format!(
        "[struct]   switch-arm goto case rewrites: {}/{}; refused non-case targets={}\n",
        g(&m.case_goto_hits),
        g(&m.case_goto_attempts),
        g(&m.case_goto_refused)
    ));
    s.push_str(&format!(
        "[struct]   flat-guard chains: {}; guard shells fused into `||`: {}; vacuous (fall arm empty): {}\n",
        g(&m.flat_guard_chains),
        g(&m.flat_guard_fused),
        g(&m.flat_guard_empty_fall)
    ));
    s.push_str(&format!(
        "[struct]     flat guards rewritten without goto: {}; refused: labels={} declarations={} depth={} (total={})\n",
        g(&m.flat_guard_structured),
        g(&m.flat_guard_refused_label),
        g(&m.flat_guard_refused_declaration),
        g(&m.flat_guard_refused_depth),
        g(&m.flat_guard_scope_refused)
    ));
    s.push_str(&format!(
        "[struct]     vacuous because: same block both edges={} | false edge leaves the region (should be 0 — carried as a goto)={} | emit_arm produced nothing={}\n",
        g(&m.flat_empty_same),
        g(&m.flat_empty_stop),
        g(&m.flat_empty_other)
    ));
    s.push_str(&format!(
        "[struct]     dropped `c == false` edges CARRIED explicitly as a goto: {}\n",
        g(&m.flat_fall_edge_kept)
    ));
    reasons.sort_by(|a, b| b.1.cmp(&a.1));
    for (r, c) in reasons {
        s.push_str(&format!(
            "[struct]     goto from {}: {} ({:.1}% of gotos)\n",
            r.label(),
            c,
            pct(c, g(&m.goto_sites))
        ));
    }
    s
}

/// Blocks the CFG can reach from the entry, over taken/fall edges plus every
/// recovered `switch` arm.
///
/// A `switch` block has no taken/fall — its arms are reachable only through the
/// jump table — so they must count as reachable here or every correctly-emitted
/// switch would look like dead code to the coverage repair.
/// Blocks reachable from the entry — see [`reachable_blocks`] for the raw set.
fn reachable_blocks_entry(sc: &StructuredCfg, insns: &[Insn]) -> Vec<bool> {
    reachable_blocks(sc, insns)
}

/// Does the block at `b` hold a branch INTO `adopted` other than the method's
/// own entry block?
///
/// The exclusion of block 0 is what keeps a THUNK out: an IL2CPP adjustor /
/// runtime-invoker stub is a bare `b <method entry>`, and such a stub sitting
/// in the gap after this method would otherwise look like this method's cold
/// region. A cold region resumes INSIDE the body, never at its first block.
fn branches_into(succs: &[Vec<usize>], b: usize, adopted: &[bool]) -> bool {
    succs[b]
        .iter()
        .any(|&s| s != 0 && s < adopted.len() && adopted[s])
}

/// The entry-reachable set PLUS the blocks of the method that no recovered edge
/// from the entry can reach: its out-of-line cold region.
///
/// LLVM lays a function's exception landing pads and cold blocks out AFTER the
/// body's last `ret`, and they are entered by the UNWINDER — which a CFG built
/// from branches cannot see. They are therefore "CFG-unreachable" while being
/// this method's own code, and every call in them was missing from the text:
/// 90 of the 198 methods still flagged as dropping calls in the reference
/// sample, holding 1,107 native `bl` between them.
///
/// Ownership is proven by an edge, never assumed from layout. A component of
/// unreachable blocks is adopted only when one of its blocks BRANCHES INTO the
/// adopted set (`branches_into`, excluding the entry block). A different
/// function never branches into the middle of this one's body, so `cbz x0,
/// 0x1DD7B60` from the cold region at 0x1DD7CA8 into
/// `ConstraintCollection.Clear`'s own body is proof they are one function —
/// which is why hand-inspecting that method and concluding "the leftover blocks
/// are a neighbour's code, they have no edge INTO them" was wrong in the other
/// direction: no edge in says nothing, an edge OUT decides.
///
/// The fixpoint matters: a cold region's landing pads can chain, so a component
/// reachable only from an already-adopted one must be adopted too.
///
/// Adoption is a measurement+emission change, not a structuring one — the
/// blocks go through [`Structurer::coverage_repair`], which renders them FLAT
/// behind a marker, so no loop body is duplicated and no call is expanded
/// twice.
fn adjoined_blocks(sc: &StructuredCfg, insns: &[Insn]) -> Vec<bool> {
    let n = sc.blocks.len();
    let mut adopted = reachable_blocks_entry(sc, insns);
    if n == 0 {
        return adopted;
    }
    // Successor lists (taken, fall, and every switch target) and their reverse,
    // built ONCE so the component walk below is linear in the CFG size.
    let mut succs: Vec<Vec<usize>> = Vec::with_capacity(n);
    for b in 0..n {
        let mut succ: Vec<usize> = [sc.blocks[b].succ_taken, sc.blocks[b].succ_fall]
            .into_iter()
            .flatten()
            .filter(|&s| s < n)
            .collect();
        if let Op::Switch(targets) = &insns[sc.blocks[b].end - 1].op {
            for t in targets {
                if let Some(&s) = sc.block_of.get(*t) {
                    if s < n {
                        succ.push(s);
                    }
                }
            }
        }
        succs.push(succ);
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    // Components of the not-yet-adopted blocks. UNDIRECTED over the CFG's edges:
    // a landing pad reached from another pad has to land in the same component,
    // or the fixpoint below needs one pass per chain link.
    let mut component: Vec<usize> = (0..n).map(|b| if adopted[b] { b } else { usize::MAX }).collect();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for root in 0..n {
        if adopted[root] || component[root] != usize::MAX {
            continue;
        }
        let id = comps.len();
        let mut members = Vec::new();
        let mut stack = vec![root];
        component[root] = id;
        while let Some(b) = stack.pop() {
            members.push(b);
            for &s in succs[b].iter().chain(preds[b].iter()) {
                if !adopted[s] && component[s] == usize::MAX {
                    component[s] = id;
                    stack.push(s);
                }
            }
        }
        comps.push(members);
    }
    loop {
        let mut changed = false;
        for members in &comps {
            if members.iter().all(|&b| adopted[b]) {
                continue;
            }
            if members.iter().any(|&b| branches_into(&succs, b, &adopted)) {
                for &b in members {
                    adopted[b] = true;
                }
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    adopted
}

fn reachable_blocks(sc: &StructuredCfg, insns: &[Insn]) -> Vec<bool> {
    let mut seen = vec![false; sc.blocks.len()];
    let mut stack: Vec<usize> = Vec::new();
    if !seen.is_empty() {
        seen[0] = true;
        stack.push(0);
    }
    while let Some(b) = stack.pop() {
        let mut succ: Vec<usize> = [sc.blocks[b].succ_taken, sc.blocks[b].succ_fall]
            .into_iter()
            .flatten()
            .collect();
        if let Op::Switch(targets) = &insns[sc.blocks[b].end - 1].op {
            for t in targets {
                if let Some(&s) = sc.block_of.get(*t) {
                    succ.push(s);
                }
            }
        }
        for s in succ {
            if s < sc.blocks.len() && !seen[s] {
                seen[s] = true;
                stack.push(s);
            }
        }
    }
    seen
}

/// Short op name, for the `ILR_COV` coverage report.
fn self_op_name(op: &Op) -> &'static str {
    match op {
        Op::CondJump(_) => "CondJump",
        Op::Jump(_) => "Jump",
        Op::Return => "Return",
        Op::Throw => "Throw",
        Op::Switch(_) => "Switch",
        Op::IndirectCall => "IndirectCall",
        Op::Call | Op::CallVoid => "Call",
        _ => "other",
    }
}

/// `(blocks never walked, total blocks)` for this thread's last render.
pub(crate) fn unemitted_blocks() -> (usize, usize) {
    (
        UNEMITTED_BLOCKS.with(|c| c.get()),
        TOTAL_BLOCKS.with(|c| c.get()),
    )
}

/// `(def resolutions, recursion high-water mark)` for this thread.
///
/// Exposed so the regression test for the def-cycle guard can assert that a
/// loop-carried def is referenced instead of expanded, without timing anything
/// or parsing the rendered text.
#[cfg(test)]
pub(crate) fn def_resolution_counts() -> (u64, u64) {
    (
        DEF_CALLS.with(|c| c.get()),
        DEF_MAXDEPTH.with(|m| m.get()),
    )
}

/// Test hook: resolve a USE of local `v` at `use_insn` through the real
/// inlining machinery and render the resulting expression.
///
/// The cycle guard lives inside `MethodCtx`, and a full render needs a
/// structurable CFG, so this drives the resolver directly — which is what the
/// guard actually protects.
#[cfg(test)]
pub(crate) fn resolve_local_use_for_test(
    ir: &super::RecoveredIr,
    use_insn: usize,
    v: u32,
) -> String {
    let parent_chain: HashMap<i32, Vec<i32>> = HashMap::new();
    let ctx = MethodCtx::build(&ir.insns, &ir.table, -1, &parent_chain, false);
    ctx.cur_insn.set(use_insn);
    let e = ctx.operand_expr(&Operand::Local(v), 0);
    let mut buf = String::new();
    e.render(&mut buf);
    buf
}

/// Zero the counters returned by [`def_resolution_counts`].
#[cfg(test)]
pub(crate) fn reset_def_resolution_counts() {
    DEF_CALLS.with(|c| c.set(0));
    DEF_DEPTH.with(|c| c.set(0));
    DEF_MAXDEPTH.with(|m| m.set(0));
}

/// `ILR_RSSDBG=1`: print the process RSS at the start and end of every method,
/// so a memory blow-up can be attributed to the method that caused it.
fn rss_probe(tag: &str, rva: u64, size: usize) {
    let rss_kb = std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| {
            s.split_whitespace()
                .nth(1)
                .and_then(|p| p.parse::<u64>().ok())
        })
        .map(|pages| pages * 4)
        .unwrap_or(0);
    eprintln!("[rss] {tag} {rva:#x} size={size} rss={rss_kb}KB");
}

/// Optional method-count cap for the determinism gate; see the strided sample
/// at the `ILR_DET_LIMIT` site in [`decompile_methods_to_csharp`].
fn det_limit() -> Option<usize> {
    std::env::var("ILR_DET_LIMIT")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
}

/// Optional wall-clock safety valve for the whole pipeline.
///
/// OFF by default, and that is the point: a wall-clock cut-off makes the
/// emitted body depend on machine load, so the same binary yields different
/// text on different runs. Set `ILR_TIME_LIMIT=<ms>` to re-enable it when a
/// hard runtime bound matters more than reproducibility — the determinism gate
/// (`scripts/det_gate.sh`) will then report the difference instead of hiding it.
fn time_limit_ms() -> Option<u64> {
    std::env::var("ILR_TIME_LIMIT")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|ms| *ms > 0)
}

/// Render-phase wall-clock budget; see [`time_limit_ms`].
fn render_deadline() -> Option<std::time::Instant> {
    time_limit_ms().map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms))
}

/// FNV-1a 64. Deliberately not `DefaultHasher`: this digest is compared across
/// SEPARATE PROCESSES (see [`det_write_manifest`]), and a hand-rolled constant
/// seed removes any question about seed stability or std's internals changing.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Determinism gate: write `<rva>\t<digest>\t<bytes>` per recovered method,
/// sorted by RVA, to `path` (`ILR_DET_MANIFEST=<path>`).
///
/// Run the dumper twice in two separate processes and diff the two manifests.
/// Every ordering leak in the pipeline — a `HashMap` iterated to build a
/// statement list, a `HashSet` seed leaking into a decision, an unstable sort,
/// a wall-clock cut-off — shows up as a differing digest next to the RVA that
/// produced it. This is the exhaustive form of "is the decompiler
/// deterministic?": it does not depend on anyone remembering which map got
/// iterated, and it cannot be satisfied by fixing only the maps we found today.
fn det_write_manifest(path: &str, results: &HashMap<u64, String>) {
    let mut rows: Vec<(u64, u64, usize)> = results
        .iter()
        .map(|(rva, text)| (*rva, fnv1a64(text.as_bytes()), text.len()))
        .collect();
    rows.sort_unstable_by_key(|r| r.0);
    let mut out = String::with_capacity(rows.len() * 34);
    for (rva, digest, len) in rows {
        out.push_str(&format!("{rva:016x}\t{digest:016x}\t{len}\n"));
    }
    match std::fs::write(path, out) {
        Ok(()) => eprintln!("[det] manifest: {} methods -> {path}", results.len()),
        Err(e) => eprintln!("[det] manifest write FAILED: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Pattern catalog — the C#-construct binding layer (data-driven)
// ---------------------------------------------------------------------------
//
// The lifters decode native instructions into the shared IR and the
// structurer/rewrite passes already rebuild statements, but each "dirty
// output" fix used to be a one-off `if` arm buried somewhere. This catalog
// makes those fixes DATA: every binding is a named, documented entry in one
// registry with a trigger shape, a condition, a rewrite, and (later) a
// before/after spec test. Coverage compounds by appending entries, exactly
// like a "C# ⇄ assembly construct table" — without paying JSON parse costs
// in the per-method hot path or losing type safety.
//
// Design notes:
//  * Post-order: children are bound first, then this list's bindings run.
//  * Each binding is iterated to a fixpoint (bounded) per list.
//  * A failed match ALWAYS leaves the lower-level sequence untouched
//    (the decompiler's "accurate lower-level > plausible-but-wrong" rule).
//  * Bindings that need *types* (e.g. `x == 0` vs `x == null`) run at
//    statement time where `MethodCtx`/local types are still reachable, not
//    here — this module only sees the pure Expr/Stmt tree.
pub mod catalog {
    use super::{CmpKind, Expr, Stmt};

    struct Binding {
        name: &'static str,
        /// How confident the rewrite is: Exact / Metadata / Pattern /
        /// Approximate. Drives optional provenance comments on emission.
        run: fn(&mut Vec<Stmt>) -> usize,
    }

    /// Registry — APPEND NEW BINDINGS HERE. Order matters: earlier entries
    /// run first. Keep each entry small + auditable + testable in isolation.
    const BINDINGS: &[Binding] = &[
        Binding {
            name: "fold_allocation_null_test",
            run: fold_allocation_null_test,
        },
        // QUEUED ENTRIES (roadmap — each gets its own binding fn):
        //  * event_subscribe_cas: Delegate.Combine/Remove retry loop →
        //    `evt += handler` / `evt -= handler` (trigger: CAS call + field
        //    reload in a single-back-edge loop)
        //  * delegate_ctor_args: new Action<T>(this, method) — fill the
        //    duplicated `Local, Local` ctor operands from the (target,
        //    method-cell) loads that precede the call
        //  * icall_signature: fn_* trampoline calls get parameter/return
        //    types from the binary's icall/rgctx tables (metadata API layer)
        //  * null_vs_zero: `x == 0` → `x == null` where the local's type
        //    is a reference (needs MethodCtx — run at statement time)
        //  * fp_field_store: FP-only method bodies whose only effects are
        //    float field stores (currently dropped pre-render)
        //  * for_loop_shape: init/cond/inc/backedge → `for` (needs the
        //    loop header's pre-header defs)
    ];

    pub fn apply(stmts: &mut Vec<Stmt>) {
        // post-order: children first
        for s in stmts.iter_mut() {
            match s {
                Stmt::If { then_body, else_body, .. } => {
                    apply(then_body);
                    apply(else_body);
                }
                Stmt::While { body, .. } => apply(body),
                Stmt::DoWhile { body, .. } => apply(body),
                Stmt::Try { try_body, catches, finally_body } => {
                    apply(try_body);
                    for (_, body) in catches.iter_mut() {
                        apply(body);
                    }
                    if let Some(f) = finally_body {
                        apply(f);
                    }
                }
                _ => {}
            }
        }
        // then this list's bindings, each to a bounded fixpoint.
        // ILR_BINDINGS=1 reports per-binding rewrite counts — the only way to
        // tell which catalog entry is actually doing work on real methods, and
        // which one silently never fires.
        let tally = std::env::var("ILR_BINDINGS").is_ok();
        for b in BINDINGS {
            let mut iters = 0;
            let mut hits = 0usize;
            loop {
                let n = (b.run)(stmts);
                hits += n;
                if n == 0 || iters >= 8 {
                    break;
                }
                iters += 1;
            }
            if tally && hits > 0 {
                eprintln!("[bindings] {}: {hits} rewrites in {iters} passes", b.name);
            }
        }
    }

    /// Is `e` an allocation expression (`new T(...)` / `new T[n]`)?
    fn is_alloc(e: &Expr) -> bool {
        matches!(e, Expr::NewObj(..) | Expr::NewArr(..))
    }

    /// Fold `new T() == 0/null` (and `!=`) to false/true.
    ///
    /// A managed allocation can never hand back a null *test that carries
    /// C#-level logic*: il2cpp may abort on OOM, but in the source language
    /// `new` never yields null, so an `if (new object() == 0) { L: }` shell
    /// left behind by failed call-merge recovery is dead and collapses. The
    /// Renderer then drops the constant-false `if` entirely (and the label
    /// inside with it), and `drop_unreachable_gotos` sweeps any dangling
    /// goto that pointed at that label.
    fn fold_allocation_null_test(stmts: &mut Vec<Stmt>) -> usize {
        let mut n = 0usize;
        for stmt in stmts.iter_mut() {
            let cond = match stmt {
                Stmt::If { cond, .. } | Stmt::While { cond, .. } => cond,
                Stmt::DoWhile { cond, .. } => cond,
                _ => continue,
            };
            let Expr::Cmp(k, a, b) = cond else { continue };
            let alloc_zero = |x: &Expr, y: &Expr| {
                is_alloc(x) && matches!(y, Expr::Int(0) | Expr::Null)
            };
            let eq = matches!(k, CmpKind::Eq);
            let ne = matches!(k, CmpKind::Ne);
            if alloc_zero(a, b) || alloc_zero(b, a) {
                if eq || ne {
                    *cond = Expr::Bool(ne); // Eq → false, Ne → true
                    n += 1;
                }            }
        }
        n
    }
}

// ---------------------------------------------------------------------------
// Reaching-definition invariant tests
// ---------------------------------------------------------------------------
//
// These lock the φ-point rule: when two DIFFERENT definitions of the same
// local reach a join, the join's IN set must contain NO entry for that local.
// Every consumer depends on it — `compute_var_reads` marks such locals as
// variables and `inline_local_at` renders the name, so the value is correct
// on every path. A meet that lets the last predecessor win instead makes a
// shared `ret` return one arbitrary path's expression (see the
// GetEffectiveSchemaType @0x1f4e79c regression below).
#[cfg(test)]
mod reaching_def_tests {
    use super::*;
    use crate::recovery::ir::{CompareKind, Op, Operand};

    fn cond(scratch: u32, target: usize, address: u64) -> Vec<Insn> {
        vec![
            Insn {
                op: Op::Compare(CompareKind::NotEqual),
                operands: vec![
                    Operand::Local(scratch),
                    Operand::Immediate(1),
                    Operand::Immediate(0),
                ],
                dest_type: None,
                address,
            },
            Insn {
                op: Op::CondJump(target),
                operands: vec![Operand::Local(scratch)],
                dest_type: None,
                address,
            },
        ]
    }

    /// THREE predecessors define `v1` differently and all reach the `Return`.
    /// The join must NOT claim a single reaching def.
    ///
    /// Three is the minimum that exposes the bug: with two, the meet's
    /// remove-on-conflict happened to leave the map empty, and only the
    /// THIRD predecessor re-inserted a def — which is exactly what happened
    /// on real methods (the shared `ret` in GetEffectiveSchemaType @0x1f4e79c
    /// has five predecessors, and the fifth won).
    #[test]
    fn conflicting_defs_at_join_are_dropped() {
        let scratch = u32::MAX - 1;
        let mut insns: Vec<Insn> = Vec::new();
        // B0: v1 = v3 ; if (… != 0) goto 9 (the join)
        insns.push(Insn::move_(Operand::Local(1), Operand::Local(3), 0x1000));
        insns.extend(cond(scratch, 9, 0x1004));
        // B1: v1 = v4 ; if (… != 0) goto 9
        insns.push(Insn::move_(Operand::Local(1), Operand::Local(4), 0x1008));
        insns.extend(cond(scratch, 9, 0x100c));
        // B2: v1 = v5 ; goto 9
        insns.push(Insn::move_(Operand::Local(1), Operand::Local(5), 0x1010));
        insns.push(Insn {
            op: Op::Jump(9),
            operands: vec![],
            dest_type: None,
            address: 0x1014,
        });
        // filler so the join gets its own block
        insns.push(Insn {
            op: Op::Nop,
            operands: vec![],
            dest_type: None,
            address: 0x1018,
        });
        // B4: the join (three predecessors)
        insns.push(Insn {
            op: Op::Return,
            operands: vec![Operand::Local(1)],
            dest_type: None,
            address: 0x101c,
        });

        let r = compute_reaching_definitions(&insns);
        // Find the join block (the one holding the Return).
        let ret_idx = insns.len() - 1;
        let join = r
            .blocks
            .iter()
            .position(|&(s, e)| ret_idx >= s && ret_idx < e)
            .expect("return block");
        assert!(
            !r.gen_kill_in[join].contains_key(&1),
            "v1 has conflicting defs reaching the join — IN must have no entry, got {:?}",
            r.gen_kill_in[join].get(&1)
        );
    }

    /// A single definition reaching a join still resolves (the fix must not
    /// throw away unambiguous merges).
    #[test]
    fn single_reaching_def_survives_join() {
        let scratch = u32::MAX - 1;
        let mut insns: Vec<Insn> = Vec::new();
        // B0: v1 = v3 ; if (… != 0) goto 7
        insns.push(Insn::move_(Operand::Local(1), Operand::Local(3), 0x2000));
        insns.extend(cond(scratch, 7, 0x2004));
        // B1: v2 = v4 ; goto 7   (does NOT touch v1)
        insns.push(Insn::move_(Operand::Local(2), Operand::Local(4), 0x2008));
        insns.push(Insn {
            op: Op::Jump(7),
            operands: vec![],
            dest_type: None,
            address: 0x200c,
        });
        insns.push(Insn {
            op: Op::Nop,
            operands: vec![],
            dest_type: None,
            address: 0x2010,
        });
        insns.push(Insn {
            op: Op::Nop,
            operands: vec![],
            dest_type: None,
            address: 0x2014,
        });
        insns.push(Insn {
            op: Op::Return,
            operands: vec![Operand::Local(1)],
            dest_type: None,
            address: 0x2018,
        });

        let r = compute_reaching_definitions(&insns);
        let ret_idx = insns.len() - 1;
        let join = r
            .blocks
            .iter()
            .position(|&(s, e)| ret_idx >= s && ret_idx < e)
            .expect("return block");
        assert_eq!(
            r.gen_kill_in[join].get(&1).copied(),
            Some(0),
            "the only def of v1 that reaches the join is insn 0"
        );
    }
}

// ---------------------------------------------------------------------------
// Structurer goto-reduction tests (Worker I)
// ---------------------------------------------------------------------------
//
// Safety proof (inert on well-formed IR): `fold_branch_over_join` keeps its
// exact gates (lone goto, noise-only else, label later in THIS list, no flat
// label inside) and only changes in two ways, both of which fold strictly
// more of what the old code already accepted: (a) siblings are visited
// reverse, so a chain `[G1, B1, G2, B2, J:]` folds G2 first and then G1
// instead of stranding G2 cross-level (forward order left 1 goto, reverse
// leaves 0); (b) recursion extends to every compound statement (Try, Switch,
// For, Using, Lock, ForEach) so guards inside handlers/arms fold with the
// same gates instead of never being visited. No write moves, no condition
// deleted (the predicate is inverted, not dropped), the join label is kept
// so every other `goto L` still lands. Well-formed diamonds without guard
// chains render identically.
#[cfg(test)]
mod structurer_goto_tests {
    use super::*;

    fn var(n: &str) -> Expr {
        Expr::Var(n.into())
    }

    fn guard(c: &str, label: &str) -> Stmt {
        Stmt::If {
            cond: var(c),
            then_body: vec![Stmt::Goto(label.into())],
            else_body: vec![],
        }
    }

    fn count_gotos(stmts: &[Stmt]) -> usize {
        let mut n = 0usize;
        for s in stmts {
            match s {
                Stmt::Goto(_) => n += 1,
                Stmt::If { then_body, else_body, .. } => {
                    n += count_gotos(then_body);
                    n += count_gotos(else_body);
                }
                Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => {
                    n += count_gotos(body);
                }
                Stmt::Try { try_body, catches, finally_body } => {
                    n += count_gotos(try_body);
                    for (_, b) in catches {
                        n += count_gotos(b);
                    }
                    if let Some(f) = finally_body {
                        n += count_gotos(f);
                    }
                }
                Stmt::Switch { cases, tail, .. } => {
                    for (_, b) in cases {
                        n += count_gotos(b);
                    }
                    n += count_gotos(tail);
                }
                Stmt::For { body, .. }
                | Stmt::Using { body, .. }
                | Stmt::Lock { body, .. }
                | Stmt::ForEach { body, .. } => n += count_gotos(body),
                _ => {}
            }
        }
        n
    }

    #[test]
    fn goto_to_void_return_folds_globally_but_never_in_finally() {
        let mut stmts = vec![
            Stmt::If {
                cond: var("nested"),
                then_body: vec![Stmt::Goto("Lexit".into())],
                else_body: vec![],
            },
            Stmt::While {
                cond: var("again"),
                body: vec![Stmt::Goto("Lexit".into())],
            },
            Stmt::Label("Lexit".into()),
            Stmt::Comment("target-adjacent comment".into()),
            Stmt::Return(None),
            Stmt::Try {
                try_body: vec![],
                catches: vec![],
                finally_body: Some(vec![Stmt::Goto("Lfinally".into())]),
            },
            Stmt::Label("Lfinally".into()),
            Stmt::Return(None),
        ];
        fold_gotos_to_void_return(&mut stmts);
        assert_eq!(count_gotos(&stmts), 1, "only finally source remains: {stmts:?}");
        let Stmt::If { then_body, .. } = &stmts[0] else { panic!("missing if"); };
        assert!(matches!(then_body.as_slice(), [Stmt::Return(None)]));
        let Stmt::While { body, .. } = &stmts[1] else { panic!("missing while"); };
        assert!(matches!(body.as_slice(), [Stmt::Return(None)]));
        let Stmt::Try { finally_body: Some(body), .. } = &stmts[5] else {
            panic!("missing finally");
        };
        assert!(matches!(body.as_slice(), [Stmt::Goto(label)] if label == "Lfinally"));
    }

    #[test]
    fn goto_to_immediate_loop_and_switch_continuations_becomes_break() {
        let mut loop_stmts = vec![
            Stmt::While {
                cond: var("loop"),
                body: vec![Stmt::If {
                    cond: var("exit"),
                    then_body: vec![Stmt::Goto("Lafter_loop".into())],
                    else_body: vec![],
                }],
            },
            Stmt::Comment("continuation anchor".into()),
            Stmt::Label("Lafter_loop".into()),
            Stmt::Eval(var("after")),
        ];
        fold_gotos_to_break(&mut loop_stmts);
        assert_eq!(count_gotos(&loop_stmts), 0, "loop continuation goto remains: {loop_stmts:?}");
        assert!(render_stmts(&loop_stmts).contains("break;"));

        let mut switch_stmts = vec![
            Stmt::Switch {
                disc: "selector".into(),
                cases: vec![(0, vec![Stmt::Goto("Lafter_switch".into())])],
                tail: vec![],
            },
            Stmt::Label("Lafter_switch".into()),
            Stmt::Return(None),
        ];
        fold_gotos_to_break(&mut switch_stmts);
        assert_eq!(count_gotos(&switch_stmts), 0, "switch continuation goto remains: {switch_stmts:?}");
        assert!(render_stmts(&switch_stmts).contains("break;"));
    }

    #[test]
    fn shared_if_suffix_is_hoisted_after_both_arms() {
        let mut stmts = vec![Stmt::If {
            cond: var("condition"),
            then_body: vec![
                Stmt::Eval(var("then_prefix")),
                Stmt::Label("Lshared".into()),
                Stmt::Eval(var("shared_effect")),
                Stmt::Return(None),
            ],
            else_body: vec![Stmt::Eval(var("else_prefix")), Stmt::Goto("Lshared".into())],
        }];
        fold_shared_if_suffixes(&mut stmts);
        let Stmt::If { then_body, else_body, .. } = &stmts[0] else { panic!("if lost"); };
        assert!(matches!(then_body.as_slice(), [Stmt::Eval(_)]));
        assert!(matches!(else_body.as_slice(), [Stmt::Eval(_)]));
        assert!(matches!(
            stmts.as_slice(),
            [Stmt::If { .. }, Stmt::Label(label), Stmt::Eval(_), Stmt::Return(None)]
                if label == "Lshared"
        ));
        assert_eq!(count_gotos(&stmts), 0);
    }

    #[test]
    fn shared_if_suffix_refuses_declarations_and_preserves_external_label_refs() {
        let declaration = vec![Stmt::If {
            cond: var("condition"),
            then_body: vec![
                Stmt::Label("Lshared".into()),
                Stmt::Assign("local".into(), Some("int".into()), var("value")),
            ],
            else_body: vec![Stmt::Goto("Lshared".into())],
        }];
        let mut declaration = declaration;
        fold_shared_if_suffixes(&mut declaration);
        assert_eq!(count_gotos(&declaration), 1, "declaration scope was moved: {declaration:?}");

        let mut external_ref = vec![
            Stmt::If {
                cond: var("condition"),
                then_body: vec![Stmt::Label("Lshared".into()), Stmt::Eval(var("effect"))],
                else_body: vec![Stmt::Goto("Lshared".into())],
            },
            Stmt::Goto("Lshared".into()),
        ];
        fold_shared_if_suffixes(&mut external_ref);
        assert_eq!(count_gotos(&external_ref), 1, "the arm continuation goto should fold: {external_ref:?}");
        assert!(matches!(
            external_ref.as_slice(),
            [Stmt::If { .. }, Stmt::Label(label), Stmt::Eval(_), Stmt::Goto(target)]
                if label == "Lshared" && target == "Lshared"
        ), "external goto must retain its target after hoisting: {external_ref:?}");
    }

    #[test]
    fn goto_to_outer_loop_continuation_refuses_nested_breakable_and_finally_sources() {
        let mut stmts = vec![
            Stmt::While {
                cond: var("outer"),
                body: vec![
                    Stmt::Switch {
                        disc: "selector".into(),
                        cases: vec![(0, vec![Stmt::Goto("Lafter_outer".into())])],
                        tail: vec![],
                    },
                    Stmt::Try {
                        try_body: vec![],
                        catches: vec![],
                        finally_body: Some(vec![Stmt::Goto("Lafter_outer".into())]),
                    },
                ],
            },
            Stmt::Label("Lafter_outer".into()),
            Stmt::Return(None),
        ];
        fold_gotos_to_break(&mut stmts);
        assert_eq!(count_gotos(&stmts), 2, "outer-scope or finally goto was folded: {stmts:?}");
        assert!(!render_stmts(&stmts).contains("break;"));
    }

    /// A 2-guard chain with a prelude between the guards must fold FULLY.
    ///
    /// `[G1, B1, G2, B2, J:]` with no flat labels is two nested guards, not
    /// two jumps. Forward order folded G1 first and stranded G2 cross-level
    /// (1 goto left); reverse folds G2 first and then G1 (0 gotos).
    #[test]
    fn guard_chain_with_prelude_folds_fully() {
        let mut stmts = vec![
            guard("c1", "J"),
            Stmt::Eval(var("prelude")),
            guard("c2", "J"),
            Stmt::Eval(var("body")),
            Stmt::Label("J".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            0,
            "both guards must fold to nested `if (!c)` shells, got: {stmts:?}"
        );
        // The predicates survive (inverted), the bodies survive nested.
        let Stmt::If { cond, then_body, .. } = &stmts[0] else {
            panic!("outer guard did not fold to an if head: {stmts:?}");
        };
        assert!(matches!(cond, Expr::Not(_)), "outer predicate not inverted: {cond:?}");
        assert!(
            then_body.iter().any(|s| matches!(s, Stmt::Eval(_))),
            "prelude lost in the fold: {stmts:?}"
        );
    }

    /// A flat label between the guard and its join blocks the fold when an edge
    /// from OUTSIDE the region targets it (that edge would become conditional
    /// under `!c`). Regression pin for 0x2CA7AC: the `v100 != 0` guard's region
    /// holds `L14:`, landed on from across the method — and a prior flat-gating
    /// variant deleted the guard itself here instead of keeping it.
    #[test]
    fn flat_label_still_blocks_the_fold() {
        let mut stmts = vec![
            guard("v100", "L3"),
            Stmt::Eval(var("cleanup")),
            Stmt::Label("L14".into()),
            Stmt::Eval(var("tail")),
            Stmt::Label("L3".into()),
            // The outside edge: in the real method many `goto L14` land here
            // from beyond this guard's region.
            Stmt::Goto("L14".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            2,
            "the guard goto and the outside edge must both survive: {stmts:?}"
        );
        let out = render_stmts(&stmts);
        assert!(
            out.contains("v100"),
            "the guard condition must survive (a prior variant deleted it):\n{out}"
        );
        assert!(
            out.contains("L14:"),
            "the outside landing site must survive:\n{out}"
        );
    }

    /// A guard whose region holds a flat label NOTHING targets must still fold.
    ///
    /// The label is a dead merge anchor: no edge in the whole method reaches
    /// it, and `strip_quiet_labels` removes it right after this fold. Refusing
    /// on it left real guard chains half-structured (one `goto` per dead
    /// anchor), while folding moves nothing observable — the predicate is
    /// inverted, never deleted, and the region moves verbatim.
    #[test]
    fn guard_with_dead_interior_label_folds_fully() {
        let mut stmts = vec![
            guard("c1", "J"),
            Stmt::Eval(var("prelude")),
            Stmt::Label("Ldead".into()),
            Stmt::Eval(var("body")),
            Stmt::Label("J".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            0,
            "a dead anchor must not block the fold, got: {stmts:?}"
        );
        let Stmt::If { cond, then_body, .. } = &stmts[0] else {
            panic!("outer guard did not fold to an if head: {stmts:?}");
        };
        assert!(matches!(cond, Expr::Not(_)), "outer predicate not inverted: {cond:?}");
        assert!(
            then_body.iter().any(|s| matches!(s, Stmt::Eval(_))),
            "prelude/body lost in the fold: {stmts:?}"
        );
        let out = render_stmts(&stmts);
        assert!(
            out.contains("c1"),
            "the guard condition must survive (inverted), got:\n{out}"
        );
    }

    #[test]
    fn flat_guard_body_nesting_refuses_scope_changes_and_labels() {
        let safe_body = vec![Stmt::Eval(var("side_effect")), Stmt::Assign("v1".into(), None, var("value"))];
        assert_eq!(flat_guard_body_nesting_refusal(&safe_body, 0, false), None);
        assert_eq!(
            flat_guard_body_nesting_refusal(
                &[Stmt::Assign("v1".into(), Some("int".into()), var("value"))],
                0,
                false,
            ),
            Some(FlatGuardNestingRefusal::Declaration)
        );
        assert_eq!(
            flat_guard_body_nesting_refusal(&[Stmt::Label("L1".into())], 0, false),
            Some(FlatGuardNestingRefusal::Label)
        );
        let nested = vec![Stmt::If {
            cond: var("c"), then_body: vec![Stmt::Eval(var("x"))], else_body: vec![]
        }];
        assert_eq!(
            flat_guard_body_nesting_refusal(&nested, MAX_NEST_DEPTH, false),
            Some(FlatGuardNestingRefusal::Depth)
        );
        let replacement = flat_guard_replacement(
            &var("condition"),
            &[Stmt::Eval(var("body"))],
            0,
            false,
        ).expect("safe guard gets structured");
        let Stmt::If { cond: Expr::Not(inner), then_body, else_body } = replacement else {
            panic!("flat guard did not become an inverted if");
        };
        assert!(matches!(*inner, Expr::Var(ref name) if name == "condition"));
        assert_eq!(then_body.len(), 1);
        assert!(else_body.is_empty());
        assert!(flat_guard_replacement(
            &var("condition"),
            &[Stmt::Label("L1".into())],
            0,
            false,
        )
        .is_none());
    }

    #[test]
    fn flat_guard_allows_labels_with_only_fall_arm_predecessors() {
        let sc = StructuredCfg {
            blocks: vec![
                BlockInfo { start: 0, end: 1, succ_taken: Some(2), succ_fall: Some(1) },
                BlockInfo { start: 1, end: 2, succ_taken: None, succ_fall: None },
                BlockInfo { start: 2, end: 3, succ_taken: None, succ_fall: None },
            ],
            block_of: vec![],
            full_succ: vec![vec![1], vec![], vec![]],
            labels: HashMap::from([(1, "L0".into())]),
            label_names: HashSet::from(["L0".into()]),
            idom: vec![None; 3],
            ipdom: vec![None; 3],
        };
        let body = vec![Stmt::Label("L0".into()), Stmt::Eval(var("inside"))];
        let fall_blocks = HashSet::from([1]);
        assert!(flat_guard_labels_are_internal(&sc, &body, &fall_blocks, 0));
        assert!(flat_guard_replacement(&var("condition"), &body, 0, true).is_some());

        let mut with_outside_edge = sc;
        with_outside_edge.full_succ[2].push(1);
        assert!(!flat_guard_labels_are_internal(
            &with_outside_edge,
            &body,
            &fall_blocks,
            0,
        ));
        assert!(flat_guard_replacement(&var("condition"), &body, 0, false).is_none());

        let mut with_taken_edge = with_outside_edge;
        with_taken_edge.full_succ[2].clear();
        with_taken_edge.blocks[0].succ_taken = Some(1);
        with_taken_edge.full_succ[0] = vec![1];
        assert!(!flat_guard_labels_are_internal(
            &with_taken_edge,
            &body,
            &fall_blocks,
            0,
        ));
    }

    /// A guard whose region holds a label reached ONLY from inside the region
    /// must still fold: the landing site and its edges move together under the
    /// inverted predicate, so no outside edge can become conditional.
    #[test]
    fn guard_with_inside_only_interior_label_folds() {
        let mut stmts = vec![
            guard("c1", "J"),
            Stmt::Goto("Linner".into()),
            Stmt::Eval(var("skipped")),
            Stmt::Label("Linner".into()),
            Stmt::Eval(var("body")),
            Stmt::Label("J".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            1,
            "only the interior jump may remain, got: {stmts:?}"
        );
        let out = render_stmts(&stmts);
        assert!(
            out.contains("c1"),
            "the guard condition must survive (inverted), got:\n{out}"
        );
        assert!(
            out.contains("Linner:"),
            "the interior landing site must survive with its edge, got:\n{out}"
        );
    }

    /// A NESTED label reached from outside the region blocks the fold too.
    ///
    /// Pulling the region under `!c` makes every landing site inside it
    /// conditional — a label nested in an arm is no more opaque than a flat
    /// one. A flat-only gate folds here and strands the outside edge; the
    /// all-depths gate keeps the guard exactly as emitted.
    #[test]
    fn guard_with_nested_outside_edge_refuses() {
        let mut stmts = vec![
            guard("c1", "J"),
            Stmt::If {
                cond: var("d"),
                then_body: vec![
                    Stmt::Label("Lnested".into()),
                    Stmt::Eval(var("x")),
                ],
                else_body: vec![],
            },
            Stmt::Eval(var("body")),
            Stmt::Label("J".into()),
            Stmt::Goto("Lnested".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            2,
            "the guard goto and the outside edge to the nested label must both survive: {stmts:?}"
        );
        let out = render_stmts(&stmts);
        assert!(
            out.contains("c1"),
            "the guard condition must survive, got:\n{out}"
        );
        assert!(
            out.contains("Lnested:"),
            "the nested landing site must survive, got:\n{out}"
        );
    }

    /// A NESTED label reached only from inside the region folds with it.
    ///
    /// The landing site and its edge move together under the inverted
    /// predicate, so no outside edge can become conditional. Guards the
    /// all-depths gate against over-blocking: nested self-contained labels
    /// must fold exactly like flat ones.
    #[test]
    fn guard_with_nested_inside_only_label_folds() {
        let mut stmts = vec![
            guard("c1", "J"),
            Stmt::If {
                cond: var("d"),
                then_body: vec![
                    Stmt::Goto("Lnested".into()),
                    Stmt::Eval(var("skipped")),
                    Stmt::Label("Lnested".into()),
                ],
                else_body: vec![],
            },
            Stmt::Eval(var("body")),
            Stmt::Label("J".into()),
            Stmt::Return(None),
        ];
        branch_repair(&mut stmts);
        assert_eq!(
            count_gotos(&stmts),
            1,
            "only the interior jump may remain, got: {stmts:?}"
        );
        let out = render_stmts(&stmts);
        assert!(
            out.contains("c1"),
            "the guard condition must survive (inverted), got:\n{out}"
        );
        assert!(
            out.contains("Lnested:"),
            "the nested landing site must survive with its edge, got:\n{out}"
        );
    }

    /// Regression pin for 0x1FE8D94: a switch whose arms each hold a guard to
    /// a shared tail. A third-arm-merge variant once dropped 16 conditions —
    /// this fold never merges arms: each arm folds (or keeps) its own guard
    /// with its own predicate, and every condition reaches the text.
    #[test]
    fn switch_arms_keep_their_own_guards() {
        let arm = |c: &str, b: &str, t: &str| {
            vec![
                guard(c, t),
                Stmt::Eval(var(b)),
                Stmt::Label(t.into()),
            ]
        };
        let mut stmts = vec![Stmt::Switch {
            disc: "v".into(),
            cases: vec![
                (0, arm("c0", "b0", "T0")),
                (1, arm("c1", "b1", "T1")),
                (2, arm("c2", "b2", "T2")),
            ],
            tail: vec![Stmt::Label("Tshared".into())],
        }];
        branch_repair(&mut stmts);
        let Stmt::Switch { cases, .. } = &stmts[0] else {
            panic!("the switch shell must survive, got: {stmts:?}");
        };
        assert_eq!(cases.len(), 3, "arms must never merge, got: {stmts:?}");
        let out = render_stmts(&stmts);
        for c in ["c0", "c1", "c2"] {
            assert!(
                out.contains(c),
                "arm condition {c} lost (a prior variant dropped 16), got:\n{out}"
            );
        }
        assert_eq!(
            count_gotos(&stmts),
            0,
            "each arm guard folds inside its own arm, got: {stmts:?}"
        );
    }

    /// A guard nested under an enclosing guard that already establishes its
    /// predicate must keep its LIVE arm, not its dead one.
    ///
    /// Regression pin for 0x1EB2F30: `apply_known` answered a `Ne` query
    /// against an established `Ne` fact with `false` (the `Eq` answer), so
    /// `fold_decided_guards` took the else-arm of a nested same-polarity
    /// guard and deleted the live then-arm — a 32-line copy loop vanished
    /// once the guard-chain fold nested it under its superset check.
    /// Predicates are inverted, never deleted; arms are kept.
    #[test]
    fn decided_guard_keeps_the_live_arm() {
        let ne0 = |v: &str| {
            Expr::Cmp(CmpKind::Ne, Box::new(var(v)), Box::new(Expr::Int(0)))
        };
        let mut stmts = vec![Stmt::If {
            cond: ne0("v125"),
            then_body: vec![Stmt::If {
                cond: ne0("v125"),
                then_body: vec![Stmt::Eval(var("live"))],
                else_body: vec![],
            }],
            else_body: vec![],
        }];
        fold_decided_guards(&mut stmts);
        let out = render_stmts(&stmts);
        assert!(
            out.contains("live"),
            "the live arm was deleted as if dead, got:\n{out}"
        );
        assert!(
            out.contains("v125"),
            "the predicate must survive, got:\n{out}"
        );
    }

    /// A jump from one case body to another case entry is expressible without
    /// a method-level label. The rewrite is limited to the enclosing switch.
    #[test]
    fn switch_case_entry_goto_uses_goto_case() {
        let mut stmts = vec![Stmt::Switch {
            disc: "selector".into(),
            cases: vec![
                (0, vec![Stmt::Label("Lcase1".into()), Stmt::Goto("Lcase2".into())]),
                (1, vec![Stmt::Label("Lcase2".into()), Stmt::Return(None)]),
            ],
            tail: vec![Stmt::Label("Loutside".into())],
        }];
        branch_repair(&mut stmts);
        let out = render_stmts(&stmts);
        assert!(out.contains("goto case 1;"), "case jump was not lowered:\n{out}");
        assert!(!out.contains("goto Lcase2;"), "old label jump remains:\n{out}");
        assert!(out.contains("return;"), "target arm was lost:\n{out}");
    }
}
