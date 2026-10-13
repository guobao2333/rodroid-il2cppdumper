//! Instruction-set-independent IR for native→CIL recovery.
//!
//! Faithful port of Cpp2IL's ISIL (`Cpp2IL.Core/ISIL/*.cs`) semantics:
//! three-address form, identity-by-index instructions (branch targets are
//! indices into the flat listing), destination positional by opcode
//! (Call: [target, returnValue, this?, params...], CallVoid: [target, this?, ...]).
//!
//! Transient ops (Phi) exist only between SSA construction and elimination.

pub use super::types::{FieldRef, ManagedType, MethodRef};
use std::fmt;

/// The IR name of the register a call's result lands in when that register is
/// NOT argument register 0.
///
/// On the ARM ABIs (AAPCS64 / AAPCS32) the result register and argument
/// register 0 are the SAME physical register (`x0` / `r0`), so one name is
/// right for both roles and the whole pipeline can say "x0". On x86-64 they
/// are two different physical registers — `rax` for the result and `rdi`
/// (SysV) / `rcx` (MSVC) for argument 0 — so naming both `x0` makes one name
/// carry two independent values. A call result then retires the receiver
/// identity, and an `x0` argument slot can resolve to a version that really
/// holds a discarded return value.
///
/// The split is what makes `x0` mean "argument 0 / `this`" on EVERY
/// architecture, which is what `locals::create` (receiver seeding),
/// `locals::rewrite_pinned_copies` (receiver pinning) and
/// `csharp::entry_register_name` already assume.
pub const RET_REG: &str = "rv";

#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    /// Hardware register or virtual stack slot ("x0", "rv", "stack_4").
    ///
    /// See [`RET_REG`] for why the x86 result register is named apart from
    /// argument 0.
    Register(String),
    /// SSA-versioned / final local variable slot.
    Local(u32),
    Immediate(i64),
    FloatLiteral(f32),
    DoubleLiteral(f64),
    StringLiteral { index: u32, value: ArcString },
    Method(MethodRef),
    Field(FieldRef),
    /// An INSTANCE FIELD access on a receiver that is NOT `this`.
    ///
    /// `Field` carries no receiver, so the typing pass could only name a field
    /// when the base was provably `this`; every other field access stayed a raw
    /// `*(nint*)(base + off)` even when both the base's class and the offset were
    /// known — the census's `vbase`/`dchain` families, ~64% of the arm64 dump.
    /// Carrying the base here lets the renderer emit `base.Field` and keeps the
    /// receiver in the output. The two must ship together: naming the field while
    /// dropping the base is exactly the `__unknown.field` corruption the old
    /// `base_is_this` gate existed to prevent.
    InstanceField {
        base: Box<Operand>,
        field: FieldRef,
    },
    Type(NamedTypeRef),
    /// `[base + index*scale + addend]`
    Memory {
        base: Box<Operand>,
        index: Option<Box<Operand>>,
        scale: u8,
        addend: i64,
        /// ACCESS WIDTH in bytes: 1/2/4/8, or 0 for "pointer-width / not
        /// stated" (the default every non-narrow access uses).
        ///
        /// This is not cosmetic. `ldrb`/`strb`/`ldrh`/`strh` were lifted
        /// identically to `ldr`/`str` — same `Move`, same `Memory` — so a
        /// one-byte `bool`/`byte`/`char` field read rendered as
        /// `*(nint*)(x+off)`: an EIGHT-byte load of an object whose declared
        /// size is one. Same slot, same address, wrong access — and a
        /// byte STORE written as an eight-byte store clobbers the seven
        /// following fields. Carrying the width lets the renderer emit the
        /// access the native code actually performs.
        width: u8,
    },
    /// Managed address-of (`&local`, `&arr[i]`).
    AddressOf(Box<Operand>),
    ArrayAccess {
        array: Box<Operand>,
        index: Box<Operand>,
    },
    ArrayLength(Box<Operand>),
    /// An ABI argument slot whose value could not be recovered. Semantically
    /// DISTINCT from `Unknown` (a genuinely unresolved value) and from a real
    /// `default(T)` — this slot exists only because ARM64 passes fewer live
    /// registers than the signature declares. The renderer drops these
    /// positionally; folding/analysis passes must NOT substitute values for
    /// them.
    MissingAbiArg,
    Unknown,
}

/// Cheap shared string for literal values (methods get cloned often).
pub type ArcString = std::sync::Arc<str>;

/// Reference to a managed type for newobj/newarr/box/typeof/castclass.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedTypeRef {
    pub ty: super::types::NamedType,
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(n) => write!(f, "%{n}"),
            Self::Local(i) => write!(f, "v{i}"),
            Self::Immediate(v) => write!(f, "#{v:#x}"),
            Self::FloatLiteral(v) => write!(f, "{v}f"),
            Self::DoubleLiteral(v) => write!(f, "{v}d"),
            Self::StringLiteral { value, .. } => write!(f, "{value:?}"),
            Self::Method(m) => write!(f, "{}", m.full_name()),
            Self::Field(fd) => write!(f, "{}::{}", fd.declaring_type_name, fd.name),
            Self::InstanceField { base, field } => write!(f, "{base}.{}", field.name),
            Self::Type(t) => write!(f, "typeof({})", t.ty),
            Self::Memory { base, index, scale, addend, width } => {
                // Only narrow accesses carry a width; `0` is the pointer-width
                // default and stays invisible so existing IR dumps are stable.
                if *width != 0 {
                    write!(f, "w{width}")?;
                }
                write!(f, "[{base}")?;
                if let Some(idx) = index {
                    write!(f, "+{idx}*{scale}")?;
                }
                write!(f, "{addend:+#x}]")
            }
            Self::AddressOf(t) => write!(f, "&{t}"),
            Self::ArrayAccess { array, index } => write!(f, "{array}[{index}]"),
            Self::ArrayLength(a) => write!(f, "len({a})"),
            Self::MissingAbiArg => write!(f, "<abi?>"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

impl Operand {
    pub fn reg(name: impl Into<String>) -> Self {
        Self::Register(name.into())
    }

    pub fn imm(v: i64) -> Self {
        Self::Immediate(v)
    }

    /// Attach an ACCESS WIDTH (in bytes) to a `Memory` operand.
    ///
    /// `0` means "not a narrow access" and returns the operand untouched, so
    /// every non-narrow site keeps the pointer-width default and nothing
    /// changes for it. A narrow access (`ldrb`/`strb` = 1, `ldrh`/`strh` = 2)
    /// has to keep its width all the way to the renderer: a one-byte field
    /// READ written as `*(nint*)` reads eight bytes, and the STORE form writes
    /// eight bytes over a one-byte slot.
    pub fn with_width(self, width: u8) -> Self {
        if width == 0 {
            return self;
        }
        match self {
            Self::Memory {
                base,
                index,
                scale,
                addend,
                ..
            } => Self::Memory {
                base,
                index,
                scale,
                addend,
                width,
            },
            other => other,
        }
    }

    pub fn as_local(&self) -> Option<u32> {
        match self {
            Self::Local(i) => Some(*i),
            _ => None,
        }
    }

    /// Registers/slots this operand READS from (for def/use analysis).
    pub fn uses(&self, out: &mut Vec<u32>) {
        match self {
            Self::Local(i) => out.push(*i),
            Self::Memory { base, index, .. } => {
                base.uses(out);
                if let Some(i) = index {
                    i.uses(out);
                }
            }
            Self::InstanceField { base, .. } => base.uses(out),
            Self::AddressOf(inner) | Self::ArrayLength(inner) => inner.uses(out),
            Self::ArrayAccess { array, index } => {
                array.uses(out);
                index.uses(out);
            }
            _ => {}
        }
    }

    /// True if operand is a pure value usable in constant folding.
    pub fn is_pure_value(&self) -> bool {
        matches!(
            self,
            Self::Immediate(_) | Self::FloatLiteral(_) | Self::DoubleLiteral(_)
        )
    }
}

/// Numeric conversion directions for [`Op::Convert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conversion {
    IntToFloat32,
    IntToFloat64,
    FloatToInt,
    Float32ToFloat64,
    Float64ToFloat32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareKind {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessOrEqual,
    GreaterOrEqual,
}

impl CompareKind {
    pub fn invert(self) -> Self {
        match self {
            Self::Equal => Self::NotEqual,
            Self::NotEqual => Self::Equal,
            Self::LessThan => Self::GreaterOrEqual,
            Self::GreaterOrEqual => Self::LessThan,
            Self::LessOrEqual => Self::GreaterThan,
            Self::GreaterThan => Self::LessOrEqual,
        }
    }

    pub fn is_signed_order(self) -> bool {
        !matches!(self, Self::Equal | Self::NotEqual)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithKind {
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
    Rol,
    Ror,
}

/// Float-specific arithmetic (single/double decided by operand types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatArithKind {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    /// Unary scalar FP square root (FSQRT / VSQRT / sqrtss / sqrtsd).
    Sqrt,
    /// Unary scalar FP absolute value (FABS / VABS / andps-mask).
    Abs,
    /// Unary FP rounding toward -infinity (FRINTM — `Math.Floor`).
    Floor,
    /// Unary FP rounding toward +infinity (FRINTP — `Math.Ceiling`).
    Ceiling,
    /// Unary FP rounding toward zero (FRINTZ — `Math.Truncate`).
    Truncate,
    /// Unary FP rounding to nearest integer (FRINTN/FRINTA — `Math.Round`).
    /// The tie rule (even vs away-from-zero) is not carried: the two encodings
    /// differ only in it and neither is recoverable from the operand.
    Round,
}

/// What a [`Op::Select`] does to its FALSE-side operand.
///
/// This is the whole of A64's conditional-select family: `CSEL` is the
/// identity, `CSINC` adds one, `CSINV` complements, `CSNEG` negates. The
/// `CSET` / `CSETM` aliases are `CSINC` / `CSINV` with both sources `xzr`,
/// which the lifter folds to a constant pair (`cond ? 1 : 0` and
/// `cond ? -1 : 0`) — that is how a managed `bool` materialises in optimized
/// A64, so getting these right is the difference between `b = x < y;` and
/// `b = __unknown;`. Same variants serve x86 `CMOVcc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectVariant {
    /// `CSEL` / `FCSEL`: `cond ? a : b`.
    Identity,
    /// `CSINC`: `cond ? a : b + 1`.
    Increment,
    /// `CSINV`: `cond ? a : ~b`.
    Invert,
    /// `CSNEG`: `cond ? a : -b`.
    Negate,
}

/// Which input lane feeds one OUTPUT lane of a vector rearrange.
///
/// This is the x86 twin of the p-code subpiece model: Ghidra addresses a
/// register as `(space, offset, size)` subpieces, so a lane permutation is a
/// list of COPYs of subpieces, and the semantics live in the CONSUMER rather
/// than in a mnemonic string. A register-granular IR cannot name a subpiece,
/// but it can name a *lane index* — which is the whole content of the operation
/// for the shuffle/unpack/shift family. Carrying the indices explicitly is what
/// turns `__shufps(a, b, 0x1b)` from a name we do not have into an operation we
/// do: a lane mapping can be folded, range-checked, and inverted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanePick {
    /// Lane `n` of source operand 1 (`operands[1]`), counting from the LOW end.
    A(u8),
    /// Lane `n` of source operand 2 (`operands[2]`).
    B(u8),
    /// A zero lane — the fill of a byte shift or a widen.
    Zero,
    /// An all-ones lane — `pcmpeq* reg, reg`.
    Ones,
}

impl LanePick {
    /// The lane index this pick reads, if it reads one.
    pub fn index(self) -> Option<u8> {
        match self {
            Self::A(n) | Self::B(n) => Some(n),
            Self::Zero | Self::Ones => None,
        }
    }

    /// True when the pick reads one of the source operands.
    pub fn is_source(self) -> bool {
        self.index().is_some()
    }

    /// True when the pick reads the SECOND source operand (`operands[2]`).
    pub fn is_second_source(self) -> bool {
        matches!(self, Self::B(_))
    }
}

/// A whole-register vector operation whose semantics are EXACT.
///
/// Deliberately not a mnemonic string: every fact a consumer needs to reason
/// about the value is present as data, so the operation can be folded, proved,
/// and rendered rather than merely named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimdOp {
    /// `dst = rearrange(a, b)` with one [`LanePick`] per OUTPUT lane, LOW lane
    /// first.
    ///
    /// Operands are `[dst, a]` or `[dst, a, b]`: `LanePick::A(n)` reads lane
    /// `n` of `operands[1]` and `LanePick::B(n)` reads lane `n` of
    /// `operands[2]`, so a one-source form (a shift, a `pshufd`, an extract)
    /// simply never emits `B`. `elem_bytes` is the LANE WIDTH the indices count
    /// in — 1 (byte shift), 2 (`pshuflw`, `pextrw`), 4 (`shufps`, `pshufd`) or
    /// 8 (`shufpd`, `unpcklpd`) — because the same index means a different
    /// field at each width, and a `psrldq 4` is a 4-BYTE shift while a
    /// `pshufd` field 4 is a DWORD.
    Arrange { elem_bytes: u8, lanes: Vec<LanePick> },
    /// `dst = a OP b` computed INDEPENDENTLY in every lane (`paddd`, `psubq`,
    /// `paddb`, …).
    ///
    /// Operands: `[dst, a, b]`. The arithmetic is the lane's own
    /// `elem_bytes`-wide modular operation: no carry ever crosses a lane
    /// boundary, which is exactly why a single register-wide add is the wrong
    /// model for it (and why these stayed stubs).
    PackedArith { elem_bytes: u8, kind: ArithKind },
}

impl SimdOp {
    /// How many lanes the operation produces.
    pub fn out_lanes(&self) -> usize {
        match self {
            Self::Arrange { lanes, .. } => lanes.len(),
            // A 128-bit register split into `elem_bytes`-wide lanes.
            Self::PackedArith { elem_bytes, .. } => 16 / (*elem_bytes).max(1) as usize,
        }
    }

    /// The width in bytes of one lane.
    pub fn elem_bytes(&self) -> u8 {
        match self {
            Self::Arrange { elem_bytes, .. } | Self::PackedArith { elem_bytes, .. } => *elem_bytes,
        }
    }

    /// The lane index of the LAST output lane, i.e. the highest index any
    /// consumer must be able to address (`out_lanes() - 1`).
    pub fn last_lane(&self) -> u8 {
        self.out_lanes().saturating_sub(1).min(u8::MAX as usize) as u8
    }

    /// True when the operation reads `operands[2]`.
    pub fn reads_second_source(&self) -> bool {
        match self {
            Self::Arrange { lanes, .. } => lanes.iter().any(|l| l.is_second_source()),
            Self::PackedArith { .. } => true,
        }
    }

    /// Every lane index this operation reads, in output order (duplicates
    /// kept). Used to prove a lowering stays inside its source register.
    pub fn source_lanes(&self) -> Vec<LanePick> {
        match self {
            Self::Arrange { lanes, .. } => lanes.clone(),
            Self::PackedArith { .. } => (0..self.out_lanes() as u8).map(LanePick::A).collect(),
        }
    }

    /// Is every lane index this operation reads a legal lane of a 128-bit
    /// register at this lane width? A lowering that selects lane 5 of a
    /// 4-lane register is a bug in the lifter, and this is the check that
    /// turns it into a reported fact instead of a silent wrong value.
    ///
    /// The bound is the REGISTER's lane count (`16 / elem_bytes`), not the
    /// number of output lanes: an extract such as `pextrw` produces ONE lane
    /// while still selecting from all eight word lanes of its source, so
    /// bounding by the output count would reject every legal extract.
    pub fn source_lanes_in_range(&self) -> bool {
        match self {
            Self::Arrange { elem_bytes, lanes } => {
                let lanes_per_reg = (16 / (*elem_bytes).max(1)) as usize;
                lanes
                    .iter()
                    .all(|l| l.index().map_or(true, |n| (n as usize) < lanes_per_reg))
            }
            // Lane i of `a` and lane i of `b` are both legal by construction.
            Self::PackedArith { .. } => true,
        }
    }
}

impl fmt::Display for SimdOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Arrange { elem_bytes, lanes } => {
                write!(f, "arrange{}", (*elem_bytes as u32) * 8)?;
                for (i, l) in lanes.iter().enumerate() {
                    write!(f, "{}", if i == 0 { "[" } else { "," })?;
                    match l {
                        LanePick::A(n) => write!(f, "a{n}")?,
                        LanePick::B(n) => write!(f, "b{n}")?,
                        LanePick::Zero => write!(f, "z")?,
                        LanePick::Ones => write!(f, "o")?,
                    }
                }
                write!(f, "]")
            }
            Self::PackedArith { elem_bytes, kind } => {
                write!(f, "p{kind:?}.{}", (*elem_bytes as u32) * 8)
            }
        }
    }
}

pub type InsnIndex = usize;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Move,
    Arith(ArithKind),
    FloatArith(FloatArithKind),
    Compare(CompareKind),
    Not,
    Negate,

    Jump(InsnIndex),
    CondJump(InsnIndex),

    /// Native stack adjustment (removed by stack analyzer).
    ShiftStack(i64),

    Call,
    CallVoid,
    /// Indirect call — rewritten to direct calls when resolvable; else diagnostic.
    IndirectCall,

    Return,
    Throw,
    NewObj(Option<NamedTypeRef>),
    NewArr(Option<ManagedType>),
    Box(Option<NamedTypeRef>),
    LoadLength,

    /// Jump-table switch: targets are instruction indices per case value.
    Switch(Vec<usize>),

    /// Numeric conversion (int↔float, widen/narrow).
    Convert(Conversion),

    /// Fused multiply-add: dest = op1 * op2 [+|−] op3 (negate flag for MSUB).
    MulAcc {
        negate_addend: bool,
        is_float: bool,
    },

    /// Count leading zeros.
    Clz,

    /// A value-producing instruction whose semantics are not modelled yet,
    /// carried as its own mnemonic over its real operands: `[dst, srcs…]`.
    ///
    /// This exists to uphold one invariant: **an instruction that writes a
    /// register must define that register.** The lifter's fall-through used to
    /// satisfy that by writing `dst = Unknown` — or, when the destination was a
    /// VECTOR register, by writing nothing at all, so the register silently
    /// kept its previous value and every later read fabricated a different
    /// expression. Both destroy dataflow. An intrinsic keeps the destination
    /// and every source, and names the operation with the opcode's own
    /// mnemonic (`__movi(...)`, `__umulh(...)`, `__tbl(...)`), which is
    /// exactly the trade `fnptr_0x…` makes for an unresolved call: a name we do
    /// not have yet must never cost the VALUE.
    Simd(String),

    /// A vector operation whose semantics ARE recovered: an explicit lane
    /// rearrange or a per-lane integer arithmetic.
    ///
    /// This is the successor to `Simd` for the one family that dominated the
    /// remaining x64 stub population — lane arrangement (`shufps` 2,385, 
    /// `psrldq` 1,255, `unpcklps` 1,026, `unpcklpd`, `unpckhpd`, `pshufd`,
    /// `pextrw`, `punpcklqdq`) and packed integer arithmetic (`paddd`,
    /// `paddq`, `psubq`). Those mnemonics have no whole-register equivalent (a
    /// register-granular model computes `addps` correctly and `paddd`
    /// incorrectly — the packed integer forms carry between lanes and the FP
    /// ones do not), so they cannot go through `Op::Arith`, but their meaning
    /// is still fully known: it is a lane mapping. Keeping that mapping as
    /// DATA lets a consumer fold it, range-check it and render it, instead of
    /// printing a name and hoping.
    ///
    /// Operands: `[dst, a, b?]` — see [`SimdOp`].
    SimdLane(SimdOp),

    /// Conditional select — `dst = cond ? a : f(b)`.
    ///
    /// Operands: `[dst, cond, a, b]`. `cond` is a BOOLEAN VALUE: exactly the
    /// kind `Op::Compare` produces and `Op::CondJump` consumes. Reusing that
    /// representation instead of inventing a second way to say "a condition"
    /// is the point — SSA renaming, type propagation, DCE and the renderer's
    /// condition inlining all already understand a boolean operand, so this op
    /// attaches to the existing machinery rather than needing its own.
    ///
    /// A64 census: 17,281 occurrences in the reference binary
    /// (`CSINC` 7,488, `CSEL` 7,113, `FCSEL` 2,247, `CSNEG` 271, `CSINV` 162),
    /// every one of which used to fall through the lifter's catch-all and
    /// become `dst = Unknown`, silently deleting the branchless min/max/select
    /// idiom the compiler emits for `Math.Min`, `?:` on constants, and every
    /// clamps/clamp-like helper.
    Select(SelectVariant),

    /// Objective-C message send with a stub-fixed selector:
    /// dest(x0 slot) = [receiver sel:arg…]. Recovered from Apple-platform
    /// selector stubs; carries the REAL operation name (setBool:forKey: …).
    ObjcMsgSend {
        selector: String,
    },

    /// SSA phi — inserted at dominance frontiers, eliminated before emit.
    Phi,

    /// Marker for analysis-internal no-ops (never emitted).
    Nop,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move => write!(f, "mov"),
            Self::Arith(k) => write!(f, "{k:?}"),
            Self::FloatArith(k) => write!(f, "f{k:?}"),
            Self::Compare(k) => write!(f, "cmp.{k:?}"),
            Self::Not => write!(f, "not"),
            Self::Negate => write!(f, "neg"),
            Self::Jump(t) => write!(f, "br -> {t}"),
            Self::CondJump(t) => write!(f, "br.true -> {t}"),
            Self::ShiftStack(d) => write!(f, "add sp, {d}"),
            Self::Call | Self::CallVoid | Self::IndirectCall => write!(f, "call"),
            Self::Return => write!(f, "ret"),
            Self::Throw => write!(f, "throw"),
            Self::NewObj(_) => write!(f, "newobj"),
            Self::NewArr(_) => write!(f, "newarr"),
            Self::Box(_) => write!(f, "box"),
            Self::LoadLength => write!(f, "ldlen"),
            Self::Switch(t) => write!(f, "switch[{}]", t.len()),
            Self::Convert(c) => write!(f, "conv.{c:?}"),
            Self::MulAcc { negate_addend, .. } => {
                write!(f, "madd{}", if *negate_addend { "-" } else { "+" })
            }
            Self::Clz => write!(f, "clz"),
            Self::Simd(name) => write!(f, "{}", name.to_lowercase()),
            Self::SimdLane(s) => write!(f, "{s}"),
            Self::Select(v) => write!(f, "sel.{v:?}"),
            Self::ObjcMsgSend { selector } => write!(f, "objc_msg[{selector}]"),
            Self::Phi => write!(f, "phi"),
            Self::Nop => write!(f, "nop"),
        }
    }
}

/// One IR instruction. Identity is its position in the flat listing; branch
/// targets are indices into the same Vec (Cpp2IL `Instruction.cs` convention:
/// no content-based Hash/Eq so targets stay stable across rewrites).
#[derive(Debug, Clone)]
pub struct Insn {
    pub op: Op,
    pub operands: Vec<Operand>,
    /// Optional type hint for the result (drives float coercion & elem types).
    pub dest_type: Option<ManagedType>,
    /// Original native address (diagnostics + debugging).
    pub address: u64,
}

impl Insn {
    pub fn new(op: Op, operands: Vec<Operand>, address: u64) -> Self {
        Self { op, operands, dest_type: None, address }
    }

    pub fn move_(dest: Operand, src: Operand, address: u64) -> Self {
        Self::new(Op::Move, vec![dest, src], address)
    }

    pub fn nop(address: u64) -> Self {
        Self::new(Op::Nop, vec![], address)
    }

    /// Operand position holding this instruction's destination, per opcode
    /// convention (None for void ops). ONE source of truth: every pass
    /// (SSA, reaching defs, renderer) must agree on which slot is the write
    /// or def/use tracking silently drops whole op families. `Convert`,
    /// `MulAcc` and `Clz` write their result to operand 0 exactly like
    /// `Arith`; `Phi` is `[dest, args…]`.
    pub fn dest_position(op: &Op) -> Option<usize> {
        match op {
            Op::Call => Some(1),
            Op::Move | Op::Arith(_) | Op::FloatArith(_) | Op::Compare(_) | Op::Not
            | Op::Negate | Op::Convert(_) | Op::MulAcc { .. } | Op::Clz | Op::NewObj(_)
            | Op::NewArr(_) | Op::Box(_) | Op::LoadLength | Op::Phi | Op::Select(_)
            | Op::Simd(_) | Op::SimdLane(_) => Some(0),
            _ => None,
        }
    }

    /// Destination operand per opcode convention (None for void ops).
    pub fn dest(&self) -> Option<&Operand> {
        Self::dest_position(&self.op).and_then(|p| self.operands.get(p))
    }

    pub fn dest_mut(&mut self) -> Option<&mut Operand> {
        Self::dest_position(&self.op).and_then(|p| self.operands.get_mut(p))
    }

    /// Whether this instruction defines (writes) its destination local.
    pub fn is_definition(&self) -> bool {
        Self::dest_position(&self.op).is_some()
            && self.dest().map(|d| d.as_local().is_some()).unwrap_or(false)
    }

    /// Locals read by this instruction (excluding its destination).
    pub fn use_locals(&self) -> Vec<u32> {
        let mut uses = Vec::new();
        for (i, op) in self.operands.iter().enumerate() {
            // skip destination positions
            let is_dest_pos = Self::dest_position(&self.op) == Some(i);
            if is_dest_pos && matches!(op, Operand::Local(_)) {
                continue;
            }
            // AddressOf(local) is a definition-like use for escape analysis but
            // still counts as a use for liveness.
            op.uses(&mut uses);
        }
        uses
    }

    /// Can dead-code elimination remove this when no one uses the result?
    pub fn is_removable_if_dead(&self) -> bool {
        matches!(
            self.op,
            Op::Move | Op::Arith(_) | Op::FloatArith(_) | Op::Compare(_) | Op::Not
                | Op::Negate | Op::Convert(_) | Op::MulAcc { .. } | Op::Clz | Op::Phi
                | Op::LoadLength | Op::Select(_)
                // A lane op is a PURE function of its operands, so unlike the
                // semantic-free `Op::Simd` intrinsic (deliberately NOT listed
                // here — an unmodelled operation must not be deleted) a dead
                // one can be removed without losing an observation.
                | Op::SimdLane(_)
        )
    }

    pub fn is_terminator(&self) -> bool {
        matches!(
            self.op,
            Op::Jump(_) | Op::CondJump(_) | Op::Return | Op::Throw | Op::IndirectCall
                | Op::Switch(_)
        )
    }

    pub fn has_side_effects(&self) -> bool {
        matches!(
            self.op,
            Op::Call | Op::CallVoid | Op::IndirectCall | Op::Return | Op::Throw
        ) || self.is_terminator()
    }
}

impl fmt::Display for Insn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.op)?;
        let start = match self.op {
            Op::Call => 2.min(self.operands.len()),
            Op::CallVoid => 1.min(self.operands.len()),
            _ => 0,
        };
        for (i, o) in self.operands.iter().enumerate() {
            if i < start {
                continue;
            }
            if i == start || (start == 0 && i == 0) {
                write!(f, " {o}")?;
            } else {
                write!(f, ", {o}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod simd_lane_tests {
    use super::*;

    fn arrange(elem_bytes: u8, lanes: Vec<LanePick>) -> SimdOp {
        SimdOp::Arrange { elem_bytes, lanes }
    }

    /// The four DWORD lanes of `shufps d, s, 0x1b` — 0x1b = 0b00_01_10_11, so
    /// lane 0 takes d[3], lane 1 d[2], lane 2 s[1], lane 3 s[0].
    #[test]
    fn a_lane_mapping_is_fully_determined_by_its_picks() {
        let op = arrange(
            4,
            vec![LanePick::A(3), LanePick::A(2), LanePick::B(1), LanePick::B(0)],
        );
        assert_eq!(op.out_lanes(), 4);
        assert_eq!(op.elem_bytes(), 4);
        assert!(op.reads_second_source());
        assert!(op.source_lanes_in_range());
        assert_eq!(op.to_string(), "arrange32[a3,a2,b1,b0]");
    }

    /// A one-source form never reads `operands[2]`.
    #[test]
    fn a_one_source_arrange_does_not_claim_a_second_source() {
        let op = arrange(4, vec![LanePick::A(0); 4]);
        assert!(!op.reads_second_source());
        assert!(op.source_lanes_in_range());
    }

    /// An out-of-range pick is INVALID — the check exists so a lifter bug is a
    /// reported fact rather than a silent wrong value.
    #[test]
    fn an_out_of_range_lane_pick_is_rejected() {
        // four DWORD lanes, so index 4 does not exist
        let op = arrange(4, vec![LanePick::A(0), LanePick::A(1), LanePick::A(2), LanePick::A(4)]);
        assert!(!op.source_lanes_in_range());
    }

    /// A byte shift is 16 BYTE lanes, so its indices are legal up to 15.
    #[test]
    fn a_byte_shift_addresses_sixteen_lanes() {
        let mut lanes: Vec<LanePick> = (4..16).map(LanePick::A).collect();
        lanes.extend(std::iter::repeat_n(LanePick::Zero, 4));
        let op = arrange(1, lanes);
        assert_eq!(op.out_lanes(), 16);
        assert_eq!(op.elem_bytes(), 1);
        assert!(op.source_lanes_in_range());
        assert_eq!(op.last_lane(), 15);
    }

    /// Packed integer arithmetic covers every lane of the register.
    #[test]
    fn packed_arithmetic_covers_the_whole_register() {
        let op = SimdOp::PackedArith { elem_bytes: 4, kind: ArithKind::Add };
        assert_eq!(op.out_lanes(), 4);
        assert!(op.reads_second_source());
        assert!(op.source_lanes_in_range());
        assert_eq!(op.to_string(), "pAdd.32");
        let q = SimdOp::PackedArith { elem_bytes: 8, kind: ArithKind::Sub };
        assert_eq!(q.out_lanes(), 2);
        assert_eq!(q.to_string(), "pSub.64");
    }

    /// An EXTRACT selects from every lane of the register it reads, even
    /// though it produces only one — the check must not confuse the two.
    #[test]
    fn an_extract_may_select_any_lane_of_its_register() {
        // 8 word lanes in a register; an extract of lane 7 is perfectly legal
        // even though the result is a single lane.
        let op = arrange(2, vec![LanePick::A(7)]);
        assert_eq!(op.out_lanes(), 1);
        assert!(op.source_lanes_in_range());
        // ...and lane 8 is not.
        assert!(!arrange(2, vec![LanePick::A(8)]).source_lanes_in_range());
    }

    /// A lane op DEFINES its destination, and it is pure.
    #[test]
    fn a_lane_op_defines_its_destination_and_is_pure() {
        // A real definition has a LOCAL destination, so the destination here is
        // a local slot and the source is the register.
        let i = Insn::new(
            Op::SimdLane(arrange(4, vec![LanePick::A(0); 4])),
            vec![Operand::Local(7), Operand::Register("vf0".into())],
            0x1000,
        );
        assert_eq!(Insn::dest_position(&i.op), Some(0));
        assert!(i.is_definition());
        assert!(i.is_removable_if_dead(), "a lane op is a pure function");
        assert!(!i.has_side_effects());
        assert!(!i.is_terminator());
        assert_eq!(i.use_locals(), Vec::<u32>::new());
        // ...and the REGISTER form of the same op is not a definition, which is
        // the invariant every def/use pass relies on.
        let r = Insn::new(
            Op::SimdLane(arrange(4, vec![LanePick::A(0); 4])),
            vec![Operand::Register("vf0".into()), Operand::Register("vf1".into())],
            0x1000,
        );
        assert!(!r.is_definition());
    }

    /// A constant lane (`Ones`) reads no source at all.
    #[test]
    fn a_constant_lane_reads_nothing() {
        let op = arrange(4, vec![LanePick::Ones; 4]);
        assert!(!op.reads_second_source());
        assert!(!op.source_lanes().iter().any(|l| l.is_source()));
        assert!(op.source_lanes_in_range());
    }
}
