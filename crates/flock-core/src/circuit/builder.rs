//! Describing circuits by *building* them, rather than by hand-writing cells.
//!
//! [`Circuit::new`] takes the wiring as raw equivalence classes of [`Cell`],
//! and the witness is built by separate code. Two artifacts that must agree
//! with nothing enforcing it — a mismatch is a silently wrong *statement*, not
//! a compile error. See `docs/circuit-wiring-design.tex`
//! §"Describing circuits: the builder".
//!
//! The fix is the principle behind [`crate::transcript_record`]: **one
//! description, both views**. A [`GateType`] carries its constraints and its
//! native evaluation together, so instantiating a gate emits the row, the
//! wiring and the witness from a single call and they cannot drift. `counts`
//! and the equivalence classes fall out of construction instead of being
//! passed in.
//!
//! ```ignore
//! let mut b = CircuitBuilder::new(nu);
//! let mult = b.slot(MultGate { kappa });
//! let mut acc = b.public_value(seed);
//! for &a in &multipliers {
//!     let a_w = b.public_value(a);
//!     acc = b.gate(mult, &[a_w, acc])[0];
//! }
//! b.publish(acc);
//! let built = b.finish();
//! ```
//!
//! ## Determinism
//!
//! Rows are allocated in gate-instantiation order and cell-slots enumerate in
//! registry order, so the same sequence of calls always produces the same
//! [`Circuit::digest`]. That matters because the digest is statement-binding:
//! a *regenerated* circuit must be the SAME circuit, not merely an equivalent
//! one.

use crate::field::F128;
use crate::schedule::{IoDirection, Registry, TableType};

use super::{Cell, Circuit, CircuitError};

/// A value in the circuit, and the cells that must hold it.
///
/// A wire IS an equivalence class under construction: binding it as a gate
/// input appends that gate's input cell to the class, and [`CircuitBuilder`]
/// hands the finished classes to [`Circuit::new`].
///
/// Wires are usable before their producer is emitted — a value can be consumed
/// by a gate declared earlier in the program than the one that defines it,
/// because a class is just a set. The Fiat–Shamir chain needs this: a squeezed
/// challenge is re-absorbed into the transcript that produced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Wire(usize);

/// A declared gate slot. Indexes the builder's DECLARATION order, which is not
/// the registry's slot order — see [`BuiltCircuit::registry_slot`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotId(usize);

struct WireData {
    value: F128,
    cells: Vec<Cell>,
}

/// One slot's committed witness, in the form its class's prover input wants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlotWitness {
    /// Element slot: the committed words in the BatchMajor rows-low layout,
    /// `word[(col << nu) + row]`, length `width << nu`. Feeds
    /// `UnionElementSlotInput`'s closure directly.
    Element(Vec<F128>),
}

/// A gate type: its constraint system and its native evaluation, together.
///
/// The pairing is the whole point. A type that could describe its constraints
/// but not evaluate them would put the witness back in separate code, which is
/// the failure this module exists to remove.
pub trait GateType {
    /// What one gate contributes to its slot's witness. Kept abstract because
    /// witnesses do not decompose uniformly per row: element slots are plain
    /// `F128` words, while boolean slots are bit-packed in bulk by their own
    /// `generate_witness`. The builder collects `Row`s in order and lets the
    /// gate type emit the slot's witness once.
    type Row;

    /// The registry type: constraints, width, and the `io_schema` whose order
    /// defines this gate's input and output positions.
    fn table(&self) -> TableType;

    /// Evaluate one gate. `inputs` are the schema's `In` words in schema
    /// order; returns the `Out` words in schema order, plus the row record.
    fn eval(&self, inputs: &[F128]) -> (Vec<F128>, Self::Row);

    /// The slot's committed witness, given every row in instantiation order
    /// and the uniform capacity `nu`. Rows `[rows.len(), 2^nu)` are dummy and
    /// must be written as zeros — the PIOP sums over the whole region.
    fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness;
}

/// Object-safe view of a declared slot, erasing `GateType::Row`.
trait SlotBuild {
    fn table(&self) -> TableType;
    fn n_in(&self) -> usize;
    fn push(&mut self, inputs: &[F128]) -> Vec<F128>;
    fn rows(&self) -> usize;
    fn witness(&self, nu: usize) -> SlotWitness;
}

struct GateSlot<G: GateType> {
    gate: G,
    table: TableType,
    rows: Vec<G::Row>,
    n_in: usize,
    n_out: usize,
}

impl<G: GateType> SlotBuild for GateSlot<G> {
    fn table(&self) -> TableType {
        self.table.clone()
    }
    fn n_in(&self) -> usize {
        self.n_in
    }
    fn push(&mut self, inputs: &[F128]) -> Vec<F128> {
        let (outputs, row) = self.gate.eval(inputs);
        assert_eq!(
            outputs.len(),
            self.n_out,
            "gate returned {} outputs, schema declares {}",
            outputs.len(),
            self.n_out
        );
        self.rows.push(row);
        outputs
    }
    fn rows(&self) -> usize {
        self.rows.len()
    }
    fn witness(&self, nu: usize) -> SlotWitness {
        self.gate.witness(&self.rows, nu)
    }
}

/// Everything [`CircuitBuilder::finish`] produces: the statement and the
/// witness, from one description.
pub struct BuiltCircuit {
    pub registry: Registry,
    pub circuit: Circuit,
    /// Declared counts per slot, in REGISTRY order — what `UnionInstance::new`
    /// wants.
    pub counts: Vec<usize>,
    /// Per-slot witnesses, in REGISTRY order.
    pub witnesses: Vec<SlotWitness>,
    /// The public segment, in publication order.
    pub public: Vec<F128>,
    /// `registry_slot[declared] = registry index`.
    registry_slot: Vec<usize>,
}

impl BuiltCircuit {
    /// Where a declared slot landed in the registry. `Registry::new` sorts
    /// class-major, area-descending, so declaration order is not slot order.
    pub fn registry_slot(&self, s: SlotId) -> usize {
        self.registry_slot[s.0]
    }
}

pub struct CircuitBuilder {
    nu: usize,
    slots: Vec<Box<dyn SlotBuild>>,
    wires: Vec<WireData>,
    public: Vec<Wire>,
}

impl CircuitBuilder {
    pub fn new(nu: usize) -> Self {
        Self {
            nu,
            slots: Vec::new(),
            wires: Vec::new(),
            public: Vec::new(),
        }
    }

    /// Declare a gate slot. Every gate of this type shares the slot, and the
    /// slot's row capacity is the registry's uniform `2^nu`.
    pub fn slot<G: GateType + 'static>(&mut self, gate: G) -> SlotId {
        let table = gate.table();
        let n_in = table
            .io_schema
            .iter()
            .filter(|w| w.dir == IoDirection::In)
            .count();
        let n_out = table.io_schema.len() - n_in;
        assert!(
            !table.io_schema.is_empty(),
            "a gate slot needs an io_schema; a type with none is unwireable"
        );
        self.slots.push(Box::new(GateSlot {
            gate,
            table,
            rows: Vec::new(),
            n_in,
            n_out,
        }));
        SlotId(self.slots.len() - 1)
    }

    /// A free value entering the circuit. It gets no producing cell, so it
    /// must be constrained by something — published, or consumed by a gate
    /// whose relation pins it.
    pub fn value(&mut self, value: F128) -> Wire {
        self.wires.push(WireData {
            value,
            cells: Vec::new(),
        });
        Wire(self.wires.len() - 1)
    }

    /// Instantiate a gate: allocate a row, bind `inputs` to its input cells,
    /// evaluate, and return wires for its outputs.
    pub fn gate(&mut self, slot: SlotId, inputs: &[Wire]) -> Vec<Wire> {
        let s = &self.slots[slot.0];
        assert_eq!(
            inputs.len(),
            s.n_in(),
            "gate takes {} inputs, got {}",
            s.n_in(),
            inputs.len()
        );
        let row = s.rows();
        assert!(
            row < (1usize << self.nu),
            "slot {} exceeded its 2^{} row capacity",
            slot.0,
            self.nu
        );

        let vals: Vec<F128> = inputs.iter().map(|w| self.wires[w.0].value).collect();
        let outputs = self.slots[slot.0].push(&vals);

        // Cells are assigned once the registry order is known; record the
        // (declared slot, schema index, row) triple and resolve in `finish`.
        for (k, w) in inputs.iter().enumerate() {
            self.wires[w.0]
                .cells
                .push(Cell::new(encode(slot.0, k), row));
        }
        let n_in = self.slots[slot.0].n_in();
        outputs
            .into_iter()
            .enumerate()
            .map(|(k, value)| {
                self.wires.push(WireData {
                    value,
                    cells: vec![Cell::new(encode(slot.0, n_in + k), row)],
                });
                Wire(self.wires.len() - 1)
            })
            .collect()
    }

    /// Publish a wire: it joins the public segment, in call order.
    pub fn publish(&mut self, w: Wire) {
        self.public.push(w);
    }

    /// A value that is both free and public — the common case for circuit
    /// inputs.
    pub fn public_value(&mut self, value: F128) -> Wire {
        let w = self.value(value);
        self.publish(w);
        w
    }

    pub fn finish(mut self) -> Result<BuiltCircuit, CircuitError> {
        // Registry::new sorts class-major, area-descending, with a STABLE sort
        // on `(is_element, Reverse(k_log))`. Replicate that key to learn where
        // each declared slot landed, then assert the result agrees with the
        // registry we actually get — so a change to the registry's ordering
        // fails loudly here rather than silently mis-wiring every circuit.
        let tables: Vec<TableType> = self.slots.iter().map(|s| s.table()).collect();
        let mut order: Vec<usize> = (0..tables.len()).collect();
        order.sort_by_key(|&i| (tables[i].is_element(), std::cmp::Reverse(tables[i].k_log)));

        let registry = Registry::new(order.iter().map(|&i| tables[i].clone()).collect(), self.nu);
        for (reg_idx, &declared) in order.iter().enumerate() {
            assert_eq!(
                registry.types()[reg_idx].k_log,
                tables[declared].k_log,
                "builder's slot ordering disagrees with Registry::new"
            );
        }
        // registry_slot[declared] = registry index
        let mut registry_slot = vec![0usize; tables.len()];
        for (reg_idx, &declared) in order.iter().enumerate() {
            registry_slot[declared] = reg_idx;
        }

        // Cell-slots enumerate in registry order, each type contributing its
        // io_schema words in schema order, then the public slots.
        let mut iota_base = vec![0usize; tables.len()];
        let mut acc = 0usize;
        for &declared in &order {
            iota_base[declared] = acc;
            acc += tables[declared].io_schema.len();
        }
        let num_gate_slots = acc;
        let rows_per_public_slot = 1usize << self.nu;

        // Resolve the placeholder cells to real cell-slot indices.
        for wd in &mut self.wires {
            for c in &mut wd.cells {
                let (declared, k) = decode(c.slot);
                *c = Cell::new(iota_base[declared] + k, c.row);
            }
        }
        // Public cells.
        let public_values: Vec<F128> = self.public.iter().map(|w| self.wires[w.0].value).collect();
        for (p, w) in self.public.iter().enumerate() {
            let slot = num_gate_slots + p / rows_per_public_slot;
            self.wires[w.0]
                .cells
                .push(Cell::new(slot, p % rows_per_public_slot));
        }

        // A class of one cell needs no copy constraint.
        let mut wires: Vec<Vec<Cell>> = self
            .wires
            .into_iter()
            .map(|wd| wd.cells)
            .filter(|c| c.len() > 1)
            .collect();
        for c in &mut wires {
            c.sort_unstable();
        }
        wires.sort_unstable();

        let counts: Vec<usize> = order.iter().map(|&d| self.slots[d].rows()).collect();
        let witnesses: Vec<SlotWitness> = order
            .iter()
            .map(|&d| self.slots[d].witness(self.nu))
            .collect();

        let circuit = Circuit::new(&registry, counts.clone(), public_values.len(), wires)?;
        Ok(BuiltCircuit {
            registry,
            circuit,
            counts,
            witnesses,
            public: public_values,
            registry_slot,
        })
    }
}

// Placeholder cell-slot encoding, used only between `gate` and `finish`: the
// registry's slot order is not known until every slot is declared, so cells
// are stamped with (declared slot, schema index) and resolved at the end.
#[inline]
fn encode(declared: usize, schema_idx: usize) -> usize {
    declared << 32 | schema_idx
}
#[inline]
fn decode(v: usize) -> (usize, usize) {
    (v >> 32, v & 0xFFFF_FFFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_r1cs::{ElementTableBuilder, ElementTableType};
    use crate::schedule::IoWord;
    use std::sync::Arc;

    /// The element `mult` gate from `circuit_wiring.rs`: columns 0,1 free
    /// wires in, column 2 = z0·z1 out.
    struct MultGate {
        ty: Arc<ElementTableType>,
    }

    impl MultGate {
        fn new(kappa: usize) -> Self {
            let mut b = ElementTableBuilder::new(kappa);
            b.free_wire(0).free_wire(1).mult(2, 0, 1);
            Self {
                ty: Arc::new(b.build().expect("mult block is valid")),
            }
        }
    }

    impl GateType for MultGate {
        type Row = (F128, F128, F128);

        fn table(&self) -> TableType {
            TableType::element(self.ty.clone()).with_io_schema(vec![
                IoWord::input(0),
                IoWord::input(1),
                IoWord::output(2),
            ])
        }

        fn eval(&self, inputs: &[F128]) -> (Vec<F128>, Self::Row) {
            let (a, b) = (inputs[0], inputs[1]);
            let c = a * b;
            (vec![c], (a, b, c))
        }

        fn witness(&self, rows: &[Self::Row], nu: usize) -> SlotWitness {
            let at = |c: usize, j: usize| (c << nu) + j;
            let mut z = vec![F128::ZERO; self.ty.width() << nu];
            for (j, &(a, b, c)) in rows.iter().enumerate() {
                z[at(0, j)] = a;
                z[at(1, j)] = b;
                z[at(2, j)] = c;
            }
            SlotWitness::Element(z)
        }
    }

    /// The builder reproduces `circuit_wiring.rs`'s hand-built element chain
    /// EXACTLY — same wiring classes, same counts, same `Circuit::digest`.
    ///
    /// This is the validation the design called for: if the builder is right,
    /// nothing moves.
    #[test]
    fn builder_reproduces_the_hand_built_element_chain() {
        const EL_A: usize = 0;
        const EL_B: usize = 1;
        const EL_C: usize = 2;
        const PUB: usize = 3;
        let (nu, kappa, n) = (12usize, 3usize, 20usize);

        let mut state = 0xC4A1_0001u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hi = state;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            F128::new(hi, state)
        };
        let seed = next();
        let a: Vec<F128> = (0..n).map(|_| next()).collect();

        // ---- built ----
        let mut b = CircuitBuilder::new(nu);
        let mult = b.slot(MultGate::new(kappa));
        let seed_w = b.public_value(seed);
        let a_w: Vec<Wire> = a.iter().map(|&x| b.public_value(x)).collect();
        let mut acc = seed_w;
        for &aw in &a_w {
            acc = b.gate(mult, &[aw, acc])[0];
        }
        b.publish(acc);
        let built = b.finish().expect("builder produces a valid circuit");

        // ---- hand-built, verbatim from circuit_wiring.rs ----
        let ty = MultGate::new(kappa);
        let registry = Registry::new(vec![ty.table()], nu);
        let mut hand = vec![vec![Cell::new(PUB, 0), Cell::new(EL_B, 0)]];
        for i in 0..n {
            hand.push(vec![Cell::new(PUB, 1 + i), Cell::new(EL_A, i)]);
        }
        for i in 0..n - 1 {
            hand.push(vec![Cell::new(EL_C, i), Cell::new(EL_B, i + 1)]);
        }
        hand.push(vec![Cell::new(EL_C, n - 1), Cell::new(PUB, 1 + n)]);
        let hand_circuit = Circuit::new(&registry, vec![n], n + 2, hand).expect("valid");

        assert_eq!(built.counts, vec![n], "counts fall out of construction");
        assert_eq!(built.public.len(), n + 2);
        assert_eq!(
            built.circuit.digest(),
            hand_circuit.digest(),
            "builder produced a DIFFERENT statement than the hand-built circuit"
        );

        // And the witness matches the hand-written chain generator.
        let at = |c: usize, j: usize| (c << nu) + j;
        let mut want = vec![F128::ZERO; ty.ty.width() << nu];
        let mut acc_v = seed;
        for (j, &aj) in a.iter().enumerate() {
            want[at(0, j)] = aj;
            want[at(1, j)] = acc_v;
            want[at(2, j)] = aj * acc_v;
            acc_v = aj * acc_v;
        }
        assert_eq!(built.witnesses[0], SlotWitness::Element(want));
        assert!(
            ty.ty.satisfies(
                match &built.witnesses[0] {
                    SlotWitness::Element(z) => z,
                },
                nu,
                n
            ),
            "built witness must satisfy the relation"
        );
    }
}
