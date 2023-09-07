use std::{collections::BTreeSet, ops::Range};

use self::utils::empty_lagrange_assigned;
use crate::{
  arithmetic::Field,
  circuit::{layouter::SyncDeps, Value},
  plonk::{
    permutation, Advice, Any, Assigned, Assignment, Challenge, Column, ConstraintSystem,
    Error as PlonkError, Fixed, Instance, Selector,
  },
  poly::{LagrangeCoeff, Polynomial},
};

/// Assembly to be used in circuit synthesis.
/// Accumulates all the necessary data in order to construct the ProvingKey.
/// Main purpose is to implement the `Assignment` trait, allowing a `Circuit` to assign a witness
/// for a constraint system. See `plonk::keygen::Assembly` for reference.
// todo(tk): ask adrian about why we store a plonk permutation assembly
#[derive(Debug)]
pub struct Assembly<F: Field> {
  // TODO(@adr1anh): k only needed for the Error, remove later
  pub k:           u32,
  pub fixed:       Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
  pub permutation: permutation::keygen::Assembly,
  /// Optimization over `Vec<Vec<_>>`;
  /// BTreeSet is sorted, so we can use it to check if a row is enabled in O(log n) time;
  pub selectors:   Vec<SelectorColumn>,
  /// A range of available rows for assignment and copies
  pub usable_rows: Range<usize>,
}

// Implements `Send` + `Sync` if `thread-safe-region` feature is enabled.
impl<F: Field> SyncDeps for Assembly<F> {}

impl<F: Field> Assembly<F> {
  // todo(tk): get adrian to comment on this logic
  pub fn new(k: u32, num_rows: usize, cs: &ConstraintSystem<F>) -> Self {
    let usable_rows = 0..num_rows - (cs.blinding_factors() + 1);

    // todo(tk): EvaluationDomain splain
    let fixed = {
      let fixed_base = empty_lagrange_assigned(num_rows);
      vec![fixed_base; cs.num_fixed_columns()]
    };
    let permutation = permutation::keygen::Assembly::new(num_rows, cs.permutation());
    let selectors = vec![SelectorColumn::default(); cs.num_selectors()];

    Self { k, fixed, permutation, selectors, usable_rows }
  }
}

// Mostly same as `plonk::keygen`, differences for optimization on Selectors.
// unimplemented, low priority: optimize out copy check on linear constraints for Protostar
impl<F: Field> Assignment<F> for Assembly<F> {
  // Same as `plonk::keygen`
  fn enter_region<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR, {
    // Do nothing; we don't care about regions in this context.
  }

  // same as `plonk::keygen`
  fn exit_region(&mut self) {
    // Do nothing; we don't care about regions in this context.
  }

  // enable the selector column with index selector.1 for `row`
  fn enable_selector<A, AR>(
    &mut self,
    _: A,
    selector: &Selector,
    row: usize,
  ) -> Result<(), PlonkError>
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    if !self.usable_rows.contains(&row) {
      return Err(PlonkError::not_enough_rows_available(self.k));
    }

    // store a Selector at the given row
    self.selectors[selector.index()].insert(row);

    Ok(())
  }

  // same as `plonk::keygen`
  fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, PlonkError> {
    let column_index = column.index();
    if !self.usable_rows.contains(&row) {
      return Err(PlonkError::not_enough_rows_available(self.k));
    }

    // There is no instance in this context.
    Ok(Value::unknown())
  }

  // same as `plonk::keygen`
  fn assign_advice<V, VR, A, AR>(
    &mut self,
    _: A,
    _column: Column<Advice>,
    _row: usize,
    _: V,
  ) -> Result<(), PlonkError>
  where
    V: FnOnce() -> Value<VR>,
    VR: Into<Assigned<F>>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    // We only care about fixed columns here
    Ok(())
  }

  // same as `plonk::keygen`
  fn assign_fixed<V, VR, A, AR>(
    &mut self,
    _: A,
    column: Column<Fixed>,
    row: usize,
    to: V,
  ) -> Result<(), PlonkError>
  where
    V: FnOnce() -> Value<VR>,
    VR: Into<Assigned<F>>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    if !self.usable_rows.contains(&row) {
      return Err(PlonkError::not_enough_rows_available(self.k));
    }

    *self
      .fixed
      .get_mut(column.index())
      .and_then(|v| v.get_mut(row))
      .ok_or(PlonkError::BoundsFailure)? = to().into_field().assign()?;

    Ok(())
  }

  // same as `plonk::keygen`
  fn copy(
    &mut self,
    left_column: Column<Any>,
    left_row: usize,
    right_column: Column<Any>,
    right_row: usize,
  ) -> Result<(), PlonkError> {
    if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
      return Err(PlonkError::not_enough_rows_available(self.k));
    }

    self.permutation.copy(left_column, left_row, right_column, right_row)
  }

  // same as `plonk::keygen`
  fn fill_from_row(
    &mut self,
    column: Column<Fixed>,
    from_row: usize,
    to: Value<Assigned<F>>,
  ) -> Result<(), PlonkError> {
    if !self.usable_rows.contains(&from_row) {
      return Err(PlonkError::not_enough_rows_available(self.k));
    }

    let col = self.fixed.get_mut(column.index()).ok_or(PlonkError::BoundsFailure)?;

    let filler = to.assign()?;
    for row in self.usable_rows.clone().skip(from_row) {
      col[row] = filler;
    }

    Ok(())
  }

  // same as `plonk::keygen`
  fn get_challenge(&self, _: Challenge) -> Value<F> { Value::unknown() }

  // same as `plonk::keygen`
  fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
  where
    A: FnOnce() -> AR,
    AR: Into<String>, {
    // Do nothing
  }

  // same as `plonk::keygen`
  fn push_namespace<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR, {
    // Do nothing; we don't care about namespaces in this context.
  }

  // same as `plonk::keygen`
  fn pop_namespace(&mut self, _: Option<String>) {
    // Do nothing; we don't care about namespaces in this context.
  }
}

/// Sorted set of rows that are enabled for a selector
#[derive(Debug, Clone, Default)]
pub(crate) struct SelectorColumn(pub BTreeSet<usize>);

impl SelectorColumn {
  pub fn insert(&mut self, row: usize) { self.0.insert(row); }
}

// todo(tk): move somewhere
mod utils {
  use crate::{
    plonk::Assigned,
    poly::{LagrangeCoeff, Polynomial},
  };

  pub(crate) fn empty_lagrange_assigned<F: ff::Field>(
    size: usize,
  ) -> Polynomial<Assigned<F>, LagrangeCoeff> {
    Polynomial { values: vec![F::ZERO.into(); size], _marker: std::marker::PhantomData }
  }
}
