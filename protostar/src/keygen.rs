//! The Accumulator Proving Key is the algebraic representation of the circuit.
use std::ops::Range;

use halo2_proofs::{
  arithmetic::Field,
  halo2curves::{group::Curve, CurveAffine},
  plonk::{
    self, permutation, Circuit, ConstraintSystem, Error as PlonkError, Expression, FloorPlanner,
    Selector,
  },
  poly::commitment::Params,
};

use self::proving_key_utils::{FoldingConstraint, LinearConstraint, SimpleSelector};
use crate::keygen::assembly::Assembly;

/// All fixed data for a circuit; the algebraic representation of a circuit. Generates an
/// Accumulator.
// DOCS(TK)
#[derive(Debug)]
pub struct AccumulatorProvingKey<C: CurveAffine> {
  /// Maximum number of rows in the trace, including blinding factors, which add ~10 or so rows
  pub num_rows:    usize,
  /// max active rows, without blinding
  pub usable_rows: Range<usize>,

  /// circuit's unmodified constraint system
  // TODO(TK): flatten necessary members of the cs into the proving key, and eliminate cs
  pub cs: ConstraintSystem<C::Scalar>,

  pub folding_constraints: FoldingConstraint<C>,
  pub linear_constraints:  LinearConstraint<C>,
  pub simple_selectors:    SimpleSelector,

  // keep track of actual number of elements in each column
  num_selector_rows: Vec<usize>,
  num_fixed_rows:    Vec<usize>,
  num_advice_rows:   Vec<usize>,
  num_instance_rows: Vec<usize>,

  // Fixed columns
  // fixed[col][row]
  // pub fixed:     Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
  // Selector columns as `bool`s
  // selectors[col][row]
  // TODO(@adr1anh): Replace with a `BTreeMap` to save memory
  pub selectors: Vec<Vec<bool>>,

  // TODO(tk): will be replaced by a permutation proving key
  //
  // Permutation columns mapping each, where
  // permutations[col][row] = sigma(col*num_rows + row)
  // TODO(@adr1anh): maybe store as Vec<Vec<(usize,usize)>)
  pub permutations: Vec<Vec<usize>>,
}

impl<C: CurveAffine> AccumulatorProvingKey<C> {
  pub fn new<'params, P, Circ>(params: &P, circuit: &Circ) -> Result<Self, PlonkError>
  where
    P: Params<'params, C>,
    Circ: Circuit<C::Scalar>, {
    let num_rows = params.n() as usize;
    // log_2(num_rows)
    let k = params.k();

    // todo(tk) what does circuit params do?
    let (cs, config) = {
      let mut cs = ConstraintSystem::default();
      #[cfg(feature = "circuit-params")]
      let config = Circ::configure_with_params(&mut cs, circuit.params());
      #[cfg(not(feature = "circuit-params"))]
      let config = Circ::configure(&mut cs);

      (cs, config)
    };

    // todo(tk): check num rows for blinding; if num_rows < cs.minimum_rows() {...
    let mut assembly = Assembly::<C::Scalar>::new(num_rows, &cs, k);
    // Synthesize the circuit to obtain URS
    Circ::FloorPlanner::synthesize(&mut assembly, circuit, config, cs.constants().clone())?;

    // let fixed = batch_invert_assigned(assembly.fixed);
    // let permutations = assembly.permutation.build_permutations(num_rows, &cs.permutation);

    // todo(tk): Adrian wtf
    // let mut linear_constraints = Vec::new();
    // let mut linear_constraints = Vec::new();
    // let mut folding_constraints = Vec::new();
    // for gate in cs.gates().iter() {
    //   for poly in gate.polynomials().iter() {
    //     // let poly = poly.clone().merge_challenge_products();
    //     // let degree = poly.folding_degree();
    //     // if degree <= 1 {
    //     //   linear_constraints.push(poly);
    //     // } else if poly.is_empty() {
    //     //   folding_constraints.push(poly);
    //     // }
    //   }
    // }

    // let (folding_constraints, simple_selectors): (Vec<_>, Vec<_>) =
    //   folding_constraints.iter().map(|polys| extract_common_simple_selector(&polys)).unzip();

    // Self { }
    todo!()
  }

  /// Obtain the maximum degree gate of the circuit. Used to compute `d`, the maximimum
  /// homogenization degree of the the folding scheme.
  // TODO(TK): doc caller
  pub fn max_degree(&self) -> usize { todo!() }

  // TODO(TK): where used?
  /// Returns the smallest `k` such that n ≤ 2^{2k}.
  /// Approximately log₂(√n)
  pub fn log2_sqrt_num_rows(&self) -> usize {
    let k = log2_ceil(self.num_rows);
    // if k is odd, add 1, and divide by 2
    (k + (k % 2)) >> 1
  }

  /// Returns a vector of same size as `num_challenges` where each entry
  /// is equal to the highest power `d` that a challenge appears over all `Gate`s
  pub fn max_challenge_powers(&self) -> Vec<usize> {
    let num_challenges = self.cs.num_challenges();
    let mut _max_challenge_power = vec![1; num_challenges];

    // for poly in self
    //   .folding_constraints
    //   .iter()
    //   .flat_map(|poly| poly.iter())
    //   .chain(self.linear_constraints.iter())
    // {
    //   for (idx, max_power) in max_challenge_power.iter_mut().enumerate() {
    //     let new_power = poly.max_challenge_power(idx);
    //     *max_power = std::cmp::max(*max_power, new_power);
    //   }
    // }

    // max_challenge_power
    todo!()
  }
}

// todo(tk): move
fn log2_ceil(num_rows: usize) -> usize { todo!() }

// TODO(TK): move to own file, rename
mod proving_key_utils {
  use halo2_proofs::{
    halo2curves::CurveAffine,
    plonk::{ConstraintSystem, Expression, Selector},
  };

  // TODO(TK): move to helper mods
  // TODO(TK): types are likely inefficient
  #[derive(Debug)]
  pub struct FoldingConstraint<C: CurveAffine>(pub Vec<Vec<Expression<C::Scalar>>>);
  impl<C: CurveAffine> FoldingConstraint<C> {
    //   // TODO(TK): annotate
    //   pub fn new(cs: ConstraintSystem<C::Scalar>) -> Self {
    //     // TODO(TK): code smell, construct LC's
    //     let mut linear_constraints = Vec::new();

    //     cs.gates()
    //       .iter()
    //       .map(|gate| {
    //         gate
    //           .polynomials()
    //           .iter()
    //           .filter_map(|poly| {
    //             let poly = poly.clone().merge_challenge_products();
    //             let degree: usize = poly.folding_degree();
    //             if degree <= 1 {
    //               linear_constraints.push(poly);
    //               None
    //             } else {
    //               Some(poly)
    //             }
    //           })
    //           .collect::<Vec<_>>()
    //       })
    //       .filter(|poly_vec| !poly_vec.is_empty())
    //       .collect()
    //       .into()
    //   }

    /// Total number of linearly-independent constraints, whose degrees are larger than 1
    pub fn num_folding_constraints(&self) -> usize { self.0.iter().map(|polys| polys.len()).sum() }
  }

  #[derive(Debug)]
  pub struct SimpleSelector(pub Vec<Option<Selector>>);
  #[derive(Debug)]
  pub struct LinearConstraint<C: CurveAffine>(pub Vec<Expression<C::Scalar>>);
}

mod assembly {
  use std::ops::Range;

  use halo2_proofs::{
    arithmetic::Field,
    circuit::{layouter::SyncDeps, Value},
    plonk::{
      permutation, Advice, Any, Assigned, Assignment, Challenge, Column, ConstraintSystem,
      Error as PlonkError, Fixed, Instance, Selector,
    },
    poly::{LagrangeCoeff, Polynomial, *},
  };

  /// Assembly to be used in circuit synthesis.
  #[derive(Debug)]
  pub struct Assembly<F: Field> {
    pub usable_rows: Range<usize>,
    // TODO(@adr1anh): Only needed for the Error, remove later
    pub k:           u32,

    pub fixed:       Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    pub permutation: permutation::keygen::Assembly,
    // TODO(@adr1anh): Replace with Vec<BTreeSet<bool>>
    pub selectors:   Vec<Vec<bool>>,

    // keep track of actual number of elements in each column
    pub num_selector_rows: Vec<usize>,
    pub num_fixed_rows:    Vec<usize>,
    pub num_advice_rows:   Vec<usize>,
    pub num_instance_rows: Vec<usize>,

    _marker: std::marker::PhantomData<F>,
  }

  impl<F: Field> Assembly<F> {
    pub fn new(num_rows: usize, cs: &ConstraintSystem<F>, k: u32) -> Self {
      todo!()
      // let mut assembly: Assembly<F> = Assembly {
      //   usable_rows: 0..num_rows - (cs.blinding_factors() + 1),
      //   k,
      //   fixed: vec![
      //     EvaluationDomain::<F>::empty_lagrange_assigned(num_rows);
      //     cs.num_fixed_columns()
      //   ],
      //   permutation: permutation::keygen::Assembly::new(num_rows, &cs.permutation()),
      //   selectors: vec![vec![false; num_rows]; cs.num_selectors()],
      //   num_selector_rows: vec![0; cs.num_selectors()],
      //   num_fixed_rows: vec![0; cs.num_fixed_columns()],
      //   num_advice_rows: vec![0; cs.num_advice_columns()],
      //   num_instance_rows: vec![0; cs.num_instance_columns()],
      //   _marker: std::marker::PhantomData,
      // };
      // assembly
    }
  }
  // let mut assembly: Assembly<C::Scalar> = Assembly {
  //   usable_rows: 0..num_rows - (cs.blinding_factors() + 1),
  //   k,
  //   fixed: vec![empty_lagrange_assigned(num_rows); cs.num_fixed_columns()],
  //   permutation: permutation::keygen::Assembly::new(num_rows, &cs.permutation()),
  //   selectors: vec![vec![false; num_rows]; cs.num_selectors()],
  //   num_selector_rows: vec![0; cs.num_selectors()],
  //   num_fixed_rows: vec![0; cs.num_fixed_columns()],
  //   num_advice_rows: vec![0; cs.num_advice_columns()],
  //   num_instance_rows: vec![0; cs.num_instance_columns()],
  //   _marker: std::marker::PhantomData,
  // };

  impl<F: Field> SyncDeps for Assembly<F> {}

  impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
      NR: Into<String>,
      N: FnOnce() -> NR, {
      // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
      // Do nothing; we don't care about regions in this context.
    }

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
      self.num_selector_rows[selector.index()] =
        std::cmp::max(self.num_selector_rows[selector.index()] + 1, row);

      if !self.usable_rows.contains(&row) {
        return Err(PlonkError::not_enough_rows_available(self.k));
      }

      self.selectors[selector.index()][row] = true;

      Ok(())
    }

    // TODO(TK): ask adrian why he made this mut
    // fn query_instance(
    //   &mut self,
    //   column: Column<Instance>,
    //   row: usize,
    // ) -> Result<Value<F>, PlonkError> { let column_index = column.index();
    //   self.num_instance_rows[column_index] = std::cmp::max(self.num_instance_rows[column_index] +
    //   1, row); if !self.usable_rows.contains(&row) { return
    //   Err(PlonkError::not_enough_rows_available(self.k)); }

    //   // There is no instance in this context.
    //   Ok(Value::unknown())
    // }

    fn query_instance(&self, column: Column<Instance>, row: usize) -> Result<Value<F>, PlonkError> {
      let column_index = column.index();
      // let num_instance_rows_at_col_idx =
      //   std::cmp::max(self.num_instance_rows[column_index] + 1, row);
      if !self.usable_rows.contains(&row) {
        return Err(PlonkError::not_enough_rows_available(self.k));
      }

      // // There is no instance in this context.
      Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
      &mut self,
      _: A,
      column: Column<Advice>,
      row: usize,
      _: V,
    ) -> Result<(), PlonkError>
    where
      V: FnOnce() -> Value<VR>,
      VR: Into<Assigned<F>>,
      A: FnOnce() -> AR,
      AR: Into<String>,
    {
      let column_index = column.index();
      self.num_advice_rows[column_index] =
        std::cmp::max(self.num_advice_rows[column_index] + 1, row);
      // We only care about fixed columns here
      Ok(())
    }

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
      let column_index = column.index();
      self.num_fixed_rows[column_index] = std::cmp::max(self.num_fixed_rows[column_index] + 1, row);

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

    fn copy(
      &mut self,
      left_column: Column<Any>,
      left_row: usize,
      right_column: Column<Any>,
      right_row: usize,
    ) -> Result<(), PlonkError> {
      let left_column_index = left_column.index();
      let right_column_index = right_column.index();

      match left_column.column_type() {
        Any::Advice(_) => {
          self.num_advice_rows[left_column_index] =
            std::cmp::max(self.num_advice_rows[left_column_index], left_row + 1);
        },
        Any::Fixed => {
          self.num_fixed_rows[left_column_index] =
            std::cmp::max(self.num_fixed_rows[left_column_index], left_row + 1);
        },
        Any::Instance => {
          self.num_instance_rows[left_column_index] =
            std::cmp::max(self.num_instance_rows[left_column_index], left_row + 1);
        },
      }
      match right_column.column_type() {
        Any::Advice(_) => {
          self.num_advice_rows[right_column_index] =
            std::cmp::max(self.num_advice_rows[right_column_index], right_row + 1);
        },
        Any::Fixed => {
          self.num_fixed_rows[right_column_index] =
            std::cmp::max(self.num_fixed_rows[right_column_index], right_row + 1);
        },
        Any::Instance => {
          self.num_instance_rows[right_column_index] =
            std::cmp::max(self.num_instance_rows[right_column_index], right_row + 1);
        },
      }

      if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
        return Err(PlonkError::not_enough_rows_available(self.k));
      }

      self.permutation.copy(left_column, left_row, right_column, right_row)
    }

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

    fn get_challenge(&self, _: Challenge) -> Value<F> { Value::unknown() }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
      A: FnOnce() -> AR,
      AR: Into<String>, {
      // Do nothing
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
      NR: Into<String>,
      N: FnOnce() -> NR, {
      // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
      // Do nothing; we don't care about namespaces in this context.
    }
  }
}

// todo(tk): wtf is this
/// Undo `Constraints::with_selector` and return the common top-level `Selector` along with the
/// `Expression` it selects. If no simple `Selector` is found, returns the original list of
/// polynomials.
fn extract_common_simple_selector<F: Field>(
  polys: &[Expression<F>],
) -> (Vec<Expression<F>>, Option<Selector>) {
  let (extracted_polys, simple_selectors): (Vec<_>, Vec<_>) = polys
    .iter()
    .map(|poly| {
      // Check whether the top node is a multiplication by a selector
      let (simple_selector, poly) = match poly {
        // If the whole polynomial is multiplied by a simple selector,
        // return it along with the expression it selects
        Expression::Product(e1, e2) => match (&**e1, &**e2) {
          (Expression::Selector(s), e) | (e, Expression::Selector(s)) => (Some(*s), e),
          _ => (None, poly),
        },
        _ => (None, poly),
      };
      (poly.clone(), simple_selector)
    })
    .unzip();

  // // Check if all simple selectors are the same and if so select it
  // let potential_selector = match simple_selectors.as_slice() {
  //   [head, tail @ ..] =>
  //     if let Some(s) = *head {
  //       tail.iter().all(|x| x.is_some_and(|x| s == x)).then(|| s)
  //     } else {
  //       None
  //     },
  //   [] => None,
  // };

  // // if we haven't found a common simple selector, then we just use the previous polys
  // if potential_selector.is_none() {
  //   (polys.to_vec(), None)
  // } else {
  //   (extracted_polys, potential_selector)
  // }
  todo!()
}
