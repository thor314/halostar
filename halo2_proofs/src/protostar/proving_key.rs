//! The Accumulator Proving Key is the algebraic representation of the circuit.
use std::ops::Range;

use self::{
  assembly::Assembly,
  proving_key_utils::{FoldingConstraint, LinearConstraint, SimpleSelector},
};
use crate::{
  arithmetic::Field,
  halo2curves::{group::Curve, CurveAffine},
  plonk::{
    self, permutation, Circuit, ConstraintSystem, Error as PlonkError, Expression, FloorPlanner,
    Selector,
  },
  poly::commitment::Params,
};

mod assembly;
mod proving_key_utils;

/// All fixed data for a circuit; the algebraic representation of a circuit. Generates an
/// Accumulator.
// DOCS(TK)
#[derive(Debug)]
pub struct ProvingKey<C: CurveAffine> {
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

impl<C: CurveAffine> ProvingKey<C> {
  /// todo(tk): doc `k`
  pub fn new<Circ: Circuit<C::Scalar>>(k: u32, circuit: &Circ) -> Result<Self, PlonkError> {
    assert!(k < 32, "k must be less than 32"); // todo(tk): error handling
    let num_rows = 1 << k;

    // todo(tk) what does circuit params do? Might be able to elim.
    let (cs, config) = {
      let mut cs = ConstraintSystem::default();
      #[cfg(feature = "circuit-params")]
      let config = Circ::configure_with_params(&mut cs, circuit.params());
      #[cfg(not(feature = "circuit-params"))]
      let config = Circ::configure(&mut cs);

      (cs, config)
    };

    // todo(tk): check num rows for blinding; if num_rows < cs.minimum_rows() {...

    let mut assembly = Assembly::<C::Scalar>::new(k, num_rows, &cs);

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
