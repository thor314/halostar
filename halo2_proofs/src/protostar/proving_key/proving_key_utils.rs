use crate::{
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
