//! todo(tk): rewire this repo for any degree of modularity
use std::{iter::zip, marker::PhantomData};

use ff::{Field, FromUniformBytes};
use halo2curves::CurveAffine;
use rand_core::RngCore;

use super::proving_key::ProvingKey;
use crate::{
  arithmetic::{lagrange_interpolate, parallelize, powers},
  plonk::{Circuit, Error as PlonkError, FixedQuery},
  poly::{
    commitment::{Blind, CommitmentScheme, Params, Prover},
    // empty_lagrange,
    LagrangeCoeff,
    Polynomial,
    Rotation,
  },
  transcript::{EncodedChallenge, TranscriptWrite},
};

// todo(tk)
pub struct Accumulator<C: CurveAffine> {
  // instance_transcript: InstanceTranscript<C>,
  // advice_transcript: AdviceTranscript<C>,
  // lookup_transcript: LookupTranscipt<C>,
  // compressed_verifier_transcript: CompressedVerifierTranscript<C>,

  // // Powers of a challenge y for taking a random linear-combination of all constraints.
  // ys: Vec<C::Scalar>,

  // // For each constraint of degree > 1, we cache its error polynomial evaluation here
  // // so we can interpolate all of them individually.
  // constraint_errors: Vec<C::Scalar>,

  // // Error value for all constraints
  error: C::Scalar,
}

impl<C: CurveAffine> Accumulator<C> {
  pub fn new(c: C) -> Self {
    todo!();
  }
}

/// Runs the IOP until the decision phase, and returns an `Accumulator` containing the entirety of
/// the transcript. The result can be folded into another `Accumulator`.
pub fn create_accumulator<
  'params,
  C: CurveAffine,
  P: Params<'params, C>,
  E: EncodedChallenge<C>,
  R: RngCore,
  T: TranscriptWrite<C, E>,
  Circ: Circuit<C::Scalar>,
>(
  params: &P,
  pk: &ProvingKey<C>,
  circuit: &Circ,
  instances: &[&[C::Scalar]],
  mut _rng: R,
  _transcript: &mut T,
) -> Result<Accumulator<C>, PlonkError> {
  todo!();
}
//     // Hash verification key into transcript
//     // pk.vk.hash_into(transcript)?;

//     // Add public inputs/outputs to the transcript, and convert them to `Polynomial`s
//     let instance_transcript = create_instance_transcript(params, &pk.cs, instances, transcript)?;

//     // Run multi-phase IOP section to generate all `Advice` columns
//     let advice_transcript = create_advice_transcript(
//         params,
//         pk,
//         circuit,
//         &instance_transcript.instance_polys,
//         &mut rng,
//         transcript,
//     )?;

//     // Run the 2-round logUp IOP for all lookup arguments
//     let lookup_transcript = create_lookup_transcript(
//         params,
//         pk,
//         &advice_transcript.challenges,
//         &advice_transcript.advice_polys,
//         &instance_transcript.instance_polys,
//         rng,
//         transcript,
//     );

//     // Generate random column(s) to multiply each constraint
//     // so that we can compress them to a single constraint
//     let compressed_verifier_transcript = create_compressed_verifier_transcript(params,
// transcript);

//     // Challenge for the RLC of all constraints (all gates and all lookups)
//     let y = *transcript.squeeze_challenge_scalar::<C::Scalar>();

//     Ok(Accumulator::new(
//         pk,
//         instance_transcript,
//         advice_transcript,
//         lookup_transcript,
//         compressed_verifier_transcript,
//         y,
//     ))
// }
