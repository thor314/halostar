#[macro_use] extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use group::ff::Field;
use halo2_proofs::*;
use halo2curves::pasta::Fp;
use rand_core::OsRng;

use crate::arithmetic::best_fft;

fn criterion_benchmark(c: &mut Criterion) {
  let mut group = c.benchmark_group("fft");
  for k in 3..19 {
    group.bench_function(BenchmarkId::new("k", k), |b| {
      let mut a = (0..(1 << k)).map(|_| Fp::random(OsRng)).collect::<Vec<_>>();
      let omega = Fp::random(OsRng); // would be weird if this mattered
      b.iter(|| {
        best_fft(&mut a, omega, k as u32);
      });
    });
  }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
