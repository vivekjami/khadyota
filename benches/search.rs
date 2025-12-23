use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use khadyota::{VectorDB, Config, DistanceMetric};

fn setup_db(size: usize, use_pq: bool, num_clusters: usize) -> VectorDB {
    let config = Config {
        dimensions: 512,
        metric: DistanceMetric::Cosine,
        use_pq,
        pq_subvectors: 8,
        num_clusters,
        num_probe: num_clusters / 10,
    };
    
    let mut db = VectorDB::new(config).unwrap();
    
    for i in 0..size {
        let vector: Vec<f32> = (0..512)
            .map(|j| ((i * 512 + j) as f32).sin())
            .collect();
        
        db.insert(vector, None).unwrap();
    }
    
    db.build_index().unwrap();
    db
}

fn bench_search_by_size(c: &mut Criterion) {
    let sizes = vec![1_000, 10_000, 100_000];
    
    for size in sizes {
        let db = setup_db(size, true, (size as f64).sqrt() as usize);
        let query: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
        
        c.bench_with_input(
            BenchmarkId::new("search", size),
            &size,
            |b, _| {
                b.iter(|| db.search(black_box(&query), 10))
            },
        );
    }
}

fn bench_search_with_without_pq(c: &mut Criterion) {
    let mut group = c.benchmark_group("pq_comparison");
    let size = 10_000;
    
    for use_pq in [false, true] {
        let db = setup_db(size, use_pq, 100);
        let query: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
        
        group.bench_with_input(
            BenchmarkId::new(if use_pq { "with_pq" } else { "without_pq" }, size),
            &size,
            |b, _| {
                b.iter(|| db.search(black_box(&query), 10))
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_search_by_size, bench_search_with_without_pq);
criterion_main!(benches);