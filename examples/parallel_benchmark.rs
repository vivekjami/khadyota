use khadyota::{VectorDB, Config};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Parallel Query Benchmark ===\n");
    
    let config = Config {
        dimensions: 512,
        use_pq: true,
        pq_subvectors: 8,
        num_clusters: 100,
        num_probe: 10,
        ..Default::default()
    };
    
    let mut db = VectorDB::new(config)?;
    
    println!("Building database with 50,000 vectors...");
    for i in 0..50_000 {
        let vector: Vec<f32> = (0..512)
            .map(|j| ((i * 512 + j) as f32).sin())
            .collect();
        db.insert(vector, None)?;
    }
    
    db.build_index()?;
    
    // Generate 100 queries
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..512).map(|j| ((i + j) as f32).cos()).collect())
        .collect();
    
    println!("\nBenchmarking 100 queries...\n");
    
    // Sequential
    let start = Instant::now();
    let mut results = Vec::new();
    for query in &queries {
        results.push(db.search(query, 10)?);
    }
    let sequential_time = start.elapsed();
    
    println!("Sequential: {:?} ({:.2} queries/sec)",
        sequential_time,
        100.0 / sequential_time.as_secs_f64()
    );
    
    // Parallel
    let start = Instant::now();
    let results_parallel = db.batch_search(&queries, 10)?;
    let parallel_time = start.elapsed();
    
    println!("Parallel:   {:?} ({:.2} queries/sec)",
        parallel_time,
        100.0 / parallel_time.as_secs_f64()
    );
    
    println!("\nSpeedup: {:.2}x", 
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );
    
    Ok(())
}