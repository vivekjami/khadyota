use khadyota::{VectorDB, Config};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════╗");
    println!("║   KHADYOTA PERFORMANCE REPORT            ║");
    println!("╚══════════════════════════════════════════╝\n");
    
    let sizes = vec![1_000, 10_000, 50_000];
    
    for &size in &sizes {
        println!("\n═══ Dataset: {} vectors ═══", size);
        
        let config = Config {
            dimensions: 512,
            use_pq: true,
            pq_subvectors: 8,
            num_clusters: (size as f64).sqrt() as usize,
            num_probe: ((size as f64).sqrt() / 10.0) as usize,
            ..Default::default()
        };
        
        let mut db = VectorDB::new(config)?;
        
        // Build database
        print!("Building... ");
        let build_start = Instant::now();
        
        for i in 0..size {
            let vector: Vec<f32> = (0..512)
                .map(|j| ((i * 512 + j) as f32).sin())
                .collect();
            db.insert(vector, None)?;
        }
        
        db.build_index()?;
        let build_time = build_start.elapsed();
        println!("{:?}", build_time);
        
        // Query performance
        let query: Vec<f32> = (0..512).map(|i| (i as f32).cos()).collect();
        
        // Warmup
        for _ in 0..10 {
            db.search(&query, 10)?;
        }
        
        // Measure p50, p95, p99
        let mut times = Vec::new();
        for _ in 0..100 {
            let start = Instant::now();
            db.search(&query, 10)?;
            times.push(start.elapsed());
        }
        
        times.sort();
        
        let p50 = times[50];
        let p95 = times[95];
        let p99 = times[99];
        
        println!("Query Latency:");
        println!("  p50: {:?}", p50);
        println!("  p95: {:?}", p95);
        println!("  p99: {:?}", p99);
        println!("  QPS: {:.0}", 1.0 / p50.as_secs_f64());
        
        // Memory estimate
        let memory_mb = (size * 512 * 4) / 1_000_000; // Original vectors
        let compressed_mb = (size * 8) / 1_000_000;   // PQ codes
        println!("Memory:");
        println!("  Original: {} MB", memory_mb);
        println!("  Compressed: {} MB", compressed_mb);
        println!("  Ratio: {:.1}x", memory_mb as f32 / compressed_mb as f32);
    }
    
    println!("\n✓ Performance report complete!\n");
    
    Ok(())
}