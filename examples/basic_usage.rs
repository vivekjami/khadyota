use khadyota::{VectorDB, Config, DistanceMetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Khadyota Basic Usage Example ===\n");
    
    // Create a new database
    let config = Config {
        dimensions: 128,
        metric: DistanceMetric::Cosine,
        use_pq: true,
        pq_subvectors: 8,
        num_clusters: 20,
        num_probe: 5,
    };
    
    let mut db = VectorDB::new(config)?;
    
    println!("1. Inserting 1,000 vectors...");
    for i in 0..1_000 {
        let vector: Vec<f32> = (0..128)
            .map(|j| ((i * 128 + j) as f32).sin())
            .collect();
        
        let metadata = serde_json::json!({
            "id": i,
            "category": if i % 3 == 0 { "A" } else if i % 3 == 1 { "B" } else { "C" },
            "timestamp": format!("2024-01-{:02}", (i % 28) + 1),
        });
        
        db.insert(vector, Some(metadata))?;
    }
    println!("✓ Inserted {} vectors\n", db.len());
    
    println!("2. Building search index (PQ + IVF)...");
    db.build_index()?;
    
    println!("\n3. Searching for similar vectors...");
    let query: Vec<f32> = (0..128).map(|i| (i as f32).cos()).collect();
    
    let results = db.search(&query, 10)?;
    
    println!("\nTop 10 Results:");
    for (rank, result) in results.iter().enumerate() {
        println!(
            "  {}. ID: {}, Distance: {:.4}, Metadata: {:?}",
            rank + 1,
            result.id,
            result.distance,
            result.metadata
        );
    }
    
    println!("\n4. Saving database to disk...");
    db.save(std::path::Path::new("example.kdb"))?;
    
    println!("\n5. Loading database from disk...");
    let loaded_db = VectorDB::load(std::path::Path::new("example.kdb"))?;
    println!("✓ Loaded database with {} vectors", loaded_db.len());
    
    println!("\n✓ Example complete!");
    
    Ok(())
}