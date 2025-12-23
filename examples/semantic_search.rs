use khadyota::{VectorDB, Config, DistanceMetric};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Semantic Search Demo ===\n");
    
    // Simulate document embeddings
    let documents = vec![
        ("Rust is a systems programming language", vec![0.8, 0.1, 0.05, 0.05]),
        ("Python is great for machine learning", vec![0.1, 0.8, 0.05, 0.05]),
        ("JavaScript runs in the browser", vec![0.05, 0.1, 0.8, 0.05]),
        ("C++ is used for game development", vec![0.7, 0.1, 0.1, 0.1]),
        ("TensorFlow is a deep learning framework", vec![0.05, 0.9, 0.03, 0.02]),
        ("React is a JavaScript library", vec![0.05, 0.1, 0.8, 0.05]),
    ];
    
    // Pad vectors to 128 dimensions (just for demo)
    let documents: Vec<_> = documents
        .into_iter()
        .map(|(text, mut vec)| {
            vec.resize(128, 0.0);
            (text, vec)
        })
        .collect();
    
    let config = Config {
        dimensions: 128,
        metric: DistanceMetric::Cosine,
        use_pq: false, // Small dataset, skip PQ
        num_clusters: 3,
        num_probe: 2,
        ..Default::default()
    };
    
    let mut db = VectorDB::new(config)?;
    
    println!("Indexing documents...");
    for (text, vector) in &documents {
        let metadata = serde_json::json!({
            "text": text,
        });
        db.insert(vector.clone(), Some(metadata))?;
    }
    
    db.build_index()?;
    
    // Query: "programming languages"
    let mut query = vec![0.7, 0.1, 0.1, 0.1];
    query.resize(128, 0.0);
    
    println!("\nQuery: 'programming languages'\n");
    let results = db.search(&query, 3)?;
    
    println!("Results:");
    for (rank, result) in results.iter().enumerate() {
        if let Some(meta) = &result.metadata {
            println!(
                "  {}. [Distance: {:.3}] {}",
                rank + 1,
                result.distance,
                meta["text"]
            );
        }
    }
    
    Ok(())
}