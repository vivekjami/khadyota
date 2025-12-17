use crate::error::Result;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub struct Serializer;

impl Serializer {
    /// Serialize data to a file
    pub fn save<T: serde::Serialize>(data: &T, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, data)?;
        Ok(())
    }
    
    /// Deserialize data from a file
    pub fn load<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let data = bincode::deserialize_from(reader)?;
        Ok(data)
    }
    
    /// Save vectors in efficient binary format
    pub fn save_vectors(vectors: &[Vec<f32>], path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write count
        let count = vectors.len() as u64;
        writer.write_all(&count.to_le_bytes())?;
        
        // Write dimensions (assuming all same size)
        if let Some(first) = vectors.first() {
            let dims = first.len() as u32;
            writer.write_all(&dims.to_le_bytes())?;
            
            // Write all vectors
            for vec in vectors {
                for &val in vec {
                    writer.write_all(&val.to_le_bytes())?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Load vectors from binary format
    pub fn load_vectors(path: &Path) -> Result<Vec<Vec<f32>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read count
        let mut count_bytes = [0u8; 8];
        reader.read_exact(&mut count_bytes)?;
        let count = u64::from_le_bytes(count_bytes) as usize;
        
        // Read dimensions
        let mut dims_bytes = [0u8; 4];
        reader.read_exact(&mut dims_bytes)?;
        let dims = u32::from_le_bytes(dims_bytes) as usize;
        
        // Read vectors
        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            let mut vec = Vec::with_capacity(dims);
            for _ in 0..dims {
                let mut val_bytes = [0u8; 4];
                reader.read_exact(&mut val_bytes)?;
                vec.push(f32::from_le_bytes(val_bytes));
            }
            vectors.push(vec);
        }
        
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_vector_serialization() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        // Create test vectors
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        // Save and load
        Serializer::save_vectors(&vectors, path).unwrap();
        let loaded = Serializer::load_vectors(path).unwrap();
        
        assert_eq!(vectors, loaded);
    }
}