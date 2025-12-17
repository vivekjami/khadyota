use crate::error::Result;
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Memory-mapped vector storage for zero-copy access
pub struct MmapVectors {
    _file: File,
    mmap: Mmap,
    dimensions: usize,
    count: usize,
}

impl MmapVectors {
    /// Open an existing memory-mapped vector file
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        
        // Read header (count + dimensions)
        let count = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let dimensions = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;
        
        Ok(Self {
            _file: file,
            mmap,
            dimensions,
            count,
        })
    }
    
    /// Get a vector by index (zero-copy)
    pub fn get(&self, index: usize) -> Option<&[f32]> {
        if index >= self.count {
            return None;
        }
        
        let offset = 12 + index * self.dimensions * 4;
        let slice = &self.mmap[offset..offset + self.dimensions * 4];
        
        // SAFETY: We know the data is properly aligned f32 values
        // because we wrote it that way
        unsafe {
            Some(std::slice::from_raw_parts(
                slice.as_ptr() as *const f32,
                self.dimensions
            ))
        }
    }
    
    pub fn len(&self) -> usize {
        self.count
    }
    
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::serialization::Serializer;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_mmap_vectors() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path();
        
        // Create and save vectors
        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        Serializer::save_vectors(&vectors, path).unwrap();
        
        // Load with mmap
        let mmap_vecs = MmapVectors::open(path).unwrap();
        
        assert_eq!(mmap_vecs.len(), 2);
        assert_eq!(mmap_vecs.dimensions(), 4);
        
        let vec0 = mmap_vecs.get(0).unwrap();
        assert_eq!(vec0, &[1.0, 2.0, 3.0, 4.0]);
    }
}