#[derive(Clone, Copy, Debug)]
struct Hole {
    /// in bytes
    offset: u64,
    /// in bytes
    size: u64,
}

struct Holes {
    holes: Vec<Hole>,
    end: u64,
}

impl Holes {
    fn new(offset: u64, size: u64) -> Self {
        Self {
            holes: vec![Hole { offset, size }],
            end: size,
        }
    }
    fn allocate(&mut self, size: u64) -> Option<u64> {
        if let Some((i, hole)) = self
            .holes
            .iter()
            .enumerate()
            .filter(|(_, h)| h.size >= size)
            .min_by_key(|(_, h)| h.size)
            .map(|(i, h)| (i, *h))
        {
            let offset = hole.offset;
            self.holes[i].size -= size;
            self.holes[i].offset += size;
            Some(offset)
        } else {
            None
        }
    }
    fn merge_holes(&mut self) {
        let mut to_remove = Vec::new();
        for i in 0..self.holes.len() {
            for j in 0..self.holes.len() {
                if i == j {
                    continue;
                }
                let hi = self.holes[i];
                let hj = self.holes[j];
                if hi.offset + hi.size == hj.offset {
                    self.holes[i].size += hj.size;
                    to_remove.push(j);
                } else if hj.offset + hj.size == hi.offset {
                    self.holes[j].size += hi.size;
                    to_remove.push(i);
                }
                if hi.size == 0 && hi.offset != self.end {
                    //to_remove.push(i);
                }
            }
        }
        to_remove.sort_unstable();
        for i in to_remove.into_iter().rev() {
            if self.holes.len() > i {
                self.holes.swap_remove(i);
            }
        }
    }
    fn expand(&mut self, size: u64) {
        let last_hole = self.holes.iter_mut().max_by_key(|h| h.offset).unwrap();
        if last_hole.offset + last_hole.size == self.end {
            last_hole.size = size - last_hole.offset;
        } else {
            let offset = last_hole.offset + last_hole.size;
            self.holes.push(Hole {
                offset,
                size: size - offset,
            })
        }
        self.end = size;
    }
    fn deallocate(&mut self, offset: u64, size: u64) {
        self.holes.push(Hole { offset, size });
        self.holes.sort_unstable_by_key(|h| h.offset);
        let mut last_end = 0;
        for hole in &self.holes {
            assert!(last_end <= hole.offset);
            last_end = hole.offset + hole.size;
        }
        self.merge_holes()
    }
}

#[test]
fn test_holes() {
    let mut holes = Holes::new(0, 120);
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(25), Some(0));
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(50), Some(25));
    holes.add_hole(Hole {
        offset: 0,
        size: 25,
    });
    assert_eq!(holes.holes.len(), 2);
    assert_eq!(holes.find(50), None);
    holes.expand(125);
    assert_eq!(holes.find(50), Some(75));
    assert_eq!(holes.holes.len(), 1);
    holes.add_hole(Hole {
        offset: 25,
        size: 50,
    });
    assert_eq!(holes.holes.len(), 1);
    holes.add_hole(Hole {
        offset: 75,
        size: 50,
    });
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(125), Some(0));
}
