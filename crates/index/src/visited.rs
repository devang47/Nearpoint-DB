//! Per-thread "visited" marks for graph search, reset in O(1).
//!
//! A `Vec<bool>` per query would cost a 1M-entry clear every search. Instead
//! each slot stores the generation that last visited it, and a reset just
//! bumps the generation. One buffer per thread, reused across searches.

use std::cell::RefCell;

pub(crate) struct Visited {
    stamps: Vec<u32>,
    generation: u32,
}

impl Visited {
    /// Starts a fresh search over ids `0..n`.
    pub fn reset(&mut self, n: usize) {
        if self.stamps.len() < n {
            self.stamps.resize(n, 0);
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped after 4B searches: old stamps could collide, so clear.
            self.stamps.fill(0);
            self.generation = 1;
        }
    }

    /// Marks `id`; returns false if it was already visited this search.
    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        let s = &mut self.stamps[id as usize];
        let fresh = *s != self.generation;
        *s = self.generation;
        fresh
    }
}

thread_local! {
    pub(crate) static VISITED: RefCell<Visited> = const {
        RefCell::new(Visited { stamps: Vec::new(), generation: 0 })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_forgets_previous_marks() {
        let mut v = Visited {
            stamps: Vec::new(),
            generation: u32::MAX - 1,
        };
        v.reset(4);
        assert!(v.insert(2));
        assert!(!v.insert(2));
        v.reset(4); // wraps to 0 → full clear
        assert!(v.insert(2));
        v.reset(8); // grows
        assert!(v.insert(7) && v.insert(2));
    }
}
