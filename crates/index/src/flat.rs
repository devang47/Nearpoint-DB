//! Exact brute-force search. Task 2.2 wraps this in the `Index` trait and
//! parallelizes it; the bench harness uses it as the exactness baseline.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A search hit: vector id and squared L2 distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

// Max-heap order: farther first; on equal distance the larger id is
// "farther", so ties keep the smaller id (the usual ground-truth convention).
impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// The `k` nearest rows of `data` (row-major, `dim` columns) to `query`
/// by squared L2, nearest first.
pub fn search(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<Hit> {
    assert_eq!(query.len(), dim);
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for (id, row) in data.chunks_exact(dim).enumerate() {
        let hit = Hit {
            id: id as u32,
            dist: kernels::l2_sq(query, row),
        };
        if heap.len() < k {
            heap.push(hit);
        } else if let Some(mut far) = heap.peek_mut()
            && hit < *far
        {
            *far = hit;
        }
    }
    heap.into_sorted_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_first_and_ties_keep_smaller_id() {
        // Rows at x = 0, 3, 1, 1, 5 (dim 1); query at 0.
        let data = [0.0, 3.0, 1.0, 1.0, 5.0];
        let ids = |k| {
            search(&data, 1, &[0.0], k)
                .iter()
                .map(|h| h.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(3), [0, 2, 3]);
        assert_eq!(ids(2), [0, 2]); // id 3 ties id 2 and loses
        assert_eq!(ids(10), [0, 2, 3, 1, 4]); // k > n returns all
        assert!(search(&data, 1, &[0.0], 0).is_empty());
    }

    #[test]
    fn matches_a_full_sort() {
        let dim = 7;
        let data: Vec<f32> = (0..dim * 500).map(|i| ((i * 37) % 101) as f32).collect();
        let q: Vec<f32> = (0..dim).map(|i| i as f32 * 9.0).collect();
        let mut all: Vec<Hit> = data
            .chunks_exact(dim)
            .enumerate()
            .map(|(id, r)| Hit {
                id: id as u32,
                dist: kernels::scalar::l2_sq(&q, r),
            })
            .collect();
        all.sort();
        assert_eq!(search(&data, dim, &q, 25), all[..25]);
    }
}
