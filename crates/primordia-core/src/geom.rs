//! Lattice geometry, face directions, the proper rotation group and the mirror
//! action on all of them.

use serde::{Deserialize, Serialize};

/// Face directions: 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z.
pub const DIRS: [[i32; 3]; 6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

#[inline]
pub const fn opposite(d: u32) -> u32 {
    d ^ 1
}
/// Reflection on axis 0 maps +x <-> -x.
#[inline]
pub const fn mirror_dir(d: u32) -> u32 {
    if d < 2 { d ^ 1 } else { d }
}

fn dir_of(v: [i32; 3]) -> u32 {
    DIRS.iter().position(|&d| d == v).expect("axis vector") as u32
}
fn cross(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub const N_ORIENT: usize = 24;

/// Rotation-group tables. Orientation o = (facing F[o], up U[o]).
#[derive(Clone, Debug)]
pub struct Orient {
    pub f: [u32; N_ORIENT],
    pub u: [u32; N_ORIENT],
    pub left: [u32; N_ORIENT],
    pub right: [u32; N_ORIENT],
    pub yaw_l: [u32; N_ORIENT],
    pub yaw_d: [u32; N_ORIENT],
    pub pitch: [u32; N_ORIENT],
    pub roll_l: [u32; N_ORIENT],
    pub roll_d: [u32; N_ORIENT],
    pub reverse: [u32; N_ORIENT],
    pub mirror: [u32; N_ORIENT],
    /// valid orientations per dimension (0..=3)
    pub valid: [Vec<u32>; 4],
}

impl Orient {
    pub fn new() -> Self {
        let mut pairs = Vec::new();
        for f in 0..6u32 {
            for u in 0..6u32 {
                if f / 2 != u / 2 {
                    pairs.push((f, u));
                }
            }
        }
        assert_eq!(pairs.len(), N_ORIENT);
        let idx = |f: u32, u: u32| pairs.iter().position(|&p| p == (f, u)).unwrap() as u32;
        let mut t = Orient {
            f: [0; N_ORIENT],
            u: [0; N_ORIENT],
            left: [0; N_ORIENT],
            right: [0; N_ORIENT],
            yaw_l: [0; N_ORIENT],
            yaw_d: [0; N_ORIENT],
            pitch: [0; N_ORIENT],
            roll_l: [0; N_ORIENT],
            roll_d: [0; N_ORIENT],
            reverse: [0; N_ORIENT],
            mirror: [0; N_ORIENT],
            valid: [vec![], vec![], vec![], vec![]],
        };
        for (o, &(f, u)) in pairs.iter().enumerate() {
            // left = rotate facing +90 deg about up = up x facing
            let l = dir_of(cross(DIRS[u as usize], DIRS[f as usize]));
            let r = opposite(l);
            t.f[o] = f;
            t.u[o] = u;
            t.left[o] = l;
            t.right[o] = r;
            t.yaw_l[o] = idx(l, u);
            t.yaw_d[o] = idx(r, u);
            // Pitch about the lateral axis is achiral (it involves only facing
            // and up), so PITCH_L and PITCH_D are the same "nose up" rotation.
            t.pitch[o] = idx(u, opposite(f));
            t.roll_l[o] = idx(f, l);
            t.roll_d[o] = idx(f, r);
            t.reverse[o] = idx(opposite(f), u);
            t.mirror[o] = idx(mirror_dir(f), mirror_dir(u));
        }
        t.valid[0] = vec![idx(0, 4), idx(1, 4)];
        t.valid[1] = vec![idx(0, 4), idx(1, 4)];
        t.valid[2] = (0..4).map(|f| idx(f, 4)).collect();
        t.valid[3] = (0..N_ORIENT as u32).collect();
        t
    }
}

impl Default for Orient {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    #[default]
    Torus,
    Reflecting,
    Absorbing,
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub dims: u32,
    pub size: [u32; 3],
    pub boundary: [Boundary; 3],
    pub n: usize,
}

impl Geometry {
    pub fn new(dims: u32, size: &[u32], boundary: &[Boundary]) -> Self {
        let mut s = [1u32; 3];
        let mut b = [Boundary::Torus; 3];
        let axes = if dims == 0 { 1 } else { dims as usize };
        for i in 0..axes {
            s[i] = size[i];
            if dims > 0 {
                b[i] = boundary.get(i).copied().unwrap_or(Boundary::Torus);
            }
        }
        let n = (s[0] as usize) * (s[1] as usize) * (s[2] as usize);
        Geometry { dims, size: s, boundary: b, n }
    }
    #[inline]
    pub fn coords(&self, c: usize) -> [u32; 3] {
        let sx = self.size[0] as usize;
        let sy = self.size[1] as usize;
        [(c % sx) as u32, ((c / sx) % sy) as u32, (c / (sx * sy)) as u32]
    }
    #[inline]
    pub fn index(&self, p: [u32; 3]) -> usize {
        p[0] as usize + self.size[0] as usize * (p[1] as usize + self.size[1] as usize * p[2] as usize)
    }
    pub fn faces(&self) -> u32 {
        2 * self.dims
    }
    /// Raw lattice neighbour (ignores wall cells).
    #[inline]
    pub fn neighbor(&self, c: usize, d: u32) -> Option<usize> {
        if self.dims == 0 || d / 2 >= self.dims {
            return None;
        }
        let axis = (d / 2) as usize;
        let mut p = self.coords(c);
        let s = self.size[axis];
        let v = p[axis] as i64 + DIRS[d as usize][axis] as i64;
        if v < 0 || v >= s as i64 {
            if self.boundary[axis] != Boundary::Torus {
                return None;
            }
            p[axis] = ((v + s as i64) % s as i64) as u32;
        } else {
            p[axis] = v as u32;
        }
        Some(self.index(p))
    }
    /// Mirror image of a cell under x -> N-1-x.
    #[inline]
    pub fn mirror_cell(&self, c: usize) -> usize {
        if self.dims == 0 {
            return c;
        }
        let mut p = self.coords(c);
        p[0] = self.size[0] - 1 - p[0];
        self.index(p)
    }
    #[inline]
    pub fn canon(&self, c: usize, mirrored: bool) -> u32 {
        if mirrored { self.mirror_cell(c) as u32 } else { c as u32 }
    }
    pub fn n_colors(&self) -> u32 {
        3u32.pow(self.dims)
    }
    #[inline]
    pub fn color(&self, c: usize) -> u32 {
        let p = self.coords(c);
        let mut col = 0;
        let mut m = 1;
        for a in 0..self.dims as usize {
            col += (p[a] % 3) * m;
            m *= 3;
        }
        col
    }
    pub fn mirror_color(&self, col: u32) -> u32 {
        if self.dims == 0 {
            return col;
        }
        let cx = col % 3;
        col - cx + (3 - 1 - cx) % 3
    }
    /// Number of cells of one colour along each axis.
    pub fn color_extent(&self) -> [u32; 3] {
        let mut e = [1; 3];
        for a in 0..self.dims as usize {
            e[a] = self.size[a] / 3;
        }
        e
    }
    /// The i-th cell of colour `col`.
    #[inline]
    pub fn color_cell(&self, col: u32, i: usize) -> usize {
        let e = self.color_extent();
        let off = [col % 3, (col / 3) % 3, col / 9];
        let ix = (i % e[0] as usize) as u32;
        let iy = ((i / e[0] as usize) % e[1] as usize) as u32;
        let iz = (i / (e[0] as usize * e[1] as usize)) as u32;
        let mut p = [0u32; 3];
        let ii = [ix, iy, iz];
        for a in 0..3 {
            p[a] = if (a as u32) < self.dims { ii[a] * 3 + off[a] } else { 0 };
        }
        self.index(p)
    }
    pub fn cells_per_color(&self) -> usize {
        let e = self.color_extent();
        (e[0] * e[1] * e[2]) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn group_closure_and_mirror() {
        let o = Orient::new();
        for i in 0..N_ORIENT {
            // four yaws return to start, yaw_l and yaw_d are inverses
            let mut j = i as u32;
            for _ in 0..4 {
                j = o.yaw_l[j as usize];
            }
            assert_eq!(j as usize, i);
            assert_eq!(o.yaw_d[o.yaw_l[i] as usize] as usize, i);
            assert_eq!(o.roll_d[o.roll_l[i] as usize] as usize, i);
            let mut j = i as u32;
            for _ in 0..4 {
                j = o.pitch[j as usize];
            }
            assert_eq!(j as usize, i);
            // mirror is an involution and conjugates L ops into D ops
            let m = o.mirror[i] as usize;
            assert_eq!(o.mirror[m] as usize, i);
            assert_eq!(o.mirror[o.yaw_l[i] as usize], o.yaw_d[m]);
            assert_eq!(o.mirror[o.roll_l[i] as usize], o.roll_d[m]);
            assert_eq!(o.mirror[o.pitch[i] as usize], o.pitch[m]);
            assert_eq!(mirror_dir(o.left[i]), o.right[m]);
            assert_eq!(o.mirror[o.reverse[i] as usize], o.reverse[m]);
        }
        for d in 0..4 {
            for &v in &o.valid[d] {
                assert!(o.valid[d].contains(&o.mirror[v as usize]));
                if d == 2 {
                    assert!(o.valid[2].contains(&o.yaw_l[v as usize]));
                }
            }
        }
    }
    #[test]
    fn colors() {
        let g = Geometry::new(2, &[9, 6], &[]);
        let mut seen = vec![0; g.n];
        for col in 0..g.n_colors() {
            for i in 0..g.cells_per_color() {
                let c = g.color_cell(col, i);
                assert_eq!(g.color(c), col);
                seen[c] += 1;
            }
            for i in 0..g.n {
                if g.color(i) == col {
                    assert_eq!(g.color(g.mirror_cell(i)), g.mirror_color(col));
                }
            }
        }
        assert!(seen.iter().all(|&s| s == 1));
    }
}
