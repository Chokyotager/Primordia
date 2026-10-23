//! CPU golden-model interpreter for PISA-1 (phase 6).
//!
//! `shaders/exec.wgsl` in `primordia-gpu` is a line-by-line port of this file;
//! any semantic change here must be made there too (the parity test enforces it).

use crate::fixed::{mul16_sat, mulshr16, ratio16};
use crate::geom::opposite;
use crate::isa::{self, op};
use crate::params::*;
use crate::rng::{Rng, phase};
use crate::state::State;

/// Lattice neighbour that is not a wall cell.
#[inline]
pub fn nb(st: &State, p: &Params, c: usize, d: u32) -> Option<usize> {
    match p.geom.neighbor(c, d) {
        Some(t) if st.site[t * SITE + S_WALL] == 0 => Some(t),
        _ => None,
    }
}

#[inline]
fn w16(x: i32) -> i32 {
    x as i16 as i32
}

#[derive(Clone, Debug)]
pub struct TraceStep {
    pub ip: u32,
    pub op: u8,
    pub h0: i32,
    pub h1: i32,
    pub stack: Vec<i32>,
    pub energy: u32,
    pub partner: Option<usize>,
    pub partner_first: bool,
    pub len_self: u32,
    pub len_partner: u32,
}

struct Cpu<'a> {
    p: &'a Params,
    snap: &'a [u32],
    rng: Rng,
    canon: u32,
    me: usize,
    mirrored: bool,
    ip: i32,
    h0: i32,
    h1: i32,
    stack: [i32; 16],
    sp: usize,
    draw: u32,
    spent: u32,
    partner_first: bool,
    fixed: Option<Option<usize>>,
    shield: [u8; 7],
}

enum Flow {
    Next(i32),
    End,
}

impl<'a> Cpu<'a> {
    #[inline]
    fn rnd(&mut self) -> u32 {
        let r = self.rng.u32(self.canon, 0, self.draw);
        self.draw += 1;
        r
    }
    #[inline]
    fn push(&mut self, v: i32) -> bool {
        if self.sp >= 16 {
            return false;
        }
        self.stack[self.sp] = w16(v);
        self.sp += 1;
        true
    }
    #[inline]
    fn pop(&mut self) -> Option<i32> {
        if self.sp == 0 {
            return None;
        }
        self.sp -= 1;
        Some(self.stack[self.sp])
    }
    fn facing(&self, st: &State) -> u32 {
        self.p.orient.f[st.orient(self.me) as usize]
    }
    /// Current partner cell (occupied) or None.
    fn partner(&self, st: &State) -> Option<usize> {
        let t = match self.fixed {
            Some(t) => t,
            None => {
                if self.p.geom.dims == 0 {
                    None
                } else {
                    nb(st, self.p, self.me, self.facing(st))
                }
            }
        };
        t.filter(|&t| st.occupied(t))
    }
    fn lens(&self, st: &State) -> (usize, usize, Option<usize>) {
        let pt = self.partner(st);
        (st.len(self.me), pt.map(|t| st.len(t)).unwrap_or(0), pt)
    }
    /// Resolve an address (already reduced mod total) to (cell, offset).
    #[inline]
    fn resolve(&self, a: usize, ls: usize, lp: usize, pt: Option<usize>) -> (usize, usize) {
        if self.partner_first {
            if a < lp { (pt.unwrap(), a) } else { (self.me, a - lp) }
        } else if a < ls {
            (self.me, a)
        } else {
            (pt.unwrap(), a - ls)
        }
    }
    #[inline]
    fn read(&self, st: &State, a: usize, ls: usize, lp: usize, pt: Option<usize>) -> u8 {
        let (c, o) = self.resolve(a, ls, lp, pt);
        st.tapes[c * st.lmax + o]
    }
    #[inline]
    fn pool(&self, st: &State, c: usize, cl: usize) -> u32 {
        st.fields[c * st.nf + self.p.f_pool(cl)]
    }
    #[inline]
    fn pool_add(&self, st: &mut State, c: usize, cl: usize, d: i32) {
        let i = c * st.nf + self.p.f_pool(cl);
        st.fields[i] = (st.fields[i] as i64 + d as i64) as u32;
    }
    fn flip_frame(&self, v: u8, bit: u32) -> u8 {
        if self.mirrored { isa::mirror(isa::mirror(v) ^ (1 << bit)) } else { v ^ (1 << bit) }
    }

    /// Soft geometric shielding for a write into foreign cell `t`.
    fn shield_ok(&mut self, st: &State, t: usize) -> bool {
        let p = self.p;
        if p.sleep_shield && t != self.me && st.h(t, H_SLEEP) > 0 {
            return false;
        }
        if !p.shield_on || t == self.me {
            return true;
        }
        let (key, dir_to_me) = if p.geom.dims == 0 || self.fixed.is_some() {
            (6usize, None)
        } else {
            let f = self.facing(st);
            (f as usize, Some(opposite(f)))
        };
        if !p.shield_byte && self.shield[key] != 0 {
            return self.shield[key] == 1;
        }
        let tb = st.bonds(t);
        let bonded_to_me = dir_to_me.map(|d| tb & (1 << d) != 0).unwrap_or(false);
        let pw = if bonded_to_me {
            u32::MAX
        } else {
            let (mut b, mut o) = (0usize, 0usize);
            for d in 0..p.geom.faces() {
                if Some(d) == dir_to_me {
                    continue;
                }
                if tb & (1 << d) != 0 {
                    b += 1;
                } else if let Some(x) = nb(st, p, t, d) {
                    if self.snap[x] & FL_OCC != 0 {
                        o += 1;
                    }
                }
            }
            p.pwrite[b * 8 + o]
        };
        let ok = if pw == u32::MAX { true } else { self.rnd() < pw };
        self.shield[key] = if ok { 1 } else { 2 };
        ok
    }

    /// Take one monomer of class `cl` for a byte written into cell `t`
    /// (destination pool first, then the executor's).
    fn take_monomer(&self, st: &mut State, t: usize, cl: u8) -> bool {
        if !self.p.matter_on || cl >= 3 {
            return true;
        }
        let cl = cl as usize;
        if self.pool(st, t, cl) > 0 {
            self.pool_add(st, t, cl, -1);
            true
        } else if self.pool(st, self.me, cl) > 0 {
            self.pool_add(st, self.me, cl, -1);
            true
        } else {
            false
        }
    }
    fn release(&self, st: &mut State, t: usize, b: u8) {
        if self.p.matter_on {
            let cl = isa::class(b);
            if cl < 3 {
                self.pool_add(st, t, cl as usize, 1);
            }
        }
    }

    /// Chiral window counts (n_L, n_D) over the w bytes before address `ip`.
    fn window(&self, st: &State, ip: usize, total: usize, ls: usize, lp: usize, pt: Option<usize>) -> (u32, u32) {
        let (mut nl, mut nd) = (0, 0);
        for k in 1..=self.p.fold_w as usize {
            let a = (ip + total * 64 - k) % total;
            let b = self.read(st, a, ls, lp, pt);
            if isa::is_chiral(b) {
                if b & 1 == 0 { nl += 1 } else { nd += 1 }
            }
        }
        (nl, nd)
    }

    /// WR / CPY write with shielding, copy errors, stereo-selectivity and
    /// matter exchange. Returns false when nothing was written.
    #[allow(clippy::too_many_arguments)]
    fn write(&mut self, st: &mut State, addr: usize, mut v: u8, ip: usize, total: usize, ls: usize, lp: usize, pt: Option<usize>) -> bool {
        let p = self.p;
        let (t, off) = self.resolve(addr, ls, lp, pt);
        if !self.shield_ok(st, t) {
            return false;
        }
        // copy errors: 0 none, 1 insert, 2 delete
        let mut mode = 0;
        if p.p_del != 0 {
            let r = self.rnd();
            if r < p.p_sub {
                let bit = self.rnd() % 8;
                v = self.flip_frame(v, bit);
            } else if r < p.p_ins {
                mode = 1;
            } else if r < p.p_del {
                mode = 2;
            }
        }
        let lmax = st.lmax;
        let base = t * lmax;
        let tlen = st.len(t);
        if mode == 2 {
            let old = st.tapes[base + off];
            self.release(st, t, old);
            st.tapes.copy_within(base + off + 1..base + tlen, base + off);
            st.tapes[base + tlen - 1] = 0;
            st.hdr[t * HDR + H_LEN] -= 1;
            return true;
        }
        if p.stereo_copy && isa::is_chiral(v) {
            let (nl, nd) = self.window(st, ip, total, ls, lp, pt);
            let (same, opp) = if nl >= nd { (nl, nd) } else { (nd, nl) };
            let smin = p.sigma_min_q16 as u32;
            let sigma = smin + ((65536 - smin) * (same - opp)) / p.fold_w;
            let h = isa::hand(v) as usize;
            let ms = self.pool(st, t, h).wrapping_add(self.pool(st, self.me, h));
            let mo = self.pool(st, t, 1 - h).wrapping_add(self.pool(st, self.me, 1 - h));
            if ms.wrapping_add(mo) > 0 {
                let pw = mul16_sat(65536 - sigma, ratio16(mo, ms.wrapping_add(mo)));
                if self.rnd() < pw {
                    v = isa::mirror(v);
                }
            }
        }
        let old = st.tapes[base + off];
        self.release(st, t, old);
        if !self.take_monomer(st, t, isa::class(v)) {
            // undo the release
            if p.matter_on && isa::class(old) < 3 {
                self.pool_add(st, t, isa::class(old) as usize, -1);
            }
            return false;
        }
        st.tapes[base + off] = v;
        if mode == 1 && tlen < lmax && self.take_monomer(st, t, isa::class(v)) {
            st.tapes.copy_within(base + off + 1..base + tlen, base + off + 2);
            st.tapes[base + off + 1] = v;
            st.hdr[t * HDR + H_LEN] += 1;
        }
        true
    }

    fn new_child(&self, st: &mut State, t: usize, len: usize, energy: u32, epoch: u64) {
        let p = self.p;
        let me = self.me;
        let lin = Rng::new(p.seed, epoch, phase::LINEAGE).u32(p.geom.canon(t, self.mirrored), 0, 0) | 1;
        let mut fl = FL_OCC | FL_EXEC | FL_NEWBORN | (st.flags(me) & ORIENT_MASK);
        if p.energy_on && energy == 0 {
            fl |= FL_HALT;
        }
        let parent = st.hdr[me * HDR + H_LINEAGE];
        let depth = st.hdr[me * HDR + H_DEPTH] + 1;
        let h = &mut st.hdr[t * HDR..t * HDR + HDR];
        h[H_LEN] = len as u32;
        h[H_FLAGS] = fl;
        h[H_E] = energy;
        h[H_SLEEP] = 0;
        h[H_LINEAGE] = lin;
        h[H_BIRTH] = epoch as u32;
        h[H_PARENT] = parent;
        h[H_DEPTH] = depth;
        h[H_ZERO] = 0;
    }

    /// Swap the individuals at a and b (header row and tape).
    fn swap_cells(st: &mut State, a: usize, b: usize) {
        for w in 0..HDR {
            st.hdr.swap(a * HDR + w, b * HDR + w);
        }
        let l = st.lmax;
        for i in 0..l {
            st.tapes.swap(a * l + i, b * l + i);
        }
    }

    fn set_orient(st: &mut State, c: usize, o: u32) {
        let f = &mut st.hdr[c * HDR + H_FLAGS];
        *f = (*f & !ORIENT_MASK) | (o << ORIENT_SHIFT);
    }

    /// State of the cell in direction d: 0 none/empty, 1 occupied, 2 bonded to me.
    fn sense(&self, st: &State, d: u32) -> i32 {
        match nb(st, self.p, self.me, d) {
            Some(t) if st.occupied(t) => {
                if st.bonds(self.me) & (1 << d) != 0 { 2 } else { 1 }
            }
            _ => 0,
        }
    }

    /// Plan a move in direction d: (cost, vote, swap target). No side effects.
    fn plan_move(&self, st: &State, d: u32) -> (u32, bool, Option<usize>) {
        let p = self.p;
        let base = p.costs[op::MOVE as usize];
        if st.bonds(self.me) != 0 {
            return (base, true, None);
        }
        match nb(st, p, self.me, d) {
            Some(t) if !st.occupied(t) => (base, false, Some(t)),
            Some(t) if st.bonds(t) == 0 => (p.move_occ_cost, false, Some(t)),
            _ => (base, false, None),
        }
    }

    fn turn(&self, st: &mut State, b: u8) {
        let p = self.p;
        let dims = p.geom.dims;
        let o = st.orient(self.me) as usize;
        let ot = &p.orient;
        let n = match b {
            0x20 | 0x21 if dims == 1 => ot.reverse[o],
            0x20 if dims >= 2 => ot.yaw_l[o],
            0x21 if dims >= 2 => ot.yaw_d[o],
            0x22 | 0x23 if dims == 3 => ot.pitch[o],
            0x24 if dims == 3 => ot.roll_l[o],
            0x25 if dims == 3 => ot.roll_d[o],
            _ => o as u32,
        };
        Self::set_orient(st, self.me, n);
    }

    fn pay(&mut self, st: &mut State, cost: u32) -> bool {
        if !self.p.energy_on {
            return true;
        }
        let e = &mut st.hdr[self.me * HDR + H_E];
        if *e < cost {
            self.spent = self.spent.wrapping_add(*e);
            *e = 0;
            st.hdr[self.me * HDR + H_FLAGS] |= FL_HALT;
            return false;
        }
        *e -= cost;
        self.spent = self.spent.wrapping_add(cost);
        true
    }

    fn cat(&mut self, st: &mut State, h: usize, sel: i32, ip: usize, total: usize, ls: usize, lp: usize, pt: Option<usize>) {
        let p = self.p;
        if !p.chem_on || p.cat.is_empty() {
            return;
        }
        let rx = p.reactions[p.cat[sel.rem_euclid(p.cat.len() as i32) as usize] as usize];
        let me = self.me;
        let nf = st.nf;
        let fi = |k: u32, hh: usize| me * nf + p.f_chem(k as usize, hh);
        let mut eta: u32;
        let mut hand = h;
        if p.stereo {
            let (nl, nd) = self.window(st, ip, total, ls, lp, pt);
            let (ns, no) = if h == 0 { (nl, nd) } else { (nd, nl) };
            let ex = ns.saturating_sub(no);
            let emin = p.eta_min_q16 as u32;
            eta = emin + ((65536 - emin) * ex) / p.fold_w;
            let cs = st.fields[fi(rx.r[0], h)];
            let co = st.fields[fi(rx.r[0], 1 - h)];
            let tot = cs.wrapping_add(co);
            if tot > 0 {
                let inh = mulshr16(p.ci_q16 as u32, ratio16(co, tot)).min(65536);
                eta = mulshr16(eta, 65536 - inh);
            }
        } else {
            eta = 65536;
        }
        let dither = self.rnd() & 0xFFFF;
        let mut ev = (p.n_cat.wrapping_mul(eta).wrapping_add(dither)) >> 16;
        let avail = |st: &State, hh: usize| -> u32 {
            let a = st.fields[fi(rx.r[0], hh)];
            if rx.r[1] == NONE {
                a
            } else if rx.r[1] == rx.r[0] {
                a / 2
            } else {
                a.min(st.fields[fi(rx.r[1], hh)])
            }
        };
        let mut cap = avail(st, hand);
        if !p.stereo && cap == 0 {
            hand = 1 - h;
            cap = avail(st, hand);
        }
        ev = ev.min(cap);
        if ev == 0 {
            return;
        }
        for k in rx.r {
            if k != NONE {
                st.fields[fi(k, hand)] -= ev;
            }
        }
        for k in rx.p {
            if k != NONE {
                st.fields[fi(k, hand)] += ev;
            }
        }
        let gain = mulshr16(ev, rx.energy_q16[hand].max(0) as u32);
        st.hdr[me * HDR + H_E] = st.hdr[me * HDR + H_E].wrapping_add(gain);
        st.site[me * SITE + S_RXN] = st.site[me * SITE + S_RXN].wrapping_add(gain);
    }

    /// Execute one instruction. Returns the IP delta or End.
    fn step(&mut self, st: &mut State, epoch: u64, trace: &mut Option<&mut Vec<TraceStep>>) -> Flow {
        let p = self.p;
        let (ls, lp, pt) = self.lens(st);
        let total = ls + lp;
        if total == 0 {
            return Flow::End;
        }
        let ip = self.ip.rem_euclid(total as i32) as usize;
        self.ip = ip as i32;
        let b = self.read(st, ip, ls, lp, pt);
        if let Some(tr) = trace {
            tr.push(TraceStep {
                ip: ip as u32,
                op: b,
                h0: self.h0,
                h1: self.h1,
                stack: self.stack[..self.sp].to_vec(),
                energy: st.h(self.me, H_E),
                partner: pt,
                partner_first: self.partner_first,
                len_self: ls as u32,
                len_partner: lp as u32,
            });
        }
        let me = self.me;
        let fixed_len = p.pair_concat;
        let ti = total as i32;
        let rem = |x: i32| x.rem_euclid(ti) as usize;
        let mut cost = p.costs[b as usize];
        let bonds_on = p.bonds_on;

        macro_rules! pay {
            ($c:expr) => {
                if !self.pay(st, $c) {
                    return Flow::End;
                }
            };
        }
        macro_rules! pop {
            () => {
                match self.pop() {
                    Some(v) => v,
                    None => return Flow::End,
                }
            };
        }
        macro_rules! push {
            ($v:expr) => {
                if !self.push($v) {
                    return Flow::End;
                }
            };
        }

        // Ops with dynamic cost or that need operands before paying.
        match b {
            op::TAKE_L | 0x2F => {
                let n = pop!().clamp(0, 255) as u32;
                cost += n / 4;
                pay!(cost);
                let h = (b & 1) as usize;
                if p.matter_on {
                    if let Some(t) = nb(st, p, me, self.facing(st)).filter(|_| p.geom.dims > 0) {
                        let amt = n.min(self.pool(st, t, h));
                        self.pool_add(st, t, h, -(amt as i32));
                        self.pool_add(st, me, h, amt as i32);
                    }
                }
                return Flow::Next(1);
            }
            op::ALLOC => {
                let n = pop!().clamp(0, st.lmax as i32) as usize;
                if p.pair_concat || p.geom.dims == 0 {
                    pay!(cost);
                    return Flow::Next(1);
                }
                pay!(cost + n as u32);
                if let Some(t) = nb(st, p, me, self.facing(st)) {
                    if !st.occupied(t) {
                        self.new_child(st, t, n, 0, epoch);
                    }
                }
                return Flow::Next(1);
            }
            op::DIVIDE => {
                if p.pair_concat || p.geom.dims == 0 {
                    pay!(cost);
                    return Flow::Next(1);
                }
                let s0 = if self.partner_first { lp } else { 0 } as i32;
                let so = (self.h0 - s0).rem_euclid(ti) as usize;
                let eo = (self.h1 - s0).rem_euclid(ti) as usize;
                let ok = so < eo && eo <= ls;
                let n = if ok { eo - so } else { 0 };
                pay!(cost + n as u32);
                if !ok {
                    return Flow::Next(1);
                }
                if let Some(t) = nb(st, p, me, self.facing(st)) {
                    if !st.occupied(t) {
                        let e = st.h(me, H_E) / 2;
                        *st.hm(me, H_E) -= e;
                        self.new_child(st, t, n, e, epoch);
                        let l = st.lmax;
                        st.tapes.copy_within(me * l + so..me * l + eo, t * l);
                        st.tapes.copy_within(me * l + eo..me * l + ls, me * l + so);
                        for i in ls - n..ls {
                            st.tapes[me * l + i] = 0;
                        }
                        *st.hm(me, H_LEN) -= n as u32;
                    }
                }
                return Flow::Next(1);
            }
            op::DIGEST => {
                let n = pop!().clamp(0, ti) as usize;
                pay!(cost * n as u32);
                if let Some(pc) = pt {
                    for k in 0..n {
                        let a = rem(self.h1 + k as i32);
                        let (c, o) = self.resolve(a, ls, lp, pt);
                        if c != pc {
                            continue;
                        }
                        if !self.shield_ok(st, pc) {
                            break;
                        }
                        let byte = st.tapes[c * st.lmax + o];
                        if byte != 0 {
                            st.tapes[c * st.lmax + o] = 0;
                            self.release(st, me, byte);
                        }
                    }
                }
                return Flow::Next(1);
            }
            op::MOVE | op::STEP_L | 0x2B => {
                if p.geom.dims == 0 || p.pair_concat {
                    pay!(cost);
                    return Flow::Next(1);
                }
                let o = st.orient(me) as usize;
                let d = match b {
                    op::MOVE => p.orient.f[o],
                    op::STEP_L => p.orient.left[o],
                    _ => p.orient.right[o],
                };
                if d / 2 >= p.geom.dims {
                    pay!(cost);
                    return Flow::Next(1);
                }
                let (c2, vote, target) = self.plan_move(st, d);
                pay!(c2);
                if vote {
                    st.site[me * SITE + S_VOTE] = d + 1;
                }
                if let Some(t) = target {
                    Self::swap_cells(st, me, t);
                    return Flow::End;
                }
                return Flow::Next(1);
            }
            _ => {}
        }

        pay!(cost);
        let mut next = 1i32;
        match b {
            op::NOP => {}
            op::DUP => {
                let a = pop!();
                push!(a);
                push!(a);
            }
            op::DROP => {
                pop!();
            }
            op::SWAP => {
                let a = pop!();
                let c = pop!();
                push!(a);
                push!(c);
            }
            op::OVER => {
                let a = pop!();
                let c = pop!();
                push!(c);
                push!(a);
                push!(c);
            }
            op::INC => {
                let a = pop!();
                push!(a + 1);
            }
            op::DEC => {
                let a = pop!();
                push!(a - 1);
            }
            op::ADD => {
                let a = pop!();
                let c = pop!();
                push!(c + a);
            }
            op::SUB => {
                let a = pop!();
                let c = pop!();
                push!(c - a);
            }
            op::SHL4 => {
                let a = pop!();
                push!(a << 4);
            }
            op::NEG => {
                let a = pop!();
                push!(-a);
            }
            op::AND => {
                let a = pop!();
                let c = pop!();
                push!(a & c);
            }
            op::SKIPNZ => {
                let c = pop!();
                if c != 0 {
                    next = 2;
                }
            }
            op::SKIPZ => {
                let c = pop!();
                if c == 0 {
                    next = 2;
                }
            }
            op::JREL => {
                next = pop!();
            }
            op::JNZ => {
                let n = pop!();
                let c = pop!();
                if c != 0 {
                    next = n;
                }
            }
            op::RD => {
                let v = self.read(st, rem(self.h0), ls, lp, pt);
                let v = if self.mirrored { isa::mirror(v) } else { v };
                push!(v as i32);
            }
            op::WR => {
                let v = (pop!() & 0xFF) as u8;
                let v = if self.mirrored { isa::mirror(v) } else { v };
                self.write(st, rem(self.h1), v, ip, total, ls, lp, pt);
            }
            op::CPY => {
                let v = self.read(st, rem(self.h0), ls, lp, pt);
                self.write(st, rem(self.h1), v, ip, total, ls, lp, pt);
            }
            op::H0INC => self.h0 += 1,
            op::H0DEC => self.h0 -= 1,
            op::H1INC => self.h1 += 1,
            op::H1DEC => self.h1 -= 1,
            op::SETH0 => self.h0 = ip as i32 + pop!(),
            op::SETH1 => self.h1 = ip as i32 + pop!(),
            op::GETH0 => push!(self.h0 - ip as i32),
            op::GETH1 => push!(self.h1 - ip as i32),
            op::XH => std::mem::swap(&mut self.h0, &mut self.h1),
            op::LEN => push!(ls as i32),
            op::PLEN => push!(lp as i32),
            op::GROW => {
                if !fixed_len && ls < st.lmax {
                    st.tapes[me * st.lmax + ls] = 0;
                    *st.hm(me, H_LEN) += 1;
                }
            }
            op::SHRINK => {
                if !fixed_len && ls > 0 {
                    let b2 = st.tapes[me * st.lmax + ls - 1];
                    st.tapes[me * st.lmax + ls - 1] = 0;
                    self.release(st, me, b2);
                    *st.hm(me, H_LEN) -= 1;
                }
            }
            0x20..=0x25 => self.turn(st, b),
            0x26 | 0x27 => {
                if bonds_on && p.geom.dims >= 2 {
                    let o = st.orient(me) as usize;
                    let d = if b == 0x26 { p.orient.left[o] } else { p.orient.right[o] };
                    self.bond(st, d, true);
                }
            }
            0x28 | 0x29 => {
                let o = st.orient(me) as usize;
                let d = if b == 0x28 { p.orient.left[o] } else { p.orient.right[o] };
                let s = if p.geom.dims >= 2 { self.sense(st, d) } else { 0 };
                push!(s);
            }
            0x2C | 0x2D => {
                let r = pop!();
                self.cat(st, (b & 1) as usize, r, ip, total, ls, lp, pt);
            }
            0x30 | 0x31 => {
                let j = pop!();
                if p.sig_on && p.s > 0 {
                    let i = me * st.nf + p.f_sig(j.rem_euclid(p.s as i32) as usize, (b & 1) as usize);
                    st.fields[i] = st.fields[i].saturating_add(p.sig_emit);
                }
            }
            0x32 | 0x33 => {
                let j = pop!();
                let v = if p.sig_on && p.s > 0 {
                    st.fields[me * st.nf + p.f_sig(j.rem_euclid(p.s as i32) as usize, (b & 1) as usize)].min(32767) as i32
                } else {
                    0
                };
                push!(v);
            }
            0x34 | 0x35 => {
                let k = pop!();
                let v = if p.chem_on && p.k > 0 {
                    st.fields[me * st.nf + p.f_chem(k.rem_euclid(p.k as i32) as usize, (b & 1) as usize)].min(32767) as i32
                } else {
                    0
                };
                push!(v);
            }
            0x36..=0x3F => {}
            0x40..=0x5F => push!((b - 0x40) as i32),
            op::BOND => {
                if bonds_on && p.geom.dims >= 1 {
                    let d = self.facing(st);
                    self.bond(st, d, true);
                }
            }
            op::UNBOND => {
                if bonds_on && p.geom.dims >= 1 {
                    let d = self.facing(st);
                    self.bond(st, d, false);
                }
            }
            op::SENSE => {
                let s = if p.geom.dims >= 1 { self.sense(st, self.facing(st)) } else { 0 };
                push!(s);
            }
            op::ENRG => push!((st.h(me, H_E) / p.e_scale).min(255) as i32),
            op::TEMP => push!(((st.f(me, F_T) / 100) as i32 - 200).clamp(0, 255)),
            op::SHARE => {
                let n = pop!().max(0) as u32;
                if p.geom.dims >= 1 && !p.pair_concat {
                    let d = self.facing(st);
                    if st.bonds(me) & (1 << d) != 0 {
                        if let Some(t) = nb(st, p, me, d) {
                            let amt = n.min(st.h(me, H_E));
                            *st.hm(me, H_E) -= amt;
                            *st.hm(t, H_E) = st.h(t, H_E).wrapping_add(amt);
                        }
                    }
                }
            }
            op::TUMBLE => {
                let r = self.rnd();
                if p.geom.dims >= 1 {
                    let v = &p.orient.valid[p.geom.dims as usize];
                    let mut o = v[(r as usize) % v.len()];
                    if self.mirrored {
                        o = p.orient.mirror[o as usize];
                    }
                    Self::set_orient(st, me, o);
                }
            }
            op::FREE => push!((st.f(me, F_F) / p.f_scale).min(255) as i32),
            op::POOL => {
                let v = if p.matter_on { self.pool(st, me, 2).min(255) as i32 } else { 0 };
                push!(v);
            }
            op::RAND => {
                let r = self.rnd();
                push!((r & 0xFF) as i32);
            }
            op::SLEEP => {
                let n = pop!();
                if n > 0 && !p.pair_concat {
                    *st.hm(me, H_SLEEP) = n.min(255) as u32 * p.sleep_scale;
                    return Flow::End;
                }
            }
            op::HALT => return Flow::End,
            0x70..=0x7F => {}
            0x80..=0xBF => next = 2 + (b & 0x3F) as i32,
            _ => next = -1 - (b & 0x3F) as i32,
        }
        if p.energy_on && st.h(me, H_E) == 0 {
            st.hdr[me * HDR + H_FLAGS] |= FL_HALT;
            return Flow::End;
        }
        Flow::Next(next)
    }

    fn bond(&self, st: &mut State, d: u32, on: bool) {
        if let Some(t) = nb(st, self.p, self.me, d) {
            if st.occupied(t) {
                let me = self.me;
                let (bm, bt) = (1 << (BOND_SHIFT + d), 1 << (BOND_SHIFT + opposite(d)));
                if on {
                    st.hdr[me * HDR + H_FLAGS] |= bm;
                    st.hdr[t * HDR + H_FLAGS] |= bt;
                } else {
                    st.hdr[me * HDR + H_FLAGS] &= !bm;
                    st.hdr[t * HDR + H_FLAGS] &= !bt;
                }
            }
        }
    }
}

/// Run one execution slice for the individual at `c`.
/// `fixed_partner`: Some(..) in 0D (the paired slot).
pub fn exec_slice(st: &mut State, p: &Params, snap: &[u32], c: usize, fixed_partner: Option<usize>, mut trace: Option<&mut Vec<TraceStep>>) {
    let fl = st.flags(c);
    if fl & FL_OCC == 0 || fl & FL_EXEC != 0 || st.h(c, H_SLEEP) > 0 {
        return;
    }
    if p.energy_on && fl & FL_HALT != 0 {
        return;
    }
    st.hdr[c * HDR + H_FLAGS] |= FL_EXEC;
    let epoch = st.epoch;
    let mirrored = st.mirrored;
    let mut cpu = Cpu {
        p,
        snap,
        rng: Rng::new(p.seed, epoch, phase::EXEC),
        canon: p.geom.canon(c, mirrored),
        me: c,
        mirrored,
        ip: 0,
        h0: 0,
        h1: 0,
        stack: [0; 16],
        sp: 0,
        draw: 0,
        spent: 0,
        partner_first: false,
        fixed: None,
        shield: [0; 7],
    };
    let dims = p.geom.dims;
    if dims >= 1 && p.random_facing {
        let r = cpu.rnd();
        let v = &p.orient.valid[dims as usize];
        let mut o = v[(r as usize) % v.len()];
        if mirrored {
            o = p.orient.mirror[o as usize];
        }
        Cpu::set_orient(st, c, o);
    }
    if p.pair_random {
        cpu.partner_first = cpu.rnd() & 1 == 1;
    }
    if dims == 0 {
        cpu.fixed = Some(fixed_partner);
    } else if p.pair_concat {
        cpu.fixed = Some(nb(st, p, c, cpu.facing(st)));
    }
    {
        let (ls, lp, _) = cpu.lens(st);
        cpu.h1 = if cpu.partner_first { lp as i32 } else { ls as i32 };
    }
    let mut steps = 0;
    while steps < p.budget {
        steps += 1;
        match cpu.step(st, epoch, &mut trace) {
            Flow::Next(d) => cpu.ip += d,
            Flow::End => break,
        }
    }
    if cpu.spent > 0 {
        st.site[c * SITE + S_SPENT] = st.site[c * SITE + S_SPENT].wrapping_add(cpu.spent);
        let heat = mulshr16(cpu.spent, p.heat_q16 as u32);
        let ti = c * st.nf + F_T;
        st.fields[ti] = st.fields[ti].saturating_add(heat);
    }
}

/// Copy of every cell's flags word, taken at the start of each sub-step.
pub fn flag_snapshot(st: &State) -> Vec<u32> {
    (0..st.n).map(|c| st.hdr[c * HDR + H_FLAGS]).collect()
}

/// Order in which the 3^D colour classes run this epoch (already mapped into
/// this world's frame).
pub fn color_order(p: &Params, epoch: u64, mirrored: bool) -> Vec<u32> {
    let nc = p.geom.n_colors();
    let rng = Rng::new(p.seed, epoch, phase::SCHED);
    let mut perm: Vec<u32> = (0..nc).collect();
    for i in (1..nc as usize).rev() {
        let j = (rng.u32(0, 0, i as u32) as usize) % (i + 1);
        perm.swap(i, j);
    }
    if mirrored {
        for c in perm.iter_mut() {
            *c = p.geom.mirror_color(*c);
        }
    }
    perm
}

/// 0D pairing: a random permutation of all slots, consecutive pairs.
pub fn pairing(p: &Params, epoch: u64) -> Vec<u32> {
    let n = p.geom.n;
    let rng = Rng::new(p.seed, epoch, phase::PAIRING);
    let mut perm: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = (rng.u32(0, 0, i as u32) as usize) % (i + 1);
        perm.swap(i, j);
    }
    perm
}

/// Whole execution phase on the CPU.
pub fn exec_phase_cpu(st: &mut State, p: &Params) {
    if p.geom.dims == 0 {
        let pairs = pairing(p, st.epoch);
        let snap = flag_snapshot(st);
        for i in 0..pairs.len() / 2 {
            exec_slice(st, p, &snap, pairs[2 * i] as usize, Some(pairs[2 * i + 1] as usize), None);
        }
        return;
    }
    let order = color_order(p, st.epoch, st.mirrored);
    for col in order {
        let snap = flag_snapshot(st);
        for i in 0..p.geom.cells_per_color() {
            let c = p.geom.color_cell(col, i);
            exec_slice(st, p, &snap, c, None, None);
        }
    }
}
