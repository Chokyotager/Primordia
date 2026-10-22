// PISA-1 interpreter, phase 6. Line-by-line port of primordia-core/src/exec.rs.
// The constant block (layout, parameter indices, rotation tables) is generated
// by primordia-core::wgsl::constants() and prepended at pipeline creation.

@group(0) @binding(0) var<storage, read_write> tapes: array<u32>;
@group(0) @binding(1) var<storage, read_write> hdr: array<u32>;
@group(0) @binding(2) var<storage, read_write> fields: array<u32>;
@group(0) @binding(3) var<storage, read_write> site: array<u32>;
@group(0) @binding(4) var<storage, read_write> snap: array<u32>;
@group(0) @binding(5) var<storage, read> nbr: array<u32>;
@group(0) @binding(6) var<storage, read> P: array<u32>;
@group(0) @binding(7) var<storage, read> pairs: array<u32>;

const NONE: u32 = 0xFFFFFFFFu;
const END: i32 = 0x40000000;

// ------------------------------------------------------------- philox

fn mulhilo(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let lo = (p00 & 0xFFFFu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(hi, lo);
}

fn philox(k0: u32, k1: u32, c: vec4<u32>) -> vec4<u32> {
    var x = c;
    var ka = k0;
    var kb = k1;
    for (var i = 0u; i < 10u; i = i + 1u) {
        let m0 = mulhilo(0xD2511F53u, x.x);
        let m1 = mulhilo(0xCD9E8D57u, x.z);
        x = vec4<u32>(m1.x ^ x.y ^ ka, m1.y, m0.x ^ x.w ^ kb, m0.y);
        ka = ka + 0x9E3779B9u;
        kb = kb + 0xBB67AE85u;
    }
    return x;
}

// ------------------------------------------------------------- fixed point

fn ratio16(a_in: u32, b_in: u32) -> u32 {
    var a = a_in;
    var b = b_in;
    if (b == 0u) { return 0u; }
    while (b >= 65536u) {
        a = a >> 1u;
        b = b >> 1u;
    }
    return (a << 16u) / b;
}

fn mulshr16(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    // (a*b) >> 16 mod 2^32
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return (hi << 16u) | (mid & 0xFFFFu);
}

fn mul16_sat(a: u32, b: u32) -> u32 {
    if (a >= 65536u && b >= 65536u) { return 0xFFFFFFFFu; }
    return a * b;
}

fn sat_add(a: u32, b: u32) -> u32 {
    let s = a + b;
    if (s < a) { return 0xFFFFFFFFu; }
    return s;
}

fn w16(x: i32) -> i32 {
    return (x << 16u) >> 16u;
}

fn rem_i(x: i32, t: i32) -> i32 {
    return ((x % t) + t) % t;
}

// ------------------------------------------------------------- isa

fn is_chiral(b: u32) -> bool { return b >= 0x20u && b <= 0x3Fu; }
fn mirror_b(b: u32) -> u32 { if (is_chiral(b)) { return b ^ 1u; } return b; }
fn class_of(b: u32) -> u32 {
    if (b == 0u) { return 3u; }
    if (is_chiral(b)) { return b & 1u; }
    return 2u;
}

// ------------------------------------------------------------- state access

var<private> g_dims: u32;
var<private> g_lmax: u32;
var<private> g_nf: u32;
var<private> g_flags: u32;
var<private> g_mirrored: bool;
var<private> g_epoch: u32;

fn has(f: u32) -> bool { return (g_flags & f) != 0u; }

fn tb(c: u32, i: u32) -> u32 {
    let a = c * g_lmax + i;
    return (tapes[a >> 2u] >> ((a & 3u) * 8u)) & 0xFFu;
}
fn tset(c: u32, i: u32, v: u32) {
    let a = c * g_lmax + i;
    let w = a >> 2u;
    let sh = (a & 3u) * 8u;
    tapes[w] = (tapes[w] & ~(0xFFu << sh)) | ((v & 0xFFu) << sh);
}
fn H(c: u32, w: u32) -> u32 { return hdr[c * HDR + w]; }
fn flags(c: u32) -> u32 { return hdr[c * HDR + H_FLAGS]; }
fn occ(c: u32) -> bool { return (flags(c) & FL_OCC) != 0u; }
fn bonds(c: u32) -> u32 { return (flags(c) & BOND_MASK) >> BOND_SHIFT; }
fn orient(c: u32) -> u32 { return (flags(c) & ORIENT_MASK) >> ORIENT_SHIFT; }
fn tlen(c: u32) -> u32 { return hdr[c * HDR + H_LEN]; }
fn fidx(c: u32, i: u32) -> u32 { return c * g_nf + i; }
fn f_chem(k: u32, h: u32) -> u32 { return 2u + 2u * k + h; }
fn f_pool(cl: u32) -> u32 { return 2u + 2u * P[P_K] + cl; }
fn f_sig(j: u32, h: u32) -> u32 { return 2u + 2u * P[P_K] + 3u + 2u * j + h; }

fn nbw(c: u32, d: u32) -> u32 {
    if (g_dims == 0u || d / 2u >= g_dims) { return NONE; }
    let t = nbr[c * 6u + d];
    if (t == NONE) { return NONE; }
    if (site[t * SITE + S_WALL] != 0u) { return NONE; }
    return t;
}

fn mirror_cell(c: u32) -> u32 {
    if (g_dims == 0u) { return c; }
    let sx = P[P_SX];
    let x = c % sx;
    return (sx - 1u - x) + (c - x);
}
fn canon(c: u32) -> u32 {
    if (g_mirrored) { return mirror_cell(c); }
    return c;
}

// ------------------------------------------------------------- cpu state

var<private> me: u32;
var<private> canon_id: u32;
var<private> ip: i32;
var<private> h0: i32;
var<private> h1: i32;
var<private> stk: array<i32, 16>;
var<private> sp: u32;
var<private> draw: u32;
var<private> spent: u32;
var<private> partner_first: bool;
var<private> fixed_mode: bool;
var<private> fixed_partner: u32;
var<private> shield: array<u32, 7>;
var<private> g_err: bool;
// per-step layout cache
var<private> ls: u32;
var<private> lp: u32;
var<private> pt: u32;
// set by every op that can change a tape length or the faced partner
var<private> dirty: bool;

fn rnd() -> u32 {
    let r = philox(P[P_SEED_LO], P[P_SEED_HI], vec4<u32>(canon_id, g_epoch, (PHASE_EXEC << 24u), draw)).x;
    draw = draw + 1u;
    return r;
}

fn push(v: i32) -> bool {
    if (sp >= 16u) { return false; }
    stk[sp] = w16(v);
    sp = sp + 1u;
    return true;
}
fn pop() -> i32 {
    if (sp == 0u) { g_err = true; return 0; }
    sp = sp - 1u;
    return stk[sp];
}

fn facing() -> u32 { return OR_F[orient(me)]; }

fn partner() -> u32 {
    var t: u32;
    if (fixed_mode) {
        t = fixed_partner;
    } else if (g_dims == 0u) {
        t = NONE;
    } else {
        t = nbw(me, facing());
    }
    if (t != NONE && !occ(t)) { t = NONE; }
    return t;
}

fn lens() {
    pt = partner();
    ls = tlen(me);
    if (pt != NONE) { lp = tlen(pt); } else { lp = 0u; }
}

// returns (cell, offset)
fn resolve(a: u32) -> vec2<u32> {
    if (partner_first) {
        if (a < lp) { return vec2<u32>(pt, a); }
        return vec2<u32>(me, a - lp);
    }
    if (a < ls) { return vec2<u32>(me, a); }
    return vec2<u32>(pt, a - ls);
}
fn rd(a: u32) -> u32 {
    let r = resolve(a);
    return tb(r.x, r.y);
}

fn pool(c: u32, cl: u32) -> u32 { return fields[fidx(c, f_pool(cl))]; }
fn pool_add(c: u32, cl: u32, d: i32) {
    let i = fidx(c, f_pool(cl));
    fields[i] = u32(i32(fields[i]) + d);
}

fn flip_frame(v: u32, bit: u32) -> u32 {
    if (g_mirrored) { return mirror_b(mirror_b(v) ^ (1u << bit)); }
    return v ^ (1u << bit);
}

fn shield_ok(t: u32) -> bool {
    if (!has(F_SHIELD) || t == me) { return true; }
    var key = 6u;
    var dir_to_me = NONE;
    if (!(g_dims == 0u || fixed_mode)) {
        let f = facing();
        key = f;
        dir_to_me = f ^ 1u;
    }
    if (!has(F_SHIELD_BYTE) && shield[key] != 0u) {
        return shield[key] == 1u;
    }
    let tbm = bonds(t);
    var bonded_to_me = false;
    if (dir_to_me != NONE) { bonded_to_me = (tbm & (1u << dir_to_me)) != 0u; }
    var pw = 0xFFFFFFFFu;
    if (!bonded_to_me) {
        var b = 0u;
        var o = 0u;
        for (var d = 0u; d < 2u * g_dims; d = d + 1u) {
            if (d == dir_to_me) { continue; }
            if ((tbm & (1u << d)) != 0u) {
                b = b + 1u;
            } else {
                let x = nbw(t, d);
                if (x != NONE) {
                    if ((snap[x] & FL_OCC) != 0u) { o = o + 1u; }
                }
            }
        }
        pw = P[P_PWRITE + b * 8u + o];
    }
    var ok = true;
    if (pw != 0xFFFFFFFFu) { ok = rnd() < pw; }
    if (ok) { shield[key] = 1u; } else { shield[key] = 2u; }
    return ok;
}

fn take_monomer(t: u32, cl: u32) -> bool {
    if (!has(F_MATTER) || cl >= 3u) { return true; }
    if (pool(t, cl) > 0u) {
        pool_add(t, cl, -1);
        return true;
    }
    if (pool(me, cl) > 0u) {
        pool_add(me, cl, -1);
        return true;
    }
    return false;
}
fn release(t: u32, b: u32) {
    if (has(F_MATTER)) {
        let cl = class_of(b);
        if (cl < 3u) { pool_add(t, cl, 1); }
    }
}

// (n_L, n_D) over the w bytes before address ipu
fn window(ipu: u32, total: u32) -> vec2<u32> {
    var nl = 0u;
    var nd = 0u;
    for (var k = 1u; k <= P[P_FOLD_W]; k = k + 1u) {
        let a = (ipu + total * 64u - k) % total;
        let b = rd(a);
        if (is_chiral(b)) {
            if ((b & 1u) == 0u) { nl = nl + 1u; } else { nd = nd + 1u; }
        }
    }
    return vec2<u32>(nl, nd);
}

fn write_byte(addr: u32, v_in: u32, ipu: u32, total: u32) -> bool {
    var v = v_in;
    let r0 = resolve(addr);
    let t = r0.x;
    let off = r0.y;
    if (!shield_ok(t)) { return false; }
    var mode = 0u;
    if (P[P_PDEL] != 0u) {
        let r = rnd();
        if (r < P[P_PSUB]) {
            let bit = rnd() % 8u;
            v = flip_frame(v, bit);
        } else if (r < P[P_PINS]) {
            mode = 1u;
        } else if (r < P[P_PDEL]) {
            mode = 2u;
        }
    }
    let tl = tlen(t);
    if (mode != 0u) { dirty = true; }
    if (mode == 2u) {
        let old = tb(t, off);
        release(t, old);
        for (var i = off; i + 1u < tl; i = i + 1u) {
            tset(t, i, tb(t, i + 1u));
        }
        tset(t, tl - 1u, 0u);
        hdr[t * HDR + H_LEN] = tl - 1u;
        return true;
    }
    if (has(F_STEREO_COPY) && is_chiral(v)) {
        let wn = window(ipu, total);
        var same = wn.x;
        var opp = wn.y;
        if (wn.x < wn.y) { same = wn.y; opp = wn.x; }
        let smin = P[P_SIGMA_MIN];
        let sigma = smin + ((65536u - smin) * (same - opp)) / P[P_FOLD_W];
        let h = v & 1u;
        let ms = pool(t, h) + pool(me, h);
        let mo = pool(t, 1u - h) + pool(me, 1u - h);
        if (ms + mo > 0u) {
            let pw = mul16_sat(65536u - sigma, ratio16(mo, ms + mo));
            if (rnd() < pw) { v = mirror_b(v); }
        }
    }
    let old = tb(t, off);
    release(t, old);
    if (!take_monomer(t, class_of(v))) {
        if (has(F_MATTER) && class_of(old) < 3u) { pool_add(t, class_of(old), -1); }
        return false;
    }
    tset(t, off, v);
    if (mode == 1u && tl < g_lmax) {
        if (take_monomer(t, class_of(v))) {
            var i = tl;
            loop {
                if (i <= off + 1u) { break; }
                tset(t, i, tb(t, i - 1u));
                i = i - 1u;
            }
            tset(t, off + 1u, v);
            hdr[t * HDR + H_LEN] = tl + 1u;
        }
    }
    return true;
}

fn new_child(t: u32, len: u32, energy: u32) {
    let lin = philox(P[P_SEED_LO], P[P_SEED_HI], vec4<u32>(canon(t), g_epoch, (PHASE_LINEAGE << 24u), 0u)).x | 1u;
    var fl = FL_OCC | FL_EXEC | FL_NEWBORN | (flags(me) & ORIENT_MASK);
    if (has(F_ENERGY) && energy == 0u) { fl = fl | FL_HALT; }
    let parent = H(me, H_LINEAGE);
    let depth = H(me, H_DEPTH) + 1u;
    let b = t * HDR;
    hdr[b + H_LEN] = len;
    hdr[b + H_FLAGS] = fl;
    hdr[b + H_E] = energy;
    hdr[b + H_SLEEP] = 0u;
    hdr[b + H_LINEAGE] = lin;
    hdr[b + H_BIRTH] = g_epoch;
    hdr[b + H_PARENT] = parent;
    hdr[b + H_DEPTH] = depth;
    hdr[b + H_ZERO] = 0u;
}

fn swap_cells(a: u32, b: u32) {
    for (var w = 0u; w < HDR; w = w + 1u) {
        let x = hdr[a * HDR + w];
        hdr[a * HDR + w] = hdr[b * HDR + w];
        hdr[b * HDR + w] = x;
    }
    let nw = g_lmax / 4u;
    for (var i = 0u; i < nw; i = i + 1u) {
        let x = tapes[a * nw + i];
        tapes[a * nw + i] = tapes[b * nw + i];
        tapes[b * nw + i] = x;
    }
}

fn set_orient(c: u32, o: u32) {
    let i = c * HDR + H_FLAGS;
    hdr[i] = (hdr[i] & ~ORIENT_MASK) | (o << ORIENT_SHIFT);
}

fn sense(d: u32) -> i32 {
    let t = nbw(me, d);
    if (t != NONE && occ(t)) {
        if ((bonds(me) & (1u << d)) != 0u) { return 2; }
        return 1;
    }
    return 0;
}

fn pay(cost: u32) -> bool {
    if (!has(F_ENERGY)) { return true; }
    let i = me * HDR + H_E;
    let e = hdr[i];
    if (e < cost) {
        spent = spent + e;
        hdr[i] = 0u;
        hdr[me * HDR + H_FLAGS] = hdr[me * HDR + H_FLAGS] | FL_HALT;
        return false;
    }
    hdr[i] = e - cost;
    spent = spent + cost;
    return true;
}

fn turn(b: u32) {
    let o = orient(me);
    var n = o;
    if ((b == 0x20u || b == 0x21u) && g_dims == 1u) {
        n = OR_REVERSE[o];
    } else if (b == 0x20u && g_dims >= 2u) {
        n = OR_YAW_L[o];
    } else if (b == 0x21u && g_dims >= 2u) {
        n = OR_YAW_D[o];
    } else if ((b == 0x22u || b == 0x23u) && g_dims == 3u) {
        n = OR_PITCH[o];
    } else if (b == 0x24u && g_dims == 3u) {
        n = OR_ROLL_L[o];
    } else if (b == 0x25u && g_dims == 3u) {
        n = OR_ROLL_D[o];
    }
    set_orient(me, n);
}

fn cat_avail(r0: u32, r1: u32, hh: u32) -> u32 {
    let a = fields[fidx(me, f_chem(r0, hh))];
    if (r1 == NONE) { return a; }
    if (r1 == r0) { return a / 2u; }
    return min(a, fields[fidx(me, f_chem(r1, hh))]);
}

fn cat(h: u32, sel: i32, ipu: u32, total: u32) {
    let ncat = P[P_NCAT];
    if (!has(F_CHEM) || ncat == 0u) { return; }
    let rb = P_CAT + u32(rem_i(sel, i32(ncat))) * CAT_STRIDE;
    let r0 = P[rb];
    let r1 = P[rb + 1u];
    let p0 = P[rb + 2u];
    let p1 = P[rb + 3u];
    var eta: u32;
    var hand = h;
    if (has(F_STEREO)) {
        let wn = window(ipu, total);
        var ns = wn.x;
        var no = wn.y;
        if (h == 1u) { ns = wn.y; no = wn.x; }
        var ex = 0u;
        if (ns > no) { ex = ns - no; }
        let emin = P[P_ETA_MIN];
        eta = emin + ((65536u - emin) * ex) / P[P_FOLD_W];
        let cs = fields[fidx(me, f_chem(r0, h))];
        let co = fields[fidx(me, f_chem(r0, 1u - h))];
        let tot = cs + co;
        if (tot > 0u) {
            let inh = min(mulshr16(P[P_CI], ratio16(co, tot)), 65536u);
            eta = mulshr16(eta, 65536u - inh);
        }
    } else {
        eta = 65536u;
    }
    let dither = rnd() & 0xFFFFu;
    var ev = (P[P_NCAT_EV] * eta + dither) >> 16u;
    var cap = cat_avail(r0, r1, hand);
    if (!has(F_STEREO) && cap == 0u) {
        hand = 1u - h;
        cap = cat_avail(r0, r1, hand);
    }
    ev = min(ev, cap);
    if (ev == 0u) { return; }
    fields[fidx(me, f_chem(r0, hand))] -= ev;
    if (r1 != NONE) { fields[fidx(me, f_chem(r1, hand))] -= ev; }
    fields[fidx(me, f_chem(p0, hand))] += ev;
    if (p1 != NONE) { fields[fidx(me, f_chem(p1, hand))] += ev; }
    let gain = mulshr16(ev, P[rb + 4u + hand]);
    hdr[me * HDR + H_E] = hdr[me * HDR + H_E] + gain;
    site[me * SITE + S_RXN] = site[me * SITE + S_RXN] + gain;
}

// Execute one instruction; returns the IP delta or END.
fn step() -> i32 {
    if (dirty) {
        lens();
        dirty = false;
    }
    let total = ls + lp;
    if (total == 0u) { return END; }
    let ti = i32(total);
    let ipu = u32(rem_i(ip, ti));
    ip = i32(ipu);
    let b = rd(ipu);
    let fixed_len = has(F_PAIR_CONCAT);
    var cost = P[P_COSTS + b];
    let bonds_on = has(F_BONDS);
    g_err = false;

    // ---- ops with dynamic cost / operands before paying
    if (b == 0x2Eu || b == 0x2Fu) { // TAKE
        let n = u32(clamp(pop(), 0, 255));
        if (g_err) { return END; }
        cost = cost + n / 4u;
        if (!pay(cost)) { return END; }
        let h = b & 1u;
        if (has(F_MATTER) && g_dims > 0u) {
            let t = nbw(me, facing());
            if (t != NONE) {
                let amt = min(n, pool(t, h));
                pool_add(t, h, -i32(amt));
                pool_add(me, h, i32(amt));
            }
        }
        return 1;
    }
    if (b == 0x67u) { // ALLOC
        let n = u32(clamp(pop(), 0, i32(g_lmax)));
        if (g_err) { return END; }
        if (fixed_len || g_dims == 0u) {
            if (!pay(cost)) { return END; }
            return 1;
        }
        if (!pay(cost + n)) { return END; }
        let t = nbw(me, facing());
        if (t != NONE) {
            if (!occ(t)) { new_child(t, n, 0u); dirty = true; }
        }
        return 1;
    }
    if (b == 0x68u) { // DIVIDE
        if (fixed_len || g_dims == 0u) {
            if (!pay(cost)) { return END; }
            return 1;
        }
        var s0 = 0;
        if (partner_first) { s0 = i32(lp); }
        let so = u32(rem_i(h0 - s0, ti));
        let eo = u32(rem_i(h1 - s0, ti));
        let ok = so < eo && eo <= ls;
        var n = 0u;
        if (ok) { n = eo - so; }
        if (!pay(cost + n)) { return END; }
        if (!ok) { return 1; }
        let t = nbw(me, facing());
        if (t != NONE) {
            if (!occ(t)) {
                let e = H(me, H_E) / 2u;
                hdr[me * HDR + H_E] = H(me, H_E) - e;
                new_child(t, n, e);
                dirty = true;
                for (var i = 0u; i < n; i = i + 1u) {
                    tset(t, i, tb(me, so + i));
                }
                for (var i = eo; i < ls; i = i + 1u) {
                    tset(me, i - n, tb(me, i));
                }
                for (var i = ls - n; i < ls; i = i + 1u) {
                    tset(me, i, 0u);
                }
                hdr[me * HDR + H_LEN] = ls - n;
            }
        }
        return 1;
    }
    if (b == 0x69u) { // DIGEST
        let n = u32(clamp(pop(), 0, ti));
        if (g_err) { return END; }
        if (!pay(cost * n)) { return END; }
        if (pt != NONE) {
            let pc = pt;
            for (var k = 0u; k < n; k = k + 1u) {
                let a = u32(rem_i(h1 + i32(k), ti));
                let r = resolve(a);
                if (r.x != pc) { continue; }
                if (!shield_ok(pc)) { break; }
                let byte = tb(r.x, r.y);
                if (byte != 0u) {
                    tset(r.x, r.y, 0u);
                    release(me, byte);
                }
            }
        }
        return 1;
    }
    if (b == 0x60u || b == 0x2Au || b == 0x2Bu) { // MOVE / STEP
        if (g_dims == 0u || fixed_len) {
            if (!pay(cost)) { return END; }
            return 1;
        }
        let o = orient(me);
        var d = OR_F[o];
        if (b == 0x2Au) { d = OR_LEFT[o]; }
        if (b == 0x2Bu) { d = OR_RIGHT[o]; }
        if (d / 2u >= g_dims) {
            if (!pay(cost)) { return END; }
            return 1;
        }
        let base = P[P_COSTS + 0x60u];
        var c2 = base;
        var vote = false;
        var tgt = NONE;
        if (bonds(me) != 0u) {
            vote = true;
        } else {
            let t = nbw(me, d);
            if (t != NONE) {
                if (!occ(t)) {
                    tgt = t;
                } else if (bonds(t) == 0u) {
                    c2 = P[P_MOVE_OCC];
                    tgt = t;
                }
            }
        }
        if (!pay(c2)) { return END; }
        if (vote) { site[me * SITE + S_VOTE] = d + 1u; }
        if (tgt != NONE) {
            swap_cells(me, tgt);
            return END;
        }
        return 1;
    }

    if (!pay(cost)) { return END; }
    var next = 1;
    if (b == 0x00u) {
    } else if (b == 0x01u) { // DUP
        let a = pop(); if (g_err) { return END; }
        if (!push(a)) { return END; }
        if (!push(a)) { return END; }
    } else if (b == 0x02u) {
        let a = pop(); if (g_err) { return END; }
    } else if (b == 0x03u) { // SWAP
        let a = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (!push(a)) { return END; }
        if (!push(c)) { return END; }
    } else if (b == 0x04u) { // OVER
        let a = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (!push(c)) { return END; }
        if (!push(a)) { return END; }
        if (!push(c)) { return END; }
    } else if (b == 0x05u) {
        let a = pop(); if (g_err) { return END; }
        if (!push(a + 1)) { return END; }
    } else if (b == 0x06u) {
        let a = pop(); if (g_err) { return END; }
        if (!push(a - 1)) { return END; }
    } else if (b == 0x07u) {
        let a = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (!push(c + a)) { return END; }
    } else if (b == 0x08u) {
        let a = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (!push(c - a)) { return END; }
    } else if (b == 0x09u) {
        let a = pop(); if (g_err) { return END; }
        if (!push(a << 4u)) { return END; }
    } else if (b == 0x0Au) {
        let a = pop(); if (g_err) { return END; }
        if (!push(-a)) { return END; }
    } else if (b == 0x0Bu) {
        let a = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (!push(a & c)) { return END; }
    } else if (b == 0x0Cu) {
        let c = pop(); if (g_err) { return END; }
        if (c != 0) { next = 2; }
    } else if (b == 0x0Du) {
        let c = pop(); if (g_err) { return END; }
        if (c == 0) { next = 2; }
    } else if (b == 0x0Eu) {
        let n = pop(); if (g_err) { return END; }
        next = n;
    } else if (b == 0x0Fu) {
        let n = pop(); if (g_err) { return END; }
        let c = pop(); if (g_err) { return END; }
        if (c != 0) { next = n; }
    } else if (b == 0x10u) { // RD
        var v = rd(u32(rem_i(h0, ti)));
        if (g_mirrored) { v = mirror_b(v); }
        if (!push(i32(v))) { return END; }
    } else if (b == 0x11u) { // WR
        let x = pop(); if (g_err) { return END; }
        var v = u32(x & 0xFF);
        if (g_mirrored) { v = mirror_b(v); }
        write_byte(u32(rem_i(h1, ti)), v, ipu, total);
    } else if (b == 0x12u) { // CPY
        let v = rd(u32(rem_i(h0, ti)));
        write_byte(u32(rem_i(h1, ti)), v, ipu, total);
    } else if (b == 0x13u) { h0 = h0 + 1;
    } else if (b == 0x14u) { h0 = h0 - 1;
    } else if (b == 0x15u) { h1 = h1 + 1;
    } else if (b == 0x16u) { h1 = h1 - 1;
    } else if (b == 0x17u) {
        let v = pop(); if (g_err) { return END; }
        h0 = i32(ipu) + v;
    } else if (b == 0x18u) {
        let v = pop(); if (g_err) { return END; }
        h1 = i32(ipu) + v;
    } else if (b == 0x19u) {
        if (!push(h0 - i32(ipu))) { return END; }
    } else if (b == 0x1Au) {
        if (!push(h1 - i32(ipu))) { return END; }
    } else if (b == 0x1Bu) {
        let x = h0; h0 = h1; h1 = x;
    } else if (b == 0x1Cu) {
        if (!push(i32(ls))) { return END; }
    } else if (b == 0x1Fu) {
        if (!push(i32(lp))) { return END; }
    } else if (b == 0x1Du) { // GROW
        dirty = true;
        if (!fixed_len && ls < g_lmax) {
            tset(me, ls, 0u);
            hdr[me * HDR + H_LEN] = ls + 1u;
        }
    } else if (b == 0x1Eu) { // SHRINK
        dirty = true;
        if (!fixed_len && ls > 0u) {
            let b2 = tb(me, ls - 1u);
            tset(me, ls - 1u, 0u);
            release(me, b2);
            hdr[me * HDR + H_LEN] = ls - 1u;
        }
    } else if (b >= 0x20u && b <= 0x25u) {
        turn(b);
        dirty = true;
    } else if (b == 0x26u || b == 0x27u) {
        if (bonds_on && g_dims >= 2u) {
            let o = orient(me);
            var d = OR_LEFT[o];
            if (b == 0x27u) { d = OR_RIGHT[o]; }
            bond(d, true);
        }
    } else if (b == 0x28u || b == 0x29u) {
        let o = orient(me);
        var d = OR_LEFT[o];
        if (b == 0x29u) { d = OR_RIGHT[o]; }
        var s = 0;
        if (g_dims >= 2u) { s = sense(d); }
        if (!push(s)) { return END; }
    } else if (b == 0x2Cu || b == 0x2Du) {
        let r = pop(); if (g_err) { return END; }
        cat(b & 1u, r, ipu, total);
    } else if (b == 0x30u || b == 0x31u) {
        let j = pop(); if (g_err) { return END; }
        let S = P[P_S];
        if (has(F_SIG) && S > 0u) {
            let i = fidx(me, f_sig(u32(rem_i(j, i32(S))), b & 1u));
            fields[i] = sat_add(fields[i], P[P_SIG_EMIT]);
        }
    } else if (b == 0x32u || b == 0x33u) {
        let j = pop(); if (g_err) { return END; }
        let S = P[P_S];
        var v = 0;
        if (has(F_SIG) && S > 0u) {
            v = i32(min(fields[fidx(me, f_sig(u32(rem_i(j, i32(S))), b & 1u))], 32767u));
        }
        if (!push(v)) { return END; }
    } else if (b == 0x34u || b == 0x35u) {
        let k = pop(); if (g_err) { return END; }
        let K = P[P_K];
        var v = 0;
        if (has(F_CHEM) && K > 0u) {
            v = i32(min(fields[fidx(me, f_chem(u32(rem_i(k, i32(K))), b & 1u))], 32767u));
        }
        if (!push(v)) { return END; }
    } else if (b >= 0x36u && b <= 0x3Fu) {
    } else if (b >= 0x40u && b <= 0x5Fu) {
        if (!push(i32(b - 0x40u))) { return END; }
    } else if (b == 0x61u) {
        if (bonds_on && g_dims >= 1u) { bond(facing(), true); }
    } else if (b == 0x62u) {
        if (bonds_on && g_dims >= 1u) { bond(facing(), false); }
    } else if (b == 0x63u) {
        var s = 0;
        if (g_dims >= 1u) { s = sense(facing()); }
        if (!push(s)) { return END; }
    } else if (b == 0x64u) {
        if (!push(i32(min(H(me, H_E) / P[P_E_SCALE], 255u)))) { return END; }
    } else if (b == 0x65u) {
        if (!push(clamp(i32(fields[fidx(me, 1u)] / 100u) - 200, 0, 255))) { return END; }
    } else if (b == 0x66u) { // SHARE
        let x = pop(); if (g_err) { return END; }
        let n = u32(max(x, 0));
        if (g_dims >= 1u && !fixed_len) {
            let d = facing();
            if ((bonds(me) & (1u << d)) != 0u) {
                let t = nbw(me, d);
                if (t != NONE) {
                    let amt = min(n, H(me, H_E));
                    hdr[me * HDR + H_E] = H(me, H_E) - amt;
                    hdr[t * HDR + H_E] = H(t, H_E) + amt;
                }
            }
        }
    } else if (b == 0x6Au) { // TUMBLE
        dirty = true;
        let r = rnd();
        if (g_dims >= 1u) {
            var o = OR_VALID[g_dims * 24u + r % OR_NVALID[g_dims]];
            if (g_mirrored) { o = OR_MIRROR[o]; }
            set_orient(me, o);
        }
    } else if (b == 0x6Bu) {
        if (!push(i32(min(fields[fidx(me, 0u)] / P[P_F_SCALE], 255u)))) { return END; }
    } else if (b == 0x6Cu) {
        var v = 0;
        if (has(F_MATTER)) { v = i32(min(pool(me, 2u), 255u)); }
        if (!push(v)) { return END; }
    } else if (b == 0x6Du) {
        let r = rnd();
        if (!push(i32(r & 0xFFu))) { return END; }
    } else if (b == 0x6Eu) { // SLEEP
        let n = pop(); if (g_err) { return END; }
        if (n > 0 && !fixed_len) {
            hdr[me * HDR + H_SLEEP] = u32(min(n, 255));
            return END;
        }
    } else if (b == 0x6Fu) {
        return END;
    } else if (b >= 0x70u && b <= 0x7Fu) {
    } else if (b >= 0x80u && b <= 0xBFu) {
        next = 2 + i32(b & 0x3Fu);
    } else {
        next = -1 - i32(b & 0x3Fu);
    }
    if (has(F_ENERGY) && H(me, H_E) == 0u) {
        hdr[me * HDR + H_FLAGS] = hdr[me * HDR + H_FLAGS] | FL_HALT;
        return END;
    }
    return next;
}

fn bond(d: u32, on: bool) {
    let t = nbw(me, d);
    if (t != NONE) {
        if (occ(t)) {
            let bm = 1u << (BOND_SHIFT + d);
            let bt = 1u << (BOND_SHIFT + (d ^ 1u));
            if (on) {
                hdr[me * HDR + H_FLAGS] = hdr[me * HDR + H_FLAGS] | bm;
                hdr[t * HDR + H_FLAGS] = hdr[t * HDR + H_FLAGS] | bt;
            } else {
                hdr[me * HDR + H_FLAGS] = hdr[me * HDR + H_FLAGS] & ~bm;
                hdr[t * HDR + H_FLAGS] = hdr[t * HDR + H_FLAGS] & ~bt;
            }
        }
    }
}

fn exec_slice(c: u32, fp: u32) {
    let fl = flags(c);
    if ((fl & FL_OCC) == 0u || (fl & FL_EXEC) != 0u || H(c, H_SLEEP) > 0u) { return; }
    if (has(F_ENERGY) && (fl & FL_HALT) != 0u) { return; }
    hdr[c * HDR + H_FLAGS] = fl | FL_EXEC;
    me = c;
    canon_id = canon(c);
    ip = 0;
    h0 = 0;
    h1 = 0;
    sp = 0u;
    draw = 0u;
    spent = 0u;
    partner_first = false;
    fixed_mode = false;
    fixed_partner = NONE;
    for (var i = 0u; i < 7u; i = i + 1u) { shield[i] = 0u; }
    if (g_dims >= 1u && has(F_RANDOM_FACING)) {
        let r = rnd();
        var o = OR_VALID[g_dims * 24u + r % OR_NVALID[g_dims]];
        if (g_mirrored) { o = OR_MIRROR[o]; }
        set_orient(c, o);
    }
    if (has(F_PAIR_RANDOM)) {
        partner_first = (rnd() & 1u) == 1u;
    }
    if (g_dims == 0u) {
        fixed_mode = true;
        fixed_partner = fp;
    } else if (has(F_PAIR_CONCAT)) {
        fixed_mode = true;
        fixed_partner = nbw(c, facing());
    }
    lens();
    dirty = false;
    if (partner_first) { h1 = i32(lp); } else { h1 = i32(ls); }
    var steps = 0u;
    let budget = P[P_BUDGET];
    loop {
        if (steps >= budget) { break; }
        steps = steps + 1u;
        let d = step();
        if (d == END) { break; }
        ip = ip + d;
    }
    if (spent > 0u) {
        site[c * SITE + S_SPENT] = site[c * SITE + S_SPENT] + spent;
        let heat = mulshr16(spent, P[P_HEAT]);
        let ti = fidx(c, 1u);
        fields[ti] = sat_add(fields[ti], heat);
    }
}

fn load_globals() {
    g_dims = P[P_DIMS];
    g_lmax = P[P_LMAX];
    g_nf = P[P_NF];
    g_flags = P[P_FLAGS];
    g_mirrored = P[P_MIRRORED] != 0u;
    g_epoch = P[P_EPOCH];
}

@compute @workgroup_size(64)
fn snap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * 65535u * 64u;
    if (i < P[P_N]) {
        snap[i] = hdr[i * HDR + H_FLAGS];
    }
}

@compute @workgroup_size(64)
fn exec_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    load_globals();
    let i = gid.x + gid.y * 65535u * 64u;
    if (g_dims == 0u) {
        if (i < P[P_NPAIRS]) {
            exec_slice(pairs[2u * i], pairs[2u * i + 1u]);
        }
        return;
    }
    if (i >= P[P_NCOLOR_CELLS]) { return; }
    let col = P[P_COLOR];
    let ex = P[P_EXT_X];
    let ey = P[P_EXT_Y];
    let ix = i % ex;
    let iy = (i / ex) % ey;
    let iz = i / (ex * ey);
    var px = ix * 3u + col % 3u;
    var py = 0u;
    var pz = 0u;
    if (g_dims >= 2u) { py = iy * 3u + (col / 3u) % 3u; }
    if (g_dims >= 3u) { pz = iz * 3u + col / 9u; }
    let c = px + P[P_SX] * (py + P[P_SY] * pz);
    exec_slice(c, NONE);
}
