//! Read-only browser viewer: a tiny HTTP server publishing the latest frame
//! of the run for the three.js page in `web/index.html`. It only answers GET
//! requests for the page, `meta.json`, `frame.bin` and `series.json`; nothing
//! can change the simulation from the browser.

use crate::Note;
use crate::render;
use anyhow::{Context, Result};
use primordia_core::Sim;
use primordia_core::params::*;
use primordia_telemetry::Recorder;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

const PAGE: &str = include_str!("web/index.html");

#[derive(Default)]
pub struct Feed {
    latest: Mutex<Option<(Arc<String>, Arc<Vec<u8>>)>>,
    /// Chart data per time window (see `SERIES_WINDOWS`).
    series: Mutex<std::collections::HashMap<&'static str, Arc<String>>>,
}

/// Time windows the charts can show: (query value, epochs; 0 = whole run).
pub const SERIES_WINDOWS: [(&str, u64); 4] = [("all", 0), ("200k", 200_000), ("50k", 50_000), ("10k", 10_000)];

/// Chart data for one window: every metric as at most `max_pts` points of
/// [epoch, mean, min, max], values rounded to 5 significant digits.
pub fn series_json(bands: &std::collections::BTreeMap<String, Vec<primordia_telemetry::Band>>, epoch: u64, window: u64, max_pts: usize) -> String {
    let r = |x: f64| -> serde_json::Value {
        if !x.is_finite() {
            return serde_json::Value::Null;
        }
        if x == 0.0 {
            return 0.into();
        }
        let m = 10f64.powi(4 - x.abs().log10().floor() as i32);
        serde_json::json!((x * m).round() / m)
    };
    let from = if window == 0 { 0.0 } else { epoch.saturating_sub(window) as f64 };
    let mut out = serde_json::Map::new();
    for (k, v) in bands {
        let v: Vec<_> = v.iter().filter(|b| b.e1 >= from).copied().collect();
        if v.is_empty() {
            continue;
        }
        let pts: Vec<serde_json::Value> = v
            .chunks(v.len().div_ceil(max_pts).max(1))
            .map(primordia_telemetry::Band::merge)
            .map(|b| serde_json::json!([b.mid().round(), r(b.mean()), r(b.lo), r(b.hi)]))
            .collect();
        out.insert(k.clone(), pts.into());
    }
    serde_json::json!({"epoch": epoch, "window": window, "series": out}).to_string()
}

impl Feed {
    pub fn new() -> Arc<Feed> {
        Arc::new(Feed::default())
    }
    /// Publish the current state (the caller must have synced the simulation).
    pub fn publish(&self, sim: &Sim, rec: &Recorder, eps: f64, sun: f32, chronicle: &[Note]) {
        let (m, f) = payload(sim, rec, eps, sun, chronicle);
        self.set(m, f);
    }
    pub fn set(&self, meta: String, frame: Vec<u8>) {
        *self.latest.lock().unwrap() = Some((Arc::new(meta), Arc::new(frame)));
    }
    fn get(&self) -> Option<(Arc<String>, Arc<Vec<u8>>)> {
        self.latest.lock().unwrap().clone()
    }
    /// Publish chart data for every window.
    pub fn set_series(&self, bands: &std::collections::BTreeMap<String, Vec<primordia_telemetry::Band>>, epoch: u64) {
        let m: std::collections::HashMap<_, _> = SERIES_WINDOWS.iter().map(|&(name, w)| (name, Arc::new(series_json(bands, epoch, w, 240)))).collect();
        *self.series.lock().unwrap() = m;
    }
    fn get_series(&self, window: &str) -> Option<Arc<String>> {
        self.series.lock().unwrap().get(window).cloned()
    }
}

/// (meta JSON, frame bytes). Frame: n x (r, g, b, height), then n flag bytes
/// (bit0 occupied, bit1 dormant, bit2 copier, bit3 bonded, bit4 wall), then n
/// colony bytes (index into meta.colonies + 1, 0 = none).
pub fn payload(sim: &Sim, rec: &Recorder, eps: f64, sun: f32, chronicle: &[Note]) -> (String, Vec<u8>) {
    let infos = render::cell_infos(&sim.st);
    let (cols, of) = render::colonies(&sim.st, &sim.p, &infos, &mut render::Tracker::default());
    payload_parts(&sim.st, &sim.p, &sim.backend.name(), &rec.summary, &rec.bands, eps, sun, chronicle, &infos, &cols, &of, &rec.last_clusters)
}

#[allow(clippy::too_many_arguments)]
pub fn payload_parts(
    st: &primordia_core::State,
    p: &primordia_core::Params,
    backend: &str,
    summary: &primordia_telemetry::Summary,
    bands: &std::collections::BTreeMap<String, Vec<primordia_telemetry::Band>>,
    eps: f64,
    sun: f32,
    chronicle: &[Note],
    infos: &[crate::CellInfo],
    colonies: &[render::Colony],
    colony_of: &[u32],
    clusters: &[primordia_telemetry::clusters::Cluster],
) -> (String, Vec<u8>) {
    let g = &p.geom;
    let (cols, h) = render::life_cells(st, p, infos, colony_of, colonies);
    let mut f = Vec::with_capacity(st.n * 5);
    for c in 0..st.n {
        f.extend_from_slice(&cols[c]);
        f.push(h[c]);
    }
    for c in 0..st.n {
        let i = &infos[c];
        let mut b = 0u8;
        if i.occ {
            b |= 1;
        }
        if i.halted {
            b |= 2;
        }
        if i.copier {
            b |= 4;
        }
        if i.bonded {
            b |= 8;
        }
        if st.site[c * SITE + S_WALL] != 0 {
            b |= 16;
        }
        f.push(b);
    }
    for c in 0..st.n {
        f.push(colony_of.get(c).map(|&k| k.min(255) as u8).unwrap_or(0));
    }
    let grid = match g.dims {
        0 => {
            let w = (g.n as f64).sqrt().ceil() as u32;
            [w, (g.n as u32).div_ceil(w), 1]
        }
        _ => g.size,
    };
    let s = summary;
    // At most ~300 points per plot: [epoch, mean, min, max] per slice of history.
    let tail = |k: &str| -> Vec<[f64; 4]> {
        let v = bands.get(k).map(|v| v.as_slice()).unwrap_or_default();
        v.chunks(v.len().div_ceil(300).max(1)).map(primordia_telemetry::Band::merge).map(|b| [b.mid(), b.mean(), b.lo, b.hi]).collect()
    };
    let meta = serde_json::json!({
        "epoch": st.epoch,
        "dims": g.dims,
        "size": g.size,
        "grid": grid,
        "n": st.n,
        "eps": eps,
        "sun": sun,
        "vents": st.vents.iter().map(|v| g.coords(v.pos as usize)).collect::<Vec<_>>(),
        "summary": s,
        "families": s.species,
        "births": st.tally.births,
        "deaths": st.tally.deaths,
        "chronicle": chronicle.iter().map(|n| serde_json::json!({"epoch": n.epoch, "icon": n.icon, "text": n.text})).collect::<Vec<_>>(),
        "series": {
            "population": tail("population"),
            "complexity": tail("high-order entropy"),
            "handedness": tail("EE code"),
            "copiers": tail("replicators (top 100)"),
            "asleep": tail("asleep %"),
            "survival": tail("night survival %"),
        },
        "backend": backend,
        "day_period": p.day_period,
        "clusters": clusters,
        "colonies": colonies.iter().map(|c| serde_json::json!({
            "id": c.id, "name": c.name, "color": c.color, "size": c.size, "center": c.center, "radius": c.radius,
            "copiers": c.copiers, "bonded": c.bonded, "dormant": c.dormant, "hand": c.hand, "energy": c.energy,
            "len": c.mean_len, "growth": c.growth, "since": c.first_seen,
            "program": c.prefix.iter().take(10).map(|&b| primordia_core::isa::mnemonic(b)).collect::<Vec<_>>().join(" "),
        })).collect::<Vec<_>>(),
    });
    (meta.to_string(), f)
}

pub fn serve(addr: &str, feed: Arc<Feed>) -> Result<()> {
    let listener = TcpListener::bind(addr).with_context(|| format!("bind {addr}"))?;
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let feed = feed.clone();
            std::thread::spawn(move || {
                let _ = handle(s, &feed);
            });
        }
    });
    Ok(())
}

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len())?;
    s.write_all(body)
}

fn handle(mut s: TcpStream, feed: &Feed) -> std::io::Result<()> {
    let mut line = String::new();
    BufReader::new(&s).read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let window = query.split('&').find_map(|kv| kv.strip_prefix("w=")).unwrap_or("all");
    if method != "GET" {
        return respond(&mut s, "405 Method Not Allowed", "text/plain", b"read-only viewer");
    }
    match path {
        "/" | "/index.html" => respond(&mut s, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        "/meta.json" => match feed.get() {
            Some((m, _)) => respond(&mut s, "200 OK", "application/json", m.as_bytes()),
            None => respond(&mut s, "503 Service Unavailable", "text/plain", b"warming up"),
        },
        "/series.json" => match feed.get_series(window).or_else(|| feed.get_series("all")) {
            Some(j) => respond(&mut s, "200 OK", "application/json", j.as_bytes()),
            None => respond(&mut s, "503 Service Unavailable", "text/plain", b"warming up"),
        },
        "/frame.bin" => match feed.get() {
            Some((_, f)) => respond(&mut s, "200 OK", "application/octet-stream", &f),
            None => respond(&mut s, "503 Service Unavailable", "text/plain", b"warming up"),
        },
        _ => respond(&mut s, "404 Not Found", "text/plain", b"not found"),
    }
}
