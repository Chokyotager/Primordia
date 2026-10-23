//! Read-only browser viewer: a tiny HTTP server publishing the latest frame
//! of the run for the three.js page in `web/index.html`. It only answers GET
//! requests for the page, `meta.json` and `frame.bin`; nothing can change
//! the simulation from the browser.

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
}

/// (meta JSON, frame bytes). Frame: n x (r, g, b, height) followed by n flag
/// bytes (bit0 occupied, bit1 dormant, bit2 copier, bit3 bonded, bit4 wall).
pub fn payload(sim: &Sim, rec: &Recorder, eps: f64, sun: f32, chronicle: &[Note]) -> (String, Vec<u8>) {
    let infos = render::cell_infos(&sim.st);
    let (cols, of) = render::colonies(&sim.st, &sim.p, &infos, &mut render::Tracker::default());
    payload_parts(&sim.st, &sim.p, &sim.backend.name(), &rec.summary, &rec.series, eps, sun, chronicle, &infos, &cols, &of)
}

#[allow(clippy::too_many_arguments)]
pub fn payload_parts(
    st: &primordia_core::State,
    p: &primordia_core::Params,
    backend: &str,
    summary: &primordia_telemetry::Summary,
    series: &std::collections::BTreeMap<String, Vec<[f64; 2]>>,
    eps: f64,
    sun: f32,
    chronicle: &[Note],
    infos: &[crate::CellInfo],
    colonies: &[render::Colony],
    colony_of: &[u32],
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
    let grid = match g.dims {
        0 => {
            let w = (g.n as f64).sqrt().ceil() as u32;
            [w, (g.n as u32).div_ceil(w), 1]
        }
        _ => g.size,
    };
    let s = summary;
    let tail = |k: &str| -> Vec<[f64; 2]> {
        let v = series.get(k).cloned().unwrap_or_default();
        let step = (v.len() / 300).max(1);
        v.into_iter().step_by(step).collect()
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
        },
        "backend": backend,
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
    let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/");
    if method != "GET" {
        return respond(&mut s, "405 Method Not Allowed", "text/plain", b"read-only viewer");
    }
    match path {
        "/" | "/index.html" => respond(&mut s, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        "/meta.json" => match feed.get() {
            Some((m, _)) => respond(&mut s, "200 OK", "application/json", m.as_bytes()),
            None => respond(&mut s, "503 Service Unavailable", "text/plain", b"warming up"),
        },
        "/frame.bin" => match feed.get() {
            Some((_, f)) => respond(&mut s, "200 OK", "application/octet-stream", &f),
            None => respond(&mut s, "503 Service Unavailable", "text/plain", b"warming up"),
        },
        _ => respond(&mut s, "404 Not Found", "text/plain", b"not found"),
    }
}
