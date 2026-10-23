//! wgpu backend. Two modes:
//! * exec-only: phase 6 on the GPU, state uploaded/downloaded every epoch;
//! * resident (default): the whole state stays on the GPU and every phase
//!   except interventions and vent bookkeeping runs there (`phys.wgsl`).
//!   The CPU copy is refreshed only on `download`.

use anyhow::{Context, Result};
use primordia_core::exec;
use primordia_core::params::{HDR, Params, SITE};
use primordia_core::phases::{self, FluxSpec};
use primordia_core::rng::phase;
use primordia_core::state::State;
use primordia_core::wgsl::{self, pi, qi, ts};
use primordia_core::world::{EpochResult, ExecBackend, StateCopy};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const EXEC_SHADER: &str = include_str!("exec.wgsl");
const PHYS_SHADER: &str = include_str!("phys.wgsl");
const WG: u32 = 64;
const CTL_LEN: usize = 96;
const BIRTH_CAP: usize = 16384;
/// Epochs that may be in flight at once.
const RING: usize = 6;
/// Asynchronous full-state copies that may be outstanding.
const COPIES: usize = 3;
const MAX_VENTS: usize = 4096;
const NONE: u32 = 0xFFFF_FFFF;
/// Union-find rounds per epoch (each gated off on the GPU once converged).
const LAB_ROUNDS: usize = 24;

#[allow(dead_code)]
struct Bufs {
    tapes: wgpu::Buffer,
    hdr: wgpu::Buffer,
    fields: wgpu::Buffer,
    site: wgpu::Buffer,
    snap: wgpu::Buffer,
    nbr: wgpu::Buffer,
    params: wgpu::Buffer,
    pairs: wgpu::Buffer,
    staging: wgpu::Buffer,
    exec_bind: wgpu::BindGroup,
    sizes: [u64; 5],
    // resident physics
    q: wgpu::Buffer,
    tables: wgpu::Buffer,
    light: wgpu::Buffer,
    rows: wgpu::Buffer,
    vents: wgpu::Buffer,
    mmap: wgpu::Buffer,
    jf: wgpu::Buffer,
    bo: wgpu::Buffer,
    outs: wgpu::Buffer,
    expo: wgpu::Buffer,
    tally: wgpu::Buffer,
    ctl: wgpu::Buffer,
    ctl_all: wgpu::Buffer,
    bw: wgpu::Buffer,
    shdr: wgpu::Buffer,
    stape: wgpu::Buffer,
    births: wgpu::Buffer,
    small: wgpu::Buffer,
    phys_bind: wgpu::BindGroup,
    iargs: wgpu::Buffer,
    iargs_w: wgpu::Buffer,
    params_all: wgpu::Buffer,
    res: Vec<Slot>,
    copies: Vec<Slot>,
}

struct Slot {
    buf: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    sub: Option<wgpu::SubmissionIndex>,
    epoch: u64,
}

struct Pipes {
    snap: wgpu::ComputePipeline,
    exec: wgpu::ComputePipeline,
    phys: std::collections::HashMap<&'static str, wgpu::ComputePipeline>,
}

pub struct GpuExec {
    device: wgpu::Device,
    queue: wgpu::Queue,
    exec_layout: wgpu::BindGroupLayout,
    phys_layout: wgpu::BindGroupLayout,
    pipes: Pipes,
    bufs: Option<Bufs>,
    layout_key: (usize, usize, usize),
    words: Vec<u32>,
    qwords: Vec<u32>,
    /// number of transport batches; ctl blocks: [0,3) normal frame,
    /// [3,6) mirrored frame, 6 = bonded energy flux
    n_batches: usize,
    map_key: u64,
    pub resident: bool,
    inflight: VecDeque<usize>,
    next_slot: usize,
    copy_q: VecDeque<usize>,
    next_copy: usize,
    done: VecDeque<EpochResult>,
    copies_done: VecDeque<StateCopy>,
    /// PRIMORDIA_PROFILE=1: flush and time each phase group (slow; for profiling only)
    profile: Option<std::collections::BTreeMap<&'static str, f64>>,
    pub adapter: String,
    /// Accumulated time (s): upload, kernels, download.
    pub timing: [f64; 3],
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

const PHYS_KERNELS: &[&str] = &[
    "k_light",
    "k_light3d",
    "k_sun",
    "k_vents",
    "k_flux_a",
    "k_flux_b",
    "k_flux_c",
    "k_flux_d",
    "k_post",
    "k_chem",
    "k_flagclear",
    "k_expo",
    "k_absorb",
    "k_wake",
    "k_site",
    "k_lab_init",
    "k_lab_hook",
    "k_lab_jump",
    "k_lab_round",
    "k_lab_gate",
    "k_body_gate",
    "k_lab_verify",
    "k_fault_gate",
    "k_body_tally",
    "k_body_decide",
    "k_body_check",
    "k_body_claim",
    "k_body_check2",
    "k_body_move1",
    "k_body_move2",
    "k_body_move3",
    "k_clear_votes",
    "k_mutate",
    "k_bondbreak",
    "k_dead",
    "k_dead_bonds",
    "k_ledger",
];
const PHYS_READ_ONLY: &[u32] = &[4, 5, 6, 8, 9, 10, 16];

impl GpuExec {
    pub fn new(p: &Params) -> Result<GpuExec> {
        Self::with_mode(p, true)
    }
    pub fn with_mode(p: &Params, resident: bool) -> Result<GpuExec> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .context("no GPU adapter")?;
        let info = adapter.get_info();
        let alimits = adapter.limits();
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = alimits.max_storage_buffer_binding_size;
        limits.max_buffer_size = alimits.max_buffer_size;
        limits.max_storage_buffers_per_shader_stage = alimits.max_storage_buffers_per_shader_stage.max(8);
        limits.max_bindings_per_bind_group = alimits.max_bindings_per_bind_group;
        anyhow::ensure!(limits.max_storage_buffers_per_shader_stage >= 22, "adapter supports only {} storage buffers per stage", limits.max_storage_buffers_per_shader_stage);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("primordia"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("request_device")?;
        // interpreter
        let src = format!("{}\n{}", wgsl::constants(), EXEC_SHADER);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("exec.wgsl"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let entries: Vec<_> = (0..8).map(|b| storage_entry(b, b == 5 || b == 6 || b == 7)).collect();
        let exec_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cells"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&exec_layout)], immediate_size: 0 });
        let mk = |module: &wgpu::ShaderModule, pl: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(pl),
                module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let snap = mk(&module, &pl, "snap_main");
        let exec = mk(&module, &pl, "exec_main");
        // physics
        let psrc = format!("{}\n{}", wgsl::phys_constants(), PHYS_SHADER);
        let pmodule = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("phys.wgsl"), source: wgpu::ShaderSource::Wgsl(psrc.into()) });
        let pentries: Vec<_> = (0..22).map(|b| storage_entry(b, PHYS_READ_ONLY.contains(&b))).collect();
        let phys_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("phys"), entries: &pentries });
        let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&phys_layout)], immediate_size: 0 });
        let phys = PHYS_KERNELS.iter().map(|k| (*k, mk(&pmodule, &ppl, k))).collect();
        let mut g = GpuExec {
            device,
            queue,
            exec_layout,
            phys_layout,
            pipes: Pipes { snap, exec, phys },
            bufs: None,
            layout_key: (0, 0, 0),
            words: wgsl::param_words(p),
            qwords: wgsl::phys_words(p),
            n_batches: 0,
            map_key: 0,
            resident,
            inflight: VecDeque::new(),
            next_slot: 0,
            copy_q: VecDeque::new(),
            next_copy: 0,
            done: VecDeque::new(),
            copies_done: VecDeque::new(),
            profile: std::env::var("PRIMORDIA_PROFILE").ok().map(|_| Default::default()),
            adapter: format!("{} ({:?}, {})", info.name, info.backend, info.driver),
            timing: [0.0; 3],
        };
        g.alloc(p);
        g.write_static(p);
        Ok(g)
    }

    fn alloc(&mut self, p: &Params) {
        let n = p.geom.n;
        let key = (n, p.lmax, p.nf);
        if self.bufs.is_some() && self.layout_key == key {
            return;
        }
        let d = &self.device;
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let mk = |label: &str, size: u64| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16).div_ceil(4) * 4, usage: st, mapped_at_creation: false });
        let sizes = [(n * p.lmax) as u64, (n * HDR * 4) as u64, (n * p.nf * 4) as u64, (n * SITE * 4) as u64, (n * 4) as u64];
        let tapes = mk("tapes", sizes[0]);
        let hdr = mk("hdr", sizes[1]);
        let fields = mk("fields", sizes[2]);
        let site = mk("site", sizes[3]);
        let snap = mk("snap", (n * 4) as u64);
        let nbr = mk("nbr", (n * 6 * 4) as u64);
        let params = mk("params", (pi::LEN * 4) as u64);
        let pairs = mk("pairs", (n * 4) as u64);
        let staging = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: sizes.iter().sum(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let small = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("small"),
            size: ((ts::N * 2 + BIRTH_CAP * 6) * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&nbr, 0, bytemuck::cast_slice(&p.nbr));
        let ebufs = [&tapes, &hdr, &fields, &site, &snap, &nbr, &params, &pairs];
        let entries: Vec<_> = ebufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let exec_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cells"), layout: &self.exec_layout, entries: &entries });
        let q = mk("q", (qi::LEN * 4) as u64);
        let tables = mk("tables", (wgsl::phys_tables(p).len() * 4) as u64);
        let light = mk("light", (n * 4) as u64);
        let rows = mk("rows", (p.geom.size[1].max(1) as usize * 4) as u64);
        let vents = mk("vents", ((MAX_VENTS + 1) * 4) as u64);
        let mmap = mk("mmap", (n * 4) as u64);
        let jf = mk("jf", (n * 3 * 8 * 4) as u64);
        let bo = mk("bo", (n * 8 * 4) as u64);
        let outs = mk("outs", (n * 8 * 4) as u64);
        let expo = mk("expo", (n * 4) as u64);
        let tally = mk("tally", (ts::N * 2 * 4) as u64);
        let ctl = mk("ctl", (CTL_LEN * 4) as u64);
        let ctl_all = mk("ctl_all", (CTL_LEN * 4 * 8) as u64);
        let bw = mk("bw", (n * 12 * 4) as u64);
        let shdr = mk("shdr", sizes[1]);
        let stape = mk("stape", sizes[0]);
        let births = mk("births", (BIRTH_CAP * 6 * 4) as u64);
        // kernels write the dispatch arguments into iargs_w (bound); a copy moves
        // them into iargs (indirect-only), since one buffer cannot be both in
        // a single dispatch
        let iargs_w = mk("iargs_w", 64);
        let iargs = d.create_buffer(&wgpu::BufferDescriptor { label: Some("iargs"), size: 64, usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let params_all = mk("params_all", (pi::LEN * 4 * 28) as u64);
        let res_size = ((ts::N * 2 + BIRTH_CAP * 6) * 4) as u64;
        let mk_slot = |label: &str, size: u64| Slot {
            buf: d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            ready: Arc::new(AtomicBool::new(false)),
            sub: None,
            epoch: 0,
        };
        let res = (0..RING).map(|_| mk_slot("result", res_size)).collect();
        let copies = (0..COPIES).map(|_| mk_slot("copy", sizes.iter().sum())).collect();
        let pb = [&tapes, &hdr, &fields, &site, &nbr, &q, &tables, &light, &rows, &vents, &mmap, &jf, &bo, &outs, &expo, &tally, &ctl, &bw, &shdr, &stape, &births, &iargs_w];
        let pentries: Vec<_> = pb.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let phys_bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("phys"), layout: &self.phys_layout, entries: &pentries });
        self.bufs = Some(Bufs {
            tapes,
            hdr,
            fields,
            site,
            snap,
            nbr,
            params,
            pairs,
            staging,
            exec_bind,
            sizes,
            q,
            tables,
            light,
            rows,
            vents,
            mmap,
            jf,
            bo,
            outs,
            expo,
            tally,
            ctl,
            ctl_all,
            bw,
            shdr,
            stape,
            births,
            small,
            phys_bind,
            iargs,
            iargs_w,
            params_all,
            res,
            copies,
        });
        self.inflight.clear();
        self.copy_q.clear();
        self.layout_key = key;
        self.map_key = 0;
    }

    /// Upload config-derived tables and the transport plans.
    fn write_static(&mut self, p: &Params) {
        let b = self.bufs.as_ref().unwrap();
        self.queue.write_buffer(&b.tables, 0, bytemuck::cast_slice(&wgsl::phys_tables(p)));
        let mut all = vec![0u32; CTL_LEN * 8];
        let mut nb = 0;
        for (bi, batch) in phases::transport_batches(p, false).iter().enumerate() {
            write_ctl(&mut all[bi * CTL_LEN..(bi + 1) * CTL_LEN], p, batch, bi as u32, phase::TRANSPORT, 0);
            nb = bi + 1;
        }
        assert!(nb <= 3, "too many transport batches");
        for (bi, batch) in phases::transport_batches(p, true).iter().enumerate() {
            write_ctl(&mut all[(3 + bi) * CTL_LEN..(4 + bi) * CTL_LEN], p, batch, bi as u32, phase::TRANSPORT, 0);
        }
        let e = FluxSpec { kappa: p.beta_q16, gamma: None, bonded_only: true, stream: 0x20000, outside: None };
        write_ctl(&mut all[6 * CTL_LEN..7 * CTL_LEN], p, &[(0, e, 0, 255)], 0x20000, phase::ABSORB, 1);
        self.queue.write_buffer(&b.ctl_all, 0, bytemuck::cast_slice(&all));
        self.n_batches = nb;
    }

    fn groups(threads: u32) -> (u32, u32) {
        let groups = threads.div_ceil(WG).max(1);
        if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups, 1) }
    }

    fn dispatch(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, threads: u32) {
        let (x, y) = Self::groups(threads);
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
    fn phys(&self, enc: &mut wgpu::CommandEncoder, k: &str, threads: u32) {
        let b = self.bufs.as_ref().unwrap();
        self.dispatch(enc, &self.pipes.phys[k], &b.phys_bind, threads);
    }
    fn phys1(&self, enc: &mut wgpu::CommandEncoder, k: &str) {
        let b = self.bufs.as_ref().unwrap();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&self.pipes.phys[k]);
        pass.set_bind_group(0, &b.phys_bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fn use_ctl(&self, enc: &mut wgpu::CommandEncoder, block: usize) {
        let b = self.bufs.as_ref().unwrap();
        enc.copy_buffer_to_buffer(&b.ctl_all, (block * CTL_LEN * 4) as u64, &b.ctl, 0, (CTL_LEN * 4) as u64);
    }

    /// Read `words` u32 from `src` at byte offset `off` through the small staging buffer.
    fn read_small(&self, src: &wgpu::Buffer, off: u64, words: usize) -> Vec<u32> {
        let b = self.bufs.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, off, &b.small, 0, (words * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = b.small.slice(0..(words * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map small"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let v = bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        b.small.unmap();
        v
    }

    fn run_exec(&self, st: &State, p: &Params) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode_exec(&mut enc, st.epoch, st.mirrored, p);
        self.queue.submit([enc.finish()]);
    }

    /// Encode phase 6 into `enc`: one parameter block per colour is uploaded
    /// once and copied into place before each sub-step.
    fn encode_exec(&self, enc: &mut wgpu::CommandEncoder, epoch: u64, mirrored: bool, p: &Params) {
        let b = self.bufs.as_ref().unwrap();
        let q = &self.queue;
        let mut w = self.words.clone();
        w[pi::EPOCH] = epoch as u32;
        w[pi::MIRRORED] = mirrored as u32;
        let blk = (pi::LEN * 4) as u64;
        if p.geom.dims == 0 {
            let pairs = exec::pairing(p, epoch);
            q.write_buffer(&b.pairs, 0, bytemuck::cast_slice(&pairs));
            w[pi::NPAIRS] = (pairs.len() / 2) as u32;
            q.write_buffer(&b.params, 0, bytemuck::cast_slice(&w));
            self.dispatch(enc, &self.pipes.snap, &b.exec_bind, p.geom.n as u32);
            self.dispatch(enc, &self.pipes.exec, &b.exec_bind, w[pi::NPAIRS]);
        } else {
            let order = exec::color_order(p, epoch, mirrored);
            let mut all = Vec::with_capacity(pi::LEN * order.len());
            for &col in &order {
                w[pi::COLOR] = col;
                all.extend_from_slice(&w);
            }
            q.write_buffer(&b.params_all, 0, bytemuck::cast_slice(&all));
            for i in 0..order.len() {
                enc.copy_buffer_to_buffer(&b.params_all, i as u64 * blk, &b.params, 0, blk);
                self.dispatch(enc, &self.pipes.snap, &b.exec_bind, p.geom.n as u32);
                self.dispatch(enc, &self.pipes.exec, &b.exec_bind, p.geom.cells_per_color() as u32);
            }
        }
    }

    fn gate(&self, enc: &mut wgpu::CommandEncoder, k: &str) {
        self.phys1(enc, k);
        let b = self.bufs.as_ref().unwrap();
        enc.copy_buffer_to_buffer(&b.iargs_w, 0, &b.iargs, 0, 64);
    }

    /// Profiling: submit what is encoded so far, wait, and charge the time to `name`.
    fn lap(&mut self, enc: &mut wgpu::CommandEncoder, name: &'static str, t: &mut Instant) {
        if self.profile.is_none() {
            return;
        }
        let done = std::mem::replace(enc, self.device.create_command_encoder(&Default::default()));
        self.queue.submit([done.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        *self.profile.as_mut().unwrap().entry(name).or_default() += t.elapsed().as_secs_f64();
        *t = Instant::now();
    }

    pub fn profile_report(&self) -> Option<Vec<(&'static str, f64)>> {
        self.profile.as_ref().map(|m| m.iter().map(|(k, v)| (*k, *v)).collect())
    }

    fn phys_indirect(&self, enc: &mut wgpu::CommandEncoder, k: &str, at: u64) {
        let b = self.bufs.as_ref().unwrap();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&self.pipes.phys[k]);
        pass.set_bind_group(0, &b.phys_bind, &[]);
        pass.dispatch_workgroups_indirect(&b.iargs, at);
    }

    /// Read a finished result slot (it must be mapped).
    fn harvest_result(&mut self, si: usize) {
        let b = self.bufs.as_ref().unwrap();
        let slot = &b.res[si];
        let (tally, births) = {
            let data = slot.buf.slice(..).get_mapped_range().expect("mapped result");
            let w: &[u32] = bytemuck::cast_slice(&data);
            let tally: Vec<u64> = (0..ts::N).map(|s| w[2 * s] as u64 | (w[2 * s + 1] as u64) << 32).collect();
            let nb = (tally[ts::BIRTH_COUNT] as usize).min(BIRTH_CAP);
            let bw = &w[ts::N * 2..];
            let mut births: Vec<[u32; 6]> = (0..nb).map(|i| [bw[i * 6], bw[i * 6 + 1], bw[i * 6 + 2], bw[i * 6 + 3], bw[i * 6 + 4], bw[i * 6 + 5]]).collect();
            births.sort_by_key(|x| x[3]);
            (tally, births)
        };
        slot.buf.unmap();
        slot.ready.store(false, Ordering::Release);
        if let Some(p) = self.profile.as_mut() {
            *p.entry("(labelling rounds used)").or_default() += tally[ts::ROUNDS] as f64;
            *p.entry("(epochs)").or_default() += 1.0;
        }
        self.done.push_back(EpochResult { epoch: slot.epoch, tally, births });
    }

    /// Harvest finished epochs in order; with `wait`, block for all of them.
    fn pump(&mut self, wait: bool) {
        let _ = self.device.poll(if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll });
        while let Some(&si) = self.inflight.front() {
            if !self.bufs.as_ref().unwrap().res[si].ready.load(Ordering::Acquire) {
                if !wait {
                    break;
                }
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                continue;
            }
            self.inflight.pop_front();
            self.harvest_result(si);
        }
    }

    fn harvest_copy(&mut self, ci: usize) {
        let b = self.bufs.as_ref().unwrap();
        let slot = &b.copies[ci];
        let s = b.sizes;
        let c = {
            let data = slot.buf.slice(..).get_mapped_range().expect("mapped copy");
            let (a, rest) = data.split_at(s[0] as usize);
            let (h, rest) = rest.split_at(s[1] as usize);
            let (f, rest) = rest.split_at(s[2] as usize);
            let (si, rest) = rest.split_at(s[3] as usize);
            StateCopy {
                epoch: slot.epoch,
                tapes: a.to_vec(),
                hdr: bytemuck::cast_slice(h).to_vec(),
                fields: bytemuck::cast_slice(f).to_vec(),
                site: bytemuck::cast_slice(si).to_vec(),
                light: bytemuck::cast_slice(&rest[..s[4] as usize]).to_vec(),
            }
        };
        slot.buf.unmap();
        slot.ready.store(false, Ordering::Release);
        self.copies_done.push_back(c);
    }

    fn pump_copies(&mut self, wait: bool) {
        let _ = self.device.poll(if wait { wgpu::PollType::wait_indefinitely() } else { wgpu::PollType::Poll });
        while let Some(&ci) = self.copy_q.front() {
            if !self.bufs.as_ref().unwrap().copies[ci].ready.load(Ordering::Acquire) {
                if !wait {
                    break;
                }
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                continue;
            }
            self.copy_q.pop_front();
            self.harvest_copy(ci);
        }
    }

    fn upload_state(&self, st: &State) {
        let b = self.bufs.as_ref().unwrap();
        let q = &self.queue;
        q.write_buffer(&b.tapes, 0, &st.tapes);
        q.write_buffer(&b.hdr, 0, bytemuck::cast_slice(&st.hdr));
        q.write_buffer(&b.fields, 0, bytemuck::cast_slice(&st.fields));
        q.write_buffer(&b.site, 0, bytemuck::cast_slice(&st.site));
        q.write_buffer(&b.light, 0, bytemuck::cast_slice(&st.light));
    }

    fn download_state(&self, st: &mut State, with_light: bool) {
        let b = self.bufs.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut off = 0;
        let srcs = [&b.tapes, &b.hdr, &b.fields, &b.site, &b.light];
        let k = if with_light { 5 } else { 4 };
        for (src, sz) in srcs.iter().zip(b.sizes).take(k) {
            enc.copy_buffer_to_buffer(src, 0, &b.staging, off, sz);
            off += sz;
        }
        self.queue.submit([enc.finish()]);
        let slice = b.staging.slice(0..off);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map staging"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        {
            let data = slice.get_mapped_range().expect("mapped range");
            let s = b.sizes;
            let (a, rest) = data.split_at(s[0] as usize);
            let (h, rest) = rest.split_at(s[1] as usize);
            let (f, rest) = rest.split_at(s[2] as usize);
            let (si, rest) = rest.split_at(s[3] as usize);
            st.tapes.copy_from_slice(a);
            st.hdr.copy_from_slice(bytemuck::cast_slice(h));
            st.fields.copy_from_slice(bytemuck::cast_slice(f));
            st.site.copy_from_slice(bytemuck::cast_slice(si));
            if with_light {
                st.light.copy_from_slice(bytemuck::cast_slice(&rest[..s[4] as usize]));
            }
        }
        b.staging.unmap();
    }
}

/// Fill one ctl block for a flux batch.
fn write_ctl(blk: &mut [u32], p: &Params, batch: &[(usize, FluxSpec, usize, u8)], stream: u32, ph: u32, mode: u32) {
    blk.iter_mut().for_each(|x| *x = 0);
    blk[0] = batch.len() as u32;
    blk[1] = stream;
    blk[2] = mode;
    blk[3] = batch.iter().any(|f| f.1.outside.is_some()) as u32;
    blk[4] = ph;
    for (k, &(off, spec, slot, kind)) in batch.iter().enumerate() {
        let b = 8 + k * 10;
        blk[b] = off as u32;
        blk[b + 1] = spec.kappa as u32;
        blk[b + 2] = spec.gamma.map(|g| g as u32).unwrap_or(0);
        blk[b + 3] = spec.bonded_only as u32;
        blk[b + 4] = spec.outside.is_some() as u32;
        blk[b + 5] = spec.outside.unwrap_or(0);
        blk[b + 6] = slot as u32;
        let (tslot, mult) = match kind {
            0 => (ts::E_SINK as u32, 1),
            2 => (ts::ATOM_SINK as u32, p.mass[(off - 2) / 2]),
            3 => ((ts::M_SINK + off - p.f_pool(0)) as u32, 1),
            _ => (NONE, 0),
        };
        blk[b + 7] = tslot;
        blk[b + 8] = mult;
    }
}

impl GpuExec {
    pub fn debug_births(&self, n: usize) -> Vec<u32> {
        let b = self.bufs.as_ref().unwrap();
        self.read_small(&b.births, 0, n)
    }
}

impl ExecBackend for GpuExec {
    fn name(&self) -> String {
        format!("gpu{}: {}", if self.resident { " (resident)" } else { "" }, self.adapter)
    }
    fn reconfigure(&mut self, p: &Params) {
        self.words = wgsl::param_words(p);
        self.qwords = wgsl::phys_words(p);
        self.alloc(p);
        self.write_static(p);
    }
    fn timing(&self) -> Option<[f64; 3]> {
        if let Some(r) = self.profile_report() {
            let tot: f64 = r.iter().map(|x| x.1).sum();
            for (k, v) in r {
                eprintln!("   profile {k:<36} {:5.1}%  {:.1} s", 100.0 * v / tot, v);
            }
        }
        Some(self.timing)
    }
    fn resident(&self) -> bool {
        self.resident
    }
    fn upload(&mut self, st: &State) {
        let t = Instant::now();
        self.upload_state(st);
        self.timing[0] += t.elapsed().as_secs_f64();
    }
    fn download(&mut self, st: &mut State) {
        let t = Instant::now();
        self.pump(true);
        self.download_state(st, true);
        self.timing[2] += t.elapsed().as_secs_f64();
    }

    fn exec(&mut self, st: &mut State, p: &Params) {
        let t0 = Instant::now();
        self.upload_state(st);
        let t1 = Instant::now();
        self.run_exec(st, p);
        let t2 = Instant::now();
        self.download_state(st, false);
        let t3 = Instant::now();
        self.timing[0] += (t1 - t0).as_secs_f64();
        self.timing[1] += (t2 - t1).as_secs_f64();
        self.timing[2] += (t3 - t2).as_secs_f64();
    }

    fn submit_resident(&mut self, epoch: u64, mirrored: bool, p: &Params, vents: &[u32], map: Option<&[u32]>) {
        let t0 = Instant::now();
        let n = p.geom.n as u32;
        let dims = p.geom.dims;
        // free the result slot we are about to reuse
        let si = self.next_slot;
        while self.inflight.contains(&si) {
            self.pump(false);
            if self.inflight.contains(&si) {
                let sub = self.bufs.as_ref().unwrap().res[self.inflight[0]].sub.clone();
                if let Some(s) = sub {
                    let _ = self.device.poll(wgpu::PollType::Wait { submission_index: Some(s), timeout: None });
                }
            }
        }
        self.next_slot = (si + 1) % RING;
        {
            let b = self.bufs.as_ref().unwrap();
            let q = &self.queue;
            let mut qw = self.qwords.clone();
            qw[qi::EPOCH] = epoch as u32;
            qw[qi::MIRRORED] = mirrored as u32;
            q.write_buffer(&b.q, 0, bytemuck::cast_slice(&qw));
            q.write_buffer(&b.rows, 0, bytemuck::cast_slice(&phases::row_lights(p, epoch)));
            let mut vw = vec![vents.len().min(MAX_VENTS) as u32];
            vw.extend(vents.iter().take(MAX_VENTS));
            q.write_buffer(&b.vents, 0, bytemuck::cast_slice(&vw));
            if let Some(m) = map {
                let key = m.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3));
                if key != self.map_key {
                    q.write_buffer(&b.mmap, 0, bytemuck::cast_slice(m));
                    self.map_key = key;
                }
            }
            q.write_buffer(&b.tally, 0, bytemuck::cast_slice(&vec![0u32; ts::N * 2]));
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut lap = Instant::now();
        if self.profile.is_some() {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            lap = Instant::now();
        }
        // phases 2-5
        if dims == 3 && p.sun_on {
            self.phys(&mut enc, "k_light3d", p.geom.size[0] * p.geom.size[1]);
        } else {
            self.phys(&mut enc, "k_light", n);
        }
        if p.sun_on {
            self.phys(&mut enc, "k_sun", n);
        }
        if p.vents_on && !vents.is_empty() {
            self.phys1(&mut enc, "k_vents");
        }
        if dims > 0 {
            let base = if mirrored { 3 } else { 0 };
            for bi in 0..self.n_batches {
                self.use_ctl(&mut enc, base + bi);
                for k in ["k_flux_a", "k_flux_b", "k_flux_c", "k_flux_d"] {
                    self.phys(&mut enc, k, n);
                }
            }
            self.phys(&mut enc, "k_post", n);
        }
        if p.chem_on || p.matter_on {
            self.phys(&mut enc, "k_chem", n);
        }
        self.phys(&mut enc, "k_flagclear", n);
        if p.energy_on {
            self.phys(&mut enc, "k_expo", n);
            self.phys(&mut enc, "k_absorb", n);
            if p.bonds_on && p.beta_q16 > 0 {
                self.use_ctl(&mut enc, 6);
                for k in ["k_flux_a", "k_flux_b", "k_flux_c", "k_flux_d"] {
                    self.phys(&mut enc, k, n);
                }
            }
            self.phys(&mut enc, "k_wake", n);
        }
        self.lap(&mut enc, "2-5 sources/transport/chem/absorb", &mut lap);
        // phase 6
        self.encode_exec(&mut enc, epoch, mirrored, p);
        self.lap(&mut enc, "6 interpreter", &mut lap);
        // site counters, births
        self.phys(&mut enc, "k_site", n);
        // phase 7: bodies, gated on the GPU by the vote count
        if p.bonds_on && dims > 0 {
            let b = self.bufs.as_ref().unwrap();
            self.gate(&mut enc, "k_body_gate");
            enc.clear_buffer(&b.bw, (n as u64) * 4, None);
            self.phys_indirect(&mut enc, "k_lab_init", 0);
            for _ in 0..LAB_ROUNDS {
                self.phys_indirect(&mut enc, "k_lab_round", 0);
                self.gate(&mut enc, "k_lab_gate");
            }
            self.phys_indirect(&mut enc, "k_lab_verify", 12);
            self.gate(&mut enc, "k_fault_gate");
            for k in ["k_body_tally", "k_body_decide", "k_body_check", "k_body_claim", "k_body_check2", "k_body_move1", "k_body_move2", "k_body_move3"] {
                self.phys_indirect(&mut enc, k, 12);
            }
        }
        self.lap(&mut enc, "7 bodies (+site)", &mut lap);
        // phases 8-9
        self.phys(&mut enc, "k_clear_votes", n);
        self.phys(&mut enc, "k_expo", n);
        self.phys(&mut enc, "k_mutate", n);
        if p.bonds_on && p.bond_break_q32 > 0 && dims > 0 {
            self.phys(&mut enc, "k_bondbreak", n);
        }
        self.phys(&mut enc, "k_dead", n);
        self.phys(&mut enc, "k_dead_bonds", n);
        self.phys(&mut enc, "k_ledger", n);
        self.lap(&mut enc, "8-9 mutation/decay/ledger", &mut lap);
        // results to the slot's staging buffer
        let b = self.bufs.as_ref().unwrap();
        let tsz = (ts::N * 2 * 4) as u64;
        enc.copy_buffer_to_buffer(&b.tally, 0, &b.res[si].buf, 0, tsz);
        enc.copy_buffer_to_buffer(&b.births, 0, &b.res[si].buf, tsz, (BIRTH_CAP * 6 * 4) as u64);
        let sub = self.queue.submit([enc.finish()]);
        let b = self.bufs.as_mut().unwrap();
        let slot = &mut b.res[si];
        slot.epoch = epoch;
        slot.sub = Some(sub);
        let ready = slot.ready.clone();
        slot.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            r.expect("map result");
            ready.store(true, Ordering::Release);
        });
        self.inflight.push_back(si);
        self.timing[1] += t0.elapsed().as_secs_f64();
    }

    fn poll_resident(&mut self, wait: bool) -> Vec<EpochResult> {
        let t = Instant::now();
        self.pump(wait);
        self.timing[2] += t.elapsed().as_secs_f64();
        self.done.drain(..).collect()
    }

    fn request_copy(&mut self, epoch: u64) {
        let ci = self.next_copy;
        while self.copy_q.contains(&ci) {
            self.pump_copies(true);
        }
        self.next_copy = (ci + 1) % COPIES;
        let b = self.bufs.as_ref().unwrap();
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut off = 0;
        for (src, sz) in [&b.tapes, &b.hdr, &b.fields, &b.site, &b.light].iter().zip(b.sizes) {
            enc.copy_buffer_to_buffer(src, 0, &b.copies[ci].buf, off, sz);
            off += sz;
        }
        let sub = self.queue.submit([enc.finish()]);
        let b = self.bufs.as_mut().unwrap();
        let slot = &mut b.copies[ci];
        slot.epoch = epoch;
        slot.sub = Some(sub);
        let ready = slot.ready.clone();
        slot.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            r.expect("map copy");
            ready.store(true, Ordering::Release);
        });
        self.copy_q.push_back(ci);
    }

    fn poll_copies(&mut self, wait: bool) -> Vec<StateCopy> {
        self.pump_copies(wait);
        self.copies_done.drain(..).collect()
    }
}

/// Construct a backend, falling back to the CPU golden model.
/// `resident`: keep the whole state on the GPU (every phase on the GPU).
pub fn backend_mode(p: &Params, want_gpu: bool, resident: bool) -> Box<dyn ExecBackend> {
    if want_gpu {
        match GpuExec::with_mode(p, resident) {
            Ok(g) => return Box::new(g),
            Err(e) => eprintln!("GPU unavailable ({e:#}); using CPU golden model"),
        }
    }
    Box::new(primordia_core::CpuExec)
}

pub fn backend(p: &Params, want_gpu: bool) -> Box<dyn ExecBackend> {
    backend_mode(p, want_gpu, true)
}
