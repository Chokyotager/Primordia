//! wgpu backend for the execution phase (phase 6). State lives on the CPU
//! between phases; each epoch uploads the cell buffers, runs the 3^D colour
//! sub-steps (or the 0D pairs) and downloads the result.

use anyhow::{Context, Result};
use primordia_core::exec;
use primordia_core::params::Params;
use primordia_core::state::State;
use primordia_core::wgsl::{self, pi};
use primordia_core::world::ExecBackend;
use std::time::Instant;

const SHADER: &str = include_str!("exec.wgsl");
const WG: u32 = 64;

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
    bind: wgpu::BindGroup,
    sizes: [u64; 4],
}

pub struct GpuExec {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    snap_pipe: wgpu::ComputePipeline,
    exec_pipe: wgpu::ComputePipeline,
    bufs: Option<Bufs>,
    layout_key: (usize, usize, usize),
    words: Vec<u32>,
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

impl GpuExec {
    pub fn new(p: &Params) -> Result<GpuExec> {
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("primordia"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context("request_device")?;
        let src = format!("{}\n{}", wgsl::constants(), SHADER);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("exec.wgsl"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let entries: Vec<_> = (0..8).map(|b| storage_entry(b, b == 5 || b == 6 || b == 7)).collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("cells"), entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let snap_pipe = mk("snap_main");
        let exec_pipe = mk("exec_main");
        let mut g = GpuExec {
            device,
            queue,
            layout,
            snap_pipe,
            exec_pipe,
            bufs: None,
            layout_key: (0, 0, 0),
            words: wgsl::param_words(p),
            adapter: format!("{} ({:?}, {})", info.name, info.backend, info.driver),
            timing: [0.0; 3],
        };
        g.alloc(p);
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
        let mk = |label: &str, size: u64| d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage: st, mapped_at_creation: false });
        let sizes = [
            (n * p.lmax) as u64,
            (n * primordia_core::params::HDR * 4) as u64,
            (n * p.nf * 4) as u64,
            (n * primordia_core::params::SITE * 4) as u64,
        ];
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
        self.queue.write_buffer(&nbr, 0, bytemuck::cast_slice(&p.nbr));
        let bufs = [&tapes, &hdr, &fields, &site, &snap, &nbr, &params, &pairs];
        let entries: Vec<_> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("cells"), layout: &self.layout, entries: &entries });
        self.bufs = Some(Bufs { tapes, hdr, fields, site, snap, nbr, params, pairs, staging, bind, sizes });
        self.layout_key = key;
    }

    fn dispatch(&self, enc: &mut wgpu::CommandEncoder, pipe: &wgpu::ComputePipeline, threads: u32) {
        let b = self.bufs.as_ref().unwrap();
        let groups = threads.div_ceil(WG);
        let (x, y) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups.max(1), 1) };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &b.bind, &[]);
        pass.dispatch_workgroups(x, y, 1);
    }
}

impl ExecBackend for GpuExec {
    fn name(&self) -> String {
        format!("gpu: {}", self.adapter)
    }
    fn reconfigure(&mut self, p: &Params) {
        self.words = wgsl::param_words(p);
        self.alloc(p);
    }
    fn exec(&mut self, st: &mut State, p: &Params) {
        let t0 = Instant::now();
        let b = self.bufs.as_ref().unwrap();
        let q = &self.queue;
        q.write_buffer(&b.tapes, 0, &st.tapes);
        q.write_buffer(&b.hdr, 0, bytemuck::cast_slice(&st.hdr));
        q.write_buffer(&b.fields, 0, bytemuck::cast_slice(&st.fields));
        q.write_buffer(&b.site, 0, bytemuck::cast_slice(&st.site));
        let mut w = self.words.clone();
        w[pi::EPOCH] = st.epoch as u32;
        w[pi::MIRRORED] = st.mirrored as u32;
        let t1 = Instant::now();
        if p.geom.dims == 0 {
            let pairs = exec::pairing(p, st.epoch);
            q.write_buffer(&b.pairs, 0, bytemuck::cast_slice(&pairs));
            w[pi::NPAIRS] = (pairs.len() / 2) as u32;
            q.write_buffer(&b.params, 0, bytemuck::cast_slice(&w));
            let mut enc = self.device.create_command_encoder(&Default::default());
            self.dispatch(&mut enc, &self.snap_pipe, p.geom.n as u32);
            self.dispatch(&mut enc, &self.exec_pipe, w[pi::NPAIRS]);
            q.submit([enc.finish()]);
        } else {
            for col in exec::color_order(p, st.epoch, st.mirrored) {
                w[pi::COLOR] = col;
                q.write_buffer(&b.params, 0, bytemuck::cast_slice(&w));
                let mut enc = self.device.create_command_encoder(&Default::default());
                self.dispatch(&mut enc, &self.snap_pipe, p.geom.n as u32);
                self.dispatch(&mut enc, &self.exec_pipe, p.geom.cells_per_color() as u32);
                q.submit([enc.finish()]);
            }
        }
        // download
        let mut enc = self.device.create_command_encoder(&Default::default());
        let mut off = 0;
        for (src, sz) in [&b.tapes, &b.hdr, &b.fields, &b.site].iter().zip(b.sizes) {
            enc.copy_buffer_to_buffer(src, 0, &b.staging, off, sz);
            off += sz;
        }
        q.submit([enc.finish()]);
        let slice = b.staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map staging"));
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let t2 = Instant::now();
        {
            let data = slice.get_mapped_range().expect("mapped range");
            let s = b.sizes;
            let (a, rest) = data.split_at(s[0] as usize);
            let (h, rest) = rest.split_at(s[1] as usize);
            let (f, rest) = rest.split_at(s[2] as usize);
            let (si, _) = rest.split_at(s[3] as usize);
            st.tapes.copy_from_slice(a);
            st.hdr.copy_from_slice(bytemuck::cast_slice(h));
            st.fields.copy_from_slice(bytemuck::cast_slice(f));
            st.site.copy_from_slice(bytemuck::cast_slice(si));
        }
        b.staging.unmap();
        let t3 = Instant::now();
        self.timing[0] += (t1 - t0).as_secs_f64();
        self.timing[1] += (t2 - t1).as_secs_f64();
        self.timing[2] += (t3 - t2).as_secs_f64();
    }
}

/// Construct the configured backend, falling back to the CPU golden model.
pub fn backend(p: &Params, want_gpu: bool) -> Box<dyn ExecBackend> {
    if want_gpu {
        match GpuExec::new(p) {
            Ok(g) => return Box::new(g),
            Err(e) => eprintln!("GPU unavailable ({e:#}); using CPU golden model"),
        }
    }
    Box::new(primordia_core::CpuExec)
}
