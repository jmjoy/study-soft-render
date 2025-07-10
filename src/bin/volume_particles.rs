pub use anyhow::Result as Anyhow;
use softbuffer::{Context, Surface};
use std::{num::NonZeroU32, rc::Rc};
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

// 引入 glam 用于向量和矩阵计算
use glam::{Mat4, Vec2, Vec3, Vec4};

// 粒子结构
#[derive(Debug, Clone, Copy)]
struct Particle {
    position: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    color: Vec3,
    size: f32,
    life: f32,
    max_life: f32,
}

impl Particle {
    fn new(position: Vec3, velocity: Vec3, color: Vec3, size: f32, life: f32) -> Self {
        Self {
            position,
            velocity,
            acceleration: Vec3::new(0.0, -9.8, 0.0), // 重力
            color,
            size,
            life,
            max_life: life,
        }
    }

    fn update(&mut self, dt: f32) {
        self.velocity += self.acceleration * dt;
        self.position += self.velocity * dt;
        self.life -= dt;

        // 基于生命周期改变颜色和大小
        let life_ratio = self.life / self.max_life;
        self.color = Vec3::new(1.0 - life_ratio * 0.5, life_ratio, life_ratio * 0.3);
        self.size = self.size * life_ratio + 0.1;
    }

    fn is_alive(&self) -> bool {
        self.life > 0.0
    }
}

// 粒子系统
struct ParticleSystem {
    particles: Vec<Particle>,
    emitter_position: Vec3,
    emission_rate: f32,
    last_emission: f32,
}

impl ParticleSystem {
    fn new(emitter_position: Vec3, emission_rate: f32) -> Self {
        Self {
            particles: Vec::new(),
            emitter_position,
            emission_rate,
            last_emission: 0.0,
        }
    }

    fn update(&mut self, dt: f32, time: f32) {
        // 更新现有粒子
        self.particles.retain_mut(|particle| {
            particle.update(dt);
            particle.is_alive()
        });

        // 发射新粒子
        if time - self.last_emission > 1.0 / self.emission_rate {
            self.emit_particle();
            self.last_emission = time;
        }

        // 更新发射器位置（旋转）
        self.emitter_position = Vec3::new(2.0 * (time * 0.5).cos(), 0.0, 2.0 * (time * 0.5).sin());
    }

    fn emit_particle(&mut self) {
        use std::f32::consts::PI;

        // 随机方向
        let angle = fastrand::f32() * 2.0 * PI;
        let elevation = fastrand::f32() * PI * 0.3; // 向上发射

        let velocity = Vec3::new(
            angle.cos() * elevation.cos() * 5.0,
            elevation.sin() * 8.0,
            angle.sin() * elevation.cos() * 5.0,
        );

        let particle = Particle::new(
            self.emitter_position,
            velocity,
            Vec3::new(1.0, 0.8, 0.2), // 初始橙色
            0.3,
            3.0, // 生命周期3秒
        );

        self.particles.push(particle);
    }

    fn get_particles(&self) -> &[Particle] {
        &self.particles
    }
}

// 体积渲染结构
struct VolumeData {
    width: u32,
    height: u32,
    depth: u32,
    data: Vec<f32>, // 密度数据
}

impl VolumeData {
    fn new(width: u32, height: u32, depth: u32) -> Self {
        let mut data = Vec::with_capacity((width * height * depth) as usize);

        // 创建一个简单的3D噪声体积
        for z in 0..depth {
            for y in 0..height {
                for x in 0..width {
                    let fx = x as f32 / width as f32;
                    let fy = y as f32 / height as f32;
                    let fz = z as f32 / depth as f32;

                    // 多层噪声
                    let noise1 = ((fx * 4.0).sin() + (fy * 4.0).sin() + (fz * 4.0).sin()) * 0.3;
                    let noise2 = ((fx * 8.0).sin() + (fy * 8.0).sin() + (fz * 8.0).sin()) * 0.2;
                    let noise3 = ((fx * 16.0).sin() + (fy * 16.0).sin() + (fz * 16.0).sin()) * 0.1;

                    let density = (noise1 + noise2 + noise3 + 0.6).max(0.0);
                    data.push(density);
                }
            }
        }

        Self {
            width,
            height,
            depth,
            data,
        }
    }

    fn sample(&self, pos: Vec3) -> f32 {
        // 将位置从世界空间转换到体积空间
        let x = (pos.x + 1.0) * 0.5 * self.width as f32;
        let y = (pos.y + 1.0) * 0.5 * self.height as f32;
        let z = (pos.z + 1.0) * 0.5 * self.depth as f32;

        if x < 0.0
            || x >= self.width as f32 - 1.0
            || y < 0.0
            || y >= self.height as f32 - 1.0
            || z < 0.0
            || z >= self.depth as f32 - 1.0
        {
            return 0.0;
        }

        // 三线性插值
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let z0 = z.floor() as usize;

        let x1 = (x0 + 1).min(self.width as usize - 1);
        let y1 = (y0 + 1).min(self.height as usize - 1);
        let z1 = (z0 + 1).min(self.depth as usize - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let fz = z - z0 as f32;

        // 获取8个角的值
        let c000 =
            self.data[z0 * (self.width * self.height) as usize + y0 * self.width as usize + x0];
        let c001 =
            self.data[z1 * (self.width * self.height) as usize + y0 * self.width as usize + x0];
        let c010 =
            self.data[z0 * (self.width * self.height) as usize + y1 * self.width as usize + x0];
        let c011 =
            self.data[z1 * (self.width * self.height) as usize + y1 * self.width as usize + x0];
        let c100 =
            self.data[z0 * (self.width * self.height) as usize + y0 * self.width as usize + x1];
        let c101 =
            self.data[z1 * (self.width * self.height) as usize + y0 * self.width as usize + x1];
        let c110 =
            self.data[z0 * (self.width * self.height) as usize + y1 * self.width as usize + x1];
        let c111 =
            self.data[z1 * (self.width * self.height) as usize + y1 * self.width as usize + x1];

        // 三线性插值
        let c00 = c000 * (1.0 - fx) + c100 * fx;
        let c01 = c001 * (1.0 - fx) + c101 * fx;
        let c10 = c010 * (1.0 - fx) + c110 * fx;
        let c11 = c011 * (1.0 - fx) + c111 * fx;

        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;

        c0 * (1.0 - fz) + c1 * fz
    }
}

// 高级光栅化器
struct AdvancedEffectsRasterizer {
    width: u32,
    height: u32,
    depth_buffer: Vec<f32>,

    // 相机和变换
    view_matrix: Mat4,
    projection_matrix: Mat4,
    camera_pos: Vec3,

    // 体积数据
    volume_data: VolumeData,
}

impl AdvancedEffectsRasterizer {
    fn new(width: u32, height: u32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        let projection_matrix =
            Mat4::perspective_rh(std::f32::consts::PI / 4.0, aspect_ratio, 0.1, 100.0);

        let view_matrix = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 8.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        Self {
            width,
            height,
            depth_buffer: vec![f32::INFINITY; (width * height) as usize],
            view_matrix,
            projection_matrix,
            camera_pos: Vec3::new(0.0, 0.0, 8.0),
            volume_data: VolumeData::new(32, 32, 32),
        }
    }

    fn clear(&mut self) {
        self.depth_buffer.fill(f32::INFINITY);
    }

    fn update_camera(&mut self, time: f32) {
        let radius = 8.0;
        let x = radius * (time * 0.3).cos();
        let z = radius * (time * 0.3).sin();
        let y = 2.0 * (time * 0.2).sin();

        self.camera_pos = Vec3::new(x, y, z);

        self.view_matrix = Mat4::look_at_rh(
            self.camera_pos,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    // 渲染粒子
    fn render_particles(&mut self, particles: &[Particle], buffer: &mut [u32]) {
        for particle in particles {
            self.render_particle(particle, buffer);
        }
    }

    fn render_particle(&mut self, particle: &Particle, buffer: &mut [u32]) {
        // 将粒子位置转换到屏幕空间
        let world_pos = Vec4::new(
            particle.position.x,
            particle.position.y,
            particle.position.z,
            1.0,
        );
        let view_pos = self.view_matrix * world_pos;
        let clip_pos = self.projection_matrix * view_pos;

        if clip_pos.w <= 0.0 {
            return; // 在相机后面
        }

        let ndc_pos = Vec3::new(
            clip_pos.x / clip_pos.w,
            clip_pos.y / clip_pos.w,
            clip_pos.z / clip_pos.w,
        );

        if ndc_pos.z < -1.0 || ndc_pos.z > 1.0 {
            return; // 超出深度范围
        }

        let screen_pos = Vec2::new(
            (ndc_pos.x + 1.0) * 0.5 * self.width as f32,
            (1.0 - ndc_pos.y) * 0.5 * self.height as f32,
        );

        // 计算粒子在屏幕上的大小
        let size = particle.size * 50.0 / view_pos.z.abs();

        // 渲染粒子为圆形
        let min_x = (screen_pos.x - size).max(0.0) as i32;
        let max_x = (screen_pos.x + size).min(self.width as f32 - 1.0) as i32;
        let min_y = (screen_pos.y - size).max(0.0) as i32;
        let max_y = (screen_pos.y + size).min(self.height as f32 - 1.0) as i32;

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - screen_pos.x;
                let dy = y as f32 - screen_pos.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist <= size {
                    let alpha = (1.0 - dist / size).powi(2);

                    let index = (y * self.width as i32 + x) as usize;
                    if index < buffer.len() && ndc_pos.z < self.depth_buffer[index] {
                        // 简单的alpha混合
                        let old_color = buffer[index];
                        let old_r = ((old_color >> 16) & 0xFF) as f32 / 255.0;
                        let old_g = ((old_color >> 8) & 0xFF) as f32 / 255.0;
                        let old_b = (old_color & 0xFF) as f32 / 255.0;

                        let new_r = particle.color.x * alpha + old_r * (1.0 - alpha);
                        let new_g = particle.color.y * alpha + old_g * (1.0 - alpha);
                        let new_b = particle.color.z * alpha + old_b * (1.0 - alpha);

                        let r = (new_r * 255.0) as u32;
                        let g = (new_g * 255.0) as u32;
                        let b = (new_b * 255.0) as u32;

                        buffer[index] = 0xFF000000 | (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
    }

    // 体积渲染
    fn render_volume(&mut self, time: f32, buffer: &mut [u32]) {
        let inv_view_proj = (self.projection_matrix * self.view_matrix).inverse();

        for y in 0..self.height {
            for x in 0..self.width {
                let index = (y * self.width + x) as usize;

                // 屏幕空间坐标到NDC
                let ndc_x = (x as f32 / self.width as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (y as f32 / self.height as f32) * 2.0;

                // 射线起点和方向
                let ray_start = self.camera_pos;
                let ndc_near = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
                let ndc_far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

                let world_near = inv_view_proj * ndc_near;
                let world_far = inv_view_proj * ndc_far;

                let world_near = Vec3::new(
                    world_near.x / world_near.w,
                    world_near.y / world_near.w,
                    world_near.z / world_near.w,
                );
                let world_far = Vec3::new(
                    world_far.x / world_far.w,
                    world_far.y / world_far.w,
                    world_far.z / world_far.w,
                );

                let ray_dir = (world_far - world_near).normalize();

                // 体积光线投射
                let color = self.volume_raycast(ray_start, ray_dir, time);

                // 与现有颜色混合
                let old_color = buffer[index];
                let old_r = ((old_color >> 16) & 0xFF) as f32 / 255.0;
                let old_g = ((old_color >> 8) & 0xFF) as f32 / 255.0;
                let old_b = (old_color & 0xFF) as f32 / 255.0;

                let alpha = color.w;
                let new_r = color.x * alpha + old_r * (1.0 - alpha);
                let new_g = color.y * alpha + old_g * (1.0 - alpha);
                let new_b = color.z * alpha + old_b * (1.0 - alpha);

                let r = (new_r * 255.0) as u32;
                let g = (new_g * 255.0) as u32;
                let b = (new_b * 255.0) as u32;

                buffer[index] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
    }

    fn volume_raycast(&self, ray_start: Vec3, ray_dir: Vec3, time: f32) -> Vec4 {
        let mut color = Vec3::ZERO;
        let mut alpha = 0.0;

        let step_size = 0.1;
        let max_steps = 50;

        for i in 0..max_steps {
            let t = i as f32 * step_size;
            let pos = ray_start + ray_dir * t;

            // 检查是否在体积边界内
            if pos.x < -2.0
                || pos.x > 2.0
                || pos.y < -2.0
                || pos.y > 2.0
                || pos.z < -2.0
                || pos.z > 2.0
            {
                continue;
            }

            // 动态变化体积
            let animated_pos = Vec3::new(
                pos.x + (time * 0.5).sin() * 0.2,
                pos.y + (time * 0.3).cos() * 0.2,
                pos.z + (time * 0.7).sin() * 0.2,
            );

            let density = self.volume_data.sample(animated_pos);

            if density > 0.1 {
                // 根据密度和位置计算颜色
                let sample_color = Vec3::new(
                    0.3 + density * 0.7,
                    0.1 + density * 0.5,
                    0.8 + density * 0.2,
                );

                let sample_alpha = density * 0.3;

                // 前向alpha混合
                color = color + sample_color * sample_alpha * (1.0 - alpha);
                alpha = alpha + sample_alpha * (1.0 - alpha);

                if alpha > 0.95 {
                    break; // 早期退出
                }
            }
        }

        Vec4::new(color.x, color.y, color.z, alpha)
    }
}

// 应用程序结构
struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    rasterizer: Option<AdvancedEffectsRasterizer>,
    particle_system: Option<ParticleSystem>,
    start_time: std::time::Instant,
    last_frame_time: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            rasterizer: None,
            particle_system: None,
            start_time: std::time::Instant::now(),
            last_frame_time: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                winit::window::WindowAttributes::default()
                    .with_title("Advanced Effects - 体积渲染与粒子系统")
                    .with_inner_size(PhysicalSize::new(800, 600)),
            )
            .unwrap();
        let window = Rc::new(window);
        let context = softbuffer::Context::new(window.clone()).unwrap();
        let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

        let size = window.inner_size();
        surface
            .resize(
                NonZeroU32::new(size.width).unwrap(),
                NonZeroU32::new(size.height).unwrap(),
            )
            .unwrap();

        let rasterizer = AdvancedEffectsRasterizer::new(size.width, size.height);
        let particle_system = ParticleSystem::new(Vec3::ZERO, 30.0); // 30个粒子/秒

        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
        self.rasterizer = Some(rasterizer);
        self.particle_system = Some(particle_system);

        info!("Advanced effects window created!");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested!");
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state.is_pressed() {
                    match event.logical_key.as_ref() {
                        Key::Named(NamedKey::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(size) => {
                if let Some(surface) = &mut self.surface {
                    surface
                        .resize(
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        )
                        .unwrap();
                    if let Some(rasterizer) = &mut self.rasterizer {
                        *rasterizer = AdvancedEffectsRasterizer::new(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(rasterizer), Some(particle_system)) = (
                    &mut self.surface,
                    &mut self.rasterizer,
                    &mut self.particle_system,
                ) {
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    self.last_frame_time = now;

                    let mut buffer = surface.buffer_mut().unwrap();
                    buffer.fill(0xFF000510); // 深蓝色背景

                    // 清除深度缓冲
                    rasterizer.clear();

                    // 更新相机
                    rasterizer.update_camera(elapsed);

                    // 更新粒子系统
                    particle_system.update(dt, elapsed);

                    // 渲染体积
                    rasterizer.render_volume(elapsed, &mut buffer);

                    // 渲染粒子
                    rasterizer.render_particles(particle_system.get_particles(), &mut buffer);

                    buffer.present().unwrap();
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn init_logger() {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();
}

fn main() -> Anyhow<()> {
    init_logger();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
