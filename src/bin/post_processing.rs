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
use glam::{Mat4, Vec3, Vec4};

// 帧缓冲结构
#[derive(Debug, Clone)]
struct FrameBuffer {
    width: u32,
    height: u32,
    color_buffer: Vec<u32>,
    depth_buffer: Vec<f32>,
}

impl FrameBuffer {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color_buffer: vec![0xFF000000; (width * height) as usize],
            depth_buffer: vec![f32::INFINITY; (width * height) as usize],
        }
    }

    fn clear(&mut self, color: u32) {
        self.color_buffer.fill(color);
        self.depth_buffer.fill(f32::INFINITY);
    }

    fn get_pixel(&self, x: u32, y: u32) -> u32 {
        if x >= self.width || y >= self.height {
            return 0xFF000000;
        }
        self.color_buffer[(y * self.width + x) as usize]
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: u32) {
        if x >= self.width || y >= self.height {
            return;
        }
        self.color_buffer[(y * self.width + x) as usize] = color;
    }
}

// 后处理效果枚举
#[derive(Debug, Clone, Copy)]
enum PostProcessEffect {
    None,
    Blur,
    EdgeDetection,
    Emboss,
    Sharpen,
    Bloom,
    Vintage,
    Vignette,
}

// 后处理器
struct PostProcessor {
    temp_buffer: FrameBuffer,
    current_effect: PostProcessEffect,
    effect_index: usize,
    effects: Vec<PostProcessEffect>,
}

impl PostProcessor {
    fn new(width: u32, height: u32) -> Self {
        Self {
            temp_buffer: FrameBuffer::new(width, height),
            current_effect: PostProcessEffect::None,
            effect_index: 0,
            effects: vec![
                PostProcessEffect::None,
                PostProcessEffect::Blur,
                PostProcessEffect::EdgeDetection,
                PostProcessEffect::Emboss,
                PostProcessEffect::Sharpen,
                PostProcessEffect::Bloom,
                PostProcessEffect::Vintage,
                PostProcessEffect::Vignette,
            ],
        }
    }

    fn next_effect(&mut self) {
        self.effect_index = (self.effect_index + 1) % self.effects.len();
        self.current_effect = self.effects[self.effect_index];
        info!("切换到后处理效果: {:?}", self.current_effect);
    }

    fn apply_effect(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        match self.current_effect {
            PostProcessEffect::None => {
                output.copy_from_slice(&input.color_buffer);
            }
            PostProcessEffect::Blur => {
                self.apply_blur(input, output);
            }
            PostProcessEffect::EdgeDetection => {
                self.apply_edge_detection(input, output);
            }
            PostProcessEffect::Emboss => {
                self.apply_emboss(input, output);
            }
            PostProcessEffect::Sharpen => {
                self.apply_sharpen(input, output);
            }
            PostProcessEffect::Bloom => {
                self.apply_bloom(input, output);
            }
            PostProcessEffect::Vintage => {
                self.apply_vintage(input, output);
            }
            PostProcessEffect::Vignette => {
                self.apply_vignette(input, output);
            }
        }
    }

    fn apply_blur(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        let kernel = [
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        ];

        self.apply_convolution(input, output, &kernel, 3);
    }

    fn apply_edge_detection(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        let kernel = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];

        self.apply_convolution(input, output, &kernel, 3);
    }

    fn apply_emboss(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        let kernel = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];

        self.apply_convolution(input, output, &kernel, 3);

        // 添加偏移以避免负值
        for pixel in output.iter_mut() {
            let r = ((*pixel >> 16) & 0xFF) as f32 + 128.0;
            let g = ((*pixel >> 8) & 0xFF) as f32 + 128.0;
            let b = (*pixel & 0xFF) as f32 + 128.0;

            let r = r.min(255.0).max(0.0) as u32;
            let g = g.min(255.0).max(0.0) as u32;
            let b = b.min(255.0).max(0.0) as u32;

            *pixel = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
    }

    fn apply_sharpen(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        let kernel = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];

        self.apply_convolution(input, output, &kernel, 3);
    }

    fn apply_bloom(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        // 简化的bloom效果
        // 首先提取亮区域
        for (i, &pixel) in input.color_buffer.iter().enumerate() {
            let r = ((pixel >> 16) & 0xFF) as f32 / 255.0;
            let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
            let b = (pixel & 0xFF) as f32 / 255.0;

            let brightness = (r + g + b) / 3.0;

            if brightness > 0.7 {
                // 保持亮像素
                output[i] = pixel;
            } else {
                // 暗像素设为黑色
                output[i] = 0xFF000000;
            }
        }

        // 应用模糊
        let mut temp_output = vec![0u32; output.len()];
        temp_output.copy_from_slice(output);

        self.temp_buffer.color_buffer.copy_from_slice(&temp_output);

        // 创建临时缓冲区来避免借用问题
        let temp_buffer = self.temp_buffer.clone();
        self.apply_blur(&temp_buffer, &mut temp_output);

        // 与原图混合
        for (i, &original) in input.color_buffer.iter().enumerate() {
            let bloom_pixel = temp_output[i];

            let orig_r = ((original >> 16) & 0xFF) as f32;
            let orig_g = ((original >> 8) & 0xFF) as f32;
            let orig_b = (original & 0xFF) as f32;

            let bloom_r = ((bloom_pixel >> 16) & 0xFF) as f32;
            let bloom_g = ((bloom_pixel >> 8) & 0xFF) as f32;
            let bloom_b = (bloom_pixel & 0xFF) as f32;

            let final_r = (orig_r + bloom_r * 0.5).min(255.0) as u32;
            let final_g = (orig_g + bloom_g * 0.5).min(255.0) as u32;
            let final_b = (orig_b + bloom_b * 0.5).min(255.0) as u32;

            output[i] = 0xFF000000 | (final_r << 16) | (final_g << 8) | final_b;
        }
    }

    fn apply_vintage(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        for (i, &pixel) in input.color_buffer.iter().enumerate() {
            let r = ((pixel >> 16) & 0xFF) as f32 / 255.0;
            let g = ((pixel >> 8) & 0xFF) as f32 / 255.0;
            let b = (pixel & 0xFF) as f32 / 255.0;

            // 棕褐色调
            let new_r = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
            let new_g = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
            let new_b = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);

            let final_r = (new_r * 255.0) as u32;
            let final_g = (new_g * 255.0) as u32;
            let final_b = (new_b * 255.0) as u32;

            output[i] = 0xFF000000 | (final_r << 16) | (final_g << 8) | final_b;
        }
    }

    fn apply_vignette(&mut self, input: &FrameBuffer, output: &mut [u32]) {
        let center_x = input.width as f32 * 0.5;
        let center_y = input.height as f32 * 0.5;
        let max_dist = (center_x * center_x + center_y * center_y).sqrt();

        for y in 0..input.height {
            for x in 0..input.width {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist = (dx * dx + dy * dy).sqrt();

                let vignette_factor = 1.0 - (dist / max_dist).powf(2.0);
                let vignette_factor = vignette_factor.max(0.2);

                let index = (y * input.width + x) as usize;
                let pixel = input.color_buffer[index];

                let r = ((pixel >> 16) & 0xFF) as f32 * vignette_factor;
                let g = ((pixel >> 8) & 0xFF) as f32 * vignette_factor;
                let b = (pixel & 0xFF) as f32 * vignette_factor;

                let final_r = r.min(255.0) as u32;
                let final_g = g.min(255.0) as u32;
                let final_b = b.min(255.0) as u32;

                output[index] = 0xFF000000 | (final_r << 16) | (final_g << 8) | final_b;
            }
        }
    }

    fn apply_convolution(
        &mut self, input: &FrameBuffer, output: &mut [u32], kernel: &[f32], size: usize,
    ) {
        let half_size = size / 2;

        for y in 0..input.height {
            for x in 0..input.width {
                let mut r_sum = 0.0;
                let mut g_sum = 0.0;
                let mut b_sum = 0.0;

                for ky in 0..size {
                    for kx in 0..size {
                        let sample_x = x as i32 + kx as i32 - half_size as i32;
                        let sample_y = y as i32 + ky as i32 - half_size as i32;

                        let sample_x = sample_x.max(0).min(input.width as i32 - 1) as u32;
                        let sample_y = sample_y.max(0).min(input.height as i32 - 1) as u32;

                        let pixel = input.get_pixel(sample_x, sample_y);
                        let r = ((pixel >> 16) & 0xFF) as f32;
                        let g = ((pixel >> 8) & 0xFF) as f32;
                        let b = (pixel & 0xFF) as f32;

                        let weight = kernel[ky * size + kx];
                        r_sum += r * weight;
                        g_sum += g * weight;
                        b_sum += b * weight;
                    }
                }

                let final_r = r_sum.max(0.0).min(255.0) as u32;
                let final_g = g_sum.max(0.0).min(255.0) as u32;
                let final_b = b_sum.max(0.0).min(255.0) as u32;

                let index = (y * input.width + x) as usize;
                output[index] = 0xFF000000 | (final_r << 16) | (final_g << 8) | final_b;
            }
        }
    }
}

// 简单3D场景渲染器
struct SceneRenderer {
    width: u32,
    height: u32,
    frame_buffer: FrameBuffer,
    view_matrix: Mat4,
    projection_matrix: Mat4,
    camera_pos: Vec3,
}

impl SceneRenderer {
    fn new(width: u32, height: u32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        let projection_matrix =
            Mat4::perspective_rh(std::f32::consts::PI / 4.0, aspect_ratio, 0.1, 100.0);

        let view_matrix = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        Self {
            width,
            height,
            frame_buffer: FrameBuffer::new(width, height),
            view_matrix,
            projection_matrix,
            camera_pos: Vec3::new(0.0, 0.0, 5.0),
        }
    }

    fn update_camera(&mut self, time: f32) {
        let radius = 5.0;
        let x = radius * (time * 0.5).cos();
        let z = radius * (time * 0.5).sin();
        let y = 1.0 + (time * 0.3).sin();

        self.camera_pos = Vec3::new(x, y, z);

        self.view_matrix = Mat4::look_at_rh(
            self.camera_pos,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    fn render_scene(&mut self, time: f32) {
        self.frame_buffer.clear(0xFF111111);

        // 渲染多个彩色立方体
        for i in 0..5 {
            let offset = Vec3::new(
                (i as f32 - 2.0) * 2.0,
                (time * 0.5 + i as f32).sin() * 0.5,
                0.0,
            );

            let model = Mat4::from_translation(offset)
                * Mat4::from_rotation_y(time * 0.3 + i as f32 * 0.7)
                * Mat4::from_rotation_x(time * 0.5 + i as f32 * 0.3);

            let color = match i {
                0 => 0xFFFF0000, // 红
                1 => 0xFF00FF00, // 绿
                2 => 0xFF0000FF, // 蓝
                3 => 0xFFFFFF00, // 黄
                4 => 0xFFFF00FF, // 洋红
                _ => 0xFFFFFFFF,
            };

            self.render_cube(model, color);
        }

        // 渲染地面网格
        self.render_grid(time);
    }

    fn render_cube(&mut self, model: Mat4, color: u32) {
        let vertices = [
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
        ];

        let indices = [
            0, 1, 2, 2, 3, 0, // 前面
            4, 7, 6, 6, 5, 4, // 后面
            0, 4, 5, 5, 1, 0, // 底面
            2, 6, 7, 7, 3, 2, // 顶面
            0, 3, 7, 7, 4, 0, // 左面
            1, 5, 6, 6, 2, 1, // 右面
        ];

        for chunk in indices.chunks(3) {
            let v0 = vertices[chunk[0]];
            let v1 = vertices[chunk[1]];
            let v2 = vertices[chunk[2]];

            self.render_triangle(v0, v1, v2, model, color);
        }
    }

    fn render_triangle(&mut self, v0: Vec3, v1: Vec3, v2: Vec3, model: Mat4, color: u32) {
        let mvp = self.projection_matrix * self.view_matrix * model;

        // 变换顶点
        let v0_clip = mvp * Vec4::new(v0.x, v0.y, v0.z, 1.0);
        let v1_clip = mvp * Vec4::new(v1.x, v1.y, v1.z, 1.0);
        let v2_clip = mvp * Vec4::new(v2.x, v2.y, v2.z, 1.0);

        // 透视除法
        let v0_ndc = Vec3::new(
            v0_clip.x / v0_clip.w,
            v0_clip.y / v0_clip.w,
            v0_clip.z / v0_clip.w,
        );
        let v1_ndc = Vec3::new(
            v1_clip.x / v1_clip.w,
            v1_clip.y / v1_clip.w,
            v1_clip.z / v1_clip.w,
        );
        let v2_ndc = Vec3::new(
            v2_clip.x / v2_clip.w,
            v2_clip.y / v2_clip.w,
            v2_clip.z / v2_clip.w,
        );

        // 屏幕空间变换
        let v0_screen = Vec3::new(
            (v0_ndc.x + 1.0) * 0.5 * self.width as f32,
            (1.0 - v0_ndc.y) * 0.5 * self.height as f32,
            v0_ndc.z,
        );
        let v1_screen = Vec3::new(
            (v1_ndc.x + 1.0) * 0.5 * self.width as f32,
            (1.0 - v1_ndc.y) * 0.5 * self.height as f32,
            v1_ndc.z,
        );
        let v2_screen = Vec3::new(
            (v2_ndc.x + 1.0) * 0.5 * self.width as f32,
            (1.0 - v2_ndc.y) * 0.5 * self.height as f32,
            v2_ndc.z,
        );

        // 简单的三角形光栅化
        let min_x = (v0_screen.x.min(v1_screen.x).min(v2_screen.x).floor() as i32).max(0);
        let max_x = (v0_screen.x.max(v1_screen.x).max(v2_screen.x).ceil() as i32)
            .min(self.width as i32 - 1);
        let min_y = (v0_screen.y.min(v1_screen.y).min(v2_screen.y).floor() as i32).max(0);
        let max_y = (v0_screen.y.max(v1_screen.y).max(v2_screen.y).ceil() as i32)
            .min(self.height as i32 - 1);

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let (w0, w1, w2) = barycentric_coordinates(p, v0_screen, v1_screen, v2_screen);

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let depth = w0 * v0_screen.z + w1 * v1_screen.z + w2 * v2_screen.z;
                    let index = (y * self.width as i32 + x) as usize;

                    if depth < self.frame_buffer.depth_buffer[index] {
                        self.frame_buffer.depth_buffer[index] = depth;
                        self.frame_buffer.color_buffer[index] = color;
                    }
                }
            }
        }
    }

    fn render_grid(&mut self, time: f32) {
        let grid_size = 10;
        let spacing = 0.5;

        for i in 0..=grid_size {
            for j in 0..=grid_size {
                let x = (i as f32 - grid_size as f32 * 0.5) * spacing;
                let z = (j as f32 - grid_size as f32 * 0.5) * spacing;
                let y = -2.0 + (time * 0.3 + x + z).sin() * 0.1;

                if i < grid_size && j < grid_size {
                    let v0 = Vec3::new(x, y, z);
                    let v1 = Vec3::new(x + spacing, y, z);
                    let v2 = Vec3::new(x, y, z + spacing);
                    let v3 = Vec3::new(x + spacing, y, z + spacing);

                    let model = Mat4::IDENTITY;
                    let color = 0xFF444444;

                    self.render_triangle(v0, v1, v2, model, color);
                    self.render_triangle(v1, v3, v2, model, color);
                }
            }
        }
    }

    fn get_frame_buffer(&self) -> &FrameBuffer {
        &self.frame_buffer
    }
}

fn barycentric_coordinates(p: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> (f32, f32, f32) {
    let v0v1 = v1 - v0;
    let v0v2 = v2 - v0;
    let v0p = p - v0;

    let dot00 = v0v2.dot(v0v2);
    let dot01 = v0v2.dot(v0v1);
    let dot02 = v0v2.dot(v0p);
    let dot11 = v0v1.dot(v0v1);
    let dot12 = v0v1.dot(v0p);

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    (1.0 - u - v, v, u)
}

// 应用程序结构
struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    scene_renderer: Option<SceneRenderer>,
    post_processor: Option<PostProcessor>,
    start_time: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            scene_renderer: None,
            post_processor: None,
            start_time: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                winit::window::WindowAttributes::default()
                    .with_title("Post-Processing Effects - 后处理效果（按空格切换）")
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

        let scene_renderer = SceneRenderer::new(size.width, size.height);
        let post_processor = PostProcessor::new(size.width, size.height);

        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
        self.scene_renderer = Some(scene_renderer);
        self.post_processor = Some(post_processor);

        info!("后处理效果窗口已创建！按空格键切换不同效果。");
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
                        Key::Named(NamedKey::Space) => {
                            if let Some(post_processor) = &mut self.post_processor {
                                post_processor.next_effect();
                            }
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
                    if let Some(scene_renderer) = &mut self.scene_renderer {
                        *scene_renderer = SceneRenderer::new(size.width, size.height);
                    }
                    if let Some(post_processor) = &mut self.post_processor {
                        *post_processor = PostProcessor::new(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(scene_renderer), Some(post_processor)) = (
                    &mut self.surface,
                    &mut self.scene_renderer,
                    &mut self.post_processor,
                ) {
                    let elapsed = self.start_time.elapsed().as_secs_f32();

                    // 更新相机
                    scene_renderer.update_camera(elapsed);

                    // 渲染场景到帧缓冲
                    scene_renderer.render_scene(elapsed);

                    // 应用后处理效果
                    let mut buffer = surface.buffer_mut().unwrap();
                    post_processor.apply_effect(scene_renderer.get_frame_buffer(), &mut buffer);

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
