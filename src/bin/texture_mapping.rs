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

// 简单的纹理结构
#[derive(Debug, Clone)]
struct Texture {
    width: u32,
    height: u32,
    data: Vec<u32>, // RGBA 颜色数据
}

impl Texture {
    // 创建一个简单的棋盘纹理
    fn create_checkerboard(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size) as usize);

        for y in 0..size {
            for x in 0..size {
                let checker_size = size / 8;
                let is_black = ((x / checker_size) + (y / checker_size)) % 2 == 0;
                let color = if is_black { 0xFF000000 } else { 0xFFFFFFFF };
                data.push(color);
            }
        }

        Self {
            width: size,
            height: size,
            data,
        }
    }

    // 创建一个渐变纹理
    fn create_gradient(width: u32, height: u32) -> Self {
        let mut data = Vec::with_capacity((width * height) as usize);

        for y in 0..height {
            for x in 0..width {
                let r = (x as f32 / width as f32 * 255.0) as u32;
                let g = (y as f32 / height as f32 * 255.0) as u32;
                let b = 128;
                let a = 255;

                let color = (a << 24) | (r << 16) | (g << 8) | b;
                data.push(color);
            }
        }

        Self {
            width,
            height,
            data,
        }
    }

    // 采样纹理（双线性过滤）
    fn sample(&self, u: f32, v: f32) -> u32 {
        // 将UV坐标映射到纹理坐标
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let x = u * (self.width - 1) as f32;
        let y = v * (self.height - 1) as f32;

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        // 获取四个角的颜色
        let c00 = self.data[(y0 * self.width + x0) as usize];
        let c10 = self.data[(y0 * self.width + x1) as usize];
        let c01 = self.data[(y1 * self.width + x0) as usize];
        let c11 = self.data[(y1 * self.width + x1) as usize];

        // 双线性插值
        self.lerp_color(
            self.lerp_color(c00, c10, fx),
            self.lerp_color(c01, c11, fx),
            fy,
        )
    }

    // 颜色插值
    fn lerp_color(&self, c0: u32, c1: u32, t: f32) -> u32 {
        let a0 = ((c0 >> 24) & 0xFF) as f32;
        let r0 = ((c0 >> 16) & 0xFF) as f32;
        let g0 = ((c0 >> 8) & 0xFF) as f32;
        let b0 = (c0 & 0xFF) as f32;

        let a1 = ((c1 >> 24) & 0xFF) as f32;
        let r1 = ((c1 >> 16) & 0xFF) as f32;
        let g1 = ((c1 >> 8) & 0xFF) as f32;
        let b1 = (c1 & 0xFF) as f32;

        let a = (a0 + (a1 - a0) * t) as u32;
        let r = (r0 + (r1 - r0) * t) as u32;
        let g = (g0 + (g1 - g0) * t) as u32;
        let b = (b0 + (b1 - b0) * t) as u32;

        (a << 24) | (r << 16) | (g << 8) | b
    }
}

// 带纹理坐标的顶点结构
#[derive(Debug, Clone, Copy)]
struct TexturedVertex {
    position: Vec3, // 世界空间位置
    uv: Vec2,       // 纹理坐标
}

// 经过变换的顶点（在裁剪空间）
#[derive(Debug, Clone, Copy)]
struct TransformedTexturedVertex {
    clip_pos: Vec4, // 裁剪空间坐标
    uv: Vec2,       // 纹理坐标
}

// 屏幕空间顶点
#[derive(Debug, Clone, Copy)]
struct ScreenTexturedVertex {
    x: f32,
    y: f32,
    z: f32, // 深度值
    u: f32, // 纹理U坐标
    v: f32, // 纹理V坐标
}

// 三角形结构
#[derive(Debug)]
struct TexturedTriangle {
    v0: TexturedVertex,
    v1: TexturedVertex,
    v2: TexturedVertex,
}

impl TexturedTriangle {
    fn new(v0: TexturedVertex, v1: TexturedVertex, v2: TexturedVertex) -> Self {
        Self { v0, v1, v2 }
    }
}

// 相机结构
#[derive(Debug)]
struct Camera {
    position: Vec3,
    target: Vec3,
    up: Vec3,
    fov: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

impl Camera {
    fn new(width: f32, height: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov: 60.0_f32.to_radians(),
            aspect: width / height,
            near: 0.1,
            far: 100.0,
        }
    }

    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    fn mvp_matrix(&self, model: Mat4) -> Mat4 {
        self.projection_matrix() * self.view_matrix() * model
    }
}

// 纹理映射光栅化器
struct TextureRasterizer {
    width: u32,
    height: u32,
    camera: Camera,
    depth_buffer: Vec<f32>,
    textures: [Texture; 2],
    current_texture_index: usize,
}

impl TextureRasterizer {
    fn new(width: u32, height: u32) -> Self {
        let camera = Camera::new(width as f32, height as f32);
        let depth_buffer = vec![1.0; (width * height) as usize];
        let checkerboard = Texture::create_checkerboard(256);
        let gradient = Texture::create_gradient(256, 256);

        Self {
            width,
            height,
            camera,
            depth_buffer,
            textures: [checkerboard, gradient],
            current_texture_index: 0,
        }
    }

    fn switch_texture(&mut self) {
        self.current_texture_index = (self.current_texture_index + 1) % self.textures.len();
    }

    // 顶点着色器
    fn vertex_shader(&self, vertex: &TexturedVertex, mvp: Mat4) -> TransformedTexturedVertex {
        let world_pos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let clip_pos = mvp * world_pos;

        TransformedTexturedVertex {
            clip_pos,
            uv: vertex.uv,
        }
    }

    // 透视除法
    fn perspective_divide(
        &self, vertex: &TransformedTexturedVertex,
    ) -> Option<ScreenTexturedVertex> {
        if vertex.clip_pos.w <= 0.0 {
            return None;
        }

        let ndc = vertex.clip_pos / vertex.clip_pos.w;

        // 裁剪检查
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 {
            return None;
        }

        // 转换到屏幕空间
        let screen_x = (ndc.x + 1.0) * 0.5 * self.width as f32;
        let screen_y = (1.0 - ndc.y) * 0.5 * self.height as f32;
        let screen_z = (ndc.z + 1.0) * 0.5; // 0-1 范围

        Some(ScreenTexturedVertex {
            x: screen_x,
            y: screen_y,
            z: screen_z,
            u: vertex.uv.x,
            v: vertex.uv.y,
        })
    }

    // 重心坐标计算
    fn barycentric(
        &self, p: (f32, f32), v0: (f32, f32), v1: (f32, f32), v2: (f32, f32),
    ) -> (f32, f32, f32) {
        let (px, py) = p;
        let (v0x, v0y) = v0;
        let (v1x, v1y) = v1;
        let (v2x, v2y) = v2;

        let denom = (v1y - v2y) * (v0x - v2x) + (v2x - v1x) * (v0y - v2y);
        if denom.abs() < 1e-8 {
            return (0.0, 0.0, 0.0);
        }

        let w0 = ((v1y - v2y) * (px - v2x) + (v2x - v1x) * (py - v2y)) / denom;
        let w1 = ((v2y - v0y) * (px - v2x) + (v0x - v2x) * (py - v2y)) / denom;
        let w2 = 1.0 - w0 - w1;

        (w0, w1, w2)
    }

    // 光栅化纹理三角形
    fn rasterize_triangle(&mut self, triangle: &TexturedTriangle, model: Mat4, buffer: &mut [u32]) {
        let mvp = self.camera.mvp_matrix(model);

        // 顶点着色器
        let v0_transformed = self.vertex_shader(&triangle.v0, mvp);
        let v1_transformed = self.vertex_shader(&triangle.v1, mvp);
        let v2_transformed = self.vertex_shader(&triangle.v2, mvp);

        // 透视除法
        let v0_screen = if let Some(v) = self.perspective_divide(&v0_transformed) {
            v
        } else {
            return;
        };
        let v1_screen = if let Some(v) = self.perspective_divide(&v1_transformed) {
            v
        } else {
            return;
        };
        let v2_screen = if let Some(v) = self.perspective_divide(&v2_transformed) {
            v
        } else {
            return;
        };

        // 计算包围盒
        let min_x = v0_screen.x.min(v1_screen.x).min(v2_screen.x).max(0.0) as u32;
        let max_x = v0_screen
            .x
            .max(v1_screen.x)
            .max(v2_screen.x)
            .min(self.width as f32 - 1.0) as u32;
        let min_y = v0_screen.y.min(v1_screen.y).min(v2_screen.y).max(0.0) as u32;
        let max_y = v0_screen
            .y
            .max(v1_screen.y)
            .max(v2_screen.y)
            .min(self.height as f32 - 1.0) as u32;

        // 遍历包围盒中的每个像素
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let pixel_center = (x as f32 + 0.5, y as f32 + 0.5);
                let (w0, w1, w2) = self.barycentric(
                    pixel_center,
                    (v0_screen.x, v0_screen.y),
                    (v1_screen.x, v1_screen.y),
                    (v2_screen.x, v2_screen.y),
                );

                // 检查是否在三角形内
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    // 深度测试
                    let depth = w0 * v0_screen.z + w1 * v1_screen.z + w2 * v2_screen.z;
                    let depth_index = (y * self.width + x) as usize;

                    if depth < self.depth_buffer[depth_index] {
                        self.depth_buffer[depth_index] = depth;

                        // 透视正确的纹理坐标插值
                        let w0_correct = w0 / v0_transformed.clip_pos.w;
                        let w1_correct = w1 / v1_transformed.clip_pos.w;
                        let w2_correct = w2 / v2_transformed.clip_pos.w;
                        let w_sum = w0_correct + w1_correct + w2_correct;

                        if w_sum > 0.0 {
                            let u = (w0_correct * v0_screen.u
                                + w1_correct * v1_screen.u
                                + w2_correct * v2_screen.u)
                                / w_sum;
                            let v = (w0_correct * v0_screen.v
                                + w1_correct * v1_screen.v
                                + w2_correct * v2_screen.v)
                                / w_sum;

                            // 纹理采样
                            let color = self.textures[self.current_texture_index].sample(u, v);
                            buffer[depth_index] = color;
                        }
                    }
                }
            }
        }
    }

    fn clear_depth(&mut self) {
        self.depth_buffer.fill(1.0);
    }

    fn update_camera(&mut self, time: f32) {
        let radius = 4.0;
        let x = radius * (time * 0.5).cos();
        let z = radius * (time * 0.5).sin();
        self.camera.position = Vec3::new(x, 1.0, z);
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    rasterizer: Option<TextureRasterizer>,
    triangles: Vec<TexturedTriangle>,
    start_time: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            rasterizer: None,
            triangles: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                winit::window::WindowAttributes::default()
                    .with_title("Texture Mapping - 纹理映射")
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

        let rasterizer = TextureRasterizer::new(size.width, size.height);

        // 创建一个带纹理的立方体
        let triangles = vec![
            // 前面
            TexturedTriangle::new(
                TexturedVertex {
                    position: Vec3::new(-1.0, -1.0, 1.0),
                    uv: Vec2::new(0.0, 1.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, -1.0, 1.0),
                    uv: Vec2::new(1.0, 1.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, 1.0, 1.0),
                    uv: Vec2::new(1.0, 0.0),
                },
            ),
            TexturedTriangle::new(
                TexturedVertex {
                    position: Vec3::new(-1.0, -1.0, 1.0),
                    uv: Vec2::new(0.0, 1.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, 1.0, 1.0),
                    uv: Vec2::new(1.0, 0.0),
                },
                TexturedVertex {
                    position: Vec3::new(-1.0, 1.0, 1.0),
                    uv: Vec2::new(0.0, 0.0),
                },
            ),
            // 后面
            TexturedTriangle::new(
                TexturedVertex {
                    position: Vec3::new(-1.0, -1.0, -1.0),
                    uv: Vec2::new(1.0, 1.0),
                },
                TexturedVertex {
                    position: Vec3::new(-1.0, 1.0, -1.0),
                    uv: Vec2::new(1.0, 0.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, 1.0, -1.0),
                    uv: Vec2::new(0.0, 0.0),
                },
            ),
            TexturedTriangle::new(
                TexturedVertex {
                    position: Vec3::new(-1.0, -1.0, -1.0),
                    uv: Vec2::new(1.0, 1.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, 1.0, -1.0),
                    uv: Vec2::new(0.0, 0.0),
                },
                TexturedVertex {
                    position: Vec3::new(1.0, -1.0, -1.0),
                    uv: Vec2::new(0.0, 1.0),
                },
            ),
        ];

        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
        self.rasterizer = Some(rasterizer);
        self.triangles = triangles;

        info!("Window created with texture mapping support!");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
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
                            if let Some(rasterizer) = &mut self.rasterizer {
                                rasterizer.switch_texture();
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

                    if let Some(rasterizer) = &mut self.rasterizer {
                        *rasterizer = TextureRasterizer::new(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(rasterizer)) = (&mut self.surface, &mut self.rasterizer)
                {
                    let mut buffer = surface.buffer_mut().unwrap();
                    buffer.fill(0xFF000000); // 黑色背景

                    // 清除深度缓冲区
                    rasterizer.clear_depth();

                    // 更新相机
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    rasterizer.update_camera(elapsed);

                    // 渲染立方体
                    let model =
                        Mat4::from_rotation_y(elapsed * 0.5) * Mat4::from_rotation_x(elapsed * 0.3);

                    for triangle in &self.triangles {
                        rasterizer.rasterize_triangle(triangle, model, &mut buffer);
                    }

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
