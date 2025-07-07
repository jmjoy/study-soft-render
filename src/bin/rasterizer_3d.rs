pub use anyhow::Result as Anyhow;
use softbuffer::{Context, Surface};
use std::{num::NonZeroU32, rc::Rc};
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

// 引入 glam 用于向量和矩阵计算
use glam::{Mat4, Vec3, Vec4};

// 三维顶点结构
#[derive(Debug, Clone, Copy)]
struct Vertex3D {
    position: Vec3, // 世界空间位置
    color: u32,     // 颜色
}

// 经过变换的顶点（在裁剪空间）
#[derive(Debug, Clone, Copy)]
struct TransformedVertex {
    clip_pos: Vec4, // 裁剪空间坐标 (x, y, z, w)
    color: u32,     // 颜色
}

// 屏幕空间顶点
#[derive(Debug, Clone, Copy)]
struct ScreenVertex {
    x: f32,
    y: f32,
    z: f32, // 深度值 (0-1)
    color: u32,
}

// 三角形结构
#[derive(Debug)]
struct Triangle3D {
    v0: Vertex3D,
    v1: Vertex3D,
    v2: Vertex3D,
}

impl Triangle3D {
    fn new(v0: Vertex3D, v1: Vertex3D, v2: Vertex3D) -> Self {
        Self { v0, v1, v2 }
    }
}

// 相机结构
#[derive(Debug)]
struct Camera {
    position: Vec3, // 相机在世界空间中的位置
    target: Vec3,   // 相机观察的目标点
    up: Vec3,       // 相机的上方向向量
    fov: f32,       // 视野角度（弧度）
    aspect: f32,    // 宽高比
    near: f32,      // 近裁剪面
    far: f32,       // 远裁剪面
}

impl Camera {
    fn new(width: f32, height: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov: 45.0_f32.to_radians(),
            aspect: width / height,
            near: 0.1,
            far: 100.0,
        }
    }

    // 计算视图矩阵
    fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.target, self.up)
    }

    // 计算投影矩阵
    fn projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, self.aspect, self.near, self.far)
    }

    // 计算 MVP 矩阵
    fn mvp_matrix(&self, model: Mat4) -> Mat4 {
        self.projection_matrix() * self.view_matrix() * model
    }
}

// 3D 光栅化器
struct Rasterizer3D {
    width: u32,
    height: u32,
    camera: Camera,
    depth_buffer: Vec<f32>, // 深度缓冲区
}

impl Rasterizer3D {
    fn new(width: u32, height: u32) -> Self {
        let camera = Camera::new(width as f32, height as f32);
        let depth_buffer = vec![1.0; (width * height) as usize]; // 初始化为最大深度

        Self {
            width,
            height,
            camera,
            depth_buffer,
        }
    }

    // 顶点着色器：变换顶点到裁剪空间
    fn vertex_shader(&self, vertex: &Vertex3D, mvp: Mat4) -> TransformedVertex {
        let world_pos = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let clip_pos = mvp * world_pos;

        TransformedVertex {
            clip_pos,
            color: vertex.color,
        }
    }

    // 透视除法：从裁剪空间转到NDC空间，再转到屏幕空间
    fn perspective_divide(&self, vertex: &TransformedVertex) -> Option<ScreenVertex> {
        let w = vertex.clip_pos.w;

        // 检查是否在视锥体内
        if w <= 0.0 {
            return None;
        }

        // 透视除法得到NDC坐标 (-1 到 1)
        let ndc_x = vertex.clip_pos.x / w;
        let ndc_y = vertex.clip_pos.y / w;
        let ndc_z = vertex.clip_pos.z / w;

        // 裁剪测试：检查顶点是否在NDC空间的有效范围内
        // 注意：glam库的perspective_rh使用DirectX约定，NDC深度范围是
        // [0, 1]而不是OpenGL的[-1,1]
        // - ndc_x, ndc_y: 范围[-1, 1]，超出此范围的顶点在屏幕外
        // - ndc_z: 范围[0, 1]，0表示近平面，1表示远平面，超出此范围的顶点应被裁剪
        if ndc_x < -1.0 || ndc_x > 1.0 || ndc_y < -1.0 || ndc_y > 1.0 || ndc_z < 0.0 || ndc_z > 1.0
        {
            return None;
        }

        // 转换到屏幕空间
        let screen_x = (ndc_x + 1.0) * 0.5 * self.width as f32;
        let screen_y = (1.0 - ndc_y) * 0.5 * self.height as f32; // 翻转Y轴
        let screen_z = ndc_z; // 深度值保持在 0-1 范围

        Some(ScreenVertex {
            x: screen_x,
            y: screen_y,
            z: screen_z,
            color: vertex.color,
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

    // 混合颜色
    fn blend_color(&self, c0: u32, c1: u32, c2: u32, w0: f32, w1: f32, w2: f32) -> u32 {
        let extract_rgba = |color: u32| -> (f32, f32, f32, f32) {
            let r = ((color >> 16) & 0xFF) as f32 / 255.0;
            let g = ((color >> 8) & 0xFF) as f32 / 255.0;
            let b = (color & 0xFF) as f32 / 255.0;
            let a = ((color >> 24) & 0xFF) as f32 / 255.0;
            (r, g, b, a)
        };

        let (r0, g0, b0, a0) = extract_rgba(c0);
        let (r1, g1, b1, a1) = extract_rgba(c1);
        let (r2, g2, b2, a2) = extract_rgba(c2);

        let r = (r0 * w0 + r1 * w1 + r2 * w2).clamp(0.0, 1.0);
        let g = (g0 * w0 + g1 * w1 + g2 * w2).clamp(0.0, 1.0);
        let b = (b0 * w0 + b1 * w1 + b2 * w2).clamp(0.0, 1.0);
        let a = (a0 * w0 + a1 * w1 + a2 * w2).clamp(0.0, 1.0);

        ((a * 255.0) as u32) << 24
            | ((r * 255.0) as u32) << 16
            | ((g * 255.0) as u32) << 8
            | (b * 255.0) as u32
    }

    // 光栅化3D三角形
    fn rasterize_triangle(&mut self, triangle: &Triangle3D, model: Mat4, buffer: &mut [u32]) {
        let mvp = self.camera.mvp_matrix(model);

        // 顶点着色器处理
        let transformed_v0 = self.vertex_shader(&triangle.v0, mvp);
        let transformed_v1 = self.vertex_shader(&triangle.v1, mvp);
        let transformed_v2 = self.vertex_shader(&triangle.v2, mvp);

        // 透视除法和裁剪
        let screen_v0 = match self.perspective_divide(&transformed_v0) {
            Some(v) => v,
            None => return,
        };
        let screen_v1 = match self.perspective_divide(&transformed_v1) {
            Some(v) => v,
            None => return,
        };
        let screen_v2 = match self.perspective_divide(&transformed_v2) {
            Some(v) => v,
            None => return,
        };

        // 获取三角形的包围盒
        let min_x = screen_v0.x.min(screen_v1.x).min(screen_v2.x).max(0.0) as u32;
        let max_x = screen_v0
            .x
            .max(screen_v1.x)
            .max(screen_v2.x)
            .min(self.width as f32 - 1.0) as u32;
        let min_y = screen_v0.y.min(screen_v1.y).min(screen_v2.y).max(0.0) as u32;
        let max_y = screen_v0
            .y
            .max(screen_v1.y)
            .max(screen_v2.y)
            .min(self.height as f32 - 1.0) as u32;

        // 遍历包围盒中的每个像素
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let v0 = (screen_v0.x, screen_v0.y);
                let v1 = (screen_v1.x, screen_v1.y);
                let v2 = (screen_v2.x, screen_v2.y);

                let (w0, w1, w2) = self.barycentric(p, v0, v1, v2);

                // 检查点是否在三角形内
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    // 插值深度值
                    let depth = screen_v0.z * w0 + screen_v1.z * w1 + screen_v2.z * w2;

                    let index = (y * self.width + x) as usize;
                    if index < buffer.len() && index < self.depth_buffer.len() {
                        // 深度测试
                        if depth < self.depth_buffer[index] {
                            self.depth_buffer[index] = depth;

                            // 插值颜色
                            let color = self.blend_color(
                                screen_v0.color,
                                screen_v1.color,
                                screen_v2.color,
                                w0,
                                w1,
                                w2,
                            );
                            buffer[index] = color;
                        }
                    }
                }
            }
        }
    }

    // 清除深度缓冲区
    fn clear_depth(&mut self) {
        for depth in &mut self.depth_buffer {
            *depth = 1.0;
        }
    }

    // 更新相机位置（简单的旋转动画）
    fn update_camera(&mut self, time: f32) {
        let radius = 5.0;
        self.camera.position = Vec3::new(radius * time.cos(), 2.0, radius * time.sin());
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    rasterizer: Option<Rasterizer3D>,
    triangles: Vec<Triangle3D>,
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
                Window::default_attributes()
                    .with_title("3D Rasterizer - MVP Transform Demo")
                    .with_resizable(false)
                    .with_inner_size(PhysicalSize {
                        width: 800,
                        height: 600,
                    }),
            )
            .unwrap();
        let window = Rc::new(window);
        let context = softbuffer::Context::new(window.clone()).unwrap();
        let mut surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

        let PhysicalSize { width, height } = window.inner_size();
        info!(width, height, "window inner size");
        surface
            .resize(
                NonZeroU32::new(width).unwrap(),
                NonZeroU32::new(height).unwrap(),
            )
            .unwrap();

        // 创建3D光栅化器
        let rasterizer = Rasterizer3D::new(width, height);

        // 创建一个彩色立方体的三角形
        let mut triangles = Vec::new();

        // 立方体的顶点
        let vertices = [
            // 前面
            Vertex3D {
                position: Vec3::new(-1.0, -1.0, 1.0),
                color: 0xFFFF0000,
            }, // 红色
            Vertex3D {
                position: Vec3::new(1.0, -1.0, 1.0),
                color: 0xFF00FF00,
            }, // 绿色
            Vertex3D {
                position: Vec3::new(1.0, 1.0, 1.0),
                color: 0xFF0000FF,
            }, // 蓝色
            Vertex3D {
                position: Vec3::new(-1.0, 1.0, 1.0),
                color: 0xFFFFFF00,
            }, // 黄色
            // 后面
            Vertex3D {
                position: Vec3::new(-1.0, -1.0, -1.0),
                color: 0xFFFF00FF,
            }, // 洋红
            Vertex3D {
                position: Vec3::new(1.0, -1.0, -1.0),
                color: 0xFF00FFFF,
            }, // 青色
            Vertex3D {
                position: Vec3::new(1.0, 1.0, -1.0),
                color: 0xFFFFFFFF,
            }, // 白色
            Vertex3D {
                position: Vec3::new(-1.0, 1.0, -1.0),
                color: 0xFF808080,
            }, // 灰色
        ];

        // 立方体的三角形索引
        let indices = [
            0, 1, 2, 0, 2, 3, // 前面
            4, 6, 5, 4, 7, 6, // 后面
            4, 0, 3, 4, 3, 7, // 左面
            1, 5, 6, 1, 6, 2, // 右面
            3, 2, 6, 3, 6, 7, // 上面
            4, 5, 1, 4, 1, 0, // 下面
        ];

        // 创建三角形
        for chunk in indices.chunks(3) {
            triangles.push(Triangle3D::new(
                vertices[chunk[0]],
                vertices[chunk[1]],
                vertices[chunk[2]],
            ));
        }

        self.window = Some(window.clone());
        self.context = Some(context);
        self.surface = Some(surface);
        self.rasterizer = Some(rasterizer);
        self.triangles = triangles;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let window = self.window.as_ref().unwrap();
        let surface = self.surface.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } if window.id() == id => {
                info!("The close button or escape key was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested if window.id() == id => {
                // 计算时间
                let elapsed = self.start_time.elapsed().as_secs_f32();

                // 渲染
                let mut buffer = surface.buffer_mut().unwrap();

                // 清除背景为深蓝色
                for pixel in buffer.iter_mut() {
                    *pixel = 0xFF001122; // 深蓝色背景
                }

                // 使用3D光栅化器绘制立方体
                if let Some(rasterizer) = &mut self.rasterizer {
                    // 清除深度缓冲区
                    rasterizer.clear_depth();

                    // 更新相机位置（旋转动画）
                    rasterizer.update_camera(elapsed * 0.5);

                    // 创建旋转的模型矩阵
                    let model =
                        Mat4::from_rotation_x(elapsed * 0.8) * Mat4::from_rotation_y(elapsed * 0.6);

                    // 绘制所有三角形
                    for triangle in &self.triangles {
                        rasterizer.rasterize_triangle(triangle, model, &mut buffer);
                    }
                }

                buffer.present().unwrap();

                // 请求重绘以保持动画循环
                window.request_redraw();
            }
            _ => (),
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
