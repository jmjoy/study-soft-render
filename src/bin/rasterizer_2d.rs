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

// 顶点结构
#[derive(Debug, Clone, Copy)]
struct Vertex {
    x: f32,
    y: f32,
    color: u32,
}

// 三角形结构
#[derive(Debug)]
struct Triangle {
    v0: Vertex,
    v1: Vertex,
    v2: Vertex,
}

impl Triangle {
    fn new(v0: Vertex, v1: Vertex, v2: Vertex) -> Self {
        Self { v0, v1, v2 }
    }
}

// 简单的光栅化器
struct Rasterizer {
    width: u32,
    height: u32,
}

impl Rasterizer {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
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

    // 光栅化三角形
    fn rasterize_triangle(&self, triangle: &Triangle, buffer: &mut [u32]) {
        // 获取三角形的包围盒
        let min_x = triangle.v0.x.min(triangle.v1.x).min(triangle.v2.x).max(0.0) as u32;
        let max_x = triangle
            .v0
            .x
            .max(triangle.v1.x)
            .max(triangle.v2.x)
            .min(self.width as f32 - 1.0) as u32;
        let min_y = triangle.v0.y.min(triangle.v1.y).min(triangle.v2.y).max(0.0) as u32;
        let max_y = triangle
            .v0
            .y
            .max(triangle.v1.y)
            .max(triangle.v2.y)
            .min(self.height as f32 - 1.0) as u32;

        // 遍历包围盒中的每个像素
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let p = (x as f32 + 0.5, y as f32 + 0.5);
                let v0 = (triangle.v0.x, triangle.v0.y);
                let v1 = (triangle.v1.x, triangle.v1.y);
                let v2 = (triangle.v2.x, triangle.v2.y);

                let (w0, w1, w2) = self.barycentric(p, v0, v1, v2);

                // 检查点是否在三角形内
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    let color = self.blend_color(
                        triangle.v0.color,
                        triangle.v1.color,
                        triangle.v2.color,
                        w0,
                        w1,
                        w2,
                    );
                    let index = (y * self.width + x) as usize;
                    if index < buffer.len() {
                        buffer[index] = color;
                    }
                }
            }
        }
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    rasterizer: Option<Rasterizer>,
    triangles: Vec<Triangle>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            rasterizer: None,
            triangles: Vec::new(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title(env!("CARGO_PKG_NAME"))
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

        // 创建光栅化器
        let rasterizer = Rasterizer::new(width, height);

        // 创建一些示例三角形
        let mut triangles = Vec::new();

        // 红色三角形
        triangles.push(Triangle::new(
            Vertex {
                x: 400.0,
                y: 100.0,
                color: 0xFFFF0000,
            }, // 红色
            Vertex {
                x: 200.0,
                y: 400.0,
                color: 0xFF00FF00,
            }, // 绿色
            Vertex {
                x: 600.0,
                y: 400.0,
                color: 0xFF0000FF,
            }, // 蓝色
        ));

        // 小的彩色三角形
        triangles.push(Triangle::new(
            Vertex {
                x: 100.0,
                y: 150.0,
                color: 0xFFFFFF00,
            }, // 黄色
            Vertex {
                x: 50.0,
                y: 250.0,
                color: 0xFFFF00FF,
            }, // 洋红
            Vertex {
                x: 150.0,
                y: 250.0,
                color: 0xFF00FFFF,
            }, // 青色
        ));

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
                // 渲染
                let mut buffer = surface.buffer_mut().unwrap();

                // 清除背景为深灰色
                for pixel in buffer.iter_mut() {
                    *pixel = 0xFF202020; // 深灰色背景
                }

                // 使用光栅化器绘制三角形
                if let Some(rasterizer) = &self.rasterizer {
                    for triangle in &self.triangles {
                        rasterizer.rasterize_triangle(triangle, &mut buffer);
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
