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

// 光照模型相关结构

// 材质结构
#[derive(Debug, Clone, Copy)]
struct Material {
    ambient: Vec3,  // 环境光反射系数
    diffuse: Vec3,  // 漫反射系数
    specular: Vec3, // 镜面反射系数
    shininess: f32, // 镜面反射指数
}

impl Material {
    fn new(ambient: Vec3, diffuse: Vec3, specular: Vec3, shininess: f32) -> Self {
        Self {
            ambient,
            diffuse,
            specular,
            shininess,
        }
    }

    // 预定义材质
    fn emerald() -> Self {
        Self::new(
            Vec3::new(0.0215, 0.1745, 0.0215),
            Vec3::new(0.07568, 0.61424, 0.07568),
            Vec3::new(0.633, 0.727811, 0.633),
            0.6 * 128.0,
        )
    }

    fn gold() -> Self {
        Self::new(
            Vec3::new(0.24725, 0.1995, 0.0745),
            Vec3::new(0.75164, 0.60648, 0.22648),
            Vec3::new(0.628281, 0.555802, 0.366065),
            0.4 * 128.0,
        )
    }

    fn silver() -> Self {
        Self::new(
            Vec3::new(0.19225, 0.19225, 0.19225),
            Vec3::new(0.50754, 0.50754, 0.50754),
            Vec3::new(0.508273, 0.508273, 0.508273),
            0.4 * 128.0,
        )
    }
}

// 光源结构
#[derive(Debug, Clone, Copy)]
struct Light {
    position: Vec3, // 光源位置
    ambient: Vec3,  // 环境光强度
    diffuse: Vec3,  // 漫反射光强度
    specular: Vec3, // 镜面反射光强度
}

impl Light {
    fn new(position: Vec3, ambient: Vec3, diffuse: Vec3, specular: Vec3) -> Self {
        Self {
            position,
            ambient,
            diffuse,
            specular,
        }
    }

    fn white_light(position: Vec3) -> Self {
        Self::new(
            position,
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(1.0, 1.0, 1.0),
        )
    }
}

// 带法线的顶点结构
#[derive(Debug, Clone, Copy)]
struct LitVertex {
    position: Vec3,     // 世界空间位置
    normal: Vec3,       // 法线向量
    material: Material, // 材质
}

// 经过变换的顶点
#[derive(Debug, Clone, Copy)]
struct TransformedLitVertex {
    clip_pos: Vec4,     // 裁剪空间坐标
    world_pos: Vec3,    // 世界空间坐标
    world_normal: Vec3, // 世界空间法线
    material: Material, // 材质
}

// 屏幕空间顶点
#[derive(Debug, Clone, Copy)]
struct ScreenLitVertex {
    x: f32,
    y: f32,
    z: f32,             // 深度值
    world_pos: Vec3,    // 世界空间坐标
    world_normal: Vec3, // 世界空间法线
    material: Material, // 材质
}

// 三角形结构
#[derive(Debug)]
struct LitTriangle {
    v0: LitVertex,
    v1: LitVertex,
    v2: LitVertex,
}

impl LitTriangle {
    fn new(v0: LitVertex, v1: LitVertex, v2: LitVertex) -> Self {
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
            position: Vec3::new(0.0, 0.0, 5.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
            fov: 45.0_f32.to_radians(),
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

// 光照计算光栅化器
struct LightingRasterizer {
    width: u32,
    height: u32,
    camera: Camera,
    depth_buffer: Vec<f32>,
    lights: Vec<Light>,
}

impl LightingRasterizer {
    fn new(width: u32, height: u32) -> Self {
        let camera = Camera::new(width as f32, height as f32);
        let depth_buffer = vec![1.0; (width * height) as usize];
        let lights = vec![
            Light::white_light(Vec3::new(5.0, 5.0, 5.0)),
            Light::new(
                Vec3::new(-3.0, 2.0, 3.0),
                Vec3::new(0.1, 0.1, 0.2),
                Vec3::new(0.3, 0.3, 0.8),
                Vec3::new(0.5, 0.5, 1.0),
            ),
        ];

        Self {
            width,
            height,
            camera,
            depth_buffer,
            lights,
        }
    }

    // 顶点着色器
    fn vertex_shader(&self, vertex: &LitVertex, model: Mat4, mvp: Mat4) -> TransformedLitVertex {
        let world_pos_4 =
            model * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let world_pos = Vec3::new(world_pos_4.x, world_pos_4.y, world_pos_4.z);

        let clip_pos =
            mvp * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);

        // 变换法线（使用模型矩阵的逆转置）
        let normal_matrix = model.inverse().transpose();
        let normal_4 =
            normal_matrix * Vec4::new(vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0);
        let world_normal = Vec3::new(normal_4.x, normal_4.y, normal_4.z).normalize();

        TransformedLitVertex {
            clip_pos,
            world_pos,
            world_normal,
            material: vertex.material,
        }
    }

    // 透视除法
    fn perspective_divide(&self, vertex: &TransformedLitVertex) -> Option<ScreenLitVertex> {
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
        let screen_z = (ndc.z + 1.0) * 0.5;

        Some(ScreenLitVertex {
            x: screen_x,
            y: screen_y,
            z: screen_z,
            world_pos: vertex.world_pos,
            world_normal: vertex.world_normal,
            material: vertex.material,
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

    // Blinn-Phong 光照模型
    fn calculate_lighting(&self, frag_pos: Vec3, normal: Vec3, material: Material) -> Vec3 {
        let mut result = Vec3::ZERO;

        for light in &self.lights {
            // 环境光
            let ambient = light.ambient * material.ambient;

            // 漫反射
            let light_dir = (light.position - frag_pos).normalize();
            let diff = normal.dot(light_dir).max(0.0);
            let diffuse = light.diffuse * material.diffuse * diff;

            // 镜面反射（Blinn-Phong）
            let view_dir = (self.camera.position - frag_pos).normalize();
            let halfway_dir = (light_dir + view_dir).normalize();
            let spec = normal.dot(halfway_dir).max(0.0).powf(material.shininess);
            let specular = light.specular * material.specular * spec;

            result += ambient + diffuse + specular;
        }

        result
    }

    // 向量插值
    fn interpolate_vec3(&self, v0: Vec3, v1: Vec3, v2: Vec3, w0: f32, w1: f32, w2: f32) -> Vec3 {
        v0 * w0 + v1 * w1 + v2 * w2
    }

    // 材质插值（这里简化为选择第一个顶点的材质）
    fn interpolate_material(
        &self, m0: Material, _m1: Material, _m2: Material, w0: f32, w1: f32, w2: f32,
    ) -> Material {
        // 简化：根据重心坐标选择最接近的材质
        if w0 >= w1 && w0 >= w2 {
            m0
        } else if w1 >= w2 {
            _m1
        } else {
            _m2
        }
    }

    // 光栅化带光照的三角形
    fn rasterize_triangle(&mut self, triangle: &LitTriangle, model: Mat4, buffer: &mut [u32]) {
        let mvp = self.camera.mvp_matrix(model);

        // 顶点着色器
        let v0_transformed = self.vertex_shader(&triangle.v0, model, mvp);
        let v1_transformed = self.vertex_shader(&triangle.v1, model, mvp);
        let v2_transformed = self.vertex_shader(&triangle.v2, model, mvp);

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

                        // 插值片段属性
                        let frag_pos = self.interpolate_vec3(
                            v0_screen.world_pos,
                            v1_screen.world_pos,
                            v2_screen.world_pos,
                            w0,
                            w1,
                            w2,
                        );

                        let frag_normal = self
                            .interpolate_vec3(
                                v0_screen.world_normal,
                                v1_screen.world_normal,
                                v2_screen.world_normal,
                                w0,
                                w1,
                                w2,
                            )
                            .normalize();

                        let material = self.interpolate_material(
                            v0_screen.material,
                            v1_screen.material,
                            v2_screen.material,
                            w0,
                            w1,
                            w2,
                        );

                        // 计算光照
                        let lighting = self.calculate_lighting(frag_pos, frag_normal, material);

                        // 转换为颜色
                        let r = (lighting.x.clamp(0.0, 1.0) * 255.0) as u32;
                        let g = (lighting.y.clamp(0.0, 1.0) * 255.0) as u32;
                        let b = (lighting.z.clamp(0.0, 1.0) * 255.0) as u32;
                        let color = 0xFF000000 | (r << 16) | (g << 8) | b;

                        buffer[depth_index] = color;
                    }
                }
            }
        }
    }

    fn clear_depth(&mut self) {
        self.depth_buffer.fill(1.0);
    }

    fn update_camera(&mut self, time: f32) {
        let radius = 8.0;
        let x = radius * (time * 0.5).cos();
        let z = radius * (time * 0.5).sin();
        self.camera.position = Vec3::new(x, 3.0, z);
    }

    fn update_lights(&mut self, time: f32) {
        // 让第一个光源围绕原点旋转
        let radius = 6.0;
        let x = radius * (time * 0.8).cos();
        let z = radius * (time * 0.8).sin();
        self.lights[0].position = Vec3::new(x, 4.0, z);
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    rasterizer: Option<LightingRasterizer>,
    triangles: Vec<LitTriangle>,
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
                    .with_title("Lighting - 光照计算")
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

        let rasterizer = LightingRasterizer::new(size.width, size.height);

        // 创建不同材质的球体（简化为四面体）
        let mut triangles = Vec::new();

        // 金色四面体
        let gold_material = Material::gold();
        triangles.push(LitTriangle::new(
            LitVertex {
                position: Vec3::new(-2.0, -1.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                material: gold_material,
            },
            LitVertex {
                position: Vec3::new(-1.0, -1.0, 1.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                material: gold_material,
            },
            LitVertex {
                position: Vec3::new(-1.0, 1.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                material: gold_material,
            },
        ));

        // 翠绿四面体
        let emerald_material = Material::emerald();
        triangles.push(LitTriangle::new(
            LitVertex {
                position: Vec3::new(1.0, -1.0, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                material: emerald_material,
            },
            LitVertex {
                position: Vec3::new(1.0, 1.0, 0.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                material: emerald_material,
            },
            LitVertex {
                position: Vec3::new(2.0, -1.0, 1.0),
                normal: Vec3::new(-1.0, 0.0, 0.0),
                material: emerald_material,
            },
        ));

        // 银色四面体
        let silver_material = Material::silver();
        triangles.push(LitTriangle::new(
            LitVertex {
                position: Vec3::new(0.0, -1.0, -1.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                material: silver_material,
            },
            LitVertex {
                position: Vec3::new(0.0, 1.0, -1.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                material: silver_material,
            },
            LitVertex {
                position: Vec3::new(1.0, 0.0, -2.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                material: silver_material,
            },
        ));

        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
        self.rasterizer = Some(rasterizer);
        self.triangles = triangles;

        info!("Window created with lighting support!");
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
                        *rasterizer = LightingRasterizer::new(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(rasterizer)) = (&mut self.surface, &mut self.rasterizer)
                {
                    let mut buffer = surface.buffer_mut().unwrap();
                    buffer.fill(0xFF111111); // 深灰色背景

                    // 清除深度缓冲区
                    rasterizer.clear_depth();

                    // 更新相机和光源
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    rasterizer.update_camera(elapsed);
                    rasterizer.update_lights(elapsed);

                    // 渲染物体
                    let model = Mat4::from_rotation_y(elapsed * 0.3);

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
