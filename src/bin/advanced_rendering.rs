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

// 纹理结构
#[derive(Debug, Clone)]
struct Texture {
    width: u32,
    height: u32,
    data: Vec<u32>,
}

impl Texture {
    // 创建法线贴图纹理
    fn create_normal_map(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size) as usize);

        for y in 0..size {
            for x in 0..size {
                // 创建简单的凹凸效果
                let fx = x as f32 / size as f32;
                let fy = y as f32 / size as f32;

                let height = ((fx * 8.0).sin() + (fy * 8.0).sin()) * 0.1;

                // 计算法线
                let dx = ((fx + 1.0 / size as f32) * 8.0).sin() * 0.1 - height;
                let dy = ((fy + 1.0 / size as f32) * 8.0).sin() * 0.1 - height;

                let normal = Vec3::new(-dx, -dy, 1.0).normalize();

                // 将法线编码为颜色 (0-1 -> 0-255)
                let r = ((normal.x + 1.0) * 0.5 * 255.0) as u32;
                let g = ((normal.y + 1.0) * 0.5 * 255.0) as u32;
                let b = ((normal.z + 1.0) * 0.5 * 255.0) as u32;
                let a = 255;

                let color = (a << 24) | (r << 16) | (g << 8) | b;
                data.push(color);
            }
        }

        Self {
            width: size,
            height: size,
            data,
        }
    }

    // 创建彩色纹理
    fn create_color_texture(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size) as usize);

        for y in 0..size {
            for x in 0..size {
                let fx = x as f32 / size as f32;
                let fy = y as f32 / size as f32;

                // 创建彩色图案
                let r = ((fx * 4.0).sin() * 0.5 + 0.5) * 255.0;
                let g = ((fy * 4.0).sin() * 0.5 + 0.5) * 255.0;
                let b = ((fx * fy * 16.0).sin() * 0.5 + 0.5) * 255.0;

                let color = 0xFF000000 | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
                data.push(color);
            }
        }

        Self {
            width: size,
            height: size,
            data,
        }
    }

    // 纹理采样
    fn sample(&self, u: f32, v: f32) -> u32 {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);

        let x = (u * (self.width - 1) as f32) as u32;
        let y = (v * (self.height - 1) as f32) as u32;

        self.data[(y * self.width + x) as usize]
    }

    // 采样法线
    fn sample_normal(&self, u: f32, v: f32) -> Vec3 {
        let color = self.sample(u, v);

        let r = ((color >> 16) & 0xFF) as f32 / 255.0;
        let g = ((color >> 8) & 0xFF) as f32 / 255.0;
        let b = (color & 0xFF) as f32 / 255.0;

        Vec3::new(r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0).normalize()
    }
}

// 材质结构
#[derive(Debug, Clone)]
struct Material {
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
    shininess: f32,
    diffuse_texture: Option<Texture>,
    normal_texture: Option<Texture>,
}

impl Material {
    fn new(ambient: Vec3, diffuse: Vec3, specular: Vec3, shininess: f32) -> Self {
        Self {
            ambient,
            diffuse,
            specular,
            shininess,
            diffuse_texture: None,
            normal_texture: None,
        }
    }

    fn with_textures(mut self, diffuse_texture: Texture, normal_texture: Texture) -> Self {
        self.diffuse_texture = Some(diffuse_texture);
        self.normal_texture = Some(normal_texture);
        self
    }

    fn textured_material() -> Self {
        Self::new(
            Vec3::new(0.2, 0.2, 0.2),
            Vec3::new(0.8, 0.8, 0.8),
            Vec3::new(1.0, 1.0, 1.0),
            64.0,
        )
        .with_textures(
            Texture::create_color_texture(128),
            Texture::create_normal_map(128),
        )
    }
}

// 光源结构
#[derive(Debug, Clone, Copy)]
struct Light {
    position: Vec3,
    ambient: Vec3,
    diffuse: Vec3,
    specular: Vec3,
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
}

// 完整的顶点结构
#[derive(Debug, Clone)]
struct ComplexVertex {
    position: Vec3,
    normal: Vec3,
    tangent: Vec3, // 切线向量（用于法线映射）
    uv: Vec2,
    material: Material,
}

// 变换后的顶点
#[derive(Debug, Clone)]
struct TransformedComplexVertex {
    clip_pos: Vec4,
    world_pos: Vec3,
    world_normal: Vec3,
    world_tangent: Vec3,
    uv: Vec2,
    material: Material,
}

// 屏幕空间顶点
#[derive(Debug, Clone)]
struct ScreenComplexVertex {
    x: f32,
    y: f32,
    z: f32,
    world_pos: Vec3,
    world_normal: Vec3,
    world_tangent: Vec3,
    uv: Vec2,
    material: Material,
}

// 三角形结构
#[derive(Debug)]
struct ComplexTriangle {
    v0: ComplexVertex,
    v1: ComplexVertex,
    v2: ComplexVertex,
}

impl ComplexTriangle {
    fn new(v0: ComplexVertex, v1: ComplexVertex, v2: ComplexVertex) -> Self {
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
            position: Vec3::new(0.0, 2.0, 5.0),
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

// 高级渲染器
struct AdvancedRenderer {
    width: u32,
    height: u32,
    camera: Camera,
    depth_buffer: Vec<f32>,
    lights: Vec<Light>,
}

impl AdvancedRenderer {
    fn new(width: u32, height: u32) -> Self {
        let camera = Camera::new(width as f32, height as f32);
        let depth_buffer = vec![1.0; (width * height) as usize];
        let lights = vec![
            Light::new(
                Vec3::new(3.0, 3.0, 3.0),
                Vec3::new(0.2, 0.2, 0.2),
                Vec3::new(0.8, 0.8, 0.8),
                Vec3::new(1.0, 1.0, 1.0),
            ),
            Light::new(
                Vec3::new(-3.0, 1.0, 2.0),
                Vec3::new(0.1, 0.1, 0.1),
                Vec3::new(0.4, 0.4, 0.6),
                Vec3::new(0.5, 0.5, 0.8),
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
    fn vertex_shader(
        &self, vertex: &ComplexVertex, model: Mat4, mvp: Mat4,
    ) -> TransformedComplexVertex {
        let world_pos_4 =
            model * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let world_pos = Vec3::new(world_pos_4.x, world_pos_4.y, world_pos_4.z);

        let clip_pos =
            mvp * Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);

        // 变换法线和切线
        let normal_matrix = model.inverse().transpose();
        let normal_4 =
            normal_matrix * Vec4::new(vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0);
        let world_normal = Vec3::new(normal_4.x, normal_4.y, normal_4.z).normalize();
        let tangent_4 =
            normal_matrix * Vec4::new(vertex.tangent.x, vertex.tangent.y, vertex.tangent.z, 0.0);
        let world_tangent = Vec3::new(tangent_4.x, tangent_4.y, tangent_4.z).normalize();

        TransformedComplexVertex {
            clip_pos,
            world_pos,
            world_normal,
            world_tangent,
            uv: vertex.uv,
            material: vertex.material.clone(),
        }
    }

    // 透视除法
    fn perspective_divide(&self, vertex: &TransformedComplexVertex) -> Option<ScreenComplexVertex> {
        if vertex.clip_pos.w <= 0.0 {
            return None;
        }

        let ndc = vertex.clip_pos / vertex.clip_pos.w;

        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 {
            return None;
        }

        let screen_x = (ndc.x + 1.0) * 0.5 * self.width as f32;
        let screen_y = (1.0 - ndc.y) * 0.5 * self.height as f32;
        let screen_z = (ndc.z + 1.0) * 0.5;

        Some(ScreenComplexVertex {
            x: screen_x,
            y: screen_y,
            z: screen_z,
            world_pos: vertex.world_pos,
            world_normal: vertex.world_normal,
            world_tangent: vertex.world_tangent,
            uv: vertex.uv,
            material: vertex.material.clone(),
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

    // 高级光照计算（支持法线贴图）
    fn calculate_lighting(
        &self, frag_pos: Vec3, normal: Vec3, tangent: Vec3, uv: Vec2, material: &Material,
    ) -> Vec3 {
        let mut final_normal = normal;

        // 如果有法线贴图，计算TBN矩阵并变换法线
        if let Some(normal_texture) = &material.normal_texture {
            let bitangent = normal.cross(tangent).normalize();
            let tbn = Mat4::from_cols(
                tangent.extend(0.0),
                bitangent.extend(0.0),
                normal.extend(0.0),
                Vec4::W,
            );

            let tangent_normal = normal_texture.sample_normal(uv.x, uv.y);
            let transformed_normal = tbn * tangent_normal.extend(0.0);
            final_normal = Vec3::new(
                transformed_normal.x,
                transformed_normal.y,
                transformed_normal.z,
            )
            .normalize();
        }

        let mut result = Vec3::ZERO;

        for light in &self.lights {
            // 环境光
            let ambient = light.ambient * material.ambient;

            // 获取材质颜色（如果有纹理）
            let material_color = if let Some(diffuse_texture) = &material.diffuse_texture {
                let tex_color = diffuse_texture.sample(uv.x, uv.y);
                let r = ((tex_color >> 16) & 0xFF) as f32 / 255.0;
                let g = ((tex_color >> 8) & 0xFF) as f32 / 255.0;
                let b = (tex_color & 0xFF) as f32 / 255.0;
                Vec3::new(r, g, b)
            } else {
                material.diffuse
            };

            // 漫反射
            let light_dir = (light.position - frag_pos).normalize();
            let diff = final_normal.dot(light_dir).max(0.0);
            let diffuse = light.diffuse * material_color * diff;

            // 镜面反射
            let view_dir = (self.camera.position - frag_pos).normalize();
            let reflect_dir = (-light_dir).reflect(final_normal);
            let spec = view_dir.dot(reflect_dir).max(0.0).powf(material.shininess);
            let specular = light.specular * material.specular * spec;

            result += ambient + diffuse + specular;
        }

        result
    }

    // 向量插值
    fn interpolate_vec3(&self, v0: Vec3, v1: Vec3, v2: Vec3, w0: f32, w1: f32, w2: f32) -> Vec3 {
        v0 * w0 + v1 * w1 + v2 * w2
    }

    // Vec2 插值
    fn interpolate_vec2(&self, v0: Vec2, v1: Vec2, v2: Vec2, w0: f32, w1: f32, w2: f32) -> Vec2 {
        v0 * w0 + v1 * w1 + v2 * w2
    }

    // 光栅化复杂三角形
    fn rasterize_triangle(&mut self, triangle: &ComplexTriangle, model: Mat4, buffer: &mut [u32]) {
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

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
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

                        let frag_tangent = self
                            .interpolate_vec3(
                                v0_screen.world_tangent,
                                v1_screen.world_tangent,
                                v2_screen.world_tangent,
                                w0,
                                w1,
                                w2,
                            )
                            .normalize();

                        let frag_uv = self.interpolate_vec2(
                            v0_screen.uv,
                            v1_screen.uv,
                            v2_screen.uv,
                            w0,
                            w1,
                            w2,
                        );

                        // 计算光照
                        let lighting = self.calculate_lighting(
                            frag_pos,
                            frag_normal,
                            frag_tangent,
                            frag_uv,
                            &v0_screen.material,
                        );

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
        let radius = 6.0;
        let x = radius * (time * 0.3).cos();
        let z = radius * (time * 0.3).sin();
        self.camera.position = Vec3::new(x, 2.0, z);
    }

    fn update_lights(&mut self, time: f32) {
        let radius = 4.0;
        let x = radius * (time * 0.7).cos();
        let z = radius * (time * 0.7).sin();
        self.lights[0].position = Vec3::new(x, 3.0, z);
    }
}

struct App {
    window: Option<Rc<Window>>,
    context: Option<Context<Rc<Window>>>,
    surface: Option<Surface<Rc<Window>, Rc<Window>>>,
    renderer: Option<AdvancedRenderer>,
    triangles: Vec<ComplexTriangle>,
    start_time: std::time::Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            context: None,
            surface: None,
            renderer: None,
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
                    .with_title("Advanced Rendering - 高级渲染")
                    .with_inner_size(PhysicalSize::new(900, 700)),
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

        let renderer = AdvancedRenderer::new(size.width, size.height);

        // 创建带纹理的平面
        let material = Material::textured_material();
        let triangles = vec![
            ComplexTriangle::new(
                ComplexVertex {
                    position: Vec3::new(-2.0, 0.0, -2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(0.0, 0.0),
                    material: material.clone(),
                },
                ComplexVertex {
                    position: Vec3::new(2.0, 0.0, -2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(1.0, 0.0),
                    material: material.clone(),
                },
                ComplexVertex {
                    position: Vec3::new(2.0, 0.0, 2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(1.0, 1.0),
                    material: material.clone(),
                },
            ),
            ComplexTriangle::new(
                ComplexVertex {
                    position: Vec3::new(-2.0, 0.0, -2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(0.0, 0.0),
                    material: material.clone(),
                },
                ComplexVertex {
                    position: Vec3::new(2.0, 0.0, 2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(1.0, 1.0),
                    material: material.clone(),
                },
                ComplexVertex {
                    position: Vec3::new(-2.0, 0.0, 2.0),
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    tangent: Vec3::new(1.0, 0.0, 0.0),
                    uv: Vec2::new(0.0, 1.0),
                    material: material.clone(),
                },
            ),
        ];

        self.window = Some(window);
        self.context = Some(context);
        self.surface = Some(surface);
        self.renderer = Some(renderer);
        self.triangles = triangles;

        info!("Window created with advanced rendering support!");
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

                    if let Some(renderer) = &mut self.renderer {
                        *renderer = AdvancedRenderer::new(size.width, size.height);
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(surface), Some(renderer)) = (&mut self.surface, &mut self.renderer) {
                    let mut buffer = surface.buffer_mut().unwrap();
                    buffer.fill(0xFF001122); // 深蓝色背景

                    // 清除深度缓冲区
                    renderer.clear_depth();

                    // 更新相机和光源
                    let elapsed = self.start_time.elapsed().as_secs_f32();
                    renderer.update_camera(elapsed);
                    renderer.update_lights(elapsed);

                    // 渲染场景
                    let model = Mat4::IDENTITY;

                    for triangle in &self.triangles {
                        renderer.rasterize_triangle(triangle, model, &mut buffer);
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
