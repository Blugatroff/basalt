mod input;
use basalt::{
    image, label, vk, ColorBlendAttachment, DefaultVertex, DepthStencilInfo, DescriptorSetLayout,
    Device, Frustum, InputAssemblyState, MaterialLoadFn, Mesh, MultiSamplingState, Pipeline,
    PipelineDesc, PipelineLayout, RasterizationState, Renderable, Renderer, ShaderModule, Vertex,
    VertexInfoDescription,
};
use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3, Vector4};
use first_person_camera::FirstPersonCamera;
use gui::EruptEgui;
use input::Input;
use sdl2::{event::Event, EventPump};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex},
};

pub fn create_window() -> (sdl2::Sdl, sdl2::video::Window, sdl2::EventPump) {
    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();
    let window = video
        .window("TEST", 500, 500)
        .vulkan()
        .resizable()
        .build()
        .unwrap();
    let event_pump = sdl.event_pump().unwrap();
    (sdl, window, event_pump)
}

fn create_frustum(fov: f32, far: f32, aspect: f32, near: f32) -> Frustum {
    let hfar = 2.0 * (fov / 2.0).tan() * far;
    let wfar = hfar * aspect;

    let top_plane_normal = cgmath::Vector3::new(0.0, (fov / 2.0).tan(), 1.0)
        .cross(cgmath::Vector3::new(-1.0, 0.0, 0.0));
    let bottom_plane_normal =
        cgmath::Vector3::new(top_plane_normal.x, -top_plane_normal.y, top_plane_normal.z);
    let near_plane_normal = cgmath::Vector3::new(0.0, 0.0, 1.0);
    let left_plane_normal = cgmath::Vector3::new(wfar / 2.0, 0.0, far)
        .cross(cgmath::Vector3::new(0.0, 1.0, 0.0))
        .normalize();
    let right_plane_normal = cgmath::Vector3::new(
        -left_plane_normal.x,
        left_plane_normal.y,
        left_plane_normal.z,
    );
    let far_plane_normal = cgmath::Vector3::new(0.0, 0.0, -1.0);

    Frustum {
        top: top_plane_normal.normalize(),
        bottom: bottom_plane_normal.normalize(),
        right: right_plane_normal.normalize(),
        left: left_plane_normal.normalize(),
        far: far_plane_normal.normalize(),
        near: near_plane_normal.normalize(),
        near_distance: near,
        far_distance: far,
    }
}

fn create_frustum_model(
    renderer: &Renderer,
    far: f32,
    fov: f32,
    aspect: f32,
    near: f32,
    rgb_pipeline: usize,
    line_pipeline: usize,
) -> Vec<Renderable> {
    let hfar = 2.0 * (fov / 2.0).tan() * far;
    let wfar = hfar * aspect;
    let hnear = 2.0 * (fov / 2.0).tan() * near;
    let wnear = hnear * aspect;
    let mut far_plane = {
        let vertices = vec![
            ColorVertex {
                position: cgmath::Vector3::new(-wfar / 2.0, -hfar / 2.0, far),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0x0],
            },
            ColorVertex {
                position: cgmath::Vector3::new(wfar / 2.0, hfar / 2.0, far),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0x0],
            },
            ColorVertex {
                position: cgmath::Vector3::new(-wfar / 2.0, hfar / 2.0, far),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0x0],
            },
            ColorVertex {
                position: cgmath::Vector3::new(wfar / 2.0, -hfar / 2.0, far),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0x0],
            },
        ];
        let indices = vec![0, 1, 2, 3, 1, 0, 1, 0, 2, 1, 3, 0];
        Mesh::new(
            &vertices,
            &indices,
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
            true,
            String::from("Frustum Far Plane"),
        )
    };
    let mut near_plane = {
        let vertices = vec![
            ColorVertex {
                position: cgmath::Vector3::new(-wnear / 2.0, -hnear / 2.0, near),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0xFF],
            },
            ColorVertex {
                position: cgmath::Vector3::new(wnear / 2.0, hnear / 2.0, near),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0xFF],
            },
            ColorVertex {
                position: cgmath::Vector3::new(-wnear / 2.0, hnear / 2.0, near),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0xFF],
            },
            ColorVertex {
                position: cgmath::Vector3::new(wnear / 2.0, -hnear / 2.0, near),
                normal: cgmath::Vector3::new(0.0, 0.0, -1.0),
                color: [0x0, 0xFF, 0xFF],
            },
        ];
        let indices = vec![0, 1, 2, 3, 1, 0, 1, 0, 2, 1, 3, 0];
        Mesh::new(
            &vertices,
            &indices,
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
            true,
            String::from("Frustum Near Plane"),
        )
    };
    let mut lines = {
        let top_plane_normal = cgmath::Vector3::new(0.0, (fov / 2.0).tan(), 1.0)
            .cross(cgmath::Vector3::new(-1.0, 0.0, 0.0));
        let top_plane_line_start = cgmath::Vector3::new(0.0, hfar / 4.0, far / 2.0);
        let top_plane_line_end = top_plane_line_start + top_plane_normal;
        let bottom_plane_normal =
            cgmath::Vector3::new(top_plane_normal.x, -top_plane_normal.y, top_plane_normal.z);
        let bottom_plane_line_start = cgmath::Vector3::new(0.0, -hfar / 4.0, far / 2.0);
        let bottom_plane_line_end = bottom_plane_line_start + bottom_plane_normal;
        let near_plane_normal = cgmath::Vector3::new(0.0, 0.0, 1.0);
        let near_plane_line_start = cgmath::Vector3::new(0.0, 0.0, near);
        let near_plane_line_end = near_plane_line_start + near_plane_normal;
        let left_plane_normal = cgmath::Vector3::new(wfar / 2.0, 0.0, far)
            .cross(cgmath::Vector3::new(0.0, 1.0, 0.0))
            .normalize();
        let left_plane_line_start = cgmath::Vector3::new(wfar / 4.0, 0.0, far / 2.0);
        let left_plane_line_end = left_plane_line_start + left_plane_normal;
        let right_plane_normal = cgmath::Vector3::new(
            -left_plane_normal.x,
            left_plane_normal.y,
            left_plane_normal.z,
        );
        let right_plane_line_start = cgmath::Vector3::new(-wfar / 4.0, 0.0, far / 2.0);
        let right_plane_line_end = right_plane_line_start + right_plane_normal;
        let far_plane_normal = cgmath::Vector3::new(0.0, 0.0, -1.0);
        let far_plane_line_start = cgmath::Vector3::new(0.0, 0.0, far);
        let far_plane_line_end = far_plane_line_start + far_plane_normal;
        let vertices = vec![
            LineVertex {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(-wfar / 2.0, -hfar / 2.0, far),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(wfar / 2.0, hfar / 2.0, far),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(-wfar / 2.0, hfar / 2.0, far),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(0.0, 0.0, 0.0),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: cgmath::Vector3::new(wfar / 2.0, -hfar / 2.0, far),
                color: [0xFF, 0, 0],
            },
            LineVertex {
                position: top_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: top_plane_line_end,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: bottom_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: bottom_plane_line_end,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: near_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: near_plane_line_end,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: left_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: left_plane_line_end,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: right_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: right_plane_line_end,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: far_plane_line_start,
                color: [0x0, 0, 0],
            },
            LineVertex {
                position: far_plane_line_end,
                color: [0x0, 0, 0],
            },
        ];
        assert!(vertices.len() % 2 == 0);
        let indices = (0..vertices.len() as u32).collect::<Vec<u32>>();
        Mesh::new(
            &vertices,
            &indices,
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
            true,
            String::from("Frustum Lines"),
        )
    };
    Mesh::combine_meshes(
        [&mut far_plane, &mut near_plane, &mut lines],
        renderer.allocator().clone(),
        renderer.transfer_context(),
        renderer.device(),
    );
    let far_plane = Renderable {
        mesh: Arc::new(far_plane),
        pipeline: rgb_pipeline,
        transform: cgmath::Matrix4::identity(),
        custom_set: None,
        custom_id: 0,
        uncullable: false,
    };
    let near_plane = Renderable {
        mesh: Arc::new(near_plane),
        pipeline: rgb_pipeline,
        transform: cgmath::Matrix4::identity(),
        custom_set: None,
        custom_id: 0,
        uncullable: false,
    };
    let lines = Renderable {
        mesh: Arc::new(lines),
        pipeline: line_pipeline,
        transform: cgmath::Matrix4::identity(),
        custom_set: None,
        custom_id: 0,
        uncullable: false,
    };
    vec![far_plane, near_plane, lines]
}

struct State {
    frames_in_flight: usize,
    vsync: bool,
    last_time: std::time::Instant,
    start: std::time::Instant,
    egui: EruptEgui,
    renderables: Vec<Renderable>,
    mouse_captured: bool,
    input: Input,
    camera: FirstPersonCamera,
    last_frame_times: std::collections::VecDeque<(f32, f32, f32, f32)>,
    workvec: Vec<Renderable>,
    window: Arc<Mutex<sdl2::video::Window>>,
    event_pump: EventPump,
    sdl: sdl2::Sdl,
    renderer: Renderer,
    egui_enabled: bool,
    tx: std::sync::mpsc::Sender<()>,
    frustum_renderables: Vec<Renderable>,
    frustum_transform: Matrix4<f32>,
}

impl State {
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let (sdl, window, event_pump) = create_window();
        let window = Arc::new(Mutex::new(window));
        sdl.mouse().set_relative_mouse_mode(true);
        let validation_layers = std::env::args().any(|s| s == "v");
        let frames_in_flight = 3;
        let mut renderer = Renderer::new(
            Arc::clone(&window),
            true,
            frames_in_flight,
            validation_layers,
        );
        {
            let mut window = window.try_lock().unwrap();
            window.set_grab(true);
        }
        let vertices = vec![
            DefaultVertex {
                position: cgmath::Vector3::new(1.0, 1.0, 0.0),
                normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
                uv: cgmath::Vector2::new(1.0, 0.0),
            },
            DefaultVertex {
                position: cgmath::Vector3::new(-1.0, 1.0, 0.0),
                normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
                uv: cgmath::Vector2::new(0.0, 1.0),
            },
            DefaultVertex {
                position: cgmath::Vector3::new(0.0, -1.0, 0.0),
                normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
                uv: cgmath::Vector2::new(1.0, 1.0),
            },
        ];
        let threads = std::env::args()
            .nth(1)
            .unwrap_or_else(String::new)
            .parse::<u32>()
            .unwrap_or(0);

        let (tx, rx) = std::sync::mpsc::channel::<()>();
        {
            let device = renderer.device().clone();
            let descriptor_set_manager = renderer.descriptor_set_manager().clone();
            let sampler = renderer.default_sampler().clone();
            let image_loader = renderer.image_loader().clone();
            let transfer_context = renderer.transfer_context().clone();
            let allocator = renderer.allocator().clone();
            std::thread::spawn(move || {
                let sender: Vec<std::sync::mpsc::Sender<()>> = (0..threads)
                    .map(|_| {
                        let device = device.clone();
                        let descriptor_set_manager = descriptor_set_manager.clone();
                        let sampler = sampler.clone();
                        let image_loader = image_loader.clone();
                        let transfer_context = transfer_context.clone();
                        let allocator = allocator.clone();
                        let (tx, rx) = std::sync::mpsc::channel::<()>();
                        std::thread::spawn(move || {
                            let width = 128;
                            let height = 128;
                            let data = vec![0u8; width * height * 4];
                            for _ in 0..300 {
                                if rx.try_iter().count() > 0 {
                                    break;
                                }
                                std::thread::sleep(std::time::Duration::from_millis(100));
                                let mesh = Mesh::load::<ColorVertex>(
                                    allocator.clone(),
                                    "./assets/suzanne.obj".into(),
                                    &transfer_context,
                                    &device,
                                )
                                .unwrap();
                                let image = image::Allocated::load(
                                    &image_loader,
                                    &data,
                                    width as u32,
                                    height as u32,
                                    String::from("ConcurrencyTestImage"),
                                );
                                image::Texture::new(
                                    &device,
                                    &descriptor_set_manager,
                                    image,
                                    sampler.clone(),
                                );
                                drop(mesh);
                            }
                        });
                        tx
                    })
                    .collect();
                rx.recv().unwrap();
                for sender in sender {
                    sender.send(()).ok();
                }
            });
        }
        let rgb_pipeline = renderer.register_pipeline(rgb_pipeline(renderer.device()));
        let mesh_pipeline = renderer.register_pipeline(mesh_pipeline(renderer.device()));
        let line_pipeline = renderer.register_pipeline(line_pipeline(renderer.device()));
        let mut triangle_mesh = Mesh::new(
            &vertices,
            &[0, 1, 2, 2, 1, 0],
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
            true,
            String::from("TriangleMesh"),
        );
        let mut suzanne_mesh = Mesh::load::<ColorVertex>(
            renderer.allocator().clone(),
            "./assets/suzanne.obj".into(),
            renderer.transfer_context(),
            renderer.device(),
        )
        .unwrap()
        .swap_remove(0);
        Mesh::combine_meshes(
            [&mut suzanne_mesh, &mut triangle_mesh],
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
        );
        let image = image::Allocated::open(
            renderer.image_loader(),
            &PathBuf::from("./assets/lost_empire-RGBA.png"),
        );
        let mut renderables = Vec::new();
        let sampler = renderer.default_sampler().clone();
        let texture = Arc::new(image::Texture::new(
            &renderer.device().clone(),
            renderer.descriptor_set_manager(),
            image,
            sampler,
        ));
        let mut suzanne = Renderable {
            mesh: Arc::new(suzanne_mesh),
            pipeline: rgb_pipeline,
            transform: cgmath::Matrix4::identity(),
            custom_set: None,
            custom_id: 0,
            uncullable: false,
        };

        let s = 10;
        for x in -s..s {
            for y in -s..s {
                for z in -s..s {
                    suzanne.transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                        x as f32,
                        y as f32 - 15.0,
                        z as f32,
                    )) * cgmath::Matrix4::from_scale(1.5);
                    renderables.push(suzanne.clone());
                }
            }
        }

        let empire_meshes = Mesh::load::<DefaultVertex>(
            renderer.allocator().clone(),
            "./assets/lost_empire.obj".into(),
            renderer.transfer_context(),
            renderer.device(),
        )
        .unwrap();
        let empire = empire_meshes.into_iter().map(|mesh| Renderable {
            mesh: Arc::new(mesh),
            pipeline: mesh_pipeline,
            transform: cgmath::Matrix4::identity(),
            custom_set: Some(texture.set.clone()),
            custom_id: 0,
            uncullable: false,
        });

        let mut triangle = Renderable {
            mesh: Arc::new(triangle_mesh),
            pipeline: mesh_pipeline,
            transform: cgmath::Matrix4::from_scale(1.0),
            custom_set: Some(texture.set.clone()),
            custom_id: 0,
            uncullable: false,
        };
        triangle.transform =
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(-3.0, 0.0, 0.0))
                * cgmath::Matrix4::from_scale(2.0);
        renderables.push(triangle.clone());
        triangle.transform =
            cgmath::Matrix4::from_translation(cgmath::Vector3::new(-8.0, 0.0, 0.0))
                * cgmath::Matrix4::from_scale(4.0);
        renderables.push(triangle);

        for mut empire in empire {
            empire.transform =
                cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.0));
            renderables.push(empire);
        }
        let fov = 50.0 * std::f32::consts::PI / 180.0;
        let near = 1.0;
        let far = 20.0;
        let aspect = 1.0;
        let frustum_renderables = create_frustum_model(
            &renderer,
            far,
            fov,
            aspect,
            near,
            rgb_pipeline,
            line_pipeline,
        );
        let last_time = std::time::Instant::now();
        let input = Input::default();
        let camera = FirstPersonCamera::new(
            cgmath::Vector3::new(0.0, 0.0, -2.0),
            cgmath::Vector3::new(0.0, 0.0, 1.0),
        );

        let last_frame_times = VecDeque::new();
        let egui = EruptEgui::new(&mut renderer, frames_in_flight);
        let workvec = Vec::new();
        let vsync = true;
        let egui_enabled = false;
        let start = std::time::Instant::now();
        let frustum_transform = cgmath::Matrix4::<f32>::identity();
        Self {
            start,
            vsync,
            frames_in_flight,
            egui,
            workvec,
            last_frame_times,
            renderables,
            renderer,
            last_time,
            mouse_captured: true,
            input,
            camera,
            sdl,
            window,
            event_pump,
            egui_enabled,
            tx,
            frustum_renderables,
            frustum_transform,
        }
    }
    fn ui(&mut self) {
        let (width, height) = self.window.lock().unwrap().size();
        self.egui
            .run(&mut self.renderer, (width as f32, height as f32), |ctx| {
                ctx.request_repaint();
                egui::Window::new("Debug Window").show(ctx, |ui| {
                    let dt = self
                        .last_frame_times
                        .get(self.last_frame_times.len().max(1) - 1)
                        .map_or(0.0, |t| t.1);
                    ui.label(format!("{}ms", dt * 1000.0));
                    ui.label(format!("{}fps", 1.0 / dt));
                    ui.checkbox(&mut self.vsync, "Vsync");
                    ui.add(egui::Slider::new(&mut self.frames_in_flight, 1..=15));

                    egui::plot::Plot::new(0)
                        .legend(egui::plot::Legend::default())
                        .allow_drag(false)
                        .allow_zoom(false)
                        .show(ui, |plot| {
                            plot.line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, v, _, _)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::BLUE)
                                .name("Frametime"),
                            );
                            plot.line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, _, v, _)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::RED)
                                .name("GPU-Wait"),
                            );
                            plot.line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, _, _, v)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::GREEN)
                                .name("Prerender Processing"),
                            );
                        })
                        .response;
                });
            });
    }

    fn resize(&mut self) -> bool {
        let (width, height) = self.window.lock().unwrap().size();
        if self.frames_in_flight != self.egui.frames_in_flight() {
            self.egui.adjust_frames_in_flight(self.frames_in_flight);
            assert_eq!(self.frames_in_flight, self.egui.frames_in_flight());
        }
        self.renderer
            .resize(width, height, self.frames_in_flight, self.vsync)
    }
    fn update(&mut self) -> bool {
        let now = std::time::Instant::now();
        let dt = now - self.last_time;
        let dt = dt.as_secs_f32();
        let time = self.start.elapsed().as_secs_f32();
        while self.last_frame_times.len() > 1000 {
            self.last_frame_times.pop_front();
        }
        self.resize();
        let mut window = self.window.lock().unwrap();
        self.camera.update(&self.input.make_controls(dt));
        let (width, height) = window.size();
        let aspect = width as f32 / height as f32;
        let fov = 90.0 * std::f32::consts::PI / 180.0;
        let near = 0.1;
        let far = 200.0;
        let view_proj = self
            .camera
            .create_view_projection_matrix(aspect, fov, near, far);

        for event in self.event_pump.poll_iter().collect::<Vec<_>>() {
            if self.mouse_captured {
                self.input.process_event(&event);
            }
            self.egui.process_event(&event);
            match event {
                Event::Quit { .. } => {
                    return false;
                }
                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => match keycode {
                    sdl2::keyboard::Keycode::Z => {
                        log::info!(
                            "Device reference count: {}",
                            Arc::strong_count(self.renderer.device())
                        );
                    }
                    sdl2::keyboard::Keycode::V => {
                        self.vsync = true;
                    }
                    sdl2::keyboard::Keycode::E => {
                        self.mouse_captured = !self.mouse_captured;
                        self.sdl.mouse().capture(self.mouse_captured);
                        window.set_grab(self.mouse_captured);
                        self.sdl
                            .mouse()
                            .set_relative_mouse_mode(self.mouse_captured);
                    }
                    sdl2::keyboard::Keycode::C => {
                        self.egui_enabled = !self.egui_enabled;
                    }
                    sdl2::keyboard::Keycode::R => {
                        self.frustum_transform =
                            cgmath::Matrix4::from_translation(self.camera.get_position())
                                * cgmath::Matrix4::from_angle_y(cgmath::Rad(self.camera.yaw()))
                                * cgmath::Matrix4::from_angle_x(-cgmath::Rad(self.camera.pitch()));
                        for r in &mut self.frustum_renderables {
                            r.transform = self.frustum_transform;
                        }
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        drop(window);
        self.ui();
        self.resize();
        self.workvec.clear();
        /* let fov = 50.0 * std::f32::consts::PI / 180.0;
        let near = 1.0;
        let far = 20.0;
        let aspect = 1.0; */
        let frustum = create_frustum(fov, far, aspect, near);
        self.workvec.extend(
            self.renderables
                .iter()
                //.filter(|r| check_frustum(r, frustum, self.frustum_transform, near, far))
                .cloned(),
        );
        self.workvec.extend(self.egui.renderables().iter().cloned());
        self.workvec
            .extend(self.frustum_renderables.iter().cloned());
        if let Some((gpu_wait, cpu_work_time)) =
            self.renderer
                .render(&self.workvec, view_proj, frustum, self.camera.transform())
        {
            self.last_frame_times.push_back((
                time,
                dt,
                gpu_wait.as_secs_f32(),
                cpu_work_time.as_secs_f32(),
            ));
        }

        self.last_time = now;
        true
    }
}

fn check_frustum(
    r: &Renderable,
    frustum: Frustum,
    camera: cgmath::Matrix4<f32>,
    near: f32,
    far: f32,
) -> bool {
    let scale = r.transform.diagonal();
    let scale = scale.x.max(scale.y).max(scale.z);
    let radius = r.mesh.bounds.sphere_bounds * scale;

    let center = r.transform * Vector4::new(0.0, 0.0, 0.0, 1.0);
    let center_in_frustum_space = camera.invert().unwrap() * center;
    let center_in_frustum_space = center_in_frustum_space.truncate();
    let near_plane_position = Vector3::new(0.0, 0.0, near);
    let far_plane_position = Vector3::new(0.0, 0.0, far);

    (center_in_frustum_space - Vector3::new(0.0, 0.0, -radius) - near_plane_position)
        .dot(frustum.near)
        > 0.0
        && (center_in_frustum_space + Vector3::new(0.0, 0.0, -radius) - far_plane_position)
            .dot(frustum.far)
            > 0.0
        && (center_in_frustum_space + frustum.top * radius).dot(frustum.top) > 0.0
        && (center_in_frustum_space + frustum.bottom * radius).dot(frustum.bottom) > 0.0
        && (center_in_frustum_space + frustum.left * radius).dot(frustum.left) > 0.0
        && (center_in_frustum_space + frustum.right * radius).dot(frustum.right) > 0.0
}

impl Drop for State {
    fn drop(&mut self) {
        self.tx.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

fn main() {
    pretty_env_logger::init();
    let mut state = State::new();
    loop {
        if !state.update() {
            break;
        }
    }
}

fn mesh_pipeline(device: &Arc<Device>) -> MaterialLoadFn {
    let vert_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/mesh.vert.spv"),
        String::from("MeshPipelineVertexShader"),
        vk::ShaderStageFlagBits::VERTEX,
    );
    let frag_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/mesh.frag.spv"),
        String::from("MeshPipelineFragmentShader"),
        vk::ShaderStageFlagBits::FRAGMENT,
    );

    let texture_set_layout = DescriptorSetLayout::from_shader(device, &frag_shader)
        .remove(&1)
        .unwrap();

    Box::new(move |args| {
        let set_layouts = [&*args.global_set_layout, &texture_set_layout].map(|l| **l);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("MeshPipelineLayout"),
        ));
        let width = args.width;
        let height = args.height;
        let shader_stages = [&vert_shader, &frag_shader];
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new::<DefaultVertex>(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment: ColorBlendAttachment {
                    blend_enable: false,
                    ..Default::default()
                },
                shader_stages: &shader_stages,
                input_assembly_state: InputAssemblyState {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                },
                rasterization_state: RasterizationState {
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                },
                multisample_state: MultiSamplingState {},
                layout: pipeline_layout,
                depth_stencil: DepthStencilInfo {
                    write: true,
                    test: Some(vk::CompareOp::LESS),
                },
            },
            &label!("MeshPipeline"),
        )
    })
}

#[allow(dead_code)]
struct ColorVertex {
    position: cgmath::Vector3<f32>,
    normal: cgmath::Vector3<f32>,
    color: [u8; 3],
}

impl Vertex for ColorVertex {
    fn position(&self) -> cgmath::Vector3<f32> {
        self.position
    }
    fn description() -> VertexInfoDescription {
        let bindings = vec![vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<Self>().try_into().unwrap())
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = vec![
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 3]>().try_into().unwrap()),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(2)
                .format(vk::Format::R8G8B8_UNORM)
                .offset(std::mem::size_of::<[f32; 6]>().try_into().unwrap()),
        ];
        VertexInfoDescription {
            bindings,
            attributes,
        }
    }
    fn new(v: basalt::LoadingVertex) -> Self {
        Self {
            position: v.position,
            normal: v.normal,
            color: v.color.map(|a| a[0..3].try_into().unwrap()).unwrap_or([
                (v.normal.x * 256.0) as u8,
                (v.normal.y * 256.0) as u8,
                (v.normal.z * 256.0) as u8,
            ]),
        }
    }
}

#[allow(dead_code)]
struct LineVertex {
    position: cgmath::Vector3<f32>,
    color: [u8; 3],
}

impl Vertex for LineVertex {
    fn position(&self) -> cgmath::Vector3<f32> {
        self.position
    }
    fn description() -> VertexInfoDescription {
        let bindings = vec![vk::VertexInputBindingDescriptionBuilder::new()
            .binding(0)
            .stride(std::mem::size_of::<Self>().try_into().unwrap())
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = vec![
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .format(vk::Format::R8G8B8_UNORM)
                .offset(std::mem::size_of::<[f32; 3]>().try_into().unwrap()),
        ];
        VertexInfoDescription {
            bindings,
            attributes,
        }
    }
}

fn rgb_pipeline(device: &Arc<Device>) -> MaterialLoadFn {
    let frag_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/rgb_triangle.frag.spv"),
        String::from(label!("RgbPipelineFragmentShader")),
        vk::ShaderStageFlagBits::FRAGMENT,
    );
    let vert_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/rgb_triangle.vert.spv"),
        String::from("RgbPipelineVertexShader"),
        vk::ShaderStageFlagBits::VERTEX,
    );

    Box::new(move |args| {
        let width = args.width;
        let height = args.height;
        let shader_stages = [&vert_shader, &frag_shader];
        let set_layouts = vec![args.global_set_layout];
        let set_layouts = set_layouts
            .iter()
            .map(|l| ****l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("RgbPipelineLayout"),
        ));
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new::<ColorVertex>(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment: Default::default(),
                shader_stages: &shader_stages,
                input_assembly_state: InputAssemblyState {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                },
                rasterization_state: RasterizationState {
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                },
                multisample_state: MultiSamplingState {},
                layout: Arc::clone(&pipeline_layout),
                depth_stencil: DepthStencilInfo {
                    write: true,
                    test: Some(vk::CompareOp::LESS),
                },
            },
            &label!("RgbPipeline"),
        )
    })
}

fn line_pipeline(device: &Arc<Device>) -> MaterialLoadFn {
    let frag_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/line.frag.spv"),
        String::from(label!("LinePipelineFragmentShader")),
        vk::ShaderStageFlagBits::FRAGMENT,
    );
    let vert_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/line.vert.spv"),
        String::from("LinePipelineVertexShader"),
        vk::ShaderStageFlagBits::VERTEX,
    );

    Box::new(move |args| {
        let width = args.width;
        let height = args.height;
        let shader_stages = [&vert_shader, &frag_shader];
        let set_layouts = vec![args.global_set_layout];
        let set_layouts = set_layouts
            .iter()
            .map(|l| ****l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("LinePipelineLayout"),
        ));
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new::<LineVertex>(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment: Default::default(),
                shader_stages: &shader_stages,
                input_assembly_state: InputAssemblyState {
                    topology: vk::PrimitiveTopology::LINE_LIST,
                },
                rasterization_state: RasterizationState {
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::NONE,
                },
                multisample_state: MultiSamplingState {},
                layout: Arc::clone(&pipeline_layout),
                depth_stencil: DepthStencilInfo {
                    write: true,
                    test: Some(vk::CompareOp::LESS),
                },
            },
            &label!("RgbPipeline"),
        )
    })
}
