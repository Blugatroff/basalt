mod input;
use basalt::{
    image, label, vk, ColorBlendAttachment, DefaultVertex, DepthStencilInfo, DescriptorSetLayout,
    Device, InputAssemblyState, MaterialLoadFn, Mesh, MultiSamplingState, Pipeline, PipelineDesc,
    PipelineLayout, RasterizationState, Renderable, Renderer, ShaderModule, Vertex,
};
use cgmath::SquareMatrix;
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
                uv: cgmath::Vector2::new(1.0, 1.0),
            },
            DefaultVertex {
                position: cgmath::Vector3::new(-1.0, 1.0, 0.0),
                normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
                uv: cgmath::Vector2::new(0.0, 1.0),
            },
            DefaultVertex {
                position: cgmath::Vector3::new(0.0, -1.0, 0.0),
                normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
                uv: cgmath::Vector2::new(0.5, 0.0),
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
                                let mesh = Mesh::load(
                                    allocator.clone(),
                                    "./assets/suzanne.obj",
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
        let mut triangle_mesh = Mesh::new(
            &vertices,
            &[0, 1, 2, 2, 1, 0],
            renderer.allocator().clone(),
            renderer.transfer_context(),
            renderer.device(),
            true,
        );
        let mut suzanne_mesh = Mesh::load(
            renderer.allocator().clone(),
            "./assets/suzanne.obj",
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

        let s = 5;
        for x in 0..s {
            for y in 0..s {
                for z in 0..s {
                    suzanne.transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                        x as f32, y as f32, z as f32,
                    ));
                    renderables.push(suzanne.clone());
                }
            }
        }

        let empire_meshes = Mesh::load(
            renderer.allocator().clone(),
            "./assets/lost_empire.obj",
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
                    ui.add({
                        egui::plot::Plot::new(0)
                            .line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, v, _, _)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::BLUE)
                                .name("Frametime"),
                            )
                            .line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, _, v, _)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::RED)
                                .name("GPU-Wait"),
                            )
                            .line(
                                egui::plot::Line::new(egui::plot::Values::from_values_iter(
                                    self.last_frame_times.iter().map(|(t, _, _, v)| {
                                        egui::plot::Value::new(*t, *v * 1000.0)
                                    }),
                                ))
                                .color(egui::Color32::GREEN)
                                .name("Prerender Processing"),
                            )
                            .legend(egui::plot::Legend::default())
                            .allow_drag(false)
                            .allow_zoom(false)
                    })
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
        let view_proj = self.camera.create_view_projection_matrix(
            aspect,
            90.0 * std::f32::consts::PI / 180.0,
            0.1,
            200.0,
        );
        for event in self.event_pump.poll_iter().collect::<Vec<_>>() {
            if self.mouse_captured {
                self.input.process_event(&event);
            }
            self.egui.process_event(&event);
            match event {
                Event::Quit { .. } => return false,
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
                    _ => {}
                },
                _ => {}
            }
        }
        drop(window);
        self.ui();
        self.resize();
        self.workvec.clear();
        self.workvec.extend(self.renderables.iter().cloned());
        self.workvec.extend(self.egui.renderables().iter().cloned());
        if let Some((gpu_wait, cpu_work_time)) = self.renderer.render(&self.workvec, view_proj) {
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
        let vertex_description = DefaultVertex::description();
        let shader_stages = [&vert_shader, &frag_shader];
        /* let color_blend_attachment = color_blend_attachment
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::SUBTRACT); */
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new(
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
                vertex_description,
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
            String::from(label!("MeshPipeline")),
        )
    })
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
        let vertex_description = DefaultVertex::description();
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment: Default::default(),
                shader_stages: &shader_stages,
                vertex_description,
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
            String::from(label!("RgbPipeline")),
        )
    })
}
