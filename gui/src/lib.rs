use basalt::{
    buffer,
    image::{Loader, Texture},
    label, vk::{self, PipelineDepthStencilStateCreateInfo}, Allocator, DescriptorSetLayout,
    InputAssemblyState, Mesh, MultiSamplingState, Pipeline, PipelineDesc, PipelineHandle,
    PipelineLayout, RasterizationState, Renderable, Renderer, Sampler, ShaderModule, vma,
};
use egui::FontImage;
use sdl2::event::Event;
use std::sync::Arc;

pub use egui;

enum KeyOrEvent {
    Key(egui::Key),
    Event(egui::Event),
}

fn sdl2_key_to_egui_key(key: sdl2::keyboard::Keycode) -> Option<KeyOrEvent> {
    Some(KeyOrEvent::Key(match key {
        sdl2::keyboard::Keycode::Backspace => egui::Key::Backspace,
        sdl2::keyboard::Keycode::Tab => egui::Key::Tab,
        sdl2::keyboard::Keycode::Return => egui::Key::Enter,
        sdl2::keyboard::Keycode::Escape => egui::Key::Escape,
        sdl2::keyboard::Keycode::Space => egui::Key::Space,
        sdl2::keyboard::Keycode::Num0 | sdl2::keyboard::Keycode::Kp0 => egui::Key::Num0,
        sdl2::keyboard::Keycode::Num1 | sdl2::keyboard::Keycode::Kp1 => egui::Key::Num1,
        sdl2::keyboard::Keycode::Num2 | sdl2::keyboard::Keycode::Kp2 => egui::Key::Num2,
        sdl2::keyboard::Keycode::Num3 | sdl2::keyboard::Keycode::Kp3 => egui::Key::Num3,
        sdl2::keyboard::Keycode::Num4 | sdl2::keyboard::Keycode::Kp4 => egui::Key::Num4,
        sdl2::keyboard::Keycode::Num5 | sdl2::keyboard::Keycode::Kp5 => egui::Key::Num5,
        sdl2::keyboard::Keycode::Num6 | sdl2::keyboard::Keycode::Kp6 => egui::Key::Num6,
        sdl2::keyboard::Keycode::Num7 | sdl2::keyboard::Keycode::Kp7 => egui::Key::Num7,
        sdl2::keyboard::Keycode::Num8 | sdl2::keyboard::Keycode::Kp8 => egui::Key::Num8,
        sdl2::keyboard::Keycode::Num9 | sdl2::keyboard::Keycode::Kp9 => egui::Key::Num9,
        sdl2::keyboard::Keycode::A => egui::Key::A,
        sdl2::keyboard::Keycode::B => egui::Key::B,
        sdl2::keyboard::Keycode::C => egui::Key::C,
        sdl2::keyboard::Keycode::D => egui::Key::D,
        sdl2::keyboard::Keycode::E => egui::Key::E,
        sdl2::keyboard::Keycode::F => egui::Key::F,
        sdl2::keyboard::Keycode::G => egui::Key::G,
        sdl2::keyboard::Keycode::H => egui::Key::H,
        sdl2::keyboard::Keycode::I => egui::Key::I,
        sdl2::keyboard::Keycode::J => egui::Key::J,
        sdl2::keyboard::Keycode::K => egui::Key::K,
        sdl2::keyboard::Keycode::L => egui::Key::L,
        sdl2::keyboard::Keycode::M => egui::Key::M,
        sdl2::keyboard::Keycode::N => egui::Key::N,
        sdl2::keyboard::Keycode::O => egui::Key::O,
        sdl2::keyboard::Keycode::P => egui::Key::P,
        sdl2::keyboard::Keycode::Q => egui::Key::Q,
        sdl2::keyboard::Keycode::R => egui::Key::R,
        sdl2::keyboard::Keycode::S => egui::Key::S,
        sdl2::keyboard::Keycode::T => egui::Key::T,
        sdl2::keyboard::Keycode::U => egui::Key::U,
        sdl2::keyboard::Keycode::V => egui::Key::V,
        sdl2::keyboard::Keycode::W => egui::Key::W,
        sdl2::keyboard::Keycode::X => egui::Key::X,
        sdl2::keyboard::Keycode::Y => egui::Key::Y,
        sdl2::keyboard::Keycode::Z => egui::Key::Z,
        sdl2::keyboard::Keycode::Delete => egui::Key::Delete,
        sdl2::keyboard::Keycode::Insert => egui::Key::Insert,
        sdl2::keyboard::Keycode::Home => egui::Key::Home,
        sdl2::keyboard::Keycode::PageUp => egui::Key::PageUp,
        sdl2::keyboard::Keycode::End => egui::Key::End,
        sdl2::keyboard::Keycode::PageDown => egui::Key::PageDown,
        sdl2::keyboard::Keycode::Right => egui::Key::ArrowRight,
        sdl2::keyboard::Keycode::Left => egui::Key::ArrowLeft,
        sdl2::keyboard::Keycode::Down => egui::Key::ArrowDown,
        sdl2::keyboard::Keycode::Up => egui::Key::ArrowUp,
        sdl2::keyboard::Keycode::Copy => return Some(KeyOrEvent::Event(egui::Event::Copy)),
        sdl2::keyboard::Keycode::Cut => return Some(KeyOrEvent::Event(egui::Event::Cut)),
        _ => return None,
    }))
}

#[repr(transparent)]
struct EguiVertex(egui::epaint::Vertex);

impl basalt::Vertex for EguiVertex {
    fn description() -> basalt::VertexInfoDescription {
        assert_eq!(
            std::mem::size_of::<Self>(),
            std::mem::size_of::<egui::epaint::Vertex>()
        );
        let binding_0 = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>().try_into().unwrap())
            .input_rate(vk::VertexInputRate::VERTEX);
        let binding_1 = vk::VertexInputBindingDescription::builder()
            .binding(1)
            .stride(std::mem::size_of::<shaders::Object>().try_into().unwrap())
            .input_rate(vk::VertexInputRate::INSTANCE);
        let bindings = vec![binding_0.build(), binding_1.build()];
        let attributes = vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 2]>().try_into().unwrap())
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(std::mem::size_of::<[f32; 4]>().try_into().unwrap())
                .build(),
        ];
        basalt::VertexInfoDescription {
            bindings,
            attributes,
        }
    }

    fn position(&self) -> cgmath::Vector3<f32> {
        cgmath::Vector3::new(self.0.pos.x, self.0.pos.y, 0.0)
    }
}

pub struct EruptEgui {
    ctx: egui::CtxRef,
    raw_input: egui::RawInput,
    pipeline: PipelineHandle,
    last_meshes: Vec<(Arc<Mesh>, Arc<basalt::DescriptorSet>, cgmath::Matrix4<f32>)>,
    font_texture_version: u64,
    buffers: Vec<Arc<buffer::Allocated>>,
    frame: usize,
    texture: Option<Arc<basalt::image::Texture>>,
    sampler: Arc<Sampler>,
    allocator: Arc<Allocator>,
    user_textures: slab::Slab<Arc<basalt::image::Texture>>,
}
impl EruptEgui {
    #[allow(clippy::missing_panics_doc)]
    pub fn new(app: &mut Renderer, frames_in_flight: usize) -> Self {
        let vert_shader = ShaderModule::new(
            app.device().clone(),
            include_bytes!("../../shaders/egui.vert.spv"),
            String::from(label!("EguiVertexShader")),
            vk::ShaderStageFlags::VERTEX,
        );
        let frag_shader = ShaderModule::new(
            app.device().clone(),
            include_bytes!("../../shaders/egui.frag.spv"),
            String::from("EguiFragmentShader"),
            vk::ShaderStageFlags::FRAGMENT,
        );

        let texture_set_layout = Arc::new(
            DescriptorSetLayout::from_shader(app.device(), &frag_shader)
                .remove(&1)
                .unwrap(),
        );

        let pipeline = app.register_pipeline(Box::new(move |params| {
            let width = params.width;
            let height = params.height;
            let shader_stages = [&vert_shader, &frag_shader];
            let pipeline_layout = Arc::new(PipelineLayout::new(
                params.device.clone(),
                vec![params.global_set_layout.clone(), texture_set_layout.clone()],
                (),
                &label!("EguiPipelineLayout"),
            ));
            let view_port = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            Pipeline::new::<EguiVertex>(
                params.device.clone(),
                **params.render_pass,
                &PipelineDesc {
                    view_port,
                    scissor: Arc::new(
                        vk::Rect2D::builder()
                            .offset(vk::Offset2D { x: 0, y: 0 })
                            .extent(vk::Extent2D { width, height }),
                    ),
                    color_blend_attachment: vk::PipelineColorBlendAttachmentState {
                        blend_enable: vk::TRUE,
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        alpha_blend_op: vk::BlendOp::ADD,
                        color_write_mask: vk::ColorComponentFlags::RGBA,
                    },
                    shader_stages: &shader_stages,
                    input_assembly_state: InputAssemblyState {
                        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    },
                    rasterization_state: RasterizationState {
                        polygon_mode: vk::PolygonMode::FILL,
                        cull_mode: vk::CullModeFlags::NONE,
                        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    },
                    multisample_state: MultiSamplingState {},
                    layout: Arc::clone(&pipeline_layout),
                    depth_stencil: PipelineDepthStencilStateCreateInfo::builder().depth_write_enable(false).depth_test_enable(false).build()
                         ,
                },
                &label!("EguiPipeline"),
            )
        }));
        let buffers = (0..=frames_in_flight)
            .map(|_| Arc::new(Self::create_buffer(app.allocator().clone())))
            .collect();
        let filter = vk::Filter::NEAREST;
        let address_mode = vk::SamplerAddressMode::REPEAT;
        let sampler = vk::SamplerCreateInfo::builder()
            .mag_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode);
        let sampler = Arc::new(Sampler::new(
            app.device().clone(),
            &sampler,
            label!("EguiSampler"),
        ));
        let user_textures = slab::Slab::new();
        Self {
            ctx: egui::CtxRef::default(),
            raw_input: Self::default_input(),
            pipeline,
            last_meshes: Vec::new(),
            font_texture_version: 0,
            buffers,
            frame: 0,
            texture: None,
            sampler,
            allocator: app.allocator().clone(),
            user_textures,
        }
    }
    fn create_buffer(allocator: Arc<Allocator>) -> buffer::Allocated {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(2_u64.pow(12))
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER);
        buffer::Allocated::new(
            allocator,
            *buffer_info,
            vma::MemoryUsage::AutoPreferDevice,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            label!("EguiVertexIndexBuffer"),
        )
    }
    pub fn adjust_frames_in_flight(&mut self, frames_in_flight: usize) {
        self.buffers = (0..=frames_in_flight)
            .map(|_| Arc::new(Self::create_buffer(self.allocator.clone())))
            .collect();
    }
    pub fn frames_in_flight(&self) -> usize {
        self.buffers.len() - 1
    }
    fn default_input() -> egui::RawInput {
        egui::RawInput {
            //scroll_delta: egui::Vec2::new(0.0, 0.0),
            //zoom_delta: 1.0,
            screen_rect: Some(egui::Rect::from_min_max(
                egui::pos2(0.0, 0.0),
                egui::pos2(100.0, 100.0),
            )),
            pixels_per_point: None,
            time: None,
            predicted_dt: 1.0 / 60.0,
            modifiers: egui::Modifiers::default(),
            events: Vec::new(),
            hovered_files: Vec::new(),
            dropped_files: Vec::new(),
        }
    }
    fn upload_font_texture(
        &mut self,
        image_loader: &Loader,
        descriptor_set_manager: &basalt::DescriptorSetManager,
        texture: &egui::FontImage,
    ) {
        let data = texture
            .pixels
            .iter()
            .flat_map(|r| [*r; 4])
            .collect::<Vec<_>>();
        let width = texture.width.try_into().unwrap();
        let height = texture.height.try_into().unwrap();
        let image = basalt::image::Allocated::load(
            image_loader,
            &data,
            width,
            height,
            label!("EguiTexture").into(),
        );
        let texture = basalt::image::Texture::new(
            &image_loader.device,
            descriptor_set_manager,
            image,
            self.sampler.clone(),
        );
        self.texture = Some(Arc::new(texture));
    }
    pub fn run(
        &mut self,
        renderer: &mut Renderer,
        window_size: (f32, f32),
        f: impl FnOnce(&egui::CtxRef),
    ) {
        let (width, height) = window_size;
        self.raw_input.screen_rect = Some(egui::Rect {
            min: egui::pos2(0.0, 0.0),
            max: egui::pos2(width, height),
        });
        self.ctx.begin_frame(std::mem::replace(
            &mut self.raw_input,
            Self::default_input(),
        ));
        let texture: Arc<FontImage> = self.ctx.fonts().font_image();

        if texture.version != self.font_texture_version {
            self.font_texture_version = texture.version;
            self.upload_font_texture(
                &renderer.image_loader().clone(),
                renderer.descriptor_set_manager(),
                &texture,
            );
        }
        f(&self.ctx);
        let (output, shapes) = self.ctx.end_frame();
        if !output.needs_repaint {
            return;
        }
        let clipped_meshes = self.ctx.tessellate(shapes);
        self.last_meshes = self.draw(&clipped_meshes, renderer.allocator(), width, height);
    }
    pub fn renderables(&self) -> impl Iterator<Item = Renderable<'_>> {
        self.last_meshes
            .iter()
            .map(|(mesh, custom_set, transform)| Renderable {
                transform,
                mesh,
                custom_set: Some(&*custom_set),
                custom_id: 0,
                uncullable: true,
                pipeline: &self.pipeline,
            })
    }
    fn draw(
        &mut self,
        clipped_meshes: &[egui::ClippedMesh],
        allocator: &Arc<Allocator>,
        width: f32,
        height: f32,
    ) -> Vec<(Arc<Mesh>, Arc<basalt::DescriptorSet>, cgmath::Matrix4<f32>)> {
        assert_eq!(
            std::mem::size_of::<EguiVertex>(),
            std::mem::size_of::<egui::epaint::Vertex>()
        );
        let font_texture = if let Some(font_texture) = self.texture.as_ref() {
            font_texture.set.clone()
        } else {
            return Vec::new();
        };
        let l = self.buffers.len();
        let buffer = &mut self.buffers[self.frame % l];
        self.frame += 1;
        let mut offset = 0;
        let transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(-1.0, -1.0, 0.0))
            * cgmath::Matrix4::from_nonuniform_scale(2.0 / width, 2.0 / height, 1.0);
        'outer: loop {
            let mut meshes = Vec::new();
            for mesh in clipped_meshes.iter() {
                let egui::ClippedMesh(rect, mesh) = mesh;
                let texture = match mesh.texture_id {
                    egui::TextureId::Egui => font_texture.clone(),
                    egui::TextureId::User(id) => self.user_textures[id as usize].set.clone(),
                };
                let min = cgmath::Vector3::new(rect.min.x, rect.min.y, 0.0);
                let max = cgmath::Vector3::new(rect.max.x, rect.max.y, 0.0);
                let sphere_bounds = *[min, max]
                    .map(cgmath::InnerSpace::magnitude)
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let (mesh, size) = if let Some((mesh, size)) = Mesh::new_into_buffer(
                    unsafe {
                        std::slice::from_raw_parts(
                            mesh.vertices.as_ptr() as *const EguiVertex,
                            mesh.vertices.len(),
                        )
                    },
                    &mesh.indices,
                    basalt::Bounds {
                        max,
                        min,
                        sphere_bounds,
                    },
                    buffer.clone(),
                    offset,
                    String::from("EguiMesh"),
                ) {
                    (mesh, size)
                } else {
                    let buffer_info = vk::BufferCreateInfo::builder()
                        .size(
                            ((offset
                                + mesh.vertices.len()
                                    * std::mem::size_of::<egui::epaint::Vertex>()
                                + mesh.indices.len() * std::mem::size_of::<u32>())
                                * 2) as u64,
                        )
                        .usage(
                            vk::BufferUsageFlags::VERTEX_BUFFER
                                | vk::BufferUsageFlags::INDEX_BUFFER,
                        );
                    *buffer = Arc::new(buffer::Allocated::new(
                        allocator.clone(),
                        *buffer_info,
                        vma::MemoryUsage::AutoPreferDevice,
                        vk::MemoryPropertyFlags::HOST_VISIBLE,
                        vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                        label!("EguiVertexIndexBuffer"),
                    ));
                    continue 'outer;
                };
                offset += size;
                offset += std::mem::size_of::<egui::epaint::Vertex>()
                    - offset % std::mem::size_of::<egui::epaint::Vertex>();
                assert_eq!(offset % std::mem::size_of::<egui::epaint::Vertex>(), 0);
                let mesh = Arc::new(mesh);
                meshes.push((mesh.clone(), texture.clone(), transform))
            }
            break meshes;
        }
    }
    pub fn load_texture(&mut self, texture: Arc<Texture>) -> egui::TextureId {
        let id = self.user_textures.insert(texture) as u64;
        egui::TextureId::User(id)
    }
    fn update_modifiers(&mut self, keymod: &sdl2::keyboard::Mod) -> &egui::Modifiers {
        self.raw_input.modifiers = egui::Modifiers {
            alt: keymod.contains(sdl2::keyboard::Mod::LALTMOD)
                | keymod.contains(sdl2::keyboard::Mod::RALTMOD),
            ctrl: keymod.contains(sdl2::keyboard::Mod::LCTRLMOD)
                | keymod.contains(sdl2::keyboard::Mod::RCTRLMOD),
            shift: keymod.contains(sdl2::keyboard::Mod::LSHIFTMOD)
                | keymod.contains(sdl2::keyboard::Mod::RSHIFTMOD),
            mac_cmd: false,
            command: keymod.contains(sdl2::keyboard::Mod::LCTRLMOD)
                | keymod.contains(sdl2::keyboard::Mod::RCTRLMOD),
        };
        &self.raw_input.modifiers
    }
    pub fn process_event(&mut self, event: &sdl2::event::Event) {
        let mut event = || match event {
            Event::Window { win_event, .. } => match win_event {
                sdl2::event::WindowEvent::Minimized
                | sdl2::event::WindowEvent::Leave
                | sdl2::event::WindowEvent::FocusLost => Some(egui::Event::PointerGone),
                _ => None,
            },
            Event::KeyDown {
                keymod, keycode, ..
            } => Some(egui::Event::Key {
                key: match keycode {
                    Some(key) => match sdl2_key_to_egui_key(*key) {
                        Some(KeyOrEvent::Key(k)) => k,
                        Some(KeyOrEvent::Event(e)) => return Some(e),
                        None => return None,
                    },
                    None => return None,
                },
                pressed: true,
                modifiers: *self.update_modifiers(keymod),
            }),
            Event::KeyUp {
                keycode, keymod, ..
            } => Some(egui::Event::Key {
                key: match keycode {
                    Some(key) => match sdl2_key_to_egui_key(*key) {
                        Some(KeyOrEvent::Key(k)) => k,
                        Some(KeyOrEvent::Event(e)) => return Some(e),
                        None => return None,
                    },
                    None => return None,
                },
                pressed: false,
                modifiers: *self.update_modifiers(keymod),
            }),
            Event::MouseMotion { x, y, .. } => Some(egui::Event::PointerMoved(egui::Pos2::new(
                *x as f32, *y as f32,
            ))),
            Event::MouseButtonDown {
                mouse_btn, x, y, ..
            } => Some(egui::Event::PointerButton {
                pos: egui::Pos2::new(*x as f32, *y as f32),
                button: match mouse_btn {
                    sdl2::mouse::MouseButton::Left => egui::PointerButton::Primary,
                    sdl2::mouse::MouseButton::Middle => egui::PointerButton::Middle,
                    sdl2::mouse::MouseButton::Right => egui::PointerButton::Secondary,
                    _ => return None,
                },
                pressed: true,
                modifiers: self.raw_input.modifiers,
            }),
            Event::MouseButtonUp {
                mouse_btn, x, y, ..
            } => Some(egui::Event::PointerButton {
                pos: egui::Pos2::new(*x as f32, *y as f32),
                button: match mouse_btn {
                    sdl2::mouse::MouseButton::Left => egui::PointerButton::Primary,
                    sdl2::mouse::MouseButton::Middle => egui::PointerButton::Middle,
                    sdl2::mouse::MouseButton::Right => egui::PointerButton::Secondary,
                    _ => return None,
                },
                pressed: false,
                modifiers: self.raw_input.modifiers,
            }),
            Event::TextInput { text, .. } => Some(egui::Event::Text(text.clone())),
            _ => None,
        };
        if let Some(event) = event() {
            self.raw_input.events.push(event);
        }
    }
}
