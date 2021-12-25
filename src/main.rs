#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::struct_excessive_bools,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr
)]
macro_rules! label {
    ( $name:expr ) => {
        concat!(label!(), " -> ", $name)
    };
    ( ) => {
        concat!(file!(), ":", line!())
    };
}
mod buffer;
mod descriptor_sets;
mod frame;
mod gui;
mod handles;
mod image;
mod input;
mod mesh;
mod utils;
use crate::descriptor_sets::DescriptorSet;
use crate::descriptor_sets::DescriptorSetLayoutBinding;
use cgmath::SquareMatrix;
use descriptor_sets::{DescriptorSetLayout, DescriptorSetManager};
use erupt::{cstr, vk};
use frame::Frames;
use gui::EruptEgui;
use handles::{
    Allocator, CommandPool, Device, Fence, Framebuffer, ImageView, Instance, Pipeline,
    PipelineDesc, PipelineLayout, RenderPass, Sampler, Semaphore, ShaderModule, Surface, Swapchain,
};
use input::Input;
use mesh::{Mesh, Vertex};
use sdl2::{event::Event, EventPump};
use std::{
    collections::VecDeque,
    ffi::{c_void, CStr},
    os::raw::c_char,
    path::PathBuf,
    sync::Arc,
};
use structopt::StructOpt;
use utils::{
    create_command_buffers, create_debug_messenger, create_debug_messenger_info,
    create_depth_image_views, create_depth_images, create_descriptor_sets, create_device_and_queue,
    create_framebuffers, create_full_view_port, create_global_descriptor_set_layout,
    create_instance, create_mesh_buffer, create_mesh_buffer_set, create_physical_device,
    create_pipeline_color_blend_attachment_state, create_render_pass, create_renderables_buffer,
    create_swapchain, create_uniform_buffer, create_window, depth_stencil_create_info,
    input_assembly_create_info, multisampling_state_create_info, pad_uniform_buffer_size,
    pipeline_shader_stage_create_info, rasterization_state_create_info,
};

const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    println!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );
    if message_severity.0
        & (vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT.0
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT.0)
        != 0
    {
        std::process::exit(-1);
    }
    vk::FALSE
}
#[derive(Copy, Clone, Debug, StructOpt)]
struct Opt {
    /// attempt to enable validation layers
    #[structopt(short, long)]
    validation_layers: bool,
    #[structopt(short, long)]
    /// render wireframe of meshes
    wireframe: bool,
    /// how many frames are rendered to at the same time
    frames_in_flight: usize,
    test: usize,
}

#[derive(Debug)]
pub struct RenderPipeline {
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,
}

#[derive(Clone)]
pub struct Renderable {
    pub transform: cgmath::Matrix4<f32>,
    pub mesh: Arc<Mesh>,
    pub custom_set: Arc<DescriptorSet>,
    pub custom_id: u32,
    pub uncullable: bool,
    pub pipeline: usize,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct GlobalUniform {
    view: cgmath::Matrix4<f32>,
    projection: cgmath::Matrix4<f32>,
    view_proj: cgmath::Matrix4<f32>,
    time: f32,
    renderables_count: u32,
    screen_width: f32,
    screen_height: f32,
}

struct PipelineCreationParams<'a> {
    set_layouts: Vec<&'a DescriptorSetLayout>,
    texture_set_layout: Arc<DescriptorSetLayout>,
    device: &'a Arc<Device>,
    width: u32,
    height: u32,
    render_pass: &'a RenderPass,
    opt: &'a Opt,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GpuMesh {
    max: cgmath::Vector3<f32>,
    first_index: u32,
    min: cgmath::Vector3<f32>,
    index_count: u32,
    vertex_offset: i32,
    _padding_0: u32,
    _padding_1: u32,
    _padding_2: u32,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Copy, Debug, Clone)]
struct GpuDataRenderable {
    transform: cgmath::Matrix4<f32>,
    batch: u32,
    draw: u32,
    first_instance: u32,
    uncullable: u32,
    unused_3: u32,
    custom_id: u32,
    mesh: u32,
    redirect: u32,
}

#[allow(dead_code)]
#[repr(C)]
#[derive(Clone, Copy)]
struct IndirectDrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
    pub draw_count: u32,
    pub padding: [u32; 2],
}

type MaterialLoadFn = Box<dyn for<'a> Fn(PipelineCreationParams<'a>) -> RenderPipeline>;

pub struct TransferContext {
    pub transfer_queue: vk::Queue,
    pub command_pool: CommandPool,
    pub fence: Fence,
}

#[allow(clippy::struct_excessive_bools)]
pub struct App {
    #[allow(dead_code)]
    graphics_queue_family: u32,
    opt: Opt,
    mouse_captured: bool,
    input: Input,
    camera: first_person_camera::FirstPersonCamera,
    start: std::time::Instant,
    #[allow(dead_code)]
    supported_present_modes: Vec<vk::PresentModeKHR>,
    present_mode: vk::PresentModeKHR,
    swapchain: Swapchain,
    #[allow(dead_code)]
    sdl: sdl2::Sdl,
    mesh_set_layout: DescriptorSetLayout,
    sampler: Arc<Sampler>,
    event_pump: EventPump,
    window: sdl2::video::Window,
    width: u32,
    height: u32,
    vsync: bool,
    graphics_queue: vk::Queue,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    descriptor_set_manager: DescriptorSetManager,
    texture_set_layout: Arc<DescriptorSetLayout>,
    command_pool: CommandPool,
    frame_buffers: Vec<Framebuffer>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<ImageView>,
    render_pass: RenderPass,
    egui: Option<EruptEgui>,
    frames: Frames,
    uniform_buffer: buffer::Allocated,
    frames_in_flight: usize,
    last_frames_in_flight: usize,
    last_vsync: bool,
    frame_number: usize,
    image_loader: image::Loader,
    #[allow(dead_code)]
    transfer_context: Arc<TransferContext>,
    last_time: std::time::Instant,
    physical_device_properties: vk::PhysicalDeviceProperties,
    materials: Vec<Option<(RenderPipeline, MaterialLoadFn)>>,
    global_uniform: GlobalUniform,
    global_set_layout: DescriptorSetLayout,
    surface: Surface,
    egui_on: bool,
    print_fps: bool,
    format: vk::SurfaceFormatKHR,
    physical_device: vk::PhysicalDevice,
    allocator: Arc<Allocator>,
    device: Arc<Device>,
    instance: Arc<Instance>,
    #[allow(dead_code)]
    entry: erupt::EntryLoader,
    last_frame_times: std::collections::VecDeque<(f32, f32, f32, f32)>,
}

fn create_sync_objects(
    device: &Arc<Device>,
    num: usize,
) -> (Vec<Fence>, Vec<Semaphore>, Vec<Semaphore>) {
    let fence_create_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let render_fences = (0..num)
        .map(|_| Fence::new(device.clone(), &fence_create_info, label!("RenderFence")))
        .collect();

    let semaphore_create_info = vk::SemaphoreCreateInfoBuilder::new();
    let present_semaphores = (0..num)
        .map(|_| {
            Semaphore::new(
                device.clone(),
                &semaphore_create_info,
                label!("PresentSemaphore"),
            )
        })
        .collect();
    let render_semaphores = (0..num)
        .map(|_| {
            Semaphore::new(
                device.clone(),
                &semaphore_create_info,
                label!("RenderSemaphore"),
            )
        })
        .collect();
    (render_fences, present_semaphores, render_semaphores)
}

impl App {
    #[allow(clippy::too_many_lines)]
    fn new(opt: Opt) -> Self {
        let (sdl, mut window, event_pump) = create_window();
        let frames_in_flight = opt.frames_in_flight;
        let messenger_info = create_debug_messenger_info();

        let (instance, entry, mut device_extensions) =
            create_instance(&window, opt.validation_layers, &messenger_info);
        device_extensions.push(vk::KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME);
        device_extensions.push(vk::KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        //device_extensions.push(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        //device_extensions.push(vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        let instance = Arc::new(Instance::new(instance));
        let mut device_layers = Vec::new();

        let messenger = if opt.validation_layers {
            device_layers.push(LAYER_KHRONOS_VALIDATION);
            Some(create_debug_messenger(
                &instance,
                &messenger_info,
                opt.validation_layers,
            ))
        } else {
            None
        };

        let surface = Surface::new(instance.clone(), &window);
        let present_mode = vk::PresentModeKHR::MAILBOX_KHR;

        let (
            physical_device,
            graphics_queue_family,
            format,
            present_mode,
            physical_device_properties,
            supported_present_modes,
            transfer_queue_family,
        ) = create_physical_device(
            &instance,
            *surface,
            &device_extensions,
            present_mode,
            vk::PresentModeKHR::FIFO_KHR,
        );

        println!("Using physical device: {:?}", unsafe {
            CStr::from_ptr(physical_device_properties.device_name.as_ptr())
        });

        let (device, graphics_queue, transfer_queue) = create_device_and_queue(
            &instance,
            graphics_queue_family,
            transfer_queue_family,
            &device_extensions,
            &device_layers,
            physical_device,
        );
        let device = Arc::new(device);
        let (swapchain, swapchain_images, swapchain_image_views) = create_swapchain(
            &instance,
            physical_device,
            *surface,
            format,
            &device,
            present_mode,
            None,
        );

        let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(graphics_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = CommandPool::new(device.clone(), &command_pool_info, label!());
        let command_buffers =
            create_command_buffers(&device, &command_pool, frames_in_flight.try_into().unwrap());
        let render_pass = create_render_pass(device.clone(), format);

        let width = window.size().0;
        let height = window.size().1;
        let (render_fences, present_semaphores, render_semaphores) =
            create_sync_objects(&device, frames_in_flight);

        let allocator_info = vk_mem_erupt::AllocatorCreateInfo {
            physical_device,
            device: device.raw(),
            instance: instance.raw(),
            flags: vk_mem_erupt::AllocatorCreateFlags::NONE,
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };

        let allocator = Arc::new(Allocator::new(&allocator_info));
        let depth_images = create_depth_images(&allocator, width, height, frames_in_flight);
        let depth_image_views = create_depth_image_views(&device, &depth_images);
        let frame_buffers = create_framebuffers(
            &device,
            width,
            height,
            *render_pass,
            &swapchain_image_views,
            &depth_image_views,
        );

        let last_time = std::time::Instant::now();
        let frame_number = 0;
        let start = std::time::Instant::now();
        let uniform_buffer = create_uniform_buffer(
            allocator.clone(),
            frames_in_flight as u64,
            &physical_device_properties.limits,
        );

        let global_uniform = GlobalUniform {
            view: cgmath::Matrix4::from_scale(0.1),
            projection: cgmath::Matrix4::from_scale(0.1),
            view_proj: cgmath::Matrix4::from_scale(0.1),
            time: 0.0,
            renderables_count: 0,
            screen_width: width as f32,
            screen_height: height as f32,
        };
        let max_objects = (0..frames_in_flight).map(|_| 4).collect::<Vec<_>>();
        let renderables_buffers = max_objects
            .iter()
            .map(|size| create_renderables_buffer(allocator.clone(), *size as u64))
            .collect::<Vec<_>>();
        let global_set_layout = create_global_descriptor_set_layout(device.clone());
        let filter = vk::Filter::NEAREST;
        let address_mode = vk::SamplerAddressMode::REPEAT;
        let sampler = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode);
        let sampler = Sampler::new(device.clone(), &sampler, label!());
        let mut descriptor_set_manager = DescriptorSetManager::new(device.clone());
        let texture_set_layout = Arc::new(DescriptorSetLayout::new(
            device.clone(),
            vec![DescriptorSetLayoutBinding {
                binding: 0,
                count: 1,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                immutable_samplers: None,
            }],
            None,
        ));
        let descriptor_sets = (0..frames_in_flight)
            .map(|i| {
                create_descriptor_sets(
                    &device,
                    &mut descriptor_set_manager,
                    &global_set_layout,
                    &uniform_buffer,
                    &renderables_buffers[i],
                    max_objects[i],
                )
            })
            .collect::<Vec<_>>();
        let mesh_buffers = (0..frames_in_flight)
            .map(|_| create_mesh_buffer(allocator.clone(), std::mem::size_of::<GpuMesh>() as u64))
            .collect::<Vec<_>>();
        let mesh_set_layout = DescriptorSetLayout::new(
            device.clone(),
            vec![DescriptorSetLayoutBinding {
                binding: 0,
                count: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                immutable_samplers: None,
            }],
            None,
        );

        let mesh_sets = mesh_buffers
            .iter()
            .map(|buffer| {
                create_mesh_buffer_set(
                    &device,
                    &mut descriptor_set_manager,
                    buffer,
                    &mesh_set_layout,
                )
            })
            .collect::<Vec<_>>();
        let cleanup = (0..frames_in_flight).map(|_| None).collect();
        let frames = Frames {
            present_semaphores,
            render_fences,
            render_semaphores,
            command_buffers,
            depth_images,
            depth_image_views,
            descriptor_sets,
            renderables_buffers,
            max_objects,
            mesh_buffers,
            mesh_sets,
            cleanup,
        };
        let materials: Vec<MaterialLoadFn> = Vec::new();

        let camera = first_person_camera::FirstPersonCamera::new(
            cgmath::Vector3::new(0.0, 0.0, -2.0),
            cgmath::Vector3::new(0.0, 0.0, 1.0),
        );
        let input = Input::default();
        window.set_grab(true);
        sdl.mouse().set_relative_mouse_mode(true);
        let materials = materials
            .into_iter()
            .map(|f| {
                std::thread::sleep(std::time::Duration::from_millis(50000));
                Some((
                    f(PipelineCreationParams {
                        set_layouts: vec![&global_set_layout],
                        texture_set_layout: texture_set_layout.clone(),
                        device: &device,
                        width,
                        height,
                        render_pass: &render_pass,
                        opt: &opt,
                    }),
                    f,
                ))
            })
            .collect();
        let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::empty());
        let fence = Fence::new(device.clone(), &fence_info, label!("TransferContextFence"));
        let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(transfer_queue_family)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );
        let transfer_command_pool = CommandPool::new(
            device.clone(),
            &command_pool_info,
            label!("TransferContextCommandPool"),
        );
        let transfer_context = Arc::new(TransferContext {
            transfer_queue,
            command_pool: transfer_command_pool,
            fence,
        });
        let show_fps = false;
        let egui = None;
        let image_loader = image::Loader {
            device: device.clone(),
            transfer_context: transfer_context.clone(),
            allocator: allocator.clone(),
        };
        let sampler = Arc::new(sampler);
        let mouse_captured = true;
        let last_frame_times = VecDeque::new();
        let vsync = false;
        let egui_on = false;
        let last_frames_in_flight = frames_in_flight;
        let last_vsync = vsync;
        let mut app = Self {
            texture_set_layout,
            vsync,
            last_vsync,
            width,
            height,
            last_frames_in_flight,
            last_frame_times,
            mouse_captured,
            image_loader,
            egui,
            mesh_set_layout,
            graphics_queue_family,
            egui_on,
            sampler,
            descriptor_set_manager,
            global_set_layout,
            frames_in_flight,
            materials,
            opt,
            camera,
            global_uniform,
            print_fps: show_fps,
            start,
            present_mode,
            sdl,
            format,
            physical_device,
            event_pump,
            window,
            instance,
            graphics_queue,
            device,
            entry,
            surface,
            messenger,
            transfer_context,
            command_pool,
            frame_buffers,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            render_pass,
            frames,
            frame_number,
            last_time,
            allocator,
            supported_present_modes,
            input,
            physical_device_properties,
            uniform_buffer,
        };
        let f = app.frames_in_flight;
        let egui = EruptEgui::new(&mut app, f);
        app.egui = Some(egui);
        app
    }
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
    fn pipeline_creation_params(&self) -> PipelineCreationParams {
        let (width, height) = self.window.size();
        PipelineCreationParams {
            set_layouts: vec![&self.global_set_layout],
            texture_set_layout: self.texture_set_layout.clone(),
            device: &self.device,
            width,
            height,
            render_pass: &self.render_pass,
            opt: &self.opt,
        }
    }
    fn register_pipeline(&mut self, create_fn: MaterialLoadFn) -> usize {
        let material = create_fn(self.pipeline_creation_params());
        if let Some(slot) = self.materials.iter_mut().find(|s| s.is_none()) {
            *slot = Some((material, create_fn));
        } else {
            self.materials.push(Some((material, create_fn)));
        }
        self.materials.len() - 1
    }
    /// returns whether a resize was necessary
    #[allow(clippy::too_many_lines)]
    fn resize(&mut self) -> bool {
        let (w, h) = self.window.size();
        let num_frames_in_flight_changed = self.last_frames_in_flight != self.frames_in_flight;
        if w == self.width
            && h == self.height
            && !num_frames_in_flight_changed
            && self.last_vsync == self.vsync
        {
            return false;
        }
        self.width = w;
        self.height = h;
        self.last_frames_in_flight = self.frames_in_flight;
        self.last_vsync = self.vsync;
        if self.vsync {
            self.present_mode = vk::PresentModeKHR::FIFO_KHR;
        } else {
            self.present_mode = if self
                .supported_present_modes
                .contains(&vk::PresentModeKHR::MAILBOX_KHR)
            {
                vk::PresentModeKHR::MAILBOX_KHR
            } else {
                vk::PresentModeKHR::IMMEDIATE_KHR
            };
        }
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .free_command_buffers(*self.command_pool, &self.frames.command_buffers);
        };
        if num_frames_in_flight_changed {
            if let Some(egui) = self.egui.as_mut() {
                egui.adjust_frames_in_flight(self.frames_in_flight);
            }
            let (render_fences, present_semaphores, render_semaphores) =
                create_sync_objects(&self.device, self.frames_in_flight);
            self.frames.render_fences = render_fences;
            self.frames.present_semaphores = present_semaphores;
            self.frames.render_semaphores = render_semaphores;

            let uniform_buffer = create_uniform_buffer(
                self.allocator.clone(),
                self.frames_in_flight as u64,
                &self.physical_device_properties.limits,
            );
            self.uniform_buffer = uniform_buffer;
            let max_objects = self.frames.max_objects.get(0).copied().unwrap_or(8);
            self.frames
                .max_objects
                .resize_with(self.frames_in_flight, || max_objects);
            self.frames
                .renderables_buffers
                .resize_with(self.frames_in_flight, || {
                    create_renderables_buffer(self.allocator.clone(), max_objects as u64)
                });
            self.frames
                .cleanup
                .resize_with(self.frames_in_flight, || None);
            //self.frames.descriptor_sets.clear();
            let descriptor_sets = (0..self.frames_in_flight)
                .zip(&mut self.frames.renderables_buffers)
                .zip(&self.frames.max_objects)
                .map(|((_, object_buffer), max_objects)| {
                    create_descriptor_sets(
                        &self.device,
                        &mut self.descriptor_set_manager,
                        &self.global_set_layout,
                        &self.uniform_buffer,
                        object_buffer,
                        *max_objects,
                    )
                })
                .collect::<Vec<_>>();
            self.frames.descriptor_sets = descriptor_sets;
            let size = self.frames.mesh_buffers[0].size;
            self.frames
                .mesh_buffers
                .resize_with(self.frames_in_flight, || {
                    create_mesh_buffer(self.allocator.clone(), size)
                });
            self.frames.mesh_sets = self
                .frames
                .mesh_buffers
                .iter()
                .map(|buffer| {
                    create_mesh_buffer_set(
                        &self.device,
                        &mut self.descriptor_set_manager,
                        buffer,
                        &self.mesh_set_layout,
                    )
                })
                .collect();
        }
        let (swapchain, swapchain_images, swapchain_image_views) = create_swapchain(
            &self.instance,
            self.physical_device,
            *self.surface,
            self.format,
            &self.device,
            self.present_mode,
            Some(*self.swapchain),
        );
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = create_render_pass(self.device.clone(), self.format);
        self.frames.depth_images =
            create_depth_images(&self.allocator, w, h, self.frames_in_flight);
        self.frames.depth_image_views =
            create_depth_image_views(&self.device, &self.frames.depth_images);
        self.frame_buffers = create_framebuffers(
            &self.device,
            w,
            h,
            *self.render_pass,
            &self.swapchain_image_views,
            &self.frames.depth_image_views,
        );
        self.reload_pipelines();
        self.frames.command_buffers = create_command_buffers(
            &self.device,
            &self.command_pool,
            self.frames_in_flight.try_into().unwrap(),
        );
        true
    }
    fn reload_pipelines(&mut self) {
        for i in 0..self.materials.len() {
            let material = unsafe {
                // safe as long as material_create_params doesn't access self.materials[i] i think :D
                &mut *(self.materials[i].as_mut().unwrap() as *mut (RenderPipeline, MaterialLoadFn))
            };
            let params = self.pipeline_creation_params();
            let new_material = material.1(params);
            material.0 = new_material;
        }
    }
    fn ui(&mut self) {
        let (width, height) = self.window.size();
        self.egui.as_mut().unwrap().run(
            &self.allocator,
            &self.image_loader,
            &mut self.descriptor_set_manager,
            &self.texture_set_layout,
            (width as f32, height as f32),
            |ctx| {
                ctx.request_repaint();
                egui::Window::new("Debug Window").show(ctx, |ui| {
                    let dt = self
                        .last_frame_times
                        .get(self.last_frame_times.len() - 1)
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
            },
        );
    }
    fn update(&mut self, renderables: &[Renderable]) -> bool {
        for event in self.event_pump.poll_iter().collect::<Vec<_>>() {
            if self.mouse_captured {
                self.input.process_event(&event);
            }
            self.egui.as_mut().unwrap().process_event(&event);
            match event {
                Event::Quit { .. } => {
                    return false;
                }
                Event::KeyDown {
                    keycode: Some(keycode),
                    ..
                } => match keycode {
                    sdl2::keyboard::Keycode::Z => {
                        dbg!(Arc::strong_count(&self.device));
                    }
                    sdl2::keyboard::Keycode::F => {
                        self.print_fps = true;
                    }
                    sdl2::keyboard::Keycode::V => {
                        self.vsync = true;
                    }
                    sdl2::keyboard::Keycode::E => {
                        self.mouse_captured = !self.mouse_captured;
                        self.sdl.mouse().capture(self.mouse_captured);
                        self.window.set_grab(self.mouse_captured);
                        self.sdl
                            .mouse()
                            .set_relative_mouse_mode(self.mouse_captured);
                    }
                    sdl2::keyboard::Keycode::C => {
                        self.egui_on = !self.egui_on;
                    }
                    _ => {}
                },
                Event::KeyUp {
                    keycode: Some(sdl2::keyboard::Keycode::F),
                    ..
                } => {
                    self.print_fps = false;
                }
                Event::Window { .. } => {
                    self.resize();
                }
                _ => {}
            }
        }
        self.render(renderables);
        true
    }
    #[allow(clippy::too_many_lines)]
    fn render(&mut self, renderables: &[Renderable]) {
        #[derive(Debug)]
        struct InstancingBatch<'a> {
            pipeline: &'a RenderPipeline,
            index_vertex_buffer: &'a buffer::Allocated,
            index_count: u32,
            index_start: u32,
            instance_count: u32,
            vertex_start: u32,
            custom_set: &'a DescriptorSet,
        }

        let now = std::time::Instant::now();
        let dt = now - self.last_time;
        let time = self.start.elapsed().as_secs_f32();
        self.last_time = now;
        self.frame_number += 1;
        let (width, height) = self.window.size();
        self.camera
            .update(&self.input.make_controls(dt.as_secs_f32()));
        let aspect = width as f32 / height as f32;
        self.global_uniform.view_proj = cgmath::Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0)
            * self.camera.create_view_projection_matrix(
                aspect,
                90.0 * std::f32::consts::PI / 180.0,
                0.1,
                200.0,
            );
        self.global_uniform.time = self.start.elapsed().as_secs_f32();
        self.global_uniform.screen_width = width as f32;
        self.global_uniform.screen_height = height as f32;
        unsafe {
            if self.resize() {
                return;
            }
            let frame_index = self.frame_number % self.frames_in_flight;
            let frame = self.frames.get(frame_index);
            let gpu_wait_start = std::time::Instant::now();
            let res = self
                .device
                .wait_for_fences(&[**frame.render_fence], true, 5_000_000_000);
            if res.raw == vk::Result::TIMEOUT {
                dbg!(res.raw, "TIMEOUT");
                std::process::exit(-1);
            }
            if self.egui_on {
                self.ui();
            }
            if self.resize() {
                return;
            }
            let frame = self.frames.get(frame_index);
            let gpu_wait = gpu_wait_start.elapsed().as_secs_f32();
            self.device.reset_fences(&[**frame.render_fence]).unwrap();
            drop(frame);

            let egui_renderables = self.egui.as_ref().unwrap().renderables();
            let renderables_len = renderables.len() + egui_renderables.len();
            let renderables = renderables.iter();
            let renderables = if self.egui_on {
                renderables.chain(self.egui.as_ref().unwrap().renderables())
            } else {
                renderables.chain(&[])
            };
            self.global_uniform.renderables_count = renderables.clone().count().try_into().unwrap();

            let frame = self.frames.get(frame_index);
            if let Some(f) = frame.cleanup.take() {
                f();
            }

            let pre_render_processing_start = std::time::Instant::now();
            if renderables_len as u64 * std::mem::size_of::<GpuDataRenderable>() as u64
                > frame.renderables_buffer.size
            {
                *frame.max_objects = (renderables_len / 2 * 3).max(16);
                *frame.renderables_buffer =
                    create_renderables_buffer(self.allocator.clone(), *frame.max_objects as u64);
                *frame.descriptor_set = create_descriptor_sets(
                    &self.device,
                    &mut self.descriptor_set_manager,
                    &self.global_set_layout,
                    &self.uniform_buffer,
                    &*frame.renderables_buffer,
                    *frame.max_objects,
                );
            }
            let swapchain_image_index = {
                let res = self.device.acquire_next_image_khr(
                    *self.swapchain,
                    1_000_000_000,
                    Some(**frame.present_semaphore),
                    None,
                );
                if let Some(value) = res.value {
                    value as usize
                } else {
                    return;
                }
            };

            let clear_value = vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.15, 0.6, 0.9, 1.0],
                },
            };

            let (width, height) = self.window.size();

            let ptr = self.uniform_buffer.map();
            let global_uniform_offset: u32 = (pad_uniform_buffer_size(
                &self.physical_device_properties.limits,
                std::mem::size_of::<GlobalUniform>() as u64,
            ) * frame_index as u64)
                .try_into()
                .unwrap();
            let current_global_uniform_ptr = ptr.add(global_uniform_offset as usize);
            std::ptr::write(
                current_global_uniform_ptr as *mut GlobalUniform,
                self.global_uniform,
            );
            self.uniform_buffer.unmap();
            let objects_ptr = frame.renderables_buffer.map() as *mut GpuDataRenderable;
            let mut instancing_batches = Vec::new();
            let mut last_pipeline = usize::MAX;
            let mut last_mesh = std::ptr::null();
            let mut indirect_draw = 0;
            let meshes_len = renderables
                .clone()
                .fold((0, std::ptr::null()), |(mut i, last), r| {
                    let this_mesh = Arc::as_ptr(&r.mesh);
                    if this_mesh != last {
                        i += 1;
                    }
                    (i, this_mesh)
                })
                .0;
            let mesh_buffer_min_size = meshes_len * std::mem::size_of::<GpuMesh>();
            if mesh_buffer_min_size as u64 > frame.mesh_buffer.size {
                *frame.mesh_buffer =
                    create_mesh_buffer(self.allocator.clone(), mesh_buffer_min_size as u64 / 2 * 3);
                *frame.mesh_set = create_mesh_buffer_set(
                    &self.device,
                    &mut self.descriptor_set_manager,
                    frame.mesh_buffer,
                    &self.mesh_set_layout,
                );
            }
            let mesh_ptr = frame.mesh_buffer.map();
            let mesh_slice = std::slice::from_raw_parts_mut(mesh_ptr as *mut GpuMesh, meshes_len);
            let mut last_buffer = std::ptr::null();
            let mut last_custom_set = std::ptr::null();
            let mut meshes = Vec::new();
            let mut custom_sets = Vec::new();
            let mut mesh_index = 0;
            let renderables_slice = std::slice::from_raw_parts_mut(
                objects_ptr.cast::<GpuDataRenderable>(),
                renderables_len,
            );
            for (i, renderable) in renderables.clone().enumerate() {
                let this_pipeline = renderable.pipeline;
                let this_mesh = &*renderable.mesh;
                let this_buffer = Arc::as_ptr(&renderable.mesh.buffer);
                let this_custom_set = &*renderable.custom_set;
                if this_pipeline != last_pipeline || this_buffer != last_buffer {
                    indirect_draw += 1;
                }
                if this_custom_set as *const DescriptorSet != last_custom_set {
                    custom_sets.push(renderable.custom_set.clone());
                }
                if this_mesh as *const Mesh != last_mesh {
                    meshes.push(renderable.mesh.clone());
                    mesh_slice[mesh_index] = GpuMesh {
                        max: this_mesh.bounds.max,
                        first_index: this_mesh.index_start,
                        min: this_mesh.bounds.min,
                        index_count: this_mesh.index_count,
                        vertex_offset: this_mesh.vertex_start.try_into().unwrap(),
                        _padding_0: Default::default(),
                        _padding_1: Default::default(),
                        _padding_2: Default::default(),
                    };
                    mesh_index += 1;
                }
                renderables_slice[i].custom_id = 0;
                if this_pipeline != last_pipeline
                    || this_mesh as *const Mesh != last_mesh
                    || this_buffer != last_buffer
                    || this_custom_set as *const DescriptorSet != last_custom_set
                {
                    renderables_slice[instancing_batches.len()].custom_id = 0;
                    renderables_slice[instancing_batches.len()].first_instance =
                        i.try_into().unwrap();
                    instancing_batches.push(InstancingBatch {
                        pipeline: &self.materials[renderable.pipeline].as_ref().unwrap().0,
                        index_vertex_buffer: &this_mesh.buffer,
                        index_count: this_mesh.index_count,
                        index_start: this_mesh.index_start,
                        vertex_start: this_mesh.vertex_start,
                        instance_count: 0,
                        custom_set: &*renderable.custom_set,
                    });
                }
                last_pipeline = this_pipeline;
                last_buffer = this_buffer;
                last_mesh = this_mesh as *const Mesh;
                last_custom_set = this_custom_set as *const DescriptorSet;
                renderables_slice[i].transform = renderable.transform;
                renderables_slice[i].custom_id = renderable.custom_id;
                renderables_slice[i].uncullable = if renderable.uncullable { 1 } else { 0 };
                renderables_slice[i].mesh = TryInto::<u32>::try_into(mesh_index).unwrap() - 1;
                let instancing_batch =
                    TryInto::<u32>::try_into(instancing_batches.len()).unwrap() - 1;
                renderables_slice[i].draw = indirect_draw - 1;
                renderables_slice[i].batch = instancing_batch;
                renderables_slice[i].redirect = i.try_into().unwrap();
                instancing_batches.last_mut().unwrap().instance_count += 1;
            }
            frame.mesh_buffer.unmap();
            frame.renderables_buffer.unmap();
            *frame.cleanup = Some(Box::new(|| {
                drop(meshes);
                drop(custom_sets);
            }) as Box<dyn FnOnce()>);
            self.last_frame_times.push_back((
                time,
                dt.as_secs_f32(),
                gpu_wait,
                pre_render_processing_start.elapsed().as_secs_f32(),
            ));
            while self.last_frame_times.len() > 1000 {
                self.last_frame_times.pop_front();
            }
            let cmd = *frame.command_buffer;
            self.device.reset_command_buffer(cmd, None).unwrap();
            let begin_info = vk::CommandBufferBeginInfoBuilder::new()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin_info).unwrap();
            let clear_values = &[
                clear_value,
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        ..erupt::vk1_0::ClearDepthStencilValue::default()
                    },
                },
            ];
            let rp_begin_info = vk::RenderPassBeginInfoBuilder::new()
                .clear_values(clear_values)
                .render_pass(*self.render_pass)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D { width, height },
                })
                .framebuffer(*self.frame_buffers[swapchain_image_index]);
            self.device
                .cmd_begin_render_pass(cmd, &rp_begin_info, vk::SubpassContents::INLINE);
            let mut first_instance = 0;
            //std::thread::sleep(std::time::Duration::from_millis(50));
            //dbg!(&instancing_batches);
            for batch in &instancing_batches {
                self.device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[**batch.index_vertex_buffer, **frame.renderables_buffer],
                    &[0, 0],
                );
                self.device.cmd_bind_index_buffer(
                    cmd,
                    **batch.index_vertex_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    *batch.pipeline.pipeline,
                );

                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    *batch.pipeline.pipeline_layout,
                    0,
                    &[**frame.descriptor_set, **batch.custom_set],
                    &[global_uniform_offset],
                );

                self.device.cmd_draw_indexed(
                    cmd,
                    batch.index_count,
                    batch.instance_count,
                    batch.index_start,
                    batch.vertex_start.try_into().unwrap(),
                    first_instance,
                );
                first_instance += batch.instance_count;
            }
            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd).unwrap();

            let wait_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
            let wait_semaphores = &[**frame.present_semaphore];
            let signal_semaphores = &[**frame.render_semaphore];
            let command_buffers = &[cmd];
            let wait_dst_stage_mask = &[wait_stage];
            let submit = vk::SubmitInfoBuilder::new()
                .wait_dst_stage_mask(wait_dst_stage_mask)
                .wait_semaphores(wait_semaphores)
                .signal_semaphores(signal_semaphores)
                .command_buffers(command_buffers);

            self.device
                .queue_submit(self.graphics_queue, &[submit], Some(**frame.render_fence))
                .unwrap();

            let image_indices = &[swapchain_image_index.try_into().unwrap()];
            let swapchains = &[*self.swapchain];
            let wait_semaphores = &[**frame.render_semaphore];
            let present_info = vk::PresentInfoKHRBuilder::new()
                .swapchains(swapchains)
                .wait_semaphores(wait_semaphores)
                .image_indices(image_indices);
            let res = self
                .device
                .queue_present_khr(self.graphics_queue, &present_info);
            if res.is_err() {
                println!("{:#?}", res);
            }
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            println!(label!("Waiting for Fences"));
            self.device
                .wait_for_fences(
                    self.frames
                        .render_fences
                        .iter()
                        .map(|f| **f)
                        .collect::<Vec<_>>()
                        .as_slice(),
                    true,
                    1_000_000_000,
                )
                .unwrap();
            println!(label!("Waiting for Idle"));
            self.device.device_wait_idle().unwrap();
            if self.messenger.is_some() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.messenger, None);
            }
            println!(label!("DROPPING App"));
        };
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let opt = Opt::from_args();
    let mut app = App::new(opt);
    let vertices = vec![
        Vertex {
            position: cgmath::Vector3::new(1.0, 1.0, 0.0),
            normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
            uv: cgmath::Vector2::new(1.0, 1.0),
        },
        Vertex {
            position: cgmath::Vector3::new(-1.0, 1.0, 0.0),
            normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
            uv: cgmath::Vector2::new(0.0, 1.0),
        },
        Vertex {
            position: cgmath::Vector3::new(0.0, -1.0, 0.0),
            normal: cgmath::Vector3::new(0.0, 0.0, 0.0),
            uv: cgmath::Vector2::new(0.5, 0.0),
        },
    ];
    let rgb_pipeline = app.register_pipeline(rgb_pipeline(app.device.clone()));
    let mesh_pipeline = app.register_pipeline(mesh_pipeline(app.device().clone()));
    let mut triangle_mesh = Mesh::new(
        &vertices,
        &[0, 1, 2, 2, 1, 0],
        app.allocator.clone(),
        &app.transfer_context,
        &app.device,
        true,
    );
    dbg!("suzanne");
    let mut suzanne_mesh = Mesh::load(
        app.allocator.clone(),
        "./assets/suzanne.obj",
        &app.transfer_context,
        &app.device,
    )
    .unwrap()
    .swap_remove(0);
    Mesh::combine_meshes(
        [&mut suzanne_mesh, &mut triangle_mesh],
        app.allocator.clone(),
        &app.transfer_context,
        &app.device,
    );
    let image = image::Allocated::open(
        &app.image_loader,
        &PathBuf::from("./assets/lost_empire-RGBA.png"),
    );
    let mut renderables = Vec::new();
    let texture = Arc::new(image::Texture::new(
        &app.device().clone(),
        &mut app.descriptor_set_manager,
        &app.texture_set_layout,
        image,
        app.sampler.clone(),
    ));
    let mut suzanne = Renderable {
        mesh: Arc::new(suzanne_mesh),
        pipeline: rgb_pipeline,
        transform: cgmath::Matrix4::identity(),
        custom_set: texture.set.clone(),
        custom_id: 0,
        uncullable: false,
    };

    let s = opt.test;
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
        app.allocator.clone(),
        "./assets/lost_empire.obj",
        &app.transfer_context,
        &app.device,
    )
    .unwrap();
    dbg!("empire");
    let empire = empire_meshes.into_iter().map(|mesh| Renderable {
        mesh: Arc::new(mesh),
        pipeline: mesh_pipeline,
        transform: cgmath::Matrix4::identity(),
        custom_set: texture.set.clone(),
        custom_id: 0,
        uncullable: false,
    });

    let mut triangle = Renderable {
        mesh: Arc::new(triangle_mesh),
        pipeline: mesh_pipeline,
        transform: cgmath::Matrix4::from_scale(1.0),
        custom_set: texture.set.clone(),
        custom_id: 0,
        uncullable: false,
    };
    triangle.transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(-3.0, 0.0, 0.0))
        * cgmath::Matrix4::from_scale(2.0);
    renderables.push(triangle.clone());
    triangle.transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(-8.0, 0.0, 0.0))
        * cgmath::Matrix4::from_scale(4.0);
    renderables.push(triangle);

    for mut empire in empire {
        empire.transform = cgmath::Matrix4::from_translation(cgmath::Vector3::new(0.0, 0.0, 0.0));
        renderables.push(empire);
    }
    let mut last_time = std::time::Instant::now();
    loop {
        let now = std::time::Instant::now();
        let dt = now - last_time;
        let _dt = dt.as_secs_f32();
        if !app.update(&renderables) {
            break;
        }
        last_time = now;
    }
}
fn mesh_pipeline(device: Arc<Device>) -> MaterialLoadFn {
    let vert_shader = ShaderModule::load(device.clone(), "./shaders/mesh.vert.spv").unwrap();
    let frag_shader = ShaderModule::load(device, "./shaders/mesh.frag.spv").unwrap();
    Box::new(move |args| {
        let mut set_layouts = args
            .set_layouts
            .iter()
            .map(|l| ***l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        set_layouts.push(**args.texture_set_layout);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("MeshPipelineLayout"),
        );
        let width = args.width;
        let height = args.height;
        let pipeline = {
            let vertex_description = Vertex::get_vertex_description();
            let vertex_input_info = vk::PipelineVertexInputStateCreateInfoBuilder::new()
                .vertex_attribute_descriptions(&vertex_description.attributes)
                .vertex_binding_descriptions(&vertex_description.bindings);
            let vertex_input_info = &vertex_input_info;
            let shader_stages = [
                pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::VERTEX, &vert_shader),
                pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::FRAGMENT, &frag_shader),
            ];
            let color_blend_attachment = create_pipeline_color_blend_attachment_state();
            /* let color_blend_attachment = color_blend_attachment
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::SUBTRACT); */
            let view_port = create_full_view_port(args.width, args.height);
            Pipeline::new(
                args.device.clone(),
                **args.render_pass,
                PipelineDesc {
                    view_port,
                    scissor: vk::Rect2DBuilder::new()
                        .offset(vk::Offset2D { x: 0, y: 0 })
                        .extent(vk::Extent2D { width, height }),
                    color_blend_attachment,
                    shader_stages: &shader_stages,
                    vertex_input_info,
                    input_assembly_state: &input_assembly_create_info(
                        vk::PrimitiveTopology::TRIANGLE_LIST,
                    ),
                    rasterization_state: &rasterization_state_create_info(if args.opt.wireframe {
                        vk::PolygonMode::LINE
                    } else {
                        vk::PolygonMode::FILL
                    }),
                    multisample_state: &multisampling_state_create_info(),
                    layout: *pipeline_layout,
                    depth_stencil: &depth_stencil_create_info(true, true, vk::CompareOp::LESS),
                },
                String::from(label!("MeshPipeline")),
            )
        };
        RenderPipeline {
            pipeline_layout,
            pipeline,
        }
    })
}

fn rgb_pipeline(device: Arc<Device>) -> MaterialLoadFn {
    let frag_shader =
        ShaderModule::load(device.clone(), "./shaders/rgb_triangle.frag.spv").unwrap();
    let vert_shader = ShaderModule::load(device, "./shaders/rgb_triangle.vert.spv").unwrap();
    let vertex_description = Vertex::get_vertex_description();
    Box::new(move |args| {
        let width = args.width;
        let height = args.height;
        let shader_stages = [
            pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::VERTEX, &vert_shader),
            pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::FRAGMENT, &frag_shader),
        ];
        let mut set_layouts = args
            .set_layouts
            .into_iter()
            .map(|l| **l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        set_layouts.push(**args.texture_set_layout);
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("RgbPipelineLayout"),
        );
        let view_port = create_full_view_port(width, height);
        let color_blend_attachment = create_pipeline_color_blend_attachment_state();
        let pipeline = Pipeline::new(
            args.device.clone(),
            **args.render_pass,
            PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment,
                shader_stages: &shader_stages,
                vertex_input_info: &vk::PipelineVertexInputStateCreateInfoBuilder::new()
                    .vertex_attribute_descriptions(&vertex_description.attributes)
                    .vertex_binding_descriptions(&vertex_description.bindings),
                input_assembly_state: &input_assembly_create_info(
                    vk::PrimitiveTopology::TRIANGLE_LIST,
                ),
                rasterization_state: &rasterization_state_create_info(if args.opt.wireframe {
                    vk::PolygonMode::LINE
                } else {
                    vk::PolygonMode::FILL
                }),
                multisample_state: &multisampling_state_create_info(),
                layout: *pipeline_layout,
                depth_stencil: &depth_stencil_create_info(true, true, vk::CompareOp::LESS),
            },
            String::from(label!("RgbPipeline")),
        );
        RenderPipeline {
            pipeline_layout,
            pipeline,
        }
    })
}
