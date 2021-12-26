#![deny(clippy::all, clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::struct_excessive_bools,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr,
    clippy::items_after_statements
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
mod shader_types;
mod utils;
use crate::descriptor_sets::DescriptorSet;
use crate::descriptor_sets::DescriptorSetLayoutBinding;
use cgmath::SquareMatrix;
use descriptor_sets::{DescriptorSetLayout, DescriptorSetManager};
use erupt::{cstr, vk};
use first_person_camera::FirstPersonCamera;
use frame::FrameData;
use frame::Frames;
use gui::EruptEgui;
use handles::{
    Allocator, CommandPool, Device, Fence, Framebuffer, ImageView, Instance, Pipeline,
    PipelineDesc, PipelineLayout, RenderPass, Sampler, Semaphore, ShaderModule, Surface, Swapchain,
};
use input::Input;
use mesh::{Mesh, Vertex};
use sdl2::{event::Event, EventPump};
use std::sync::Mutex;
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
    let msg = CStr::from_ptr((*p_callback_data).p_message).to_string_lossy();
    if (message_severity.0 & vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO_EXT.0) != 0 {
        log::info!("{}", msg);
    }
    if (message_severity.0 & vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING_EXT.0) != 0 {
        log::warn!("{}", msg);
        panic!();
    }
    if (message_severity.0 & vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT.0) != 0 {
        log::error!("{}", msg);
        panic!();
    }

    vk::FALSE
}

#[derive(Copy, Clone, Debug, StructOpt)]
struct Opt {
    /// attempt to enable validation layers
    #[structopt(short, long)]
    validation_layers: bool,
    /// how many frames are rendered to at the same time
    #[structopt(short, long)]
    frames_in_flight: usize,
    test: u32,
}

#[derive(Clone, Debug)]
pub struct Renderable {
    pub transform: cgmath::Matrix4<f32>,
    pub mesh: Arc<Mesh>,
    pub custom_set: Option<Arc<DescriptorSet>>,
    pub custom_id: u32,
    pub uncullable: bool,
    pub pipeline: usize,
}

struct PipelineCreationParams<'a> {
    device: &'a Arc<Device>,
    width: u32,
    height: u32,
    render_pass: &'a RenderPass,
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

#[derive(Debug)]
struct InstancingBatch<'a> {
    pipeline: &'a Pipeline,
    index_vertex_buffer: &'a buffer::Allocated,
    index_count: u32,
    index_start: u32,
    instance_count: u32,
    vertex_start: u32,
    renderables: Vec<&'a Renderable>,
    custom_set: Option<&'a DescriptorSet>,
    mesh: u32,
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

type MaterialLoadFn = Box<dyn for<'a> Fn(PipelineCreationParams<'a>) -> Pipeline>;

pub struct TransferContext {
    pub transfer_queue: vk::Queue,
    pub command_pool: CommandPool,
    pub fence: Fence,
}

#[allow(clippy::struct_excessive_bools)]
pub struct Renderer {
    #[allow(dead_code)]
    graphics_queue_family: u32,
    start: std::time::Instant,
    supported_present_modes: Vec<vk::PresentModeKHR>,
    present_mode: vk::PresentModeKHR,
    swapchain: Swapchain,
    mesh_set_layout: DescriptorSetLayout,
    sampler: Arc<Sampler>,
    width: u32,
    height: u32,
    vsync: bool,
    graphics_queue: vk::Queue,
    messenger: Option<vk::DebugUtilsMessengerEXT>,
    descriptor_set_manager: DescriptorSetManager,
    command_pool: CommandPool,
    frame_buffers: Vec<Framebuffer>,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<ImageView>,
    render_pass: RenderPass,
    frames: Frames,
    uniform_buffer: buffer::Allocated,
    frames_in_flight: usize,
    frame_number: usize,
    image_loader: image::Loader,
    transfer_context: Arc<TransferContext>,
    physical_device_properties: vk::PhysicalDeviceProperties,
    materials: Vec<(Pipeline, MaterialLoadFn)>,
    global_uniform: shader_types::GlobalUniform,
    global_set_layout: DescriptorSetLayout,
    surface: Surface,
    format: vk::SurfaceFormatKHR,
    physical_device: vk::PhysicalDevice,
    allocator: Arc<Allocator>,
    device: Arc<Device>,
    instance: Arc<Instance>,
    /// This field must be dropped last because other fields might rely on the objects inside it being alive.
    #[allow(dead_code)]
    keep_alive: Vec<Box<dyn std::any::Any + 'static>>,
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

impl Renderer {
    #[allow(clippy::too_many_lines)]
    fn new(
        window: Arc<Mutex<sdl2::video::Window>>,
        vsync: bool,
        frames_in_flight: usize,
        validation_layers: bool,
    ) -> Self {
        let w = window.lock().unwrap();
        let messenger_info = create_debug_messenger_info();

        let (instance, entry, mut device_extensions) =
            create_instance(&w, validation_layers, &messenger_info);
        device_extensions.push(vk::KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME);
        device_extensions.push(vk::KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        //device_extensions.push(vk::KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        //device_extensions.push(vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        let instance = Arc::new(Instance::new(instance, entry));
        let mut device_layers = Vec::new();

        let messenger = if validation_layers {
            device_layers.push(LAYER_KHRONOS_VALIDATION);
            Some(create_debug_messenger(
                &instance,
                &messenger_info,
                validation_layers,
            ))
        } else {
            None
        };

        let surface = Surface::new(instance.clone(), &w);
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

        log::info!("Using physical device: {:?}", unsafe {
            CStr::from_ptr(physical_device_properties.device_name.as_ptr())
        });

        let (device, graphics_queue, transfer_queue) = create_device_and_queue(
            Arc::clone(&instance),
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

        let width = w.size().0;
        let height = w.size().1;
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

        let frame_number = 0;
        let start = std::time::Instant::now();
        let uniform_buffer = create_uniform_buffer(
            allocator.clone(),
            frames_in_flight as u64,
            &physical_device_properties.limits,
        );

        let global_uniform = shader_types::GlobalUniform {
            view: cgmath::Matrix4::from_scale(0.1),
            proj: cgmath::Matrix4::from_scale(0.1),
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
        let descriptor_sets = (0..frames_in_flight)
            .map(|i| {
                create_descriptor_sets(
                    &device,
                    &mut descriptor_set_manager,
                    &global_set_layout,
                    &uniform_buffer,
                    &renderables_buffers[i],
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

        let materials = materials
            .into_iter()
            .map(|f| {
                std::thread::sleep(std::time::Duration::from_millis(50000));
                (
                    f(PipelineCreationParams {
                        device: &device,
                        width,
                        height,
                        render_pass: &render_pass,
                    }),
                    f,
                )
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
        let image_loader = image::Loader {
            device: device.clone(),
            transfer_context: transfer_context.clone(),
            allocator: allocator.clone(),
        };
        let sampler = Arc::new(sampler);
        drop(w);
        let keep_alive = Vec::from([Box::new(window) as Box<dyn std::any::Any + 'static>]);
        Self {
            graphics_queue_family,
            start,
            supported_present_modes,
            present_mode,
            swapchain,
            mesh_set_layout,
            sampler,
            width,
            height,
            vsync,
            graphics_queue,
            messenger,
            descriptor_set_manager,
            command_pool,
            frame_buffers,
            swapchain_images,
            swapchain_image_views,
            render_pass,
            frames,
            uniform_buffer,
            frames_in_flight,
            frame_number,
            image_loader,
            transfer_context,
            physical_device_properties,
            materials,
            global_uniform,
            global_set_layout,
            surface,
            format,
            physical_device,
            allocator,
            device,
            instance,
            keep_alive,
        }
    }
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
    fn pipeline_creation_params(&self) -> PipelineCreationParams {
        PipelineCreationParams {
            device: &self.device,
            width: self.width,
            height: self.height,
            render_pass: &self.render_pass,
        }
    }
    fn register_pipeline(&mut self, create_fn: MaterialLoadFn) -> usize {
        let material = create_fn(self.pipeline_creation_params());
        self.materials.push((material, create_fn));
        self.materials.len() - 1
    }
    fn set_present_mode(&mut self) {
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
    }
    fn wait_idle(&self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .free_command_buffers(*self.command_pool, &self.frames.command_buffers);
        }
    }
    fn change_num_frames_in_flight(&mut self) {
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
        let descriptor_sets = self
            .frames
            .renderables_buffers
            .iter()
            .map(|object_buffer| {
                create_descriptor_sets(
                    &self.device,
                    &mut self.descriptor_set_manager,
                    &self.global_set_layout,
                    &self.uniform_buffer,
                    object_buffer,
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
    fn recreate_swapchain(&mut self) {
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
        self.frames.depth_images = create_depth_images(
            &self.allocator,
            self.width,
            self.height,
            self.frames_in_flight,
        );
        self.frames.depth_image_views =
            create_depth_image_views(&self.device, &self.frames.depth_images);
        self.frame_buffers = create_framebuffers(
            &self.device,
            self.width,
            self.height,
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
    }
    /// returns whether a resize was necessary
    fn resize(&mut self, width: u32, height: u32, frames_in_flight: usize, vsync: bool) -> bool {
        let num_frames_in_flight_changed = frames_in_flight != self.frames_in_flight;
        if width == self.width
            && height == self.height
            && !num_frames_in_flight_changed
            && vsync == self.vsync
        {
            return false;
        }
        self.width = width;
        self.height = height;
        self.vsync = vsync;
        self.frames_in_flight = frames_in_flight;
        self.set_present_mode();
        self.wait_idle();
        if num_frames_in_flight_changed {
            self.change_num_frames_in_flight();
        }
        self.recreate_swapchain();
        true
    }
    fn reload_pipelines(&mut self) {
        for i in 0..self.materials.len() {
            let material = unsafe {
                // safe as long as pipeline_create_params() doesn't access self.materials[i] i think :D
                &mut *(&mut self.materials[i] as *mut (Pipeline, MaterialLoadFn))
            };
            let params = self.pipeline_creation_params();
            let new_material = material.1(params);
            material.0 = new_material;
        }
    }
    fn update_global_uniform(
        &mut self,
        frame_index: usize,
        view_proj: cgmath::Matrix4<f32>,
        renderables: u32,
    ) -> u32 {
        self.global_uniform.view_proj =
            cgmath::Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0) * view_proj;
        self.global_uniform.time = self.start.elapsed().as_secs_f32();
        self.global_uniform.screen_width = self.width as f32;
        self.global_uniform.screen_height = self.height as f32;
        self.global_uniform.renderables_count = renderables;
        let ptr = self.uniform_buffer.map();
        let global_uniform_offset: u32 = (pad_uniform_buffer_size(
            &self.physical_device_properties.limits,
            std::mem::size_of::<shader_types::GlobalUniform>() as u64,
        ) * frame_index as u64)
            .try_into()
            .unwrap();
        unsafe {
            let current_global_uniform_ptr = ptr.add(global_uniform_offset as usize);
            std::ptr::write(
                current_global_uniform_ptr as *mut shader_types::GlobalUniform,
                self.global_uniform,
            );
        }
        self.uniform_buffer.unmap();
        global_uniform_offset
    }
    fn batch_renderables<'a>(
        materials: &'a [(Pipeline, MaterialLoadFn)],
        renderables: impl IntoIterator<Item = &'a Renderable>,
    ) -> (
        Vec<InstancingBatch<'a>>,
        Vec<Arc<Mesh>>,
        Vec<Option<Arc<DescriptorSet>>>,
    ) {
        let mut last_mesh: *const Mesh = std::ptr::null();
        let mut last_pipeline: Option<usize> = None;
        let mut last_custom_set: *const DescriptorSet = std::ptr::null();
        let mut batches: Vec<InstancingBatch> = Vec::new();
        let mut meshes: Vec<Arc<Mesh>> = Vec::new();
        let mut custom_descriptor_sets: Vec<Option<Arc<DescriptorSet>>> = Vec::new();
        for renderable in renderables {
            let this_mesh = &*renderable.mesh as *const Mesh;
            let this_pipeline = Some(renderable.pipeline);
            let this_custom_set = renderable
                .custom_set
                .as_ref()
                .map(|s| Arc::as_ptr(s))
                .unwrap_or(std::ptr::null());
            if this_mesh != last_mesh {
                meshes.push(Arc::clone(&renderable.mesh));
            }
            if this_custom_set != last_custom_set {
                custom_descriptor_sets.push(renderable.custom_set.as_ref().map(Arc::clone));
            }
            if this_mesh == last_mesh
                && this_pipeline == last_pipeline
                && this_custom_set == last_custom_set
            {
                let last = batches.last_mut().unwrap();
                last.instance_count += 1;
                last.renderables.push(renderable);
                continue;
            }
            last_mesh = this_mesh;
            last_pipeline = this_pipeline;
            last_custom_set = this_custom_set;

            let pipeline = &materials.get(renderable.pipeline).unwrap().0;
            batches.push(InstancingBatch {
                custom_set: renderable.custom_set.as_ref().map(|d| &**d),
                index_count: renderable.mesh.index_count,
                index_start: renderable.mesh.index_start,
                index_vertex_buffer: &renderable.mesh.buffer,
                instance_count: 1,
                pipeline,
                vertex_start: renderable.mesh.vertex_start,
                mesh: TryInto::<u32>::try_into(meshes.len()).unwrap() - 1,
                renderables: Vec::from([renderable]),
            });
        }
        (batches, meshes, custom_descriptor_sets)
    }
    fn write_renderables(buffer: &buffer::Allocated, instancing_batches: &[InstancingBatch]) {
        let ptr = buffer.map().cast::<shader_types::Object>() as *mut shader_types::Object;
        let num_renderables = instancing_batches.iter().map(|b| b.renderables.len()).sum();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, num_renderables) };
        let mut i = 0;
        for (batch_id, batch) in instancing_batches.iter().enumerate() {
            for renderable in &batch.renderables {
                slice[i as usize].redirect = i;
                slice[i as usize].transform = renderable.transform;
                slice[i as usize].batch = batch_id.try_into().unwrap();
                slice[i as usize].draw = batch_id.try_into().unwrap();
                slice[i as usize].uncullable = if renderable.uncullable { 1 } else { 0 };
                slice[i as usize].unused_3 = 0;
                slice[i as usize].custom_set = renderable.custom_id;
                slice[i as usize].mesh = batch.mesh;
                i += 1;
            }
        }
        buffer.unmap();
    }
    fn write_meshes(buffer: &buffer::Allocated, meshes: &[Arc<Mesh>]) {
        let ptr = buffer.map().cast::<GpuMesh>() as *mut GpuMesh;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, meshes.len()) };
        for (i, mesh) in meshes.iter().enumerate() {
            slice[i] = GpuMesh {
                max: mesh.bounds.max,
                first_index: mesh.index_start,
                min: mesh.bounds.min,
                index_count: mesh.index_count,
                vertex_offset: mesh.vertex_start.try_into().unwrap(),
                _padding_0: Default::default(),
                _padding_1: Default::default(),
                _padding_2: Default::default(),
            };
        }
        buffer.unmap();
    }
    fn wait_for_next_frame(&mut self) -> (usize, frame::FrameDataMut) {
        let frame_index = self.frame_number % self.frames_in_flight;
        let frame = self.frames.get_mut(frame_index);
        unsafe {
            self.device
                .wait_for_fences(&[**frame.render_fence], true, 5_000_000_000)
                .unwrap();
        }
        let frame = self.frames.get_mut(frame_index);
        unsafe {
            self.device.reset_fences(&[**frame.render_fence]).unwrap();
        }
        (frame_index, frame)
    }
    fn resize_renderable_buffer_if_necessary(&mut self, frame: usize, renderables_count: usize) {
        let frame = self.frames.get_mut(frame);
        if renderables_count as u64 * std::mem::size_of::<shader_types::Object>() as u64
            > frame.renderables_buffer.size
        {
            let new_length = (renderables_count / 2 * 3).max(16) as u64;
            *frame.renderables_buffer =
                create_renderables_buffer(Arc::clone(&self.allocator), new_length);
            *frame.descriptor_set = create_descriptor_sets(
                &self.device,
                &mut self.descriptor_set_manager,
                &self.global_set_layout,
                &self.uniform_buffer,
                &*frame.renderables_buffer,
            );
        }
    }
    fn acquire_swapchain_image(
        device: &Device,
        swapchain: &Swapchain,
        frame: &FrameData,
    ) -> Option<usize> {
        unsafe {
            let res = device.acquire_next_image_khr(
                **swapchain,
                1_000_000_000,
                Some(**frame.present_semaphore),
                None,
            );
            res.value.map(|value| value as usize)
        }
    }
    fn resize_mesh_buffer_if_necessary<'a>(
        &mut self,
        renderables: impl IntoIterator<Item = &'a Renderable>,
        frame_index: usize,
    ) {
        let frame = self.frames.get_mut(frame_index);
        let meshes = renderables
            .into_iter()
            .map(|r| Arc::as_ptr(&r.mesh))
            .fold((0, std::ptr::null()), |(i, last), this| {
                if this == last {
                    (i, this)
                } else {
                    (i + 1, this)
                }
            })
            .0;
        let mesh_buffer_min_size = meshes * std::mem::size_of::<GpuMesh>();
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
    }
    fn begin_render_pass(
        &self,
        frame_index: usize,
        swapchain_image_index: usize,
        clear_color: [f32; 4],
    ) -> vk::CommandBuffer {
        let frame = self.frames.get(frame_index);
        let cmd = *frame.command_buffer;
        unsafe {
            self.device.reset_command_buffer(cmd, None).unwrap();
            let begin_info = vk::CommandBufferBeginInfoBuilder::new()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin_info).unwrap();
            let clear_values = &[
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_color,
                    },
                },
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
                    extent: vk::Extent2D {
                        width: self.width,
                        height: self.height,
                    },
                })
                .framebuffer(*self.frame_buffers[swapchain_image_index]);
            self.device
                .cmd_begin_render_pass(cmd, &rp_begin_info, vk::SubpassContents::INLINE);
        }
        cmd
    }
    fn record_draws(
        &self,
        frame_index: usize,
        global_uniform_offset: u32,
        instancing_batches: &[InstancingBatch],
        cmd: vk::CommandBuffer,
    ) {
        let frame = self.frames.get(frame_index);
        let mut first_instance = 0;
        unsafe {
            for batch in instancing_batches {
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
                    **batch.pipeline,
                );

                if let Some(custom_set) = batch.custom_set {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        **batch.pipeline.layout(),
                        0,
                        &[**frame.descriptor_set, **custom_set],
                        &[global_uniform_offset],
                    );
                } else {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        **batch.pipeline.layout(),
                        0,
                        &[**frame.descriptor_set],
                        &[global_uniform_offset],
                    );
                };
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
        }
    }
    fn submit_cmd(&self, frame_index: usize, cmd: vk::CommandBuffer) {
        let frame = self.frames.get(frame_index);
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

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &[submit], Some(**frame.render_fence))
                .unwrap();
        }
    }
    fn present(&self, frame_index: usize, swapchain_image_index: usize) {
        let frame = self.frames.get(frame_index);
        let image_indices = &[swapchain_image_index.try_into().unwrap()];
        let swapchains = &[*self.swapchain];
        let wait_semaphores = &[**frame.render_semaphore];
        let present_info = vk::PresentInfoKHRBuilder::new()
            .swapchains(swapchains)
            .wait_semaphores(wait_semaphores)
            .image_indices(image_indices);
        let res = unsafe {
            self.device
                .queue_present_khr(self.graphics_queue, &present_info)
        };
        if res.is_err() {
            log::warn!("{:#?}", res);
        }
    }
    fn render(
        &mut self,
        renderables: &[Renderable],
        view_proj: cgmath::Matrix4<f32>,
    ) -> Option<(std::time::Duration, std::time::Duration)> {
        self.frame_number += 1;
        if self.resize(self.width, self.height, self.frames_in_flight, self.vsync) {
            return None;
        }
        let gpu_wait_start = std::time::Instant::now();
        let (frame_index, frame) = self.wait_for_next_frame();
        let gpu_wait = gpu_wait_start.elapsed();
        let cpu_work_start = std::time::Instant::now();
        if let Some(f) = frame.cleanup.take() {
            f(); // Drop the resources which were being used while rendering the previous frame
        }
        drop(frame);
        let global_uniform_offset = self.update_global_uniform(
            frame_index,
            view_proj,
            renderables.len().try_into().unwrap(),
        );
        self.resize_mesh_buffer_if_necessary(renderables, frame_index);
        self.resize_renderable_buffer_if_necessary(frame_index, renderables.len());
        let (instancing_batches, meshes, custom_sets) =
            Self::batch_renderables(&self.materials, renderables);
        let frame = self.frames.get_mut(frame_index);
        let swapchain_image_index =
            match Self::acquire_swapchain_image(&self.device, &self.swapchain, &frame.immu()) {
                Some(i) => i,
                None => return None,
            };
        Self::write_renderables(frame.renderables_buffer, &instancing_batches);
        Self::write_meshes(frame.mesh_buffer, &meshes);
        *frame.cleanup = Some(Box::new(|| {
            drop(meshes);
            drop(custom_sets);
        }) as Box<dyn FnOnce()>);
        drop(frame);
        let cmd = self.begin_render_pass(frame_index, swapchain_image_index, [0.15, 0.6, 0.9, 1.0]);
        self.record_draws(frame_index, global_uniform_offset, &instancing_batches, cmd);
        self.submit_cmd(frame_index, cmd);
        self.present(frame_index, swapchain_image_index);
        let cpu_work_time = cpu_work_start.elapsed();
        Some((gpu_wait, cpu_work_time))
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            log::info!(label!("Waiting for Fences"));
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
            log::info!(label!("Waiting for Idle"));
            self.device.device_wait_idle().unwrap();
            if self.messenger.is_some() {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.messenger, None);
            }
            log::info!(label!("DROPPING App"));
        };
    }
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
}

impl State {
    #[allow(clippy::too_many_lines)]
    pub fn new() -> Self {
        let (sdl, window, event_pump) = create_window();
        let window = Arc::new(Mutex::new(window));
        sdl.mouse().set_relative_mouse_mode(true);
        let opt = Opt::from_args();
        let mut renderer = Renderer::new(
            Arc::clone(&window),
            true,
            opt.frames_in_flight,
            opt.validation_layers,
        );
        {
            let mut window = window.try_lock().unwrap();
            window.set_grab(true);
        }
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
        let rgb_pipeline = renderer.register_pipeline(rgb_pipeline(renderer.device()));
        let mesh_pipeline = renderer.register_pipeline(mesh_pipeline(renderer.device()));
        let mut triangle_mesh = Mesh::new(
            &vertices,
            &[0, 1, 2, 2, 1, 0],
            renderer.allocator.clone(),
            &renderer.transfer_context,
            &renderer.device,
            true,
        );
        let mut suzanne_mesh = Mesh::load(
            renderer.allocator.clone(),
            "./assets/suzanne.obj",
            &renderer.transfer_context,
            &renderer.device,
        )
        .unwrap()
        .swap_remove(0);
        Mesh::combine_meshes(
            [&mut suzanne_mesh, &mut triangle_mesh],
            renderer.allocator.clone(),
            &renderer.transfer_context,
            &renderer.device,
        );
        let image = image::Allocated::open(
            &renderer.image_loader,
            &PathBuf::from("./assets/lost_empire-RGBA.png"),
        );
        let mut renderables = Vec::new();
        let texture = Arc::new(image::Texture::new(
            &renderer.device().clone(),
            &mut renderer.descriptor_set_manager,
            image,
            renderer.sampler.clone(),
        ));
        let mut suzanne = Renderable {
            mesh: Arc::new(suzanne_mesh),
            pipeline: rgb_pipeline,
            transform: cgmath::Matrix4::identity(),
            custom_set: None,
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
            renderer.allocator.clone(),
            "./assets/lost_empire.obj",
            &renderer.transfer_context,
            &renderer.device,
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
        let egui = EruptEgui::new(&mut renderer, opt.frames_in_flight);
        let frames_in_flight = opt.frames_in_flight;
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
        }
    }
    fn ui(&mut self) {
        let (width, height) = self.window.lock().unwrap().size();
        self.egui.run(
            &self.renderer.allocator,
            &self.renderer.image_loader,
            &mut self.renderer.descriptor_set_manager,
            (width as f32, height as f32),
            |ctx| {
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
            },
        );
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
                            Arc::strong_count(&self.renderer.device)
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
    let vert_shader = ShaderModule::load(device.clone(), "./shaders/mesh.vert.spv").unwrap();
    let frag_shader = ShaderModule::load(device.clone(), "./shaders/mesh.frag.spv").unwrap();
    let set_layouts: Vec<DescriptorSetLayout> = [
        DescriptorSetLayout::from_shader(device, &vert_shader),
        DescriptorSetLayout::from_shader(device, &frag_shader),
    ]
    .into_iter()
    .flatten()
    .collect();

    Box::new(move |args| {
        let set_layouts = set_layouts
            .iter()
            .map(|l| **l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
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
            &PipelineDesc {
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
                rasterization_state: &rasterization_state_create_info(vk::PolygonMode::FILL),
                multisample_state: &multisampling_state_create_info(),
                layout: pipeline_layout,
                depth_stencil: &depth_stencil_create_info(true, true, vk::CompareOp::LESS),
            },
            String::from(label!("MeshPipeline")),
        )
    })
}

fn rgb_pipeline(device: &Arc<Device>) -> MaterialLoadFn {
    let frag_shader =
        ShaderModule::load(device.clone(), "./shaders/rgb_triangle.frag.spv").unwrap();
    let vert_shader =
        ShaderModule::load(device.clone(), "./shaders/rgb_triangle.vert.spv").unwrap();
    let set_layouts: Vec<DescriptorSetLayout> = [
        DescriptorSetLayout::from_shader(device, &vert_shader),
        DescriptorSetLayout::from_shader(device, &frag_shader),
    ]
    .into_iter()
    .flatten()
    .collect();

    let vertex_description = Vertex::get_vertex_description();
    Box::new(move |args| {
        let width = args.width;
        let height = args.height;
        let shader_stages = [
            pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::VERTEX, &vert_shader),
            pipeline_shader_stage_create_info(vk::ShaderStageFlagBits::FRAGMENT, &frag_shader),
        ];
        let set_layouts = set_layouts
            .iter()
            .map(|l| **l)
            .collect::<Vec<vk::DescriptorSetLayout>>();
        let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&[]);
        let pipeline_layout = Arc::new(PipelineLayout::new(
            args.device.clone(),
            &pipeline_layout_info,
            label!("RgbPipelineLayout"),
        ));
        let view_port = create_full_view_port(width, height);
        let color_blend_attachment = create_pipeline_color_blend_attachment_state();
        Pipeline::new(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
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
                rasterization_state: &rasterization_state_create_info(vk::PolygonMode::FILL),
                multisample_state: &multisampling_state_create_info(),
                layout: Arc::clone(&pipeline_layout),
                depth_stencil: &depth_stencil_create_info(true, true, vk::CompareOp::LESS),
            },
            String::from(label!("RgbPipeline")),
        )
    })
}
