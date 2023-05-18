#![deny(clippy::all)]
#![allow(
    clippy::cast_precision_loss,
    clippy::struct_excessive_bools,
    clippy::cast_ptr_alignment,
    clippy::ptr_as_ptr,
    clippy::items_after_statements,
    clippy::drop_non_drop
)]
#[macro_export]
macro_rules! label {
    ( $name:expr ) => {
        concat!(label!(), " -> ", $name)
    };
    ( ) => {
        concat!(file!(), ":", line!())
    };
}
use cgmath::Matrix4;
use cgmath::SquareMatrix;
use cgmath::Vector3;
use cgmath::Vector4;
pub use descriptor_sets::DescriptorSetLayout;
pub use descriptor_sets::DescriptorSetLayoutBinding;
pub use descriptor_sets::DescriptorSetManager;
pub use erupt::vk;
use frame::FrameDataMut;
pub use handles::Allocator;
pub use handles::CommandPool;
pub use handles::Device;
pub use handles::Pipeline;
pub use handles::PipelineDesc;
pub use handles::PipelineLayout;
use handles::Queue;
use handles::QueueFamily;
pub use handles::Sampler;
pub use handles::ShaderModule;
pub use mesh::Bounds;
pub use mesh::DefaultVertex;
pub use mesh::Mesh;
pub use mesh::Vertex;
pub use mesh::VertexInfoDescription;
use shaders::IndirectDrawCommand;
pub use utils::ColorBlendAttachment;
pub use utils::DepthStencilInfo;
pub use utils::InputAssemblyState;
pub use utils::MultiSamplingState;
pub use utils::RasterizationState;
pub use vk_mem_3_erupt as vma;
pub mod buffer;
pub mod image;
pub use descriptor_sets::DescriptorSet;
pub use mesh::LoadingVertex;
pub use puffin;

mod descriptor_sets;
mod frame;
mod handles;
mod mesh;
mod utils;
use crate::handles::ComputePipeline;
use crate::handles::DebugUtilsMessenger;
use crate::utils::create_cull_set;
use crate::utils::create_indirect_buffer;
use crate::utils::create_sync_objects;
use erupt::cstr;
use frame::FrameData;
use frame::Frames;
use handles::{Fence, ImageView, Instance, RenderPass, Surface, Swapchain};
use std::fmt::Debug;
use std::sync::Mutex;
use std::{
    ffi::{c_void, CStr},
    os::raw::c_char,
    sync::Arc,
};
use utils::{
    create_command_buffers, create_debug_messenger_info, create_depth_image_views,
    create_depth_images, create_descriptor_sets, create_device_and_queue, create_framebuffers,
    create_global_descriptor_set_layout, create_instance, create_mesh_buffer,
    create_physical_device, create_render_pass, create_renderables_buffer, create_swapchain,
    create_uniform_buffer, pad_uniform_buffer_size,
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
        //panic!();
    }
    if (message_severity.0 & vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR_EXT.0) != 0 {
        log::error!("{}", msg);
        panic!();
    }

    vk::FALSE
}

#[derive(Clone, Debug)]
pub struct Renderable<'a> {
    pub transform: &'a cgmath::Matrix4<f32>,
    pub mesh: &'a Arc<Mesh>,
    pub custom_set: Option<&'a Arc<DescriptorSet>>,
    pub custom_id: u32,
    pub uncullable: bool,
    pub pipeline: &'a PipelineHandle,
}

pub struct PipelineCreationParams<'a> {
    pub device: &'a Arc<Device>,
    pub width: u32,
    pub height: u32,
    pub render_pass: &'a RenderPass,
    pub global_set_layout: &'a Arc<DescriptorSetLayout>,
}

#[allow(dead_code)]
#[derive(Debug)]
struct InstancingBatch<'a> {
    pipeline: &'a Pipeline,
    index_vertex_buffer: &'a buffer::Allocated,
    index_count: u32,
    index_start: u32,
    instance_count: u32,
    vertex_start: u32,
    renderables: Vec<Renderable<'a>>,
    custom_set: Option<&'a DescriptorSet>,
    mesh: u32,
    first_instance: u32,
}

#[derive(Debug)]
struct IndirectDraw<'a> {
    pipeline: &'a Pipeline,
    index_vertex_buffer: &'a buffer::Allocated,
    custom_set: Option<&'a DescriptorSet>,
    instancing_batches: &'a [InstancingBatch<'a>],
    first_batch: u32,
}

struct BatchingResult<'a> {
    batches: Vec<InstancingBatch<'a>>,
    meshes: Vec<Arc<Mesh>>,
    sets: Vec<Option<Arc<DescriptorSet>>>,
    pipelines: Vec<Arc<Pipeline>>,
    num_renderables: usize,
}

pub type MaterialLoadFn = Box<dyn for<'a> Fn(PipelineCreationParams<'a>) -> Pipeline>;

type PipelineStorage = slab::Slab<(Arc<Pipeline>, MaterialLoadFn)>;

pub struct PipelineHandle(Arc<Mutex<PipelineStorage>>, usize);

impl PipelineHandle {
    pub fn key(&self) -> usize {
        self.1
    }
}

impl Debug for PipelineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PipelineHandle").field(&self.1).finish()
    }
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        drop(self.0.lock().unwrap().remove(self.key()));
    }
}

pub struct TransferContext {
    pub transfer_queue: Arc<Mutex<Queue>>,
    pub graphics_family: QueueFamily,
    pub transfer_family: QueueFamily,
    pub command_pool: Mutex<CommandPool>,
    pub fence: Mutex<Fence>,
    pub device: Arc<Device>,
}

impl TransferContext {
    pub fn immediate_submit(&self, f: impl FnOnce(vk::CommandBuffer)) {
        unsafe {
            let fence = self.fence.lock().unwrap();
            let command_pool = self.command_pool.lock().unwrap();
            self.device.reset_fences(&[**fence]).unwrap();
            let alloc_info = vk::CommandBufferAllocateInfoBuilder::new()
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(**command_pool);
            let cmd = self.device.allocate_command_buffers(&alloc_info).unwrap()[0];
            let begin_info = vk::CommandBufferBeginInfoBuilder::new()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin_info).unwrap();
            f(cmd);
            self.device.end_command_buffer(cmd).unwrap();
            let cmds = &[cmd];
            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(cmds);
            let queue = self.transfer_queue.lock().unwrap();
            self.device
                .queue_submit(**queue, &[submit_info], **fence)
                .unwrap();
            drop(queue);
            self.device
                .wait_for_fences(&[**fence], true, 1_000_000_000)
                .unwrap();
            self.device
                .reset_command_pool(**command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
            drop(command_pool);
            drop(fence);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub top: Vector3<f32>,
    pub bottom: Vector3<f32>,
    pub right: Vector3<f32>,
    pub left: Vector3<f32>,
    pub far: Vector3<f32>,
    pub near: Vector3<f32>,
    pub far_distance: f32,
    pub near_distance: f32,
}

pub struct Renderer {
    compute_pipeline: ComputePipeline,
    compute_set_layout: Arc<DescriptorSetLayout>,
    start: std::time::Instant,
    supported_present_modes: Vec<vk::PresentModeKHR>,
    present_mode: vk::PresentModeKHR,
    swapchain: Swapchain,
    max_image_count: u32,
    sampler: Arc<Sampler>,
    width: u32,
    height: u32,
    vsync: bool,
    graphics_queue: Arc<Mutex<Queue>>,
    messenger: Option<DebugUtilsMessenger>,
    descriptor_set_manager: Arc<DescriptorSetManager>,
    command_pool: CommandPool,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    render_pass: RenderPass,
    frames: Frames,
    uniform_buffer: buffer::Allocated,
    frames_in_flight: usize,
    frame_number: usize,
    image_loader: image::Loader,
    transfer_context: Arc<TransferContext>,
    physical_device_properties: vk::PhysicalDeviceProperties,
    materials: Arc<Mutex<PipelineStorage>>,
    global_uniform: shaders::GlobalUniform,
    global_set_layout: Arc<DescriptorSetLayout>,
    surface: Surface,
    format: vk::SurfaceFormatKHR,
    physical_device: vk::PhysicalDevice,
    allocator: Arc<Allocator>,
    device: Arc<Device>,
    instance: Arc<Instance>,
    /// This field must be dropped last because other fields might rely on the objects inside being alive.
    #[allow(dead_code)]
    keep_alive: Vec<Box<dyn std::any::Any + 'static>>,
}

impl Renderer {
    #[must_use]
    pub fn new(
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
            Some(DebugUtilsMessenger::new(instance.clone(), &messenger_info))
        } else {
            None
        };

        let surface = Surface::new(instance.clone(), &w);
        let present_mode = vk::PresentModeKHR::MAILBOX_KHR;

        let (
            physical_device,
            graphics_family,
            format,
            present_mode,
            physical_device_properties,
            supported_present_modes,
            transfer_family,
        ) = create_physical_device(
            &instance,
            *surface,
            &device_extensions,
            present_mode,
            vk::PresentModeKHR::FIFO_KHR,
        );
        //let transfer_family = None;
        log::info!("Using physical device: {:?}", unsafe {
            CStr::from_ptr(physical_device_properties.device_name.as_ptr())
        });
        let (device, graphics_queue, transfer_queue) = create_device_and_queue(
            Arc::clone(&instance),
            graphics_family,
            transfer_family,
            &device_extensions,
            &device_layers,
            physical_device,
        );
        let device = Arc::new(device);
        let (swapchain, swapchain_images, swapchain_image_views, max_image_count) =
            create_swapchain(
                &instance,
                physical_device,
                *surface,
                format,
                &device,
                present_mode,
                None,
                frames_in_flight as u32,
            );
        let swapchain_image_views: Vec<Arc<ImageView>> =
            swapchain_image_views.into_iter().map(Arc::new).collect();

        let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(graphics_family.0)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = CommandPool::new(device.clone(), &command_pool_info, label!());
        let command_buffers =
            create_command_buffers(&device, &command_pool, frames_in_flight.try_into().unwrap());
        let render_pass = create_render_pass(device.clone(), format);

        let width = w.size().0;
        let height = w.size().1;
        let (render_fences, present_semaphores, render_semaphores) =
            create_sync_objects(&device, frames_in_flight);

        let allocator_info = vma::AllocatorCreateInfo {
            physical_device,
            device: device.raw(),
            instance: instance.raw(),
            flags: vma::AllocatorCreateFlags::NONE,
            preferred_large_heap_block_size: 0,
            heap_size_limits: None,
            allocation_callbacks: None,
            device_memory_callbacks: None,
            vulkan_api_version: vk::API_VERSION_1_2
        };

        let allocator = Arc::new(Allocator::new(device.clone(), &allocator_info));
        let depth_images: Vec<(Arc<image::Allocated>, Arc<image::Allocated>)> =
            create_depth_images(&allocator, width, height, frames_in_flight)
                .into_iter()
                .map(|(a, b)| (Arc::new(a), Arc::new(b)))
                .collect();
        let descriptor_set_manager = Arc::new(DescriptorSetManager::new(device.clone()));
        let depth_image_views = create_depth_image_views(&device, &depth_images)
            .into_iter()
            .map(|(a, b)| (Arc::new(a), Arc::new(b)))
            .collect();

        let framebuffers = create_framebuffers(
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

        let global_uniform = shaders::GlobalUniform {
            view: cgmath::Matrix4::from_scale(0.1),
            proj: cgmath::Matrix4::from_scale(0.1),
            view_proj: cgmath::Matrix4::from_scale(0.1),
            time: 0.0,
            renderables_count: 0,
            screen_width: width as f32,
            screen_height: height as f32,
            frustum_top_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            frustum_bottom_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            frustum_right_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            frustum_left_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            frustum_far_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            frustum_near_normal: Vector4::new(0.0, 0.0, 0.0, 0.0),
            camera_transform: Matrix4::identity(),
            near: 0.0,
            far: 0.0,
        };
        let max_objects = (0..frames_in_flight).map(|_| 4).collect::<Vec<_>>();
        let renderables_buffers = max_objects
            .iter()
            .map(|size| create_renderables_buffer(allocator.clone(), *size as u64))
            .collect::<Vec<_>>();
        let global_set_layout = Arc::new(create_global_descriptor_set_layout(device.clone()));
        let filter = vk::Filter::NEAREST;
        let address_mode = vk::SamplerAddressMode::REPEAT;
        let sampler = vk::SamplerCreateInfoBuilder::new()
            .mag_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode);
        let sampler = Sampler::new(device.clone(), &sampler, label!());
        let descriptor_sets = (0..frames_in_flight)
            .map(|i| {
                create_descriptor_sets(
                    &device,
                    &descriptor_set_manager,
                    &global_set_layout,
                    &uniform_buffer,
                    &renderables_buffers[i],
                )
            })
            .collect::<Vec<_>>();
        let mesh_buffers = (0..frames_in_flight)
            .map(|_| {
                Arc::new(create_mesh_buffer(
                    allocator.clone(),
                    std::mem::size_of::<shaders::Mesh>() as u64,
                ))
            })
            .collect::<Vec<_>>();
        let compute_set_layout = Arc::new(DescriptorSetLayout::new(
            device.clone(),
            vec![
                DescriptorSetLayoutBinding {
                    binding: 0,
                    count: 1,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: None,
                },
                DescriptorSetLayoutBinding {
                    binding: 1,
                    count: 1,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    immutable_samplers: None,
                },
            ],
            None,
        ));

        let indirect_buffers = (0..frames_in_flight)
            .map(|_| Arc::new(create_indirect_buffer(allocator.clone(), 2)))
            .collect();
        let cull_sets = mesh_buffers
            .iter()
            .zip(&indirect_buffers)
            .map(
                |(mesh_buffer, indirect_buffer): (
                    &Arc<buffer::Allocated>,
                    &Arc<buffer::Allocated>,
                )| {
                    create_cull_set(
                        &device,
                        &descriptor_set_manager,
                        &compute_set_layout,
                        mesh_buffer.clone(),
                        indirect_buffer.clone(),
                    )
                },
            )
            .collect::<Vec<_>>();
        let sampler = Arc::new(sampler);
        let cleanup = (0..frames_in_flight).map(|_| None).collect();

        let frames = Frames {
            present_semaphores,
            render_fences,
            render_semaphores,
            command_buffers,
            depth_image_views,
            descriptor_sets,
            renderables_buffers,
            max_objects,
            mesh_buffers,
            cull_sets,
            cleanup,
            indirect_buffers,
            framebuffers,
        };
        let materials = Default::default();
        let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::empty());
        let fence = Mutex::new(Fence::new(
            device.clone(),
            &fence_info,
            label!("TransferContextFence"),
        ));

        let graphics_queue = Arc::new(Mutex::new(graphics_queue));
        let transfer_family = transfer_family.unwrap_or(graphics_family);
        let transfer_queue = transfer_queue
            .map(|q| Arc::new(Mutex::new(q)))
            .unwrap_or_else(|| graphics_queue.clone());
        let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(transfer_family.0)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );
        let transfer_command_pool = Mutex::new(CommandPool::new(
            device.clone(),
            &command_pool_info,
            label!("TransferContextCommandPool"),
        ));
        let transfer_context = Arc::new(TransferContext {
            transfer_queue,
            command_pool: transfer_command_pool,
            graphics_family,
            transfer_family,
            fence,
            device: device.clone(),
        });
        let image_loader = image::Loader {
            device: device.clone(),
            transfer_context: transfer_context.clone(),
            allocator: allocator.clone(),
        };
        drop(w);
        let cull_shader = ShaderModule::new(
            device.clone(),
            include_bytes!("../../shaders/cull.comp.spv"),
            String::from("CullShader"),
            vk::ShaderStageFlagBits::VERTEX,
        );

        let compute_pipeline_layout = Arc::new(PipelineLayout::new(
            device.clone(),
            vec![global_set_layout.clone(), compute_set_layout.clone()],
            (),
            &label!("CullPipelineLayout"),
        ));
        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            compute_pipeline_layout,
            &cull_shader,
            "CullPipeline",
        );
        let keep_alive = Vec::from([Box::new(window) as Box<dyn std::any::Any + 'static>]);
        Self {
            compute_pipeline,
            start,
            supported_present_modes,
            compute_set_layout,
            present_mode,
            swapchain,
            sampler,
            width,
            height,
            vsync,
            max_image_count,
            graphics_queue,
            messenger,
            descriptor_set_manager,
            command_pool,
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
    #[must_use]
    pub fn allocator(&self) -> &Arc<Allocator> {
        &self.allocator
    }
    #[must_use]
    pub fn image_loader(&self) -> &image::Loader {
        &self.image_loader
    }
    #[must_use]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    #[must_use]
    pub fn transfer_context(&self) -> &Arc<TransferContext> {
        &self.transfer_context
    }
    #[must_use]
    pub fn default_sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }
    #[must_use]
    pub fn descriptor_set_manager(&self) -> &Arc<DescriptorSetManager> {
        &self.descriptor_set_manager
    }
    #[must_use]
    pub fn pipeline_creation_params(&self) -> PipelineCreationParams {
        PipelineCreationParams {
            device: &self.device,
            width: self.width,
            height: self.height,
            render_pass: &self.render_pass,
            global_set_layout: &self.global_set_layout,
        }
    }
    pub fn register_pipeline(&mut self, create_fn: MaterialLoadFn) -> PipelineHandle {
        let mut materials = self.materials.lock().unwrap();
        let material = Arc::new(create_fn(self.pipeline_creation_params()));
        let k = materials.insert((material, create_fn));
        PipelineHandle(self.materials.clone(), k)
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
            let guard = self.graphics_queue.lock().unwrap();
            let guard_2 = self.transfer_context.transfer_queue.lock().unwrap();
            self.device.device_wait_idle().unwrap();
            self.device
                .free_command_buffers(*self.command_pool, &self.frames.command_buffers);
            drop(guard_2);
            drop(guard);
        }
    }
    fn change_num_frames_in_flight(&mut self) {
        if self.frames_in_flight > self.max_image_count as usize {
            panic!(
                "Cannot resize to {} frames in flight the maximum is: {}",
                self.frames_in_flight, self.max_image_count
            );
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
        let max_objects = self.frames.max_objects.first().copied().unwrap_or(8);
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
                    &self.descriptor_set_manager,
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
                Arc::new(create_mesh_buffer(self.allocator.clone(), size))
            });
        let size = self
            .frames
            .indirect_buffers
            .get(0)
            .map(|b| b.size)
            .unwrap_or(100) as usize
            / std::mem::size_of::<IndirectDrawCommand>();
        self.frames
            .indirect_buffers
            .resize_with(self.frames_in_flight, || {
                Arc::new(create_indirect_buffer(self.allocator.clone(), size))
            });
        self.frames.cull_sets = self
            .frames
            .mesh_buffers
            .iter()
            .zip(&self.frames.indirect_buffers)
            .map(|(mesh_buffer, indirect_buffer)| {
                create_cull_set(
                    &self.device,
                    &self.descriptor_set_manager,
                    &self.compute_set_layout,
                    mesh_buffer.clone(),
                    indirect_buffer.clone(),
                )
            })
            .collect();
    }
    fn recreate_swapchain(&mut self) {
        let (swapchain, swapchain_images, swapchain_image_views, max_image_count) =
            create_swapchain(
                &self.instance,
                self.physical_device,
                *self.surface,
                self.format,
                &self.device,
                self.present_mode,
                Some(*self.swapchain),
                self.frames_in_flight as u32,
            );
        let swapchain_image_views = swapchain_image_views.into_iter().map(Arc::new).collect();
        self.max_image_count = max_image_count;
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.render_pass = create_render_pass(self.device.clone(), self.format);
        let depth_images: Vec<(Arc<image::Allocated>, Arc<image::Allocated>)> =
            create_depth_images(
                &self.allocator,
                self.width,
                self.height,
                self.frames_in_flight,
            )
            .into_iter()
            .map(|(a, b)| (Arc::new(a), Arc::new(b)))
            .collect();
        self.frames.depth_image_views = create_depth_image_views(&self.device, &depth_images)
            .into_iter()
            .map(|(a, b)| (Arc::new(a), Arc::new(b)))
            .collect();
        self.frames.framebuffers = create_framebuffers(
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
    pub fn max_frames_in_flight(&self) -> u32 {
        self.max_image_count
    }
    /// returns whether a resize was necessary
    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
        frames_in_flight: usize,
        vsync: bool,
    ) -> bool {
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
        self.frames.check_valid();
        true
    }
    fn reload_pipelines(&mut self) {
        let mut materials = self.materials.lock().unwrap();
        for (_, (pipeline, f)) in materials.iter_mut() {
            let params = self.pipeline_creation_params();
            *pipeline = Arc::new(f(params));
        }
    }
    #[rustfmt::skip]
    fn update_global_uniform(
        &mut self,
        frame_index: usize,
        view_proj: cgmath::Matrix4<f32>,
        frustum: Frustum,
        renderables: u32,
        camera_transform: Matrix4<f32>
    ) -> u32 {
        self.global_uniform.view_proj =
            cgmath::Matrix4::from_nonuniform_scale(1.0, -1.0, 1.0) * view_proj;
        self.global_uniform.time = self.start.elapsed().as_secs_f32();
        self.global_uniform.screen_width = self.width as f32;
        self.global_uniform.screen_height = self.height as f32;
        self.global_uniform.renderables_count = renderables;
        self.global_uniform.frustum_top_normal = frustum.top.extend(42.69);
        self.global_uniform.frustum_bottom_normal = frustum.bottom.extend(42.69);
        self.global_uniform.frustum_right_normal = frustum.right.extend(42.69);
        self.global_uniform.frustum_left_normal = frustum.left.extend(42.69);
        self.global_uniform.frustum_far_normal = frustum.far.extend(42.69);
        self.global_uniform.frustum_near_normal = frustum.near.extend(42.69);
        self.global_uniform.camera_transform = camera_transform;
        self.global_uniform.near = frustum.near_distance;
        self.global_uniform.far = frustum.far_distance;

        let ptr = self.uniform_buffer.map();
        let global_uniform_offset: u32 = (pad_uniform_buffer_size(
            &self.physical_device_properties.limits,
            std::mem::size_of::<shaders::GlobalUniform>() as u64,
        ) * frame_index as u64)
            .try_into()
            .unwrap();
        unsafe {
            let current_global_uniform_ptr = ptr.add(global_uniform_offset as usize);
            std::ptr::write(
                current_global_uniform_ptr as *mut shaders::GlobalUniform,
                self.global_uniform,
            );
        }
        self.uniform_buffer.unmap();
        global_uniform_offset
    }
    fn batch_renderables<'a, 'b, 'c>(
        materials: &'a PipelineStorage,
        renderables: impl IntoIterator<Item = Renderable<'b>>,
    ) -> BatchingResult<'c>
    where
        'a: 'c,
        'b: 'c,
    {
        let mut last_mesh: *const Mesh = std::ptr::null();
        let mut last_pipeline: *const PipelineHandle = std::ptr::null();
        let mut last_custom_set: *const DescriptorSet = std::ptr::null();
        let mut batches: Vec<InstancingBatch> = Vec::new();
        let mut meshes: Vec<Arc<Mesh>> = Vec::new();
        let mut custom_descriptor_sets: Vec<Option<Arc<DescriptorSet>>> = Vec::new();
        let mut pipelines: Vec<Arc<Pipeline>> = Vec::new();
        let mut num_renderables = 0;
        let mut pipeline: Option<&Pipeline> = None;
        for (i, renderable) in renderables.into_iter().enumerate() {
            let this_mesh = &**renderable.mesh as *const Mesh;
            let this_pipeline = renderable.pipeline as *const PipelineHandle;
            let this_custom_set = renderable.custom_set.map_or(std::ptr::null(), Arc::as_ptr);
            if this_mesh != last_mesh {
                meshes.push(Arc::clone(renderable.mesh));
            }
            if this_custom_set != last_custom_set {
                custom_descriptor_sets.push(renderable.custom_set.map(Arc::clone));
            }
            if last_pipeline != this_pipeline {
                let p = &materials.get(renderable.pipeline.key()).unwrap().0;
                pipeline = Some(&**p);
                pipelines.push(p.clone());
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

            let pipeline = pipeline.unwrap();
            if pipeline.vertex_type() != renderable.mesh.vertex_type() {
                panic!(
                    "Different vertex types! Pipeline ({}): {:?}, mesh({}): {:?}",
                    pipeline.name(),
                    pipeline.vertex_name(),
                    renderable.mesh.name(),
                    renderable.mesh.vertex_name()
                );
            }
            batches.push(InstancingBatch {
                custom_set: renderable.custom_set.map(|a| &**a),
                index_count: renderable.mesh.index_count,
                index_start: renderable.mesh.index_start,
                index_vertex_buffer: &renderable.mesh.buffer,
                instance_count: 1,
                pipeline,
                vertex_start: renderable.mesh.vertex_start,
                mesh: TryInto::<u32>::try_into(meshes.len()).unwrap() - 1,
                renderables: Vec::from([renderable]),
                first_instance: i as u32,
            });
            num_renderables = i;
        }
        BatchingResult {
            batches,
            meshes,
            sets: custom_descriptor_sets,
            pipelines,
            num_renderables,
        }
    }
    fn batch_batches<'a>(batches: &'a [InstancingBatch<'a>]) -> Vec<IndirectDraw<'a>> {
        let mut draws = Vec::new();
        let mut last_index_vertex_buffer: *const buffer::Allocated = std::ptr::null();
        let mut last_custom_set: *const DescriptorSet = std::ptr::null();
        let mut last_pipeline: *const Pipeline = std::ptr::null();
        for (i, batch) in batches.iter().enumerate() {
            let this_custom_set: *const DescriptorSet = batch
                .custom_set
                .map(|c| c as *const DescriptorSet)
                .unwrap_or_else(std::ptr::null);
            if last_index_vertex_buffer != batch.index_vertex_buffer
                || this_custom_set != last_custom_set
                || last_pipeline != batch.pipeline
            {
                draws.push(IndirectDraw {
                    pipeline: batch.pipeline,
                    index_vertex_buffer: batch.index_vertex_buffer,
                    custom_set: batch.custom_set,
                    first_batch: i as u32,
                    instancing_batches: &batches[i..i + 1],
                })
            } else {
                let last = draws.last_mut().unwrap();
                last.instancing_batches = &batches[last.first_batch as usize..i + 1];
            }
            last_index_vertex_buffer = batch.index_vertex_buffer;
            last_custom_set = this_custom_set;
            last_pipeline = batch.pipeline;
        }
        draws
    }
    fn write_renderables(buffer: &buffer::Allocated, instancing_batches: &[InstancingBatch]) {
        puffin::profile_function!();
        let ptr = buffer.map().cast::<shaders::Object>() as *mut shaders::Object;
        let num_renderables = instancing_batches.iter().map(|b| b.renderables.len()).sum();
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, num_renderables) };
        let mut i = 0;
        for (batch_id, batch) in instancing_batches.iter().enumerate() {
            slice[batch_id].first_instance = batch.first_instance;
            for renderable in &batch.renderables {
                //slice[i].redirect = i as u32;
                slice[i].transform = *renderable.transform;
                slice[i].batch = batch_id.try_into().unwrap();
                slice[i].draw = batch_id.try_into().unwrap();
                slice[i].uncullable = if renderable.uncullable { 1 } else { 0 };
                slice[i].unused_3 = 0;
                slice[i].custom_set = renderable.custom_id;
                slice[i].mesh = batch.mesh;
                i += 1;
            }
        }
        buffer.unmap();
    }
    fn write_meshes(buffer: &buffer::Allocated, meshes: &[Arc<Mesh>]) {
        puffin::profile_function!();
        let ptr = buffer.map().cast::<shaders::Mesh>() as *mut shaders::Mesh;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, meshes.len()) };
        for (i, mesh) in meshes.iter().enumerate() {
            slice[i] = shaders::Mesh {
                sphere_bounds: mesh.bounds.sphere_bounds,
                first_index: mesh.index_start,
                index_count: mesh.index_count,
                vertex_offset: mesh.vertex_start.try_into().unwrap(),
            };
        }
        buffer.unmap();
    }
    fn wait_for_next_frame(&mut self) -> (usize, frame::FrameDataMut) {
        puffin::profile_function!();
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
        if renderables_count as u64 * std::mem::size_of::<shaders::Object>() as u64
            > frame.renderables_buffer.size
        {
            let new_length = (renderables_count / 2 * 3).max(16) as u64;
            *frame.renderables_buffer =
                create_renderables_buffer(Arc::clone(&self.allocator), new_length);
            *frame.descriptor_set = create_descriptor_sets(
                &self.device,
                &self.descriptor_set_manager,
                &self.global_set_layout,
                &self.uniform_buffer,
                &*frame.renderables_buffer,
            );
        }
    }
    fn resize_indirect_buffer_if_necessary(
        device: &Device,
        descriptor_set_manager: &DescriptorSetManager,
        compute_set_layout: &DescriptorSetLayout,
        allocator: Arc<Allocator>,
        frame: &mut FrameDataMut,
        batches: &[InstancingBatch],
    ) {
        if batches.len() as u64 * std::mem::size_of::<shaders::IndirectDrawCommand>() as u64
            > frame.indirect_buffer.size
        {
            let new_length = (batches.len() / 2 * 3).max(16);
            *frame.indirect_buffer = Arc::new(create_indirect_buffer(allocator, new_length));
            *frame.cull_set = create_cull_set(
                device,
                descriptor_set_manager,
                compute_set_layout,
                frame.mesh_buffer.clone(),
                frame.indirect_buffer.clone(),
            );
        }
    }
    fn acquire_swapchain_image(
        device: &Device,
        swapchain: &Swapchain,
        frame: &FrameData,
    ) -> Option<usize> {
        puffin::profile_function!();
        unsafe {
            let res = device.acquire_next_image_khr(
                **swapchain,
                1_000_000_000,
                **frame.present_semaphore,
                vk::Fence::default(),
            );
            res.value.map(|value| value as usize)
        }
    }
    fn resize_mesh_buffer_if_necessary(&mut self, num_meshes: usize, frame_index: usize) {
        let frame = self.frames.get_mut(frame_index);
        let mesh_buffer_min_size = num_meshes * std::mem::size_of::<shaders::Mesh>();
        if mesh_buffer_min_size as u64 > frame.mesh_buffer.size {
            *frame.mesh_buffer = Arc::new(create_mesh_buffer(
                self.allocator.clone(),
                mesh_buffer_min_size as u64 / 2 * 3,
            ));
            *frame.cull_set = create_cull_set(
                &self.device,
                &self.descriptor_set_manager,
                &self.compute_set_layout,
                frame.mesh_buffer.clone(),
                frame.indirect_buffer.clone(),
            );
        }
    }
    fn begin_render_pass(
        &mut self,
        frame_index: usize,
        swapchain_image_index: usize,
        clear_color: [f32; 4],
    ) -> vk::CommandBuffer {
        let frame = self.frames.get(frame_index);
        let cmd = *frame.command_buffer;
        let framebuffer = &self.frames.framebuffers[swapchain_image_index].0;
        unsafe {
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
                .framebuffer(**framebuffer);
            self.device
                .cmd_begin_render_pass(cmd, &rp_begin_info, vk::SubpassContents::INLINE);
        }
        fn flip<T>(t: &mut (T, T)) {
            std::mem::swap(&mut t.0, &mut t.1)
        }
        let frame = self.frames.get_mut(frame_index);
        flip(frame.framebuffer);
        //flip(&mut frame.depth_image_view);
        cmd
    }
    fn begin_command_buffer(&self, frame: &FrameData) {
        unsafe {
            let cmd = *frame.command_buffer;
            self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
            let begin_info = vk::CommandBufferBeginInfoBuilder::new()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(cmd, &begin_info).unwrap();
        }
    }
    fn record_cull_dispatch(
        &self,
        frame: &FrameData,
        instancing_batches: &[InstancingBatch],
        renderables_len: usize,
        global_uniform_offset: u32,
    ) {
        puffin::profile_function!();
        let cmd = *frame.command_buffer;
        unsafe {
            self.device.cmd_fill_buffer(
                cmd,
                ***frame.indirect_buffer,
                0,
                (std::mem::size_of::<IndirectDrawCommand>() * instancing_batches.len()) as u64,
                0,
            );
            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                *self.compute_pipeline,
            );

            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                ***self.compute_pipeline.layout(),
                0,
                &[**frame.descriptor_set, **frame.cull_set],
                &[global_uniform_offset],
            );

            let group_count_x = (renderables_len as u32 / 256) + 1;
            self.device.cmd_dispatch(cmd, group_count_x, 1, 1);

            let buffer_memory_barrier = vk::BufferMemoryBarrierBuilder::new()
                .buffer(***frame.indirect_buffer)
                .size(frame.indirect_buffer.size)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ);
            let src_stage_mask = vk::PipelineStageFlags::COMPUTE_SHADER;
            let dst_stage_mask = vk::PipelineStageFlags::DRAW_INDIRECT;
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage_mask,
                dst_stage_mask,
                vk::DependencyFlags::empty(),
                &[],
                &[buffer_memory_barrier],
                &[],
            );
        }
    }
    fn record_draws(
        &self,
        frame_index: usize,
        global_uniform_offset: u32,
        indirect_draws: &[IndirectDraw],
        cmd: vk::CommandBuffer,
    ) {
        puffin::profile_function!();
        let frame = self.frames.get(frame_index);
        unsafe {
            for draw in indirect_draws {
                self.device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    &[**draw.index_vertex_buffer, **frame.renderables_buffer],
                    &[0, 0],
                );
                self.device.cmd_bind_index_buffer(
                    cmd,
                    **draw.index_vertex_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    **draw.pipeline,
                );

                if let Some(custom_set) = draw.custom_set {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        **draw.pipeline.layout(),
                        0,
                        &[**frame.descriptor_set, **custom_set],
                        &[global_uniform_offset],
                    );
                } else {
                    self.device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        **draw.pipeline.layout(),
                        0,
                        &[**frame.descriptor_set],
                        &[global_uniform_offset],
                    );
                };
                let stride = std::mem::size_of::<shaders::IndirectDrawCommand>() as u32;
                let offset = draw.first_batch as u64 * stride as u64;
                self.device.cmd_draw_indexed_indirect(
                    cmd,
                    ***frame.indirect_buffer,
                    offset,
                    draw.instancing_batches.len() as u32,
                    stride,
                );
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
        let queue = self.graphics_queue.lock().unwrap();
        unsafe {
            self.device
                .queue_submit(**queue, &[submit], **frame.render_fence)
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
        let queue = self.graphics_queue.lock().unwrap();
        let res = unsafe { self.device.queue_present_khr(**queue, &present_info) };
        if res.is_err() {
            log::warn!("{:#?}", res);
        }
    }
    pub fn render<'a>(
        &mut self,
        renderables: impl Iterator<Item = Renderable<'a>>,
        view_proj: Matrix4<f32>,
        frustum: Frustum,
        camera_transform: Matrix4<f32>,
    ) -> Option<(std::time::Duration, std::time::Duration)> {
        puffin::profile_function!();
        self.frames.check_valid();
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
        let materials = self.materials.clone();
        let materials = materials.lock().unwrap();
        let BatchingResult {
            batches,
            meshes,
            sets: custom_sets,
            pipelines,
            num_renderables,
        } = Self::batch_renderables(&materials, renderables);
        let global_uniform_offset = self.update_global_uniform(
            frame_index,
            view_proj,
            frustum,
            num_renderables.try_into().unwrap(),
            camera_transform,
        );

        self.resize_mesh_buffer_if_necessary(meshes.len(), frame_index);
        self.resize_renderable_buffer_if_necessary(frame_index, num_renderables);
        let mut frame = self.frames.get_mut(frame_index);
        Self::resize_indirect_buffer_if_necessary(
            &self.device,
            &self.descriptor_set_manager,
            &self.compute_set_layout,
            self.allocator.clone(),
            &mut frame,
            &batches,
        );
        let swapchain_image_index =
            match Self::acquire_swapchain_image(&self.device, &self.swapchain, &frame.immu()) {
                Some(i) => i,
                None => return None,
            };
        Self::write_renderables(frame.renderables_buffer, &batches);
        Self::write_meshes(frame.mesh_buffer, &meshes);
        *frame.cleanup = Some(Box::new(|| {
            drop(meshes);
            drop(custom_sets);
            drop(pipelines);
        }) as Box<dyn FnOnce()>);
        drop(frame);
        let frame = self.frames.get(frame_index);
        self.begin_command_buffer(&frame);
        self.record_cull_dispatch(&frame, &batches, num_renderables, global_uniform_offset);
        let cmd = self.begin_render_pass(frame_index, swapchain_image_index, [0.15, 0.6, 0.9, 1.0]);
        let indirect_draws = Self::batch_batches(&batches);
        self.record_draws(frame_index, global_uniform_offset, &indirect_draws, cmd);
        self.submit_cmd(frame_index, cmd);
        self.present(frame_index, swapchain_image_index);
        let cpu_work_time = cpu_work_start.elapsed();
        Some((gpu_wait, cpu_work_time))
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            log::info!(label!("Waiting for exclusive access to queues"));
            let guard_0 = self.graphics_queue.lock().unwrap();
            let guard_1 = self.transfer_context.transfer_queue.lock().unwrap();
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
            drop(self.messenger.take());
            drop((guard_0, guard_1));
            log::info!(label!("DROPPING Renderer"));
        };
    }
}
