use crate::buffer;
use crate::handles::{Device, Fence, Instance, Queue, QueueFamily, Semaphore, Surface, Trash};
use crate::image;
use crate::{
    debug_callback,
    descriptor_sets::{
        DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetManager,
    },
    handles::{Allocator, CommandPool, Framebuffer, ImageView, RenderPass, Swapchain},
};
use ash::vk;
use raw_window_handle::HasRawDisplayHandle;
use std::{
    ffi::{CStr, CString},
    sync::Arc,
};

#[derive(Debug, Clone, Copy)]
pub struct RasterizationState {
    pub polygon_mode: vk::PolygonMode,
    pub front_face: vk::FrontFace,
    pub cull_mode: vk::CullModeFlags,
}

impl RasterizationState {
    pub fn builder(&self) -> vk::PipelineRasterizationStateCreateInfoBuilder<'static> {
        vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .line_width(1.0)
            .cull_mode(self.cull_mode)
            .depth_bias_enable(false)
            .depth_bias_constant_factor(0.0)
            .depth_bias_clamp(0.0)
            .depth_bias_slope_factor(0.0)
            .front_face(self.front_face)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct InputAssemblyState {
    pub topology: vk::PrimitiveTopology,
}

impl InputAssemblyState {
    pub fn builder(&self) -> vk::PipelineInputAssemblyStateCreateInfoBuilder<'static> {
        vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(self.topology)
            .primitive_restart_enable(false)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DepthStencilInfo {
    /// The compare function used to determine whether to write to the DepthBuffer.
    /// None = always write to buffer
    pub test: Option<vk::CompareOp>,
    /// disable / enable writing to the DepthBuffer
    pub write: bool,
}

pub const fn pad_uniform_buffer_size(
    physical_device_limits: &vk::PhysicalDeviceLimits,
    original_size: u64,
) -> u64 {
    let min_alignment = physical_device_limits.min_uniform_buffer_offset_alignment;
    let mut aligned_size = original_size;
    if min_alignment > 0 {
        aligned_size = (aligned_size + min_alignment - 1) & !(min_alignment - 1);
    }
    aligned_size
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MultiSamplingState {}

impl MultiSamplingState {
    #[must_use]
    pub fn builder(&self) -> vk::PipelineMultisampleStateCreateInfoBuilder<'static> {
        vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .min_sample_shading(1.0)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false)
    }
}

pub fn create_global_descriptor_set_layout(device: Arc<Device>) -> DescriptorSetLayout {
    DescriptorSetLayout::new(
        device,
        vec![
            DescriptorSetLayoutBinding {
                binding: 0,
                count: 1,
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                immutable_samplers: None,
            },
            DescriptorSetLayoutBinding {
                binding: 1,
                count: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                immutable_samplers: None,
            },
            DescriptorSetLayoutBinding {
                binding: 2,
                count: 1,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                immutable_samplers: None,
            },
        ],
        None,
    )
}

pub fn create_instance(
    window: &sdl2::video::Window,
    validation_layers: bool,
    messenger_info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> (Instance, ash::Entry, Vec<*const i8>) {
    let application_name = CString::new("test").unwrap();
    let engine_name = CString::new("no engine").unwrap();
    let app_info = vk::ApplicationInfo::builder()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let needed_extensions: Vec<CString> = window
        .vulkan_instance_extensions()
        .unwrap()
        .into_iter()
        .map(|v| CString::new(v).unwrap())
        .collect();
    let needed_extensions: Vec<*const i8> = needed_extensions.iter().map(|v| v.as_ptr()).collect();
    let needed_extensions = needed_extensions.iter().copied().map(|mut v| {
        let mut s = String::new();
        loop {
            let c = unsafe { *v };
            if c == 0 {
                break;
            }
            let c = TryInto::<u8>::try_into(c).unwrap() as char;
            v = unsafe { (v).add(1) };
            s.push(c);
        }
        s
    });

    for extension in needed_extensions {
        log::info!("required EXT: {}", extension);
    }
    let entry = unsafe { ash::Entry::load() }.unwrap();

    let instance_extensions = window
        .vulkan_instance_extensions()
        .unwrap()
        .into_iter()
        .map(|v| CString::new(v).unwrap())
        .collect::<Vec<CString>>();
    let mut instance_extensions: Vec<*const i8> = instance_extensions
        .iter()
        .map(|v| v.as_ptr())
        .collect::<Vec<*const i8>>();
    let mut instance_layers = Vec::new();

    if validation_layers {
        instance_extensions.push(cstr::cstr!(b"VK_EXT_debug_utils").as_ptr());
        instance_layers.push(cstr::cstr!(b"VK_LAYER_KHRONOS_validation").as_ptr());
    }
    ash_window::enumerate_required_extensions(window.raw_display_handle())
        .unwrap()
        .to_vec();

    let device_extensions =
        vec![unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_KHR_swapchain\0").as_ptr() }];

    let mut instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    if validation_layers {
        instance_info.p_next = (messenger_info as *const vk::DebugUtilsMessengerCreateInfoEXT)
            .cast::<std::ffi::c_void>();
    }
    let instance = Instance::new(&entry, &instance_info);

    (instance, entry, device_extensions)
}

pub fn create_uniform_buffer(
    allocator: Arc<Allocator>,
    frames_in_flight: u64,
    limits: &vk::PhysicalDeviceLimits,
) -> buffer::Allocated {
    let size =
        pad_uniform_buffer_size(limits, std::mem::size_of::<shaders::GlobalUniform>() as u64)
            * frames_in_flight;

    buffer::Allocated::new(
        allocator,
        *vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
        vma::MemoryUsage::AutoPreferDevice,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        label!("UniformBuffer"),
    )
}

pub fn create_descriptor_sets(
    device: &Device,
    descriptor_set_manager: &DescriptorSetManager,
    layout: &DescriptorSetLayout,
    uniform_buffer: &buffer::Allocated,
    object_buffer: &buffer::Allocated,
) -> DescriptorSet {
    let set = descriptor_set_manager.allocate(layout, label!("BasaltInternalSet"));

    let global_uniform_buffer_info = vk::DescriptorBufferInfo::builder()
        .buffer(**uniform_buffer)
        .range(std::mem::size_of::<shaders::GlobalUniform>() as u64);
    let global_uniform_buffer_info = &[*global_uniform_buffer_info];

    let uniform_write = vk::WriteDescriptorSet::builder()
        .dst_binding(0)
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .buffer_info(global_uniform_buffer_info);

    let object_buffer_info = vk::DescriptorBufferInfo::builder()
        .buffer(**object_buffer)
        .range(object_buffer.size);
    let object_buffer_info = [*object_buffer_info];
    let object_write = vk::WriteDescriptorSet::builder()
        .dst_binding(1)
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&object_buffer_info);

    unsafe {
        device.update_descriptor_sets(&[*uniform_write, *object_write], &[]);
    }
    set
}

pub fn create_depth_images(
    allocator: &Arc<Allocator>,
    width: u32,
    height: u32,
    frames_in_flight: usize,
) -> Vec<(image::Allocated, image::Allocated)> {
    let format = vk::Format::D32_SFLOAT;
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let f = || {
        image::Allocated::new(
            allocator.clone(),
            format,
            usage,
            extent,
            vk::ImageTiling::OPTIMAL,
            label!("DepthImage").into(),
        )
    };
    (0..frames_in_flight).map(|_| (f(), f())).collect()
}
pub fn create_depth_image_views<'a>(
    device: &Arc<Device>,
    images: impl IntoIterator<Item = &'a (Arc<image::Allocated>, Arc<image::Allocated>)>,
) -> Vec<(ImageView, ImageView)> {
    let f = |image: &Arc<image::Allocated>| {
        let info =
            image.image_view_create_info(vk::Format::D32_SFLOAT, vk::ImageAspectFlags::DEPTH);
        ImageView::new(
            device.clone(),
            &info,
            Some(image.clone()),
            label!("DepthImageView"),
        )
    };
    images.into_iter().map(|(a, b)| (f(a), f(b))).collect()
}
pub fn create_device_and_queue(
    instance: Arc<Instance>,
    graphics_queue_family: QueueFamily,
    transfer_queue_family: Option<QueueFamily>,
    device_extensions: &[*const i8],
    physical_device: vk::PhysicalDevice,
) -> (Device, Queue, Option<Queue>) {
    let mut queue_info = vec![*vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(graphics_queue_family.0)
        .queue_priorities(&[1.0])];
    if let Some(transfer_queue_family) = transfer_queue_family {
        queue_info.push(
            *vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(transfer_queue_family.0)
                .queue_priorities(&[0.5]),
        );
    }
    let features = vk::PhysicalDeviceFeatures::builder()
        .fill_mode_non_solid(true)
        .multi_draw_indirect(true);

    let mut device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(device_extensions);

    let indexing_features = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT {
        shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
        descriptor_binding_partially_bound: vk::TRUE,
        descriptor_binding_variable_descriptor_count: vk::TRUE,
        runtime_descriptor_array: vk::TRUE,
        ..vk::PhysicalDeviceDescriptorIndexingFeatures::default()
    };
    device_info.p_next = (&indexing_features
        as *const vk::PhysicalDeviceDescriptorIndexingFeatures)
        .cast::<std::ffi::c_void>();

    let device = Device::new(instance, physical_device, &device_info);
    let graphics_queue = Queue::new(
        unsafe { device.get_device_queue(graphics_queue_family.0, 0) },
        label!("GraphicsQueue"),
    );
    let transfer_queue = transfer_queue_family.map(|transfer_queue_family| {
        Queue::new(
            unsafe { device.get_device_queue(transfer_queue_family.0, 0) },
            label!("TransferQueue"),
        )
    });
    (device, graphics_queue, transfer_queue)
}

pub fn create_command_buffers(
    device: &Device,
    command_pool: &CommandPool,
    number: u32,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(**command_pool)
        .command_buffer_count(number)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
        .unwrap()
        .to_vec()
}
pub fn create_render_pass(device: Arc<Device>, format: vk::SurfaceFormatKHR) -> RenderPass {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachments = &[*color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_attachment_ref);

    let attachments = &[*color_attachment, *depth_attachment];
    let subpasses = &[*subpass];
    let render_pass_info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses);

    RenderPass::new(device, &render_pass_info, label!("MainRenderPass"))
}
pub fn create_swapchain(
    device: Arc<Device>,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
    swapchain_image_count: u32,
) -> (Swapchain, Vec<vk::Image>, Vec<ImageView>, u32) {
    let surface_caps = unsafe {
        surface
            .loader()
            .get_physical_device_surface_capabilities(physical_device, **surface)
    }
    .unwrap();
    let mut image_count = swapchain_image_count.max(surface_caps.min_image_count);
    let max_image_count = surface_caps.max_image_count;
    if max_image_count > 0 && image_count > max_image_count {
        image_count = max_image_count;
    }
    let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(**surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(match old_swapchain {
            Some(sc) => sc,
            None => vk::SwapchainKHR::null(),
        });

    let swapchain = Swapchain::new(device.instance(), device.clone(), &swapchain_info, label!());
    let swapchain_images: Vec<vk::Image> =
        unsafe { swapchain.loader().get_swapchain_images(*swapchain) }
            .unwrap()
            .to_vec();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let image_view_info = vk::ImageViewCreateInfo::builder()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            ImageView::new(
                device.clone(),
                &image_view_info,
                None,
                label!("SwapchainImageView"),
            )
        })
        .collect();
    (
        swapchain,
        swapchain_images,
        swapchain_image_views,
        max_image_count,
    )
}
pub fn create_framebuffers<'a, I>(
    device: &Arc<Device>,
    width: u32,
    height: u32,
    render_pass: vk::RenderPass,
    swapchain_image_views: impl IntoIterator<Item = &'a Arc<ImageView>>,
    depth_views: I,
) -> Vec<(Framebuffer, Framebuffer)>
where
    I: IntoIterator<Item = &'a (Arc<ImageView>, Arc<ImageView>)>,
    I::IntoIter: Clone,
{
    let depth_views = depth_views.into_iter().cycle();
    let attachments: Vec<[([vk::ImageView; 2], Trash); 2]> = swapchain_image_views
        .into_iter()
        .zip(depth_views)
        .map(|(swapchain_image_view, (depth_view_1, depth_view_2))| {
            [
                (
                    [***swapchain_image_view, ***depth_view_1],
                    Arc::new((swapchain_image_view.clone(), depth_view_1.clone())) as Trash,
                ),
                (
                    [***swapchain_image_view, ***depth_view_2],
                    Arc::new((swapchain_image_view.clone(), depth_view_2.clone())) as Trash,
                ),
            ]
        })
        .collect::<Vec<[([vk::ImageView; 2], Trash); 2]>>();

    let f = |(attachments, trash): (&[vk::ImageView; 2], Trash)| {
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .width(width)
            .height(height)
            .layers(1)
            .attachments(attachments);
        Framebuffer::new(
            device.clone(),
            &framebuffer_info,
            label!("SwapchainFrameBuffer"),
        )
        .with_trash(trash.clone())
    };

    attachments
        .iter()
        .map(|a: &[([vk::ImageView; 2], Trash); 2]| {
            let [a, b]: &[([vk::ImageView; 2], Trash); 2] = a;
            (f((&a.0, a.1.clone())), f((&b.0, b.1.clone())))
        })
        .collect::<Vec<_>>()
}

#[allow(clippy::too_many_lines)]
pub fn create_physical_device(
    instance: &Instance,
    surface: &Surface,
    device_extensions: &[*const i8],
    preferred_present_mode: vk::PresentModeKHR,
    backup_present_mode: vk::PresentModeKHR,
) -> (
    vk::PhysicalDevice,
    QueueFamily,
    vk::SurfaceFormatKHR,
    vk::PresentModeKHR,
    vk::PhysicalDeviceProperties,
    Vec<vk::PresentModeKHR>,
    Option<QueueFamily>,
) {
    let (
        physical_device,
        graphics_queue_family,
        format,
        present_mode,
        device_properties,
        present_modes,
        transfer_queue_family
    ) = unsafe { instance.enumerate_physical_devices() }
        .unwrap()
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let physical_device_queue_familiy_properties =
                instance.get_physical_device_queue_family_properties(physical_device);
            let graphics_family_i = match physical_device_queue_familiy_properties
                .iter()
                .copied()
                .enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && surface.loader()
                            .get_physical_device_surface_support(
                                physical_device,
                                i.try_into().unwrap(),
                                **surface,
                            )
                            .unwrap()
                }) {
                Some(queue_family) => queue_family.try_into().unwrap(),
                None => return None,
            };
            let graphics_family = QueueFamily(graphics_family_i);
            let features = instance.get_physical_device_features(physical_device);
            if features.multi_draw_indirect == vk::FALSE {
                return None;
            }
            //let features = instance.get_physical_device_features2(physical_device, None);
            let transfer_family: Option<usize>= physical_device_queue_familiy_properties
                .into_iter().enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::TRANSFER) && queue_family_properties
                        .queue_flags.contains(vk::QueueFlags::COMPUTE) && i != graphics_family_i as usize
                });

            let transfer_family = transfer_family.map(|a| a as u32).map(QueueFamily);
            let formats = surface.loader()
                .get_physical_device_surface_formats(physical_device, **surface)
                .unwrap();
            let format = match formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| formats.get(0))
            {
                Some(surface_format) => *surface_format,
                None => return None,
            };
            let present_modes = surface.loader()
                .get_physical_device_surface_present_modes(physical_device, **surface)
                .unwrap().to_vec();
            let present_mode = *present_modes
                .iter()
                .find(|present_mode| present_mode == &&preferred_present_mode)
                .unwrap_or(&backup_present_mode);

            let supported_device_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();
            let device_extensions_supported = device_extensions.iter().all(|device_extension| {
                let device_extension = CStr::from_ptr(*device_extension);
                supported_device_extensions.iter().any(|properties| {
                    CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                })
            });

            if !device_extensions_supported {
                return None;
            }

            let device_properties = instance.get_physical_device_properties(physical_device);
            Some((
                physical_device,
                graphics_family,
                format,
                present_mode,
                device_properties,
                present_modes,
                transfer_family
            ))
        })
        .max_by_key(|(_, _, _, _, properties, _, transfer_queue)| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 0,
        } + if transfer_queue.is_some() { 1 } else { 0 })
        .expect("No suitable physical device found");
    (
        physical_device,
        graphics_queue_family,
        format,
        present_mode,
        device_properties,
        present_modes,
        transfer_queue_family,
    )
}

#[derive(Debug, Clone, Copy)]
pub struct ColorBlendAttachment {
    pub blend_enable: bool,
    pub src_color_factor: vk::BlendFactor,
    pub dst_color_factor: vk::BlendFactor,
}

impl Default for ColorBlendAttachment {
    fn default() -> Self {
        Self {
            blend_enable: false,
            src_color_factor: vk::BlendFactor::ZERO,
            dst_color_factor: vk::BlendFactor::ZERO,
        }
    }
}

impl ColorBlendAttachment {
    pub fn builder(&self) -> vk::PipelineColorBlendAttachmentStateBuilder<'static> {
        vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(self.blend_enable)
            .src_color_blend_factor(self.src_color_factor)
            .dst_color_blend_factor(self.dst_color_factor)
    }
}

pub fn create_debug_messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXTBuilder<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

pub fn create_renderables_buffer(allocator: Arc<Allocator>, max_objects: u64) -> buffer::Allocated {
    let size = std::mem::size_of::<shaders::Object>() as u64 * max_objects;
    buffer::Allocated::new(
        allocator,
        *vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER),
        vma::MemoryUsage::AutoPreferDevice,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        label!("RenderablesBuffer"),
    )
}

pub fn create_mesh_buffer(allocator: Arc<Allocator>, size: u64) -> buffer::Allocated {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
    buffer::Allocated::new(
        allocator,
        *buffer_info,
        vma::MemoryUsage::AutoPreferDevice,
        vk::MemoryPropertyFlags::HOST_VISIBLE,
        vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
        label!("MeshBuffer"),
    )
}

pub fn round_to<T>(v: T, a: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Rem<Output = T> + std::ops::Sub<Output = T> + Copy,
{
    v + (a - (v % a))
}

pub fn create_sync_objects(
    device: &Arc<Device>,
    num: usize,
) -> (Vec<Fence>, Vec<Semaphore>, Vec<Semaphore>) {
    let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
    let render_fences = (0..num)
        .map(|_| Fence::new(device.clone(), &fence_create_info, label!("RenderFence")))
        .collect();

    let semaphore_create_info = vk::SemaphoreCreateInfo::builder();
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

pub fn log_resource_created(typename: &'static str, name: &str) {
    use ::colored::*;
    ::log::info!("{} {}! {}", "CREATED".green(), typename, name);
}

pub fn log_resource_dropped(typename: &'static str, name: &str) {
    use ::colored::*;
    ::log::info!("{} {}! {}", "DROPPED".red(), typename, name);
}

pub fn create_indirect_buffer(allocator: Arc<Allocator>, size: usize) -> buffer::Allocated {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(std::mem::size_of::<shaders::IndirectDrawCommand>() as u64 * size as u64)
        .usage(
            vk::BufferUsageFlags::INDIRECT_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
    buffer::Allocated::new(
        allocator,
        *buffer_info,
        vma::MemoryUsage::AutoPreferDevice,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vma::AllocationCreateFlags::empty(),
        label!("IndirectBuffer"),
    )
}

pub fn create_cull_set(
    device: &Device,
    descriptor_set_manager: &DescriptorSetManager,
    layout: &DescriptorSetLayout,
    mesh_buffer: Arc<buffer::Allocated>,
    indirect_buffer: Arc<buffer::Allocated>,
) -> DescriptorSet {
    let mut set = descriptor_set_manager.allocate(layout, label!("CullSet"));
    let buffer_info = vk::DescriptorBufferInfo::builder()
        .buffer(**mesh_buffer)
        .offset(0)
        .range(mesh_buffer.size);
    let buffer_info = [*buffer_info];
    let mesh_buffer_write = vk::WriteDescriptorSet::builder()
        .buffer_info(&buffer_info)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_set(*set)
        .dst_binding(0)
        .dst_array_element(0);

    let buffer_info = vk::DescriptorBufferInfo::builder()
        .buffer(**indirect_buffer)
        .offset(0)
        .range(indirect_buffer.size);
    let buffer_info = [*buffer_info];
    let indirect_buffer_write = vk::WriteDescriptorSet::builder()
        .buffer_info(&buffer_info)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_set(*set)
        .dst_binding(1)
        .dst_array_element(0);
    let writes = [*mesh_buffer_write, *indirect_buffer_write];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
    set.attach_resources(Box::new((mesh_buffer, indirect_buffer)));
    set
}
