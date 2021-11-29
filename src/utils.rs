use crate::handles::Device;
use crate::{
    buffer::AllocatedBuffer,
    debug_callback,
    descriptor_sets::{
        DescriptorSet, DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetManager,
    },
    handles::*,
    image::AllocatedImage,
    GlobalUniform, GpuSceneData, TransferContext, LAYER_KHRONOS_VALIDATION,
};
use egui::Output;
use erupt::{cstr, vk, DeviceLoader, InstanceLoader};
use std::{
    ffi::{c_void, CStr, CString},
    sync::Arc,
};

pub fn rasterization_state_create_info<'a>(
    polygon_mode: vk::PolygonMode,
) -> vk::PipelineRasterizationStateCreateInfoBuilder<'a> {
    vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .depth_bias_constant_factor(0.0)
        .depth_bias_clamp(0.0)
        .depth_bias_slope_factor(0.0)
}

pub fn input_assembly_create_info<'a>(
    topology: vk::PrimitiveTopology,
) -> vk::PipelineInputAssemblyStateCreateInfoBuilder<'a> {
    vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(topology)
        .primitive_restart_enable(false)
}

pub fn depth_stencil_create_info<'a>(
    depth_test: bool,
    depth_write: bool,
    compare_op: vk::CompareOp,
) -> vk::PipelineDepthStencilStateCreateInfoBuilder<'a> {
    let compare_op = if depth_test {
        compare_op
    } else {
        vk::CompareOp::ALWAYS
    };
    vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_write)
        .depth_compare_op(compare_op)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false)
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

pub fn multisampling_state_create_info<'a>() -> vk::PipelineMultisampleStateCreateInfoBuilder<'a> {
    vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1)
        .min_sample_shading(1.0)
        .alpha_to_coverage_enable(false)
        .alpha_to_one_enable(false)
}

pub fn pipeline_shader_stage_create_info<'a>(
    stage: vk::ShaderStageFlagBits,
    shader_module: &'a ShaderModule,
) -> vk::PipelineShaderStageCreateInfoBuilder<'a> {
    let mut info = vk::PipelineShaderStageCreateInfoBuilder::new()
        .stage(stage)
        .module(**shader_module);
    info.p_name = cstr!("main");
    info
}

pub fn create_global_descriptor_set_layout(device: Arc<Device>) -> DescriptorSetLayout {
    DescriptorSetLayout::new(
        device.clone(),
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
                ty: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
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
) -> (erupt::InstanceLoader, erupt::EntryLoader, Vec<*const i8>) {
    let application_name = CString::new("test").unwrap();
    let engine_name = CString::new("no engine").unwrap();
    let app_info = vk::ApplicationInfoBuilder::new()
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
    needed_extensions.iter().copied().for_each(|mut v| loop {
        let c = unsafe { *v };
        if c == 0 {
            print!("\n");
            break;
        }
        print!("{}", c as u8 as char);
        v = unsafe { (v).add(1) };
    });

    let entry = erupt::EntryLoader::new().unwrap();
    println!(
        "Vulkan Instance {}.{}.{}",
        vk::api_version_major(entry.instance_version()),
        vk::api_version_minor(entry.instance_version()),
        vk::api_version_patch(entry.instance_version())
    );
    let instance_extensions = window
        .vulkan_instance_extensions()
        .unwrap()
        .into_iter()
        .map(|v| CString::new(v).unwrap())
        .collect::<Vec<CString>>();
    let mut instance_extensions = instance_extensions
        .iter()
        .map(|v| v.as_ptr())
        .collect::<Vec<*const i8>>();
    let mut instance_layers = Vec::new();
    if validation_layers {
        instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
    }

    let device_extensions = vec![vk::KHR_SWAPCHAIN_EXTENSION_NAME];

    let mut instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);

    if validation_layers {
        instance_info.p_next =
            messenger_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
    }

    let instance = unsafe { InstanceLoader::new(&entry, &instance_info, None) }.unwrap();
    (instance, entry, device_extensions)
}

pub fn create_uniform_buffer(
    allocator: Arc<Allocator>,
    frames_in_flight: u64,
    limits: &vk::PhysicalDeviceLimits,
) -> (AllocatedBuffer, u64, u64) {
    let scene_data_offset =
        pad_uniform_buffer_size(limits, std::mem::size_of::<GlobalUniform>() as u64)
            * frames_in_flight;
    let global_uniform_offset = 0;
    let size = pad_uniform_buffer_size(limits, std::mem::size_of::<GlobalUniform>() as u64)
        * frames_in_flight
        + pad_uniform_buffer_size(limits, std::mem::size_of::<GpuSceneData>() as u64)
            * frames_in_flight;

    (
        AllocatedBuffer::new(
            allocator.clone(),
            *vk::BufferCreateInfoBuilder::new()
                .size(size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER),
            vk_mem_erupt::MemoryUsage::CpuToGpu,
            Default::default(),
            label!("UniformBuffer"),
        ),
        global_uniform_offset,
        scene_data_offset,
    )
}

pub fn create_descriptor_sets<'a>(
    device: Arc<Device>,
    descriptor_set_manager: &mut DescriptorSetManager,
    global_layout: &DescriptorSetLayout,
    object_layout: &DescriptorSetLayout,
    uniform_buffer: &AllocatedBuffer,
    object_buffer: &AllocatedBuffer,
    max_objects: usize,
) -> (DescriptorSet, DescriptorSet) {
    let global_set = descriptor_set_manager.allocate(global_layout, None);
    let object_set = descriptor_set_manager.allocate(object_layout, None);

    let camera_info = vk::DescriptorBufferInfoBuilder::new()
        .buffer(**uniform_buffer)
        .range(std::mem::size_of::<GlobalUniform>() as u64);
    let buffer_info = &[camera_info];
    let scene_params_info = vk::DescriptorBufferInfoBuilder::new()
        .buffer(**uniform_buffer)
        .range(std::mem::size_of::<GpuSceneData>() as u64);
    let scene_params_info = &[scene_params_info];

    let uniform_set_write_0 = vk::WriteDescriptorSetBuilder::new()
        .dst_binding(1)
        .dst_set(*global_set)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .buffer_info(scene_params_info);
    let uniform_set_write_1 = vk::WriteDescriptorSetBuilder::new()
        .dst_binding(0)
        .dst_set(*global_set)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
        .buffer_info(buffer_info);

    let object_params_info = vk::DescriptorBufferInfoBuilder::new()
        .buffer(**object_buffer)
        .range((std::mem::size_of::<cgmath::Matrix4<f32>>() * max_objects) as u64);
    let object_params_info = [object_params_info];
    let object_set_write = vk::WriteDescriptorSetBuilder::new()
        .dst_binding(0)
        .dst_set(*object_set)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&object_params_info);

    unsafe {
        device.update_descriptor_sets(
            &[uniform_set_write_0, uniform_set_write_1, object_set_write],
            &[],
        );
    }

    (global_set, object_set)
}

pub fn create_depth_images(
    allocator: Arc<Allocator>,
    width: u32,
    height: u32,
    frames_in_flight: usize,
) -> Vec<AllocatedImage> {
    let format = vk::Format::D32_SFLOAT;
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    (0..frames_in_flight)
        .map(|_| {
            AllocatedImage::new(
                allocator.clone(),
                format,
                usage,
                extent,
                vk::ImageTiling::OPTIMAL,
            )
        })
        .collect()
}
pub fn create_depth_image_views(device: Arc<Device>, images: &[AllocatedImage]) -> Vec<ImageView> {
    images
        .iter()
        .map(|image| {
            let info = AllocatedImage::image_view_create_info(
                vk::Format::D32_SFLOAT,
                **image,
                vk::ImageAspectFlags::DEPTH,
            );
            ImageView::new(device.clone(), &info)
        })
        .collect::<Vec<ImageView>>()
}
pub fn create_device_and_queue(
    instance: &InstanceLoader,
    graphics_queue_family: u32,
    transfer_queue_family: u32,
    device_extensions: &[*const i8],
    device_layers: &[*const i8],
    physical_device: vk::PhysicalDevice,
) -> (Device, vk::Queue, vk::Queue) {
    let queue_info = vec![
        vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&[1.0]),
        vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(transfer_queue_family)
            .queue_priorities(&[0.5]),
    ];
    let features = vk::PhysicalDeviceFeaturesBuilder::new()
        .fill_mode_non_solid(true)
        .multi_draw_indirect(true);

    let mut device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers)
        .build();

    /* let synchronization_2_feature =
    vk::PhysicalDeviceSynchronization2FeaturesKHRBuilder::new().synchronization2(true); */
    let indexing_features = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT {
        //p_next: &synchronization_2_feature as *const _ as *mut std::ffi::c_void,
        shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
        descriptor_binding_partially_bound: vk::TRUE,
        descriptor_binding_variable_descriptor_count: vk::TRUE,
        runtime_descriptor_array: vk::TRUE,
        ..Default::default()
    };
    device_info.p_next = &indexing_features as *const _ as *const c_void;

    let device =
        unsafe { DeviceLoader::new(&instance, physical_device, &device_info, None) }.unwrap();
    let device = Device::new(device);
    let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };
    let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family, 0) };
    (device, graphics_queue, transfer_queue)
}
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
pub fn create_command_buffers(
    device: &DeviceLoader,
    command_pool: &CommandPool,
    number: u32,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(**command_pool)
        .command_buffer_count(number)
        .level(vk::CommandBufferLevel::PRIMARY);

    unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
        .unwrap()
        .to_vec()
}
pub fn create_render_pass(device: Arc<Device>, format: vk::SurfaceFormatKHR) -> RenderPass {
    let color_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(format.format)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescriptionBuilder::new()
        .format(vk::Format::D32_SFLOAT)
        .samples(vk::SampleCountFlagBits::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::CLEAR)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let depth_attachment_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_attachment_ref);

    let attachments = &[color_attachment, depth_attachment];
    let subpasses = &[subpass];
    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(attachments)
        .subpasses(subpasses);

    RenderPass::new(device, &render_pass_info)
}
pub fn create_swapchain(
    instance: &InstanceLoader,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    format: vk::SurfaceFormatKHR,
    device: Arc<Device>,
    present_mode: vk::PresentModeKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
) -> (Swapchain, Vec<vk::Image>, Vec<ImageView>) {
    let surface_caps =
        unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface) }
            .unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }
    let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(surface_caps.current_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(match old_swapchain {
            Some(sc) => sc,
            None => vk::SwapchainKHR::null(),
        });

    let swapchain = Swapchain::new(device.clone(), &swapchain_info);
    let swapchain_images = unsafe { device.get_swapchain_images_khr(*swapchain, None) }
        .unwrap()
        .to_vec();

    // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Image_views
    let swapchain_image_views: Vec<_> = swapchain_images
        .iter()
        .map(|swapchain_image| {
            let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                .image(*swapchain_image)
                .view_type(vk::ImageViewType::_2D)
                .format(format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(
                    vk::ImageSubresourceRangeBuilder::new()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );
            ImageView::new(device.clone(), &image_view_info)
        })
        .collect();
    (swapchain, swapchain_images, swapchain_image_views)
}
pub fn create_framebuffers(
    device: &Arc<Device>,
    width: u32,
    height: u32,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[ImageView],
    depth_views: &[ImageView],
) -> Vec<Framebuffer> {
    let mut framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
        .render_pass(render_pass)
        .width(width)
        .height(height)
        .layers(1);

    let attachments = swapchain_image_views
        .iter()
        .zip(depth_views.iter().cycle())
        .map(|(swapchain_image_view, depth_view)| [**swapchain_image_view, **depth_view])
        .collect::<Vec<[vk::ImageView; 2]>>();

    attachments
        .iter()
        .map(|attachments| {
            framebuffer_info = framebuffer_info.attachments(attachments);
            Framebuffer::new(device.clone(), &framebuffer_info)
        })
        .collect::<Vec<_>>()
}

pub fn create_physical_device(
    instance: &InstanceLoader,
    surface: vk::SurfaceKHR,
    device_extensions: &[*const i8],
    preferred_present_mode: vk::PresentModeKHR,
    backup_present_mode: vk::PresentModeKHR,
) -> (
    vk::PhysicalDevice,
    u32,
    vk::SurfaceFormatKHR,
    vk::PresentModeKHR,
    vk::PhysicalDeviceProperties,
    Vec<vk::PresentModeKHR>,
    u32,
) {
    let (
        physical_device,
        graphics_queue_family,
        format,
        present_mode,
        device_properties,
        present_modes,
        transfer_queue_family
    ) = unsafe { instance.enumerate_physical_devices(None) }
        .unwrap()
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let physical_device_queue_familiy_properties =
                instance.get_physical_device_queue_family_properties(physical_device, None);
            let graphics_queue_family = match physical_device_queue_familiy_properties
                .iter()
                .copied()
                .enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && instance
                            .get_physical_device_surface_support_khr(
                                physical_device,
                                i as u32,
                                surface,
                            )
                            .unwrap()
                }) {
                Some(queue_family) => queue_family as u32,
                None => return None,
            };
            dbg!("GOT HERE");
            let features = instance.get_physical_device_features(physical_device);
            if features.multi_draw_indirect == vk::FALSE {
                dbg!("NO MULTI DRAW INDIRECT");
                return None;
            }
            //let features = instance.get_physical_device_features2(physical_device, None);
            let transfer_queue = physical_device_queue_familiy_properties
                .into_iter().enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::TRANSFER) && queue_family_properties
                        .queue_flags.contains(vk::QueueFlags::COMPUTE) && i != graphics_queue_family as usize
                }).map(|i| i as u32);
            dbg!((graphics_queue_family, transfer_queue));
            let formats = instance
                .get_physical_device_surface_formats_khr(physical_device, surface, None)
                .unwrap();
            let format = match formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == vk::Format::B8G8R8A8_SRGB
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
                })
                .or_else(|| formats.get(0))
            {
                Some(surface_format) => *surface_format,
                None => return None,
            };
            dbg!("SURFACE IS SUPPORTED");

            let present_modes = instance
                .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
                .unwrap().to_vec();
            let present_mode = *present_modes
                .iter()
                .find(|present_mode| present_mode == &&preferred_present_mode)
                .unwrap_or(&backup_present_mode);

            let supported_device_extensions = instance
                .enumerate_device_extension_properties(physical_device, None, None)
                .unwrap();
            let device_extensions_supported = device_extensions.iter().all(|device_extension| {
                let device_extension = CStr::from_ptr(*device_extension);
                dbg!(device_extension);
                dbg!(supported_device_extensions.iter().any(|properties| {
                    CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                }))
            });

            if !device_extensions_supported {
                dbg!("EXTENSIONS NOT SUPPORTED");
                return None;
            }

            let device_properties = instance.get_physical_device_properties(physical_device);
            Some((
                physical_device,
                graphics_queue_family,
                format,
                present_mode,
                device_properties,
                present_modes,
                transfer_queue
            ))
        })
        .max_by_key(|(_, _, _, _, properties, _, transfer_queue)| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 0,
        } + if transfer_queue.is_some() { 1 } else { 0 })
        .expect("No suitable physical device found");
    let transfer_queue_family = transfer_queue_family.unwrap_or(graphics_queue_family);
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
pub fn create_full_view_port(width: u32, height: u32) -> vk::Viewport {
    vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: width as f32,
        height: height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}
pub fn create_pipeline_color_blend_attachment_state(
) -> vk::PipelineColorBlendAttachmentStateBuilder<'static> {
    vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false)
}

pub fn create_debug_messenger(
    instance: &InstanceLoader,
    messenger_info: &vk::DebugUtilsMessengerCreateInfoEXT,
    validation_layers: bool,
) -> vk::DebugUtilsMessengerEXT {
    let messenger = if validation_layers {
        unsafe { instance.create_debug_utils_messenger_ext(&messenger_info, None) }.unwrap()
    } else {
        Default::default()
    };
    messenger
}
pub fn create_debug_messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT {
    vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
        )
        .pfn_user_callback(Some(debug_callback))
        .build()
}

pub fn create_renderables_buffer(allocator: Arc<Allocator>, max_objects: u64) -> AllocatedBuffer {
    let size = std::mem::size_of::<crate::GpuDataRenderable>() as u64 * max_objects;
    AllocatedBuffer::new(
        allocator.clone(),
        *vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER),
        vk_mem_erupt::MemoryUsage::CpuToGpu,
        Default::default(),
        label!("RenderablesBuffer"),
    )
}

pub fn create_object_set_layout(device: Arc<Device>) -> DescriptorSetLayout {
    DescriptorSetLayout::new(
        device.clone(),
        vec![DescriptorSetLayoutBinding {
            binding: 0,
            count: 1,
            ty: vk::DescriptorType::STORAGE_BUFFER,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
            immutable_samplers: None,
        }],
        None,
    )
}

pub fn immediate_submit<F>(device: &Device, transfer_context: &TransferContext, f: F)
where
    F: FnOnce(vk::CommandBuffer),
{
    unsafe {
        device.reset_fences(&[*transfer_context.fence]).unwrap();
        let alloc_info = vk::CommandBufferAllocateInfoBuilder::new()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(*transfer_context.command_pool);
        let cmd = device.allocate_command_buffers(&alloc_info).unwrap()[0];
        let begin_info = vk::CommandBufferBeginInfoBuilder::new()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(cmd, &begin_info).unwrap();
        f(cmd);
        device.end_command_buffer(cmd).unwrap();
        let cmds = &[cmd];
        let submit_info = vk::SubmitInfoBuilder::new().command_buffers(cmds);
        device
            .queue_submit(
                transfer_context.transfer_queue,
                &[submit_info],
                Some(*transfer_context.fence),
            )
            .unwrap();
        device
            .wait_for_fences(&[*transfer_context.fence], true, 1000_000_000_000)
            .unwrap();
        device
            .reset_command_pool(*transfer_context.command_pool, None)
            .unwrap();
    }
}

pub fn create_textures_set_layout<'a>(
    device: &Arc<Device>,
    views_number: usize,
    sampler: &Sampler,
) -> DescriptorSetLayout {
    DescriptorSetLayout::new(
        device.clone(),
        vec![DescriptorSetLayoutBinding {
            binding: 0,
            count: views_number as u32,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            immutable_samplers: Some(vec![**sampler; views_number]),
        }],
        Some(&[vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT]),
    )
}

pub fn create_textures_set<'a>(
    device: &Arc<Device>,
    descriptor_set_manager: &mut DescriptorSetManager,
    views: impl ExactSizeIterator<Item = &'a ImageView>,
    set_layout: &DescriptorSetLayout,
) -> DescriptorSet {
    let view_number = views.len();
    let descriptor_counts = [view_number as u32];
    let descriptor_counts = descriptor_counts.as_ptr();
    let variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo {
        descriptor_set_count: 1,
        p_descriptor_counts: descriptor_counts,
        ..Default::default()
    };
    let textures_set = descriptor_set_manager.allocate(&set_layout, Some(&variable_info));
    if view_number == 0 {
        return textures_set;
    }
    let image_infos = views
        .map(|view| {
            let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            vk::DescriptorImageInfoBuilder::new()
                .image_layout(image_layout)
                .image_view(**view)
        })
        .collect::<Vec<_>>();
    let mut writes = Vec::new();
    for i in 0..image_infos.len() {
        writes.push(
            vk::WriteDescriptorSetBuilder::new()
                .dst_binding(0)
                .dst_set(*textures_set)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_infos[i..i + 1])
                .dst_array_element(i as u32),
        );
    }
    let writes = [vk::WriteDescriptorSetBuilder::new()
        .dst_binding(0)
        .dst_set(*textures_set)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)
        .dst_array_element(0)];
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    textures_set
}

pub fn create_indirect_buffer(allocator: Arc<Allocator>, size: u64) -> AllocatedBuffer {
    let buffer_info = vk::BufferCreateInfoBuilder::new().size(size).usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::INDIRECT_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST,
    );
    AllocatedBuffer::new(
        allocator,
        *buffer_info,
        vk_mem_erupt::MemoryUsage::GpuOnly,
        vk::MemoryPropertyFlags::empty(),
        label!("IndirectBuffer"),
    )
}

pub fn create_indirect_buffer_set(
    device: &Device,
    descriptor_set_manager: &mut DescriptorSetManager,
    indirect_buffer: &AllocatedBuffer,
    set_layout: &DescriptorSetLayout,
) -> DescriptorSet {
    let set = descriptor_set_manager.allocate(&set_layout, None);

    let write_buffer_info = vk::DescriptorBufferInfoBuilder::new()
        .buffer(**indirect_buffer)
        .range(indirect_buffer.size)
        .offset(0);
    let buffer_info = &[write_buffer_info];
    let indirect_buffer_write = vk::WriteDescriptorSetBuilder::new()
        .buffer_info(buffer_info)
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(0);

    let writes = [indirect_buffer_write];
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    set
}

pub fn create_mesh_buffer_set(
    device: &Device,
    descriptor_set_manager: &mut DescriptorSetManager,
    buffer: &AllocatedBuffer,
    set_layout: &DescriptorSetLayout,
) -> DescriptorSet {
    let set = descriptor_set_manager.allocate(&set_layout, None);
    let write_buffer_info = vk::DescriptorBufferInfoBuilder::new()
        .buffer(**buffer)
        .range(buffer.size)
        .offset(0);
    let buffer_info = &[write_buffer_info];
    let write = vk::WriteDescriptorSetBuilder::new()
        .buffer_info(buffer_info)
        .dst_set(*set)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .dst_binding(0);
    let writes = [write];
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
    set
}

pub fn create_mesh_buffer(allocator: Arc<Allocator>, size: u64) -> AllocatedBuffer {
    let buffer_info = vk::BufferCreateInfoBuilder::new()
        .size(size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER);
    AllocatedBuffer::new(
        allocator.clone(),
        *buffer_info,
        vk_mem_erupt::MemoryUsage::CpuToGpu,
        Default::default(),
        label!("MeshBuffer"),
    )
}

pub fn round_to<T>(v: T, a: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Rem<Output = T> + std::ops::Sub<Output = T> + Copy,
{
    v + (a - (v % a))
}
