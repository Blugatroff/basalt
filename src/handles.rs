use erupt::vk;
use std::{ffi::CString, io::Read, sync::Arc};

pub struct Allocator(vk_mem_erupt::Allocator);

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Allocator")
    }
}

impl Allocator {
    pub fn new(info: vk_mem_erupt::AllocatorCreateInfo) -> Self {
        Self(vk_mem_erupt::Allocator::new(&info).unwrap())
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        println!("DROPPED Allocator!");
        self.0.destroy();
    }
}

impl std::ops::Deref for Allocator {
    type Target = vk_mem_erupt::Allocator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait MyInto<T> {
    fn my_into(self) -> T;
}

#[rustfmt::skip]
impl MyInto<vk::DescriptorType> for spirv_reflect::types::ReflectDescriptorType {
    fn my_into(self) -> vk::DescriptorType {
        match self {
            spirv_reflect::types::ReflectDescriptorType::Undefined => todo!(),
            spirv_reflect::types::ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
            spirv_reflect::types::ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            spirv_reflect::types::ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            spirv_reflect::types::ReflectDescriptorType::StorageImage => vk::DescriptorType::SAMPLED_IMAGE,
            spirv_reflect::types::ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            spirv_reflect::types::ReflectDescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            spirv_reflect::types::ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            spirv_reflect::types::ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            spirv_reflect::types::ReflectDescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            spirv_reflect::types::ReflectDescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
            spirv_reflect::types::ReflectDescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
            spirv_reflect::types::ReflectDescriptorType::AccelerationStructureNV => vk::DescriptorType::ACCELERATION_STRUCTURE_NV,
        }
    }
}

pub struct ShaderModule {
    module: vk::ShaderModule,
    device: Arc<Device>,
}
impl ShaderModule {
    pub fn load(
        device: Arc<Device>,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, std::io::Error> {
        let mut data = Vec::new();
        let len = std::fs::File::open(path)?.read_to_end(&mut data)?;
        Ok(Self::new(device, &data[0..len]))
    }
    pub fn new(device: Arc<Device>, spv: &[u8]) -> Self {
        assert!(spv.len() % 4 == 0);
        let code = unsafe { std::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4) };
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(&code);
        let module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();
        /* let reflect = spirv_reflect::ShaderModule::load_u32_data(code).unwrap();
        let descriptor_bindings = reflect.enumerate_descriptor_bindings(Some("main")).unwrap();
        let stage = reflect.get_shader_stage();
        //dbg!(reflect.enumerate_input_variables(Some("main"))).unwrap();
        let bindings = descriptor_bindings
        .into_iter()
        .map(|binding| crate::DescriptorSetLayoutBinding {
            binding: binding.binding,
            count: binding.count,
            ty: binding.descriptor_type.my_into(),
            stage_flags: vk::ShaderStageFlags::from_bits(stage.bits()).unwrap(),
            immutable_samplers: None,
        })
        .collect::<Vec<_>>(); */
        //dbg!(bindings);
        Self { module, device }
    }
}

impl std::ops::Deref for ShaderModule {
    type Target = vk::ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        println!("DROPPED ShaderModule!");
        unsafe { self.device.destroy_shader_module(Some(self.module), None) };
    }
}

#[derive(Debug)]
pub struct Device(Arc<erupt::DeviceLoader>);

impl Device {
    pub fn new(device: erupt::DeviceLoader) -> Self {
        Self(Arc::new(device))
    }
    pub fn raw(&self) -> Arc<erupt::DeviceLoader> {
        self.0.clone()
    }
}

impl std::ops::Deref for Device {
    type Target = erupt::DeviceLoader;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        println!("DROPPED Device!");
        unsafe {
            self.0.destroy_device(None);
        }
    }
}

pub struct Instance {
    instance: Arc<erupt::InstanceLoader>,
}

impl Instance {
    pub fn new(instance: erupt::InstanceLoader) -> Self {
        let instance = Arc::new(instance);
        Self { instance }
    }
    pub fn raw(&self) -> Arc<erupt::InstanceLoader> {
        self.instance.clone()
    }
}

impl std::ops::Deref for Instance {
    type Target = erupt::InstanceLoader;
    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        println!("DROPPED Instance!");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

pub struct PipelineDesc<'a> {
    pub view_port: vk::Viewport,
    pub scissor: vk::Rect2DBuilder<'a>,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentStateBuilder<'a>,
    pub shader_stages: &'a [vk::PipelineShaderStageCreateInfoBuilder<'a>],
    pub vertex_input_info: &'a vk::PipelineVertexInputStateCreateInfo,
    pub input_assembly_state: &'a vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterization_state: &'a vk::PipelineRasterizationStateCreateInfo,
    pub multisample_state: &'a vk::PipelineMultisampleStateCreateInfo,
    pub layout: vk::PipelineLayout,
    pub depth_stencil: &'a vk::PipelineDepthStencilStateCreateInfo,
}

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}
impl ComputePipeline {
    pub fn new<'a>(
        device: Arc<Device>,
        layout: &PipelineLayout,
        shader_module: &ShaderModule,
    ) -> Self {
        let name = CString::new("main").unwrap();
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .module(**shader_module)
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .flags(vk::PipelineShaderStageCreateFlags::empty())
            .name(&name);
        let create_info = vk::ComputePipelineCreateInfoBuilder::new()
            .layout(**layout)
            .flags(vk::PipelineCreateFlags::empty())
            .stage(*shader_stage_create_info);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(None, &[create_info], None)
                .unwrap()
        };
        assert_eq!(pipeline.len(), 1);
        let pipeline = pipeline[0];
        Self { device, pipeline }
    }
}

impl std::ops::Deref for ComputePipeline {
    type Target = vk::Pipeline;
    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        println!("DROPPED ComputePipeline!");
        unsafe { self.device.destroy_pipeline(Some(self.pipeline), None) };
    }
}

#[derive(Debug)]
pub struct Pipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl Pipeline {
    pub fn new<'a>(device: Arc<Device>, pass: vk::RenderPass, desc: PipelineDesc<'a>) -> Self {
        let viewports = &[desc.view_port.into_builder()];
        let scissors = &[desc.scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
            .viewports(viewports)
            .scissors(scissors);
        let attachments = &[desc.color_blend_attachment];
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfoBuilder::new()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments);
        let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
            .stages(desc.shader_stages)
            .vertex_input_state(desc.vertex_input_info)
            .input_assembly_state(desc.input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(desc.rasterization_state)
            .multisample_state(desc.multisample_state)
            .color_blend_state(&color_blend_state_create_info)
            .layout(desc.layout)
            .render_pass(pass)
            .depth_stencil_state(&desc.depth_stencil)
            .subpass(0);

        let pipeline =
            unsafe { device.create_graphics_pipelines(None, &[pipeline_info], None) }.unwrap()[0];

        Self { device, pipeline }
    }
}

impl std::ops::Deref for Pipeline {
    type Target = vk::Pipeline;
    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            println!("DROPPED Pipeline!");
            self.device.destroy_pipeline(Some(self.pipeline), None);
        }
    }
}

pub struct Surface {
    surface: vk::SurfaceKHR,
    instance: Arc<Instance>,
}

impl Surface {
    pub fn new(instance: Arc<Instance>, window: &sdl2::video::Window) -> Self {
        let surface =
            unsafe { erupt::utils::surface::create_surface(&instance, window, None) }.unwrap();
        Self { surface, instance }
    }
}

impl std::ops::Deref for Surface {
    type Target = vk::SurfaceKHR;
    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            println!("DROPPED Surface!");
            self.instance.destroy_surface_khr(Some(self.surface), None);
        }
    }
}

macro_rules! handle {
    ( $name:ident, $t:ty, $info:ty, $create:expr, $destroy:expr ) => {
        #[derive(Debug)]
        pub struct $name {
            inner: $t,
            device: Arc<Device>,
        }

        impl $name {
            pub fn new<'a>(device: Arc<Device>, info: &$info) -> Self {
                let inner = $create(&device, info);
                Self { inner, device }
            }
        }

        impl std::ops::Deref for $name {
            type Target = $t;
            fn deref(&self) -> &Self::Target {
                &self.inner
            }
        }

        impl Drop for $name {
            fn drop(&mut self) {
                println!(concat!("DROPPED ", stringify!($name), "!"));
                $destroy(&self.device, self.inner)
            }
        }
    };
}

handle!(
    ImageView,
    vk::ImageView,
    vk::ImageViewCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_image_view(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_image_view(Some(inner), None) }
);

handle!(
    PipelineLayout,
    vk::PipelineLayout,
    vk::PipelineLayoutCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_pipeline_layout(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_pipeline_layout(Some(inner), None) }
);

handle!(
    Swapchain,
    vk::SwapchainKHR,
    vk::SwapchainCreateInfoKHRBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_swapchain_khr(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_swapchain_khr(Some(inner), None) }
);

handle!(
    Framebuffer,
    vk::Framebuffer,
    vk::FramebufferCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_framebuffer(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_framebuffer(Some(inner), None) }
);

handle!(
    RenderPass,
    vk::RenderPass,
    vk::RenderPassCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_render_pass(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_render_pass(Some(inner), None) }
);

handle!(
    Fence,
    vk::Fence,
    vk::FenceCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_fence(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_fence(Some(inner), None) }
);

handle!(
    Semaphore,
    vk::Semaphore,
    vk::SemaphoreCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_semaphore(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_semaphore(Some(inner), None) }
);

handle!(
    CommandPool,
    vk::CommandPool,
    vk::CommandPoolCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_command_pool(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_command_pool(Some(inner), None) }
);

handle!(
    Sampler,
    vk::Sampler,
    vk::SamplerCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_sampler(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_sampler(Some(inner), None) }
);
