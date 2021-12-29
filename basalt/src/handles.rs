use crate::{
    mesh::VertexInfoDescription,
    utils::{log_resource_created, log_resource_dropped, ColorBlendAttachment, DepthStencilInfo},
    utils::{InputAssemblyState, MultiSamplingState, RasterizationState},
};
use erupt::{cstr, vk};
use std::{ffi::CString, io::Read, marker::PhantomData, sync::Arc};

pub struct Allocator(vk_mem_erupt::Allocator);

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Allocator")
    }
}

impl Allocator {
    pub fn new(info: &vk_mem_erupt::AllocatorCreateInfo) -> Self {
        log_resource_created("Allocator", "");
        Self(vk_mem_erupt::Allocator::new(info).unwrap())
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log_resource_dropped("Allocator", "");
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

pub struct ShaderModule {
    module: vk::ShaderModule,
    device: Arc<Device>,
    name: String,
    shader: reflection::Shader,
    stage: vk::ShaderStageFlagBits,
}
impl ShaderModule {
    pub fn load(device: Arc<Device>, path: &'static str) -> Result<Self, std::io::Error> {
        let mut data = Vec::new();
        let len = std::fs::File::open(path)?.read_to_end(&mut data)?;
        let stage = if path.ends_with(".frag.spv") {
            vk::ShaderStageFlagBits::FRAGMENT
        } else if path.ends_with(".vert.spv") {
            vk::ShaderStageFlagBits::VERTEX
        } else if path.ends_with(".comp.spv") {
            vk::ShaderStageFlagBits::COMPUTE
        } else {
            panic!()
        };
        Ok(Self::new(device, &data[0..len], path.into(), stage))
    }
    pub fn stage(&self) -> vk::ShaderStageFlagBits {
        self.stage
    }
    pub fn new(
        device: Arc<Device>,
        spv: &[u8],
        name: String,
        stage: vk::ShaderStageFlagBits,
    ) -> Self {
        assert!(spv.len() % 4 == 0);
        let code = unsafe { std::slice::from_raw_parts(spv.as_ptr().cast(), spv.len() / 4) };
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(code);
        let module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();
        let mut shaders = reflection::Shader::from_spirv(spv).unwrap();
        assert_eq!(shaders.len(), 1);
        let shader = shaders.swap_remove(0);
        log_resource_created("ShaderMode", &name);
        Self {
            module,
            device,
            name,
            shader,
            stage,
        }
    }
    pub fn reflection(&self) -> &reflection::Shader {
        &self.shader
    }
    fn builder(&self) -> vk::PipelineShaderStageCreateInfoBuilder {
        let mut info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(self.stage)
            .module(self.module);
        info.p_name = cstr!("main");
        info
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
        log_resource_dropped("ShaderMode", &self.name);
        unsafe { self.device.destroy_shader_module(Some(self.module), None) };
    }
}

#[derive(Debug)]
pub struct Device(Arc<erupt::DeviceLoader>, Arc<Instance>);

impl Device {
    pub fn new(device: erupt::DeviceLoader, instance: Arc<Instance>) -> Self {
        log_resource_created("Device", "");
        Self(Arc::new(device), instance)
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
        log_resource_dropped("Device", "");
        unsafe {
            self.0.destroy_device(None);
        }
    }
}

pub struct DebugUtilsMessenger {
    inner: vk::DebugUtilsMessengerEXT,
    instance: Arc<Instance>,
}

impl DebugUtilsMessenger {
    pub fn new(
        instance: Arc<Instance>,
        messenger_info: &vk::DebugUtilsMessengerCreateInfoEXTBuilder,
    ) -> Self {
        let inner =
            unsafe { instance.create_debug_utils_messenger_ext(messenger_info, None) }.unwrap();
        log_resource_created("DebugUtilsMessenger", "");
        Self { inner, instance }
    }
}

impl Drop for DebugUtilsMessenger {
    fn drop(&mut self) {
        log_resource_dropped("DebugUtilsMessenger", "");
        unsafe {
            self.instance
                .destroy_debug_utils_messenger_ext(Some(self.inner), None);
        }
    }
}

#[derive(Debug)]
pub struct Instance {
    instance: Arc<erupt::InstanceLoader>,
    #[allow(dead_code)]
    entry: erupt::EntryLoader,
}

impl Instance {
    pub fn new(instance: erupt::InstanceLoader, entry: erupt::EntryLoader) -> Self {
        let instance = Arc::new(instance);
        log_resource_created("Instance", "");
        Self { instance, entry }
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
        log_resource_dropped("Instance", "");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone)]
pub struct PipelineDesc<'a> {
    pub view_port: vk::Viewport,
    pub scissor: vk::Rect2DBuilder<'a>,
    pub color_blend_attachment: ColorBlendAttachment,
    pub shader_stages: &'a [&'a ShaderModule],
    pub vertex_description: VertexInfoDescription,
    pub input_assembly_state: InputAssemblyState,
    pub rasterization_state: RasterizationState,
    pub multisample_state: MultiSamplingState,
    pub layout: Arc<PipelineLayout>,
    pub depth_stencil: DepthStencilInfo,
}

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
    layout: Arc<PipelineLayout>,
    name: String,
}
impl ComputePipeline {
    #[allow(dead_code)]
    pub fn new(
        device: Arc<Device>,
        layout: Arc<PipelineLayout>,
        shader_module: &ShaderModule,
        name: impl Into<String>,
    ) -> Self {
        let entry_name = CString::new("main").unwrap();
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfoBuilder::new()
            .module(**shader_module)
            .stage(vk::ShaderStageFlagBits::COMPUTE)
            .flags(vk::PipelineShaderStageCreateFlags::empty())
            .name(&entry_name);
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
        let name = name.into();
        log_resource_created("ComputePipeline", &name);
        Self {
            pipeline,
            device,
            name,
            layout,
        }
    }
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
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
        log_resource_dropped("ComputePipeline", &self.name);
        unsafe { self.device.destroy_pipeline(Some(self.pipeline), None) };
    }
}

#[derive(Debug)]
pub struct Pipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
    name: String,
    layout: Arc<PipelineLayout>,
}

impl Pipeline {
    pub fn new(
        device: Arc<Device>,
        pass: vk::RenderPass,
        desc: &PipelineDesc,
        name: impl Into<String>,
    ) -> Self {
        fn new(device: &Device, pass: vk::RenderPass, desc: &PipelineDesc) -> vk::Pipeline {
            let viewports = &[desc.view_port.into_builder()];
            let scissors = &[desc.scissor];
            let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
                .viewports(viewports)
                .scissors(scissors);
            let attachments = &[desc.color_blend_attachment.builder()];
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfoBuilder::new()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments);
            let depth_stencil_state = std::convert::Into::<
                vk::PipelineDepthStencilStateCreateInfoBuilder,
            >::into(desc.depth_stencil);
            let multi_sample_state = desc.multisample_state.builder();
            let rasterization_state = desc.rasterization_state.builder();
            let vertex_info = desc.vertex_description.builder();
            let input_assembly_state = desc.input_assembly_state.builder();
            let shader_stages = desc
                .shader_stages
                .iter()
                .copied()
                .map(ShaderModule::builder)
                .collect::<Vec<_>>();
            let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_info)
                .input_assembly_state(&input_assembly_state)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization_state)
                .multisample_state(&multi_sample_state)
                .color_blend_state(&color_blend_state_create_info)
                .layout(**desc.layout)
                .render_pass(pass)
                .depth_stencil_state(&depth_stencil_state)
                .subpass(0);

            unsafe { device.create_graphics_pipelines(None, &[pipeline_info], None) }.unwrap()[0]
        }
        let name = name.into();
        log_resource_created("Pipeline", &name);
        Self {
            layout: Arc::clone(&desc.layout),
            pipeline: new(&device, pass, desc),
            device,
            name,
        }
    }
    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
    pub fn name(&self) -> &str {
        &self.name
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
            log_resource_dropped("Pipeline", &self.name);
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
        log_resource_created("Surface", "");
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
            log_resource_dropped("Surface", "");
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
            name: String,
        }

        impl $name {
            pub fn new<'a>(device: Arc<Device>, info: &$info, name: impl Into<String>) -> Self {
                let inner = $create(&device, info);
                let name = name.into();
                crate::utils::log_resource_created(stringify!($name), &name);
                Self {
                    inner,
                    device,
                    name,
                }
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
                crate::utils::log_resource_dropped(stringify!($name), &self.name);
                $destroy(&self.device, self.inner)
            }
        }
    };
}

#[derive(Debug)]
pub struct CommandPool {
    inner: vk::CommandPool,
    device: Arc<Device>,
    name: String,
    _phantom: PhantomData<*const ()>,
}

unsafe impl Send for CommandPool {}

impl CommandPool {
    pub fn new(
        device: Arc<Device>,
        info: &vk::CommandPoolCreateInfoBuilder,
        name: impl Into<String>,
    ) -> Self {
        let inner = unsafe { device.create_command_pool(info, None) }.unwrap();
        let name = name.into();
        log_resource_created("CommandPool", &name);
        Self {
            inner,
            device,
            name,
            _phantom: PhantomData::default(),
        }
    }
}

impl std::ops::Deref for CommandPool {
    type Target = vk::CommandPool;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        log_resource_dropped("CommandPool", &self.name);
        unsafe {
            self.device.destroy_command_pool(Some(self.inner), None);
        }
    }
}

#[derive(Debug)]
pub struct Fence {
    inner: vk::Fence,
    device: Arc<Device>,
    name: String,
    _phantom: PhantomData<*const ()>,
}

unsafe impl Send for Fence {}

impl Fence {
    pub fn new(
        device: Arc<Device>,
        info: &vk::FenceCreateInfoBuilder,
        name: impl Into<String>,
    ) -> Self {
        let inner = unsafe { device.create_fence(info, None) }.unwrap();
        let name = name.into();
        log_resource_created("Fence", &name);
        Self {
            inner,
            device,
            name,
            _phantom: PhantomData::default(),
        }
    }
}

impl std::ops::Deref for Fence {
    type Target = vk::Fence;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        log_resource_dropped("Fence", &self.name);
        unsafe {
            self.device.destroy_fence(Some(self.inner), None);
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QueueFamily(pub u32);

#[derive(Debug)]
pub struct Queue {
    inner: vk::Queue,
    name: String,
    _phantom: PhantomData<*const ()>,
}

unsafe impl Send for Queue {}

impl Queue {
    pub fn new(queue: vk::Queue, name: impl Into<String>) -> Self {
        let name = name.into();
        log_resource_created("Queue", &name);
        Self {
            inner: queue,
            name,
            _phantom: PhantomData::default(),
        }
    }
}

impl std::ops::Deref for Queue {
    type Target = vk::Queue;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        log_resource_dropped("Queue", &self.name);
    }
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
    Semaphore,
    vk::Semaphore,
    vk::SemaphoreCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_semaphore(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_semaphore(Some(inner), None) }
);

handle!(
    Sampler,
    vk::Sampler,
    vk::SamplerCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_sampler(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_sampler(Some(inner), None) }
);
