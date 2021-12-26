use erupt::vk;
use std::{ffi::CString, io::Read, sync::Arc};

pub struct Allocator(vk_mem_erupt::Allocator);

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Allocator")
    }
}

impl Allocator {
    pub fn new(info: &vk_mem_erupt::AllocatorCreateInfo) -> Self {
        Self(vk_mem_erupt::Allocator::new(info).unwrap())
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

pub struct ShaderModule {
    module: vk::ShaderModule,
    device: Arc<Device>,
    name: String,
    shader: reflection::Shader,
    stage: vk::ShaderStageFlags,
}
impl ShaderModule {
    pub fn load(device: Arc<Device>, path: &'static str) -> Result<Self, std::io::Error> {
        let mut data = Vec::new();
        let len = std::fs::File::open(path)?.read_to_end(&mut data)?;
        let stage = if path.ends_with(".frag.spv") {
            vk::ShaderStageFlags::FRAGMENT
        } else if path.ends_with(".vert.spv") {
            vk::ShaderStageFlags::VERTEX
        } else if path.ends_with(".comp.spv") {
            vk::ShaderStageFlags::COMPUTE
        } else {
            panic!()
        };
        Ok(Self::new(device, &data[0..len], path.into(), stage))
    }
    pub fn stage(&self) -> vk::ShaderStageFlags {
        self.stage
    }
    pub fn new(device: Arc<Device>, spv: &[u8], name: String, stage: vk::ShaderStageFlags) -> Self {
        assert!(spv.len() % 4 == 0);
        let code = unsafe { std::slice::from_raw_parts(spv.as_ptr().cast(), spv.len() / 4) };
        let create_info = vk::ShaderModuleCreateInfoBuilder::new().code(code);
        let module = unsafe { device.create_shader_module(&create_info, None) }.unwrap();
        let mut shaders = reflection::Shader::from_spirv(spv).unwrap();
        assert_eq!(shaders.len(), 1);
        let shader = shaders.swap_remove(0);
        Self {
            stage,
            module,
            device,
            name,
            shader,
        }
    }
    pub fn reflection(&self) -> &reflection::Shader {
        &self.shader
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
        println!("DROPPED ShaderModule! {}", self.name);
        unsafe { self.device.destroy_shader_module(Some(self.module), None) };
    }
}

#[derive(Debug)]
pub struct Device(Arc<erupt::DeviceLoader>, Arc<Instance>);

impl Device {
    pub fn new(device: erupt::DeviceLoader, instance: Arc<Instance>) -> Self {
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
        println!("DROPPED Device!");
        unsafe {
            self.0.destroy_device(None);
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
        println!("DROPPED Instance!");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Clone)]
pub struct PipelineDesc<'a> {
    pub view_port: vk::Viewport,
    pub scissor: vk::Rect2DBuilder<'a>,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentStateBuilder<'a>,
    pub shader_stages: &'a [vk::PipelineShaderStageCreateInfoBuilder<'a>],
    pub vertex_input_info: &'a vk::PipelineVertexInputStateCreateInfo,
    pub input_assembly_state: &'a vk::PipelineInputAssemblyStateCreateInfo,
    pub rasterization_state: &'a vk::PipelineRasterizationStateCreateInfo,
    pub multisample_state: &'a vk::PipelineMultisampleStateCreateInfo,
    pub layout: Arc<PipelineLayout>,
    pub depth_stencil: &'a vk::PipelineDepthStencilStateCreateInfo,
}

pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}
impl ComputePipeline {
    #[allow(dead_code)]
    pub fn new(device: Arc<Device>, layout: &PipelineLayout, shader_module: &ShaderModule) -> Self {
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
        Self { pipeline, device }
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
                .layout(**desc.layout)
                .render_pass(pass)
                .depth_stencil_state(desc.depth_stencil)
                .subpass(0);

            unsafe { device.create_graphics_pipelines(None, &[pipeline_info], None) }.unwrap()[0]
        }
        let name = name.into();
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
            println!("DROPPED Pipeline! {}", self.name);
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
            name: String,
        }

        impl $name {
            pub fn new<'a>(device: Arc<Device>, info: &$info, name: impl Into<String>) -> Self {
                let inner = $create(&device, info);
                let name = name.into();
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
                println!(concat!("DROPPED ", stringify!($name), "! {}"), self.name);
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
