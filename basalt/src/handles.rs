use crate::{
    mesh::VertexInfoDescription,
    utils::{log_resource_created, log_resource_dropped},
    utils::{InputAssemblyState, MultiSamplingState, RasterizationState},
    DescriptorSetLayout, Vertex,
};
use ash::{
    extensions::ext::DebugUtils,
    vk::{self, DebugUtilsMessengerEXT},
    Entry,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::{any::TypeId, ffi::CString, io::Read, marker::PhantomData, sync::Arc};

pub struct Allocator(vma::Allocator, Arc<Device>);

impl std::fmt::Debug for Allocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Allocator")
    }
}

impl Allocator {
    pub fn new(device: Arc<Device>, info: vma::AllocatorCreateInfo) -> Self {
        log_resource_created("Allocator", "");
        Self(vma::Allocator::new(info).unwrap(), device)
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log_resource_dropped("Allocator", "");
    }
}

impl std::ops::Deref for Allocator {
    type Target = vma::Allocator;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait MyInto<T> {
    fn my_into(self) -> T;
}

#[derive(Debug)]
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
        let create_info = vk::ShaderModuleCreateInfo::builder().code(code);
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
        let mut info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(self.stage)
            .module(self.module);
        info.p_name = CString::new("main").unwrap().into_raw(); // leak memory
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
        unsafe { self.device.destroy_shader_module(self.module, None) };
    }
}

pub struct Device(Arc<ash::Device>, Arc<Instance>);

impl Device {
    pub fn new(
        instance: Arc<Instance>,
        physical_device: ash::vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
    ) -> Self {
        let device = unsafe {
            instance
                .create_device(physical_device, create_info, None)
                .unwrap()
        };
        log_resource_created("Device", "");
        Self(Arc::new(device), instance)
    }
}

impl std::ops::Deref for Device {
    type Target = ash::Device;
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

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Device")
    }
}

pub struct DebugUtilsMessenger {
    debug_utils: DebugUtils,
    inner: DebugUtilsMessengerEXT,
}

impl DebugUtilsMessenger {
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        messenger_info: &vk::DebugUtilsMessengerCreateInfoEXTBuilder,
    ) -> Self {
        let debug_utils = DebugUtils::new(&entry, &instance);
        let inner =
            unsafe { debug_utils.create_debug_utils_messenger(messenger_info, None) }.unwrap();
        log_resource_created("DebugUtilsMessenger", "");
        Self { debug_utils, inner }
    }
}

impl Drop for DebugUtilsMessenger {
    fn drop(&mut self) {
        log_resource_dropped("DebugUtilsMessenger", "");
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.inner, None);
        }
    }
}

pub struct Instance {
    instance: Arc<ash::Instance>,
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance").finish()
    }
}

impl Instance {
    pub fn new(entry: &ash::Entry, create_info: &vk::InstanceCreateInfo) -> Self {
        let instance = unsafe { entry.create_instance(&create_info, None) }.unwrap();
        let instance = Arc::new(instance);
        log_resource_created("Instance", "");
        Self { instance }
    }
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;
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
    pub scissor: Arc<vk::Rect2DBuilder<'a>>,
    pub color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    pub shader_stages: &'a [&'a ShaderModule],
    pub input_assembly_state: InputAssemblyState,
    pub rasterization_state: RasterizationState,
    pub multisample_state: MultiSamplingState,
    pub layout: Arc<PipelineLayout>,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfo,
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
        let shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(**shader_module)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .flags(vk::PipelineShaderStageCreateFlags::empty())
            .name(&entry_name);
        let create_info = vk::ComputePipelineCreateInfo::builder()
            .layout(**layout)
            .flags(vk::PipelineCreateFlags::empty())
            .stage(*shader_stage_create_info);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::default(), &[*create_info], None)
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
        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
    }
}

#[derive(Debug)]
pub struct Pipeline {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
    name: String,
    layout: Arc<PipelineLayout>,
    vertex_type: std::any::TypeId,
    vertex_name: &'static str,
}

impl Pipeline {
    pub fn new<V: Vertex + 'static>(
        device: Arc<Device>,
        pass: vk::RenderPass,
        desc: &PipelineDesc,
        name: &dyn ToString,
    ) -> Self {
        fn new(
            device: &Device,
            pass: vk::RenderPass,
            desc: &PipelineDesc,
            vertex_description: VertexInfoDescription,
        ) -> vk::Pipeline {
            let viewports = &[desc.view_port];
            let scissors = &[**desc.scissor];
            let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
                .viewports(viewports)
                .scissors(scissors);
            let attachments = &[desc.color_blend_attachment];
            let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .logic_op(vk::LogicOp::COPY)
                .attachments(attachments);
            let depth_stencil_state = desc.depth_stencil;
            let multi_sample_state = desc.multisample_state.builder();
            let rasterization_state = desc.rasterization_state.builder();
            let vertex_info = vertex_description.builder();
            let input_assembly_state = desc.input_assembly_state.builder();
            let shader_stages = desc
                .shader_stages
                .iter()
                .copied()
                .map(ShaderModule::builder)
                .collect::<Vec<_>>();
            let shader_stages = shader_stages
                .iter()
                .map(std::ops::Deref::deref)
                .copied()
                .collect::<Vec<_>>();
            let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
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

            unsafe {
                device.create_graphics_pipelines(
                    vk::PipelineCache::default(),
                    &[*pipeline_info],
                    None,
                )
            }
            .unwrap()[0]
        }
        let vertex_description = V::description();
        let name = name.to_string();
        let vertex_type = std::any::TypeId::of::<V>();
        let vertex_name = std::any::type_name::<V>();
        log_resource_created("Pipeline", &name);
        Self {
            layout: Arc::clone(&desc.layout),
            pipeline: new(&device, pass, desc, vertex_description),
            device,
            name,
            vertex_type,
            vertex_name,
        }
    }
    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub const fn vertex_type(&self) -> TypeId {
        self.vertex_type
    }
    pub const fn vertex_name(&self) -> &'static str {
        self.vertex_name
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
            self.device.destroy_pipeline(self.pipeline, None);
        }
    }
}

#[derive(Debug)]
pub struct PipelineLayout {
    inner: vk::PipelineLayout,
    device: Arc<Device>,
    name: String,
    set_layouts: Vec<Arc<DescriptorSetLayout>>,
}

impl PipelineLayout {
    pub fn new(
        device: Arc<Device>,
        set_layouts: Vec<Arc<DescriptorSetLayout>>,
        _push_constants: (),
        name: &dyn ToString,
    ) -> Self {
        let name = name.to_string();
        let inner = {
            let set_layouts: Vec<vk::DescriptorSetLayout> =
                set_layouts.iter().map(|l| ***l).collect::<Vec<_>>();
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[]);
            log_resource_created("PipelineLayout", &name);
            unsafe { device.create_pipeline_layout(&info, None).unwrap() }
        };
        Self {
            inner,
            device,
            name,
            set_layouts,
        }
    }
    pub fn compatible_with(&self, set_layouts: &[DescriptorSetLayout]) -> bool {
        if set_layouts.len() < self.set_layouts.len() {
            return false;
        }
        for (a, b) in self.set_layouts.iter().zip(set_layouts) {
            let a = a.bindings();
            let b = b.bindings();
            if a.len() > b.len() {
                return false;
            }
            for (a, b) in a.iter().zip(b) {
                if a.ty != b.ty {
                    return false;
                }
                if a.binding != b.binding {
                    return false;
                }
                if a.stage_flags != b.stage_flags {
                    return false;
                }
            }
        }
        true
    }
}

impl std::ops::Deref for PipelineLayout {
    type Target = vk::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        log_resource_dropped("PipelineLayout", &self.name);
        unsafe { self.device.destroy_pipeline_layout(self.inner, None) }
    }
}

pub struct Surface {
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(
        entry: &ash::Entry,
        instance: Arc<Instance>,
        window: &sdl2::video::Window,
    ) -> Self {
        let surface_loader = ash::extensions::khr::Surface::new(&*entry, &*instance);
        log_resource_created("Surface", "");
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
            .unwrap()
        };
        Self {
            surface_loader,
            surface,
        }
    }
    pub fn loader(&self) -> &ash::extensions::khr::Surface {
        &self.surface_loader
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
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }
}

pub type Trash = Arc<dyn std::any::Any + Send + Sync>;

macro_rules! handle {
    ( $name:ident, $t:ty, $info:ty, $create:expr, $destroy:expr ) => {
        #[derive(Debug)]
        pub struct $name {
            inner: $t,
            device: Arc<Device>,
            name: String,
            trash: Option<Trash>,
        }

        impl $name {
            pub fn new(device: Arc<Device>, info: &$info, name: impl Into<String>) -> Self {
                let inner = $create(&device, info);
                let name = name.into();
                crate::utils::log_resource_created(stringify!($name), &name);
                Self {
                    inner,
                    device,
                    name,
                    trash: None,
                }
            }
            #[allow(dead_code)]
            pub fn attach_trash<T: 'static + Send + Sync>(&mut self, trash: T) {
                self.trash = Some(if let Some(old) = self.trash.take() {
                    Arc::new((old, trash))
                } else {
                    Arc::new(trash)
                })
            }
            #[allow(dead_code)]
            pub fn with_trash(mut self, trash: Trash) -> Self {
                self.attach_trash(trash);
                self
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
            self.device.destroy_command_pool(self.inner, None);
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
            self.device.destroy_fence(self.inner, None);
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

pub struct ImageView {
    inner: vk::ImageView,
    device: Arc<Device>,
    image: Option<Arc<crate::image::Allocated>>,
    name: String,
}

impl std::ops::Deref for ImageView {
    type Target = vk::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ImageView {
    pub fn new(
        device: Arc<Device>,
        info: &vk::ImageViewCreateInfoBuilder,
        image: Option<Arc<crate::image::Allocated>>,
        name: impl Into<String>,
    ) -> Self {
        let inner = unsafe { device.create_image_view(info, None).unwrap() };
        let name = name.into();
        crate::utils::log_resource_created("ImageView", &name);
        Self {
            inner,
            device,
            image,
            name,
        }
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            crate::utils::log_resource_dropped("ImageView", &self.name);
            self.device.destroy_image_view(self.inner, None);
        }
        drop(self.image.take());
    }
}

pub struct Swapchain {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    device: Arc<Device>,
    name: String,
    trash: Option<Trash>,
}
impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: Arc<Device>,
        info: &vk::SwapchainCreateInfoKHRBuilder,
        name: impl Into<String>,
    ) -> Self {
        let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(info, None).unwrap() };
        let name = name.into();
        crate::utils::log_resource_created(stringify!(Swapchain), &name);
        Self {
            swapchain_loader,
            swapchain,
            device,
            name,
            trash: None,
        }
    }
    pub fn loader(&self) -> &ash::extensions::khr::Swapchain {
        &self.swapchain_loader
    }
    #[allow(dead_code)]
    pub fn attach_trash<T: 'static + Send + Sync>(&mut self, trash: T) {
        self.trash = Some(if let Some(old) = self.trash.take() {
            Arc::new((old, trash))
        } else {
            Arc::new(trash)
        })
    }
    #[allow(dead_code)]
    pub fn with_trash(mut self, trash: Trash) -> Self {
        self.attach_trash(trash);
        self
    }
}
impl std::ops::Deref for Swapchain {
    type Target = vk::SwapchainKHR;
    fn deref(&self) -> &Self::Target {
        &self.swapchain
    }
}
impl Drop for Swapchain {
    fn drop(&mut self) {
        crate::utils::log_resource_dropped(stringify!(Swapchain), &self.name);
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

handle!(
    Framebuffer,
    vk::Framebuffer,
    vk::FramebufferCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_framebuffer(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_framebuffer(inner, None) }
);

handle!(
    RenderPass,
    vk::RenderPass,
    vk::RenderPassCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_render_pass(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_render_pass(inner, None) }
);

handle!(
    Semaphore,
    vk::Semaphore,
    vk::SemaphoreCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_semaphore(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_semaphore(inner, None) }
);

handle!(
    Sampler,
    vk::Sampler,
    vk::SamplerCreateInfoBuilder,
    |device: &Arc<Device>, info| unsafe { device.create_sampler(info, None).unwrap() },
    |device: &Arc<Device>, inner| unsafe { device.destroy_sampler(inner, None) }
);
