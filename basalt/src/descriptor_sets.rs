use std::{
    collections::BTreeMap,
    marker::PhantomData,
    sync::{Arc, Mutex, RwLock},
};

use crate::{
    handles::{Device, ShaderModule},
    utils::{log_resource_created, log_resource_dropped},
};
use ash::vk;

#[derive(Debug)]
pub struct DescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
    bindings: Vec<DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayout {
    pub fn bindings(&self) -> &[DescriptorSetLayoutBinding] {
        &self.bindings
    }
}

impl std::ops::Deref for DescriptorSetLayout {
    type Target = vk::DescriptorSetLayout;
    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

#[derive(Debug, Clone)]
pub struct DescriptorSetLayoutBinding {
    pub binding: u32,
    pub count: u32,
    pub ty: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub immutable_samplers: Option<Vec<vk::Sampler>>,
}

impl DescriptorSetLayoutBinding {
    fn from_reflection(desc: &reflection::Descriptor, stage_flags: vk::ShaderStageFlags) -> Self {
        let ty = match desc.ty {
            reflection::DescriptorType::Uniform(_) => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            reflection::DescriptorType::Storage(_) => vk::DescriptorType::STORAGE_BUFFER,
            reflection::DescriptorType::Sampler(_) => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        };
        Self {
            binding: desc.binding,
            count: 1,
            ty,
            stage_flags,
            immutable_samplers: None,
        }
    }
}

impl<'a> From<&'a DescriptorSetLayoutBinding> for vk::DescriptorSetLayoutBindingBuilder<'a> {
    fn from(val: &'a DescriptorSetLayoutBinding) -> Self {
        let mut builder = vk::DescriptorSetLayoutBinding::builder()
            .binding(val.binding)
            .descriptor_type(val.ty)
            .stage_flags(val.stage_flags)
            .descriptor_count(val.count);
        if let Some(samplers) = val.immutable_samplers.as_ref() {
            builder = builder.immutable_samplers(samplers);
        }
        builder
    }
}

impl DescriptorSetLayout {
    pub fn from_shader(device: &Arc<Device>, shader: &ShaderModule) -> BTreeMap<u32, Self> {
        let reflection = shader.reflection();
        let mut sets: BTreeMap<u32, Vec<DescriptorSetLayoutBinding>> = BTreeMap::new();
        for descriptor in &reflection.descriptors {
            let set = descriptor.set;
            let descriptor =
                DescriptorSetLayoutBinding::from_reflection(descriptor, shader.stage());
            sets.entry(set).or_insert_with(Vec::new).push(descriptor);
        }
        sets.into_iter()
            .map(|(k, v)| (k, Self::new(device.clone(), v, None)))
            .collect()
    }
    pub fn new(
        device: Arc<Device>,
        bindings: Vec<DescriptorSetLayoutBinding>,
        layout_binding_flags: Option<&[vk::DescriptorBindingFlags]>,
    ) -> Self {
        let binding_builders = bindings
            .iter()
            .map(|d| Into::<vk::DescriptorSetLayoutBindingBuilder<'_>>::into(d).build())
            .collect::<Vec<_>>();
        let layout_create_flags = vk::DescriptorSetLayoutCreateFlags::empty();
        let mut create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&binding_builders)
            .flags(layout_create_flags);

        let layout_binding_flags = layout_binding_flags.map(|layout_binding_flags| {
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(layout_binding_flags)
                .build()
        });
        create_info.p_next = layout_binding_flags
            .as_ref()
            .map(|p| p as *const vk::DescriptorSetLayoutBindingFlagsCreateInfo)
            .unwrap_or(std::ptr::null::<
                vk::DescriptorSetLayoutBindingFlagsCreateInfo,
            >()) as *const std::ffi::c_void;
        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();
        Self {
            layout,
            device,
            bindings,
        }
    }
}

#[derive(Debug)]
struct DescriptorPoolInner {
    pool: vk::DescriptorPool,
    device: Arc<Device>,
    used: u32,
    active: u32,
    max_sets: u32,
    pool_sizes: Vec<(vk::DescriptorType, u32)>,
    starting_pool_sizes: Vec<(vk::DescriptorType, u32)>,
    phantom: PhantomData<*const ()>,
}
unsafe impl Send for DescriptorPoolInner {}

impl DescriptorPoolInner {
    fn new(device: Arc<Device>, max_sets: u32, pool_sizes: Vec<(vk::DescriptorType, u32)>) -> Self {
        let mut sizes: Vec<(vk::DescriptorType, u32)> = Vec::new();
        for (ty, count) in pool_sizes {
            if let Some((_, c)) = sizes.iter_mut().find(|(t, _)| *t == ty) {
                *c += count;
            } else {
                sizes.push((ty, count));
            }
        }
        let pool_size_builders = sizes
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count,
            })
            .collect::<Vec<_>>();
        let info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_size_builders)
            .max_sets(max_sets);
        let pool_sizes: Vec<(vk::DescriptorType, u32)> =
            sizes.into_iter().map(|(ty, count)| (ty, count)).collect();
        let starting_pool_sizes = pool_sizes.iter().map(|(ty, count)| (*ty, *count)).collect();
        log_resource_created("DescriptorPoolInner", "");
        DescriptorPoolInner {
            pool: unsafe { device.create_descriptor_pool(&info, None) }.unwrap(),
            device,
            used: 0,
            active: 0,
            starting_pool_sizes,
            pool_sizes,
            max_sets,
            phantom: Default::default(),
        }
    }
    pub fn free_set(&mut self) {
        self.active -= 1;
        if self.active == 0 {
            self.reset();
        }
    }
    pub fn free_slots_remaining(&self) -> u32 {
        self.max_sets - self.used
    }
    pub fn attempt_to_insert(&mut self, bindings: &[DescriptorSetLayoutBinding]) -> bool {
        if self.free_slots_remaining() == 0 {
            return false;
        }
        let mut unique_bindings: Vec<(vk::DescriptorType, u32)> = self.pool_sizes.clone();
        'outer: for binding in bindings {
            if let Some((_, free)) = unique_bindings.iter_mut().find(|(ty, _)| *ty == binding.ty) {
                if *free >= binding.count {
                    *free -= binding.count;
                    continue 'outer;
                }
                return false;
            } else {
                return false;
            }
        }
        self.pool_sizes = unique_bindings;
        true
    }
    pub fn reset(&mut self) {
        self.used = 0;
        self.active = 0;
        self.pool_sizes = self.starting_pool_sizes.clone();
        unsafe {
            self.device
                .reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())
        }
        .unwrap();
    }
    pub fn allocate_set(&mut self, layout: &DescriptorSetLayout) -> Option<vk::DescriptorSet> {
        if !self.attempt_to_insert(&layout.bindings) {
            return None;
        }
        let set_layouts = &[**layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.pool)
            .set_layouts(set_layouts);

        self.active += 1;
        self.used += 1;
        log_resource_created("DescriptorSet", "");
        Some(unsafe { self.device.allocate_descriptor_sets(&alloc_info) }.unwrap()[0])
    }
}

#[derive(Debug, Clone)]
pub struct DescriptorPool {
    inner: Arc<Mutex<DescriptorPoolInner>>,
}
impl Drop for DescriptorPoolInner {
    fn drop(&mut self) {
        unsafe {
            log_resource_dropped("DescriptorPoolInner", "");
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}
impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        max_sets: u32,
        pool_sizes: Vec<(vk::DescriptorType, u32)>,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(DescriptorPoolInner::new(
                device, max_sets, pool_sizes,
            ))),
        }
    }

    pub fn allocate_set(
        &self,
        layout: &DescriptorSetLayout,
        name: String,
    ) -> Option<DescriptorSet> {
        let mut inner = self.inner.lock().unwrap();
        let set = inner.allocate_set(layout);
        drop(inner);
        set.map(|set| DescriptorSet::new(set, self.inner.clone(), name))
    }
}

pub struct DescriptorSetManager {
    pools: RwLock<Vec<DescriptorPool>>,
    device: Arc<Device>,
}

impl DescriptorSetManager {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            pools: RwLock::new(Vec::new()),
        }
    }
    pub fn allocate(&self, set_layout: &DescriptorSetLayout, name: &str) -> DescriptorSet {
        for pool in self.pools.read().unwrap().iter() {
            if let Some(set) = DescriptorPool::allocate_set(pool, set_layout, String::from(name)) {
                return set;
            }
        }
        self.pools.write().unwrap().push(DescriptorPool::new(
            self.device.clone(),
            8,
            set_layout
                .bindings
                .iter()
                .map(|b| (b.ty, b.count * 8))
                .collect::<Vec<_>>(),
        ));
        self.allocate(set_layout, name)
    }
}

pub struct DescriptorSet {
    name: String,
    set: vk::DescriptorSet,
    pool: Arc<Mutex<DescriptorPoolInner>>,
    resources: Vec<Box<dyn std::any::Any>>,
    phantom: PhantomData<*const ()>,
}
unsafe impl Send for DescriptorSet {}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSet")
            .field("set", &self.set)
            .field("pool", &self.pool)
            .finish()
    }
}

impl DescriptorSet {
    pub fn attach_resources(&mut self, r: Box<dyn std::any::Any>) {
        self.resources.push(r);
    }

    fn new(
        set: vk::DescriptorSet,
        pool: Arc<Mutex<DescriptorPoolInner>>,
        name: String,
    ) -> DescriptorSet {
        Self {
            name,
            set,
            pool,
            resources: Vec::new(),
            phantom: PhantomData,
        }
    }
}

impl std::ops::Deref for DescriptorSet {
    type Target = vk::DescriptorSet;
    fn deref(&self) -> &Self::Target {
        &self.set
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        log_resource_dropped("DescriptorSet", &self.name);
        self.pool.lock().unwrap().free_set();
    }
}
