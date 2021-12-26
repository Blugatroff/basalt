use std::{
    ffi::c_void,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use crate::handles::Device;
use erupt::vk;

#[derive(Debug)]
pub struct DescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
    bindings: Vec<DescriptorSetLayoutBinding>,
}

impl<'a> std::ops::Deref for DescriptorSetLayout {
    type Target = vk::DescriptorSetLayout;
    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl<'a> Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(Some(self.layout), None);
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

impl<'a> From<&'a DescriptorSetLayoutBinding> for vk::DescriptorSetLayoutBindingBuilder<'a> {
    fn from(val: &'a DescriptorSetLayoutBinding) -> Self {
        let mut builder = vk::DescriptorSetLayoutBindingBuilder::new()
            .binding(val.binding)
            .descriptor_type(val.ty)
            .stage_flags(val.stage_flags)
            .descriptor_count(dbg!(val.count));
        if let Some(samplers) = val.immutable_samplers.as_ref() {
            builder = builder.immutable_samplers(samplers);
        }
        builder
    }
}

impl DescriptorSetLayout {
    pub fn new(
        device: Arc<Device>,
        bindings: Vec<DescriptorSetLayoutBinding>,
        layout_binding_flags: Option<&[vk::DescriptorBindingFlags]>,
    ) -> Self {
        let binding_builders = bindings
            .iter()
            .map(Into::<vk::DescriptorSetLayoutBindingBuilder<'_>>::into)
            .collect::<Vec<_>>();
        let layout_create_flags = vk::DescriptorSetLayoutCreateFlags::empty();
        let mut create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
            .bindings(&binding_builders)
            .flags(layout_create_flags);
        if let Some(layout_binding_flags) = layout_binding_flags {
            let layout_binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
                .binding_flags(layout_binding_flags)
                .build();
            create_info.p_next = (&layout_binding_flags
                as *const vk::DescriptorSetLayoutBindingFlagsCreateInfo)
                .cast();
        }
        let layout = unsafe { device.create_descriptor_set_layout(&create_info, None) }.unwrap();
        Self {
            layout,
            device,
            bindings,
        }
    }
}

#[derive(Debug)]
pub struct DescriptorPool {
    pool: vk::DescriptorPool,
    device: Arc<Device>,
    used: AtomicU32,
    active: AtomicU32,
    max_sets: u32,
    pool_sizes: Vec<(vk::DescriptorType, AtomicU32)>,
    starting_pool_sizes: Vec<(vk::DescriptorType, AtomicU32)>,
}

impl std::ops::Deref for DescriptorPool {
    type Target = vk::DescriptorPool;
    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.destroy_descriptor_pool(Some(self.pool), None) }
    }
}
impl DescriptorPool {
    pub fn new(
        device: Arc<Device>,
        max_sets: u32,
        pool_sizes: Vec<(vk::DescriptorType, u32)>,
    ) -> Self {
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
            .map(|(ty, count)| {
                vk::DescriptorPoolSize {
                    _type: *ty,
                    descriptor_count: *count,
                }
                .into_builder()
            })
            .collect::<Vec<_>>();
        let info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&pool_size_builders)
            .max_sets(max_sets);
        let pool_sizes: Vec<(vk::DescriptorType, AtomicU32)> = sizes
            .into_iter()
            .map(|(ty, count)| (ty, AtomicU32::new(count)))
            .collect();
        let starting_pool_sizes = pool_sizes
            .iter()
            .map(|(ty, count)| (*ty, AtomicU32::new(count.load(Ordering::SeqCst))))
            .collect();
        Self {
            pool: unsafe { device.create_descriptor_pool(&info, None) }.unwrap(),
            device,
            used: AtomicU32::new(0),
            active: AtomicU32::new(0),
            starting_pool_sizes,
            pool_sizes,
            max_sets,
        }
    }
    pub fn free_set(&self) {
        let active = self.active.fetch_sub(1, Ordering::SeqCst) - 1;
        if active == 0 {
            self.reset();
        }
    }
    pub fn free_slots_remaining(&self) -> u32 {
        self.max_sets - self.used.load(Ordering::SeqCst)
    }
    pub fn check_for_space(&self, bindings: &[DescriptorSetLayoutBinding]) -> bool {
        if self.free_slots_remaining() == 0 {
            return false;
        }
        for binding in bindings {
            if self
                .pool_sizes
                .iter()
                .any(|(ty, free)| binding.ty == *ty && free.load(Ordering::SeqCst) >= binding.count)
            {
                return true;
            }
        }
        false
    }
    pub fn allocate_set(&self, bindings: &[DescriptorSetLayoutBinding]) {
        assert!(self.check_for_space(bindings));
        for binding in bindings {
            if let Some((_, free)) = self.pool_sizes.iter().find(|(ty, _)| *ty == binding.ty) {
                free.fetch_sub(binding.count, Ordering::SeqCst);
            }
        }
        self.active.fetch_add(1, Ordering::SeqCst);
        self.used.fetch_add(1, Ordering::SeqCst);
    }
    pub fn reset(&self) {
        self.used.store(0, Ordering::SeqCst);
        self.active.store(0, Ordering::SeqCst);
        for ((_, count), (_, starting_count)) in
            self.pool_sizes.iter().zip(&self.starting_pool_sizes)
        {
            count.store(starting_count.load(Ordering::SeqCst), Ordering::SeqCst);
        }
        unsafe { self.device.reset_descriptor_pool(self.pool, None) }.unwrap();
    }
}

pub struct DescriptorSetManager {
    pools: Vec<Arc<DescriptorPool>>,
    device: Arc<Device>,
}

impl DescriptorSetManager {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            pools: Vec::new(),
        }
    }
    pub fn allocate(
        &mut self,
        set_layout: &DescriptorSetLayout,
        variable_info: Option<&vk::DescriptorSetVariableDescriptorCountAllocateInfo>,
    ) -> DescriptorSet {
        dbg!(self.pools.len());
        for pool in &self.pools {
            if pool.check_for_space(&set_layout.bindings) {
                return DescriptorSet::allocate(
                    &self.device,
                    set_layout,
                    pool.clone(),
                    variable_info,
                );
            }
        }
        self.pools.push(Arc::new(DescriptorPool::new(
            self.device.clone(),
            8,
            set_layout
                .bindings
                .iter()
                .map(|b| (b.ty, b.count * 8))
                .collect::<Vec<_>>(),
        )));
        self.allocate(set_layout, variable_info)
    }
}

impl std::fmt::Debug for DescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DescriptorSet")
            .field("set", &self.set)
            .field("pool", &self.pool)
            .finish()
    }
}

pub struct DescriptorSet {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    resources: Vec<Box<dyn std::any::Any>>,
}

impl DescriptorSet {
    pub fn allocate(
        device: &Arc<Device>,
        set_layout: &DescriptorSetLayout,
        pool: Arc<DescriptorPool>,
        variable_info: Option<&vk::DescriptorSetVariableDescriptorCountAllocateInfo>,
    ) -> DescriptorSet {
        let set_layouts = &[**set_layout];
        let mut alloc_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(**pool)
            .set_layouts(set_layouts);
        if let Some(variable_info) = variable_info {
            alloc_info.p_next = variable_info as *const _ as *const c_void;
        }
        let set = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap()[0];
        pool.allocate_set(&set_layout.bindings);
        let resources = Vec::new();
        DescriptorSet {
            set,
            pool,
            resources,
        }
    }
    pub fn attach_resources(&mut self, r: Box<dyn std::any::Any>) {
        self.resources.push(r);
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
        self.pool.free_set();
    }
}
