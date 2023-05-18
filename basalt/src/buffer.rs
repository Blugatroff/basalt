use crate::{
    handles::Allocator,
    utils::{log_resource_created, log_resource_dropped},
    TransferContext,
};
use erupt::vk;
use std::sync::Arc;
use vk_mem_3_erupt as vma;

#[derive(Debug)]
pub struct Allocated {
    buffer: vk::Buffer,
    allocation: vma::Allocation,
    allocator: Arc<Allocator>,
    pub size: u64,
    usage: vk::BufferUsageFlags,
    name: &'static str,
}

impl Allocated {
    pub fn new(
        allocator: Arc<Allocator>,
        buffer_info: vk::BufferCreateInfo,
        usage: vma::MemoryUsage,
        required_memory_properties: vk::MemoryPropertyFlags,
        flags: vma::AllocationCreateFlags,
        name: &'static str,
    ) -> Self {
        let vmaalloc_info = vma::AllocationCreateInfo {
            usage,
            required_flags: required_memory_properties,
            flags,
            ..vma::AllocationCreateInfo::default()
        };
        assert!(buffer_info.size > 0);
        let (buffer, allocation, _) = allocator
            .create_buffer(&buffer_info, &vmaalloc_info)
            .unwrap();
        let size = buffer_info.size;
        let usage = buffer_info.usage;
        log_resource_created("Buffer", &format!("{} {} {:?}", name, size, usage));
        Self {
            buffer,
            allocation,
            allocator,
            size,
            usage,
            name,
        }
    }
    pub fn map(&self) -> *const u8 {
        self.allocator.map_memory(&self.allocation).unwrap()
    }
    pub fn unmap(&self) {
        self.allocator.unmap_memory(&self.allocation);
    }
    pub fn copy_to_device_local(
        &self,
        transfer_context: &TransferContext,
        usage: vk::BufferUsageFlags,
    ) -> Allocated {
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(self.size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
        let device_buffer = Self::new(
            self.allocator.clone(),
            *buffer_info,
            vma::MemoryUsage::AutoPreferDevice,
            erupt::vk1_0::MemoryPropertyFlags::DEVICE_LOCAL,
            vma::AllocationCreateFlags::default(),
            label!("MeshBuffer"),
        );
        let device = transfer_context.device.clone();
        transfer_context.immediate_submit(|cmd| unsafe {
            device.cmd_copy_buffer(
                cmd,
                **self,
                *device_buffer,
                &[vk::BufferCopyBuilder::new()
                    .dst_offset(0)
                    .src_offset(0)
                    .size(self.size)],
            );
        });
        assert!(device_buffer.size >= self.size);
        device_buffer
    }
}

impl std::ops::Deref for Allocated {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl Drop for Allocated {
    fn drop(&mut self) {
        log_resource_dropped(
            "Buffer",
            &format!("{} {} {:?}", self.name, self.size, self.usage),
        );
        self.allocator.destroy_buffer(self.buffer, &self.allocation);
    }
}
