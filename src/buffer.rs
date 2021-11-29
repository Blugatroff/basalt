use crate::{handles::*, utils::immediate_submit, TransferContext};
use erupt::vk;
use std::sync::Arc;

#[derive(Debug)]
pub struct AllocatedBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem_erupt::Allocation,
    allocator: Arc<Allocator>,
    pub size: u64,
    usage: vk::BufferUsageFlags,
    name: &'static str,
}

impl AllocatedBuffer {
    pub fn new(
        allocator: Arc<Allocator>,
        buffer_info: vk::BufferCreateInfo,
        usage: vk_mem_erupt::MemoryUsage,
        required_memory_properties: vk::MemoryPropertyFlags,
        name: &'static str,
    ) -> Self {
        let vmaalloc_info = vk_mem_erupt::AllocationCreateInfo {
            usage,
            required_flags: required_memory_properties,
            ..Default::default()
        };
        assert!(buffer_info.size > 0);
        let (buffer, allocation, _) = allocator
            .create_buffer(&buffer_info, &vmaalloc_info)
            .unwrap();
        let size = buffer_info.size;
        let usage = buffer_info.usage;
        Self {
            allocator,
            buffer,
            allocation,
            size,
            name,
            usage,
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
        device: Arc<Device>,
        transfer_context: &TransferContext,
        usage: vk::BufferUsageFlags,
    ) -> AllocatedBuffer {
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(self.size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
        let device_buffer = Self::new(
            self.allocator.clone(),
            *buffer_info,
            vk_mem_erupt::MemoryUsage::GpuOnly,
            Default::default(),
            label!("MeshBuffer"),
        );

        immediate_submit(&device, transfer_context, |cmd| unsafe {
            device.cmd_copy_buffer(
                cmd,
                **self,
                *device_buffer,
                &[vk::BufferCopyBuilder::new()
                    .dst_offset(0)
                    .src_offset(0)
                    .size(self.size)],
            )
        });
        assert!(device_buffer.size >= self.size);
        device_buffer
    }
}

impl std::ops::Deref for AllocatedBuffer {
    type Target = vk::Buffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        println!(
            "DROPPED Buffer! {} {} {:?}",
            self.name, self.size, self.usage
        );
        self.allocator.destroy_buffer(self.buffer, &self.allocation);
    }
}
