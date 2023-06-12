use crate::{
    handles::Allocator,
    utils::{log_resource_created, log_resource_dropped},
    TransferContext,
};
use ash::vk;
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex, MutexGuard},
};
use vma::Alloc;

pub struct Allocated {
    buffer: vk::Buffer,
    allocation: Mutex<vma::Allocation>,
    allocator: Arc<Allocator>,
    pub size: u64,
    usage: vk::BufferUsageFlags,
    name: &'static str,
}

pub struct Mapped<'buffer, T> {
    allocator: &'buffer Allocator,
    allocation: MutexGuard<'buffer, vma::Allocation>,
    data: &'buffer mut [T],
}

impl<'buffer, T> Deref for Mapped<'buffer, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'buffer, T> DerefMut for Mapped<'buffer, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<'buffer, T> Drop for Mapped<'buffer, T> {
    fn drop(&mut self) {
        unsafe { self.allocator.unmap_memory(&mut self.allocation) };
    }
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
        let (buffer, allocation) = unsafe {
            allocator
                .create_buffer(&buffer_info, &vmaalloc_info)
                .unwrap()
        };
        let allocation = Mutex::new(allocation);
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
    pub fn map<T: Copy>(&self, length: usize) -> Mapped<'_, T> {
        self.map_with_offset(0, length)
    }
    pub fn map_with_offset<T: Copy>(&self, byte_offset: usize, length: usize) -> Mapped<'_, T> {
        assert_ne!(std::mem::size_of::<T>(), 0);

        let mut allocation = self.allocation.lock().unwrap();
        let ptr = unsafe { self.allocator.map_memory(&mut allocation) }.unwrap();
        let ptr = unsafe { ptr.add(byte_offset) };
        let ptr = ptr as *mut T;
        let len = length.min((self.size as usize - byte_offset) / std::mem::size_of::<T>());
        let data = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        Mapped {
            allocator: &self.allocator,
            data,
            allocation,
        }
    }
    pub fn copy_to_device_local(
        &self,
        transfer_context: &TransferContext,
        usage: vk::BufferUsageFlags,
    ) -> Allocated {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(self.size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
        let device_buffer = Self::new(
            self.allocator.clone(),
            *buffer_info,
            vma::MemoryUsage::AutoPreferDevice,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vma::AllocationCreateFlags::empty(),
            label!("MeshBuffer"),
        );
        let device = transfer_context.device.clone();
        let region = vk::BufferCopy::builder()
            .dst_offset(0)
            .src_offset(0)
            .size(self.size);
        transfer_context.immediate_submit(|cmd| unsafe {
            device.cmd_copy_buffer(cmd, **self, *device_buffer, &[*region]);
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
        let allocation = &mut self.allocation.lock().unwrap();
        unsafe {
            self.allocator.destroy_buffer(self.buffer, allocation);
        }
    }
}
