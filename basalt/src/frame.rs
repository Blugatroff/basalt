use std::sync::Arc;

use crate::{
    buffer,
    descriptor_sets::DescriptorSet,
    handles::{Fence, ImageView, Semaphore},
    image,
};
use erupt::vk;

pub struct Frames {
    pub present_semaphores: Vec<Semaphore>,
    pub render_fences: Vec<Fence>,
    pub render_semaphores: Vec<Semaphore>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub depth_images: Vec<image::Allocated>,
    pub depth_image_views: Vec<ImageView>,
    pub descriptor_sets: Vec<DescriptorSet>,
    pub renderables_buffers: Vec<buffer::Allocated>,
    pub max_objects: Vec<usize>,
    pub mesh_buffers: Vec<Arc<buffer::Allocated>>,
    pub cull_sets: Vec<DescriptorSet>,
    pub cleanup: Vec<Option<Box<dyn FnOnce()>>>,
    pub indirect_buffers: Vec<Arc<buffer::Allocated>>,
}

#[allow(clippy::module_name_repetitions)]
pub struct FrameData<'a> {
    pub present_semaphore: &'a Semaphore,
    pub render_fence: &'a Fence,
    pub render_semaphore: &'a Semaphore,
    pub command_buffer: &'a vk::CommandBuffer,
    pub depth_image: &'a image::Allocated,
    pub depth_image_view: &'a ImageView,
    pub descriptor_set: &'a DescriptorSet,
    pub renderables_buffer: &'a buffer::Allocated,
    pub max_objects: &'a usize,
    pub mesh_buffer: &'a buffer::Allocated,
    pub cull_set: &'a DescriptorSet,
    pub cleanup: &'a Option<Box<dyn FnOnce()>>,
    pub indirect_buffer: &'a Arc<buffer::Allocated>,
}

impl<'a> FrameDataMut<'a> {
    pub fn immu(&self) -> FrameData {
        FrameData {
            present_semaphore: self.present_semaphore,
            render_fence: self.render_fence,
            render_semaphore: self.render_semaphore,
            command_buffer: self.command_buffer,
            depth_image: self.depth_image,
            depth_image_view: self.depth_image_view,
            descriptor_set: self.descriptor_set,
            renderables_buffer: self.renderables_buffer,
            max_objects: self.max_objects,
            mesh_buffer: self.mesh_buffer,
            cull_set: self.cull_set,
            cleanup: self.cleanup,
            indirect_buffer: self.indirect_buffer,
        }
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct FrameDataMut<'a> {
    pub present_semaphore: &'a Semaphore,
    pub render_fence: &'a Fence,
    pub render_semaphore: &'a Semaphore,
    pub command_buffer: &'a vk::CommandBuffer,
    pub depth_image: &'a image::Allocated,
    pub depth_image_view: &'a ImageView,
    pub descriptor_set: &'a mut DescriptorSet,
    pub renderables_buffer: &'a mut buffer::Allocated,
    pub max_objects: &'a mut usize,
    pub mesh_buffer: &'a mut Arc<buffer::Allocated>,
    pub cull_set: &'a mut DescriptorSet,
    pub cleanup: &'a mut Option<Box<dyn FnOnce()>>,
    pub indirect_buffer: &'a mut Arc<buffer::Allocated>,
}

impl Frames {
    pub fn get(&self, index: usize) -> FrameData {
        FrameData {
            present_semaphore: &self.present_semaphores[index],
            render_fence: &self.render_fences[index],
            render_semaphore: &self.render_semaphores[index],
            command_buffer: &self.command_buffers[index],
            depth_image: &self.depth_images[index],
            depth_image_view: &self.depth_image_views[index],
            descriptor_set: &self.descriptor_sets[index],
            renderables_buffer: &self.renderables_buffers[index],
            max_objects: &self.max_objects[index],
            mesh_buffer: &self.mesh_buffers[index],
            cull_set: &self.cull_sets[index],
            cleanup: &self.cleanup[index],
            indirect_buffer: &self.indirect_buffers[index],
        }
    }
    pub fn get_mut(&mut self, index: usize) -> FrameDataMut {
        FrameDataMut {
            present_semaphore: &self.present_semaphores[index],
            render_fence: &self.render_fences[index],
            render_semaphore: &self.render_semaphores[index],
            command_buffer: &self.command_buffers[index],
            depth_image: &self.depth_images[index],
            depth_image_view: &self.depth_image_views[index],
            descriptor_set: &mut self.descriptor_sets[index],
            renderables_buffer: &mut self.renderables_buffers[index],
            max_objects: &mut self.max_objects[index],
            mesh_buffer: &mut self.mesh_buffers[index],
            cull_set: &mut self.cull_sets[index],
            cleanup: &mut self.cleanup[index],
            indirect_buffer: &mut self.indirect_buffers[index],
        }
    }
}
