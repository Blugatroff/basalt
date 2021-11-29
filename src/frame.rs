use crate::{
    buffer::AllocatedBuffer, descriptor_sets::DescriptorSet, handles::*, image::AllocatedImage,
};
use erupt::vk;
pub struct Frames {
    pub present_semaphores: Vec<Semaphore>,
    pub render_fences: Vec<Fence>,
    pub render_semaphores: Vec<Semaphore>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub depth_images: Vec<AllocatedImage>,
    pub depth_image_views: Vec<ImageView>,
    pub descriptor_sets: Vec<(DescriptorSet, DescriptorSet, DescriptorSet)>,
    pub renderables_buffers: Vec<AllocatedBuffer>,
    pub max_objects: Vec<usize>,
    pub indirect_buffers: Vec<AllocatedBuffer>,
    pub indirect_buffer_sets: Vec<DescriptorSet>,
    pub mesh_buffers: Vec<AllocatedBuffer>,
    pub mesh_sets: Vec<DescriptorSet>,
    pub cleanup: Vec<Option<Box<dyn FnOnce()>>>,
}

pub struct FrameData<'a> {
    pub present_semaphore: &'a Semaphore,
    pub render_fence: &'a Fence,
    pub render_semaphore: &'a Semaphore,
    pub command_buffer: &'a vk::CommandBuffer,
    pub depth_image: &'a AllocatedImage,
    pub depth_image_view: &'a ImageView,
    pub descriptor_sets: &'a mut (DescriptorSet, DescriptorSet, DescriptorSet),
    pub renderables_buffer: &'a mut AllocatedBuffer,
    pub max_objects: &'a mut usize,
    pub indirect_buffer: &'a mut AllocatedBuffer,
    pub indirect_buffer_set: &'a mut DescriptorSet,
    pub mesh_buffer: &'a mut AllocatedBuffer,
    pub mesh_set: &'a mut DescriptorSet,
    pub cleanup: &'a mut Option<Box<dyn FnOnce()>>,
}

impl Frames {
    pub fn get(&mut self, index: usize) -> FrameData {
        FrameData {
            present_semaphore: &self.present_semaphores[index],
            render_fence: &self.render_fences[index],
            render_semaphore: &self.render_semaphores[index],
            command_buffer: &self.command_buffers[index],
            depth_image: &self.depth_images[index],
            depth_image_view: &self.depth_image_views[index],
            descriptor_sets: &mut self.descriptor_sets[index],
            renderables_buffer: &mut self.renderables_buffers[index],
            max_objects: &mut self.max_objects[index],
            indirect_buffer: &mut self.indirect_buffers[index],
            indirect_buffer_set: &mut self.indirect_buffer_sets[index],
            mesh_buffer: &mut self.mesh_buffers[index],
            mesh_set: &mut self.mesh_sets[index],
            cleanup: &mut self.cleanup[index],
        }
    }
}
