use crate::descriptor_sets::DescriptorSetLayout;
use crate::utils::{log_resource_created, log_resource_dropped};
use crate::{buffer, descriptor_sets};
use crate::{
    descriptor_sets::{DescriptorSet, DescriptorSetManager},
    handles::{Allocator, Device, ImageView, Sampler},
    TransferContext,
};
use ash::vk;
use std::sync::Arc;
use vma::Alloc;

pub struct Allocated {
    image: vk::Image,
    allocation: vma::Allocation,
    allocator: Arc<Allocator>,
    name: String,
    width: u32,
    height: u32,
}

impl std::ops::Deref for Allocated {
    type Target = vk::Image;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[derive(Clone)]
pub struct Loader {
    pub device: Arc<Device>,
    pub transfer_context: Arc<TransferContext>,
    pub allocator: Arc<Allocator>,
}

impl Allocated {
    pub fn image_create_info(
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
    ) -> vk::ImageCreateInfoBuilder<'static> {
        vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
    }
    pub fn image_view_create_info<'a>(
        &self,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageViewCreateInfoBuilder<'a> {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .aspect_mask(aspect);
        vk::ImageViewCreateInfo::builder()
            .view_type(vk::ImageViewType::TYPE_2D)
            .image(**self)
            .format(format)
            .subresource_range(*subresource_range)
    }
    pub fn new(
        allocator: Arc<Allocator>,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
        name: String,
    ) -> Self {
        let info = Self::image_create_info(format, usage, extent, tiling);
        let allocation_info = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::AutoPreferDevice,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..vma::AllocationCreateInfo::default()
        };
        let (image, allocation) =
            unsafe { allocator.create_image(&info, &allocation_info) }.unwrap();
        let width = extent.width;
        let height = extent.height;
        log_resource_created("AllocatedImage", &format!("{} {}x{}", name, width, height));
        Self {
            image,
            allocation,
            allocator,
            name,
            width,
            height,
        }
    }
    pub fn open(image_loader: &Loader, path: &std::path::Path) -> Self {
        let image = image::open(path).unwrap();
        let image = image.into_rgba8();
        let width = image.width();
        let height = image.height();
        let image = image.into_raw();
        let name = format!("{:?}", path);
        Self::load(image_loader, &image, width, height, name)
    }
    pub fn load(image_loader: &Loader, data: &[u8], width: u32, height: u32, name: String) -> Self {
        let image_size = width * height * 4;
        let image_format = vk::Format::R8G8B8A8_SRGB;
        assert_eq!(data.len(), image_size as usize);
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(u64::from(image_size))
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = buffer::Allocated::new(
            image_loader.allocator.clone(),
            *buffer_info,
            vma::MemoryUsage::AutoPreferHost,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            label!("ImageStagingBuffer"),
        );
        let ptr = staging_buffer.map() as *mut _;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, image_size as usize) };
        staging_buffer.unmap();

        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image = Self::new(
            image_loader.allocator.clone(),
            image_format,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            image_extent,
            vk::ImageTiling::LINEAR,
            name,
        );

        image_loader
            .transfer_context
            .immediate_submit(|cmd| unsafe {
                let range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let image_barrier_transfer = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(*image)
                    .subresource_range(*range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                image_loader.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier_transfer],
                );
                let image_subresource = vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1);
                let copy_region = vk::BufferImageCopy::builder()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(*image_subresource)
                    .image_extent(image_extent);
                image_loader.device.cmd_copy_buffer_to_image(
                    cmd,
                    *staging_buffer,
                    *image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[*copy_region],
                );

                let image_barrier_to_read_optimal = image_barrier_transfer
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                image_loader.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*image_barrier_to_read_optimal],
                );

                let ownership_transfer_barrier = vk::ImageMemoryBarrier::builder()
                    .image(*image)
                    .subresource_range(*range)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(image_loader.transfer_context.transfer_family.0)
                    .dst_queue_family_index(image_loader.transfer_context.graphics_family.0);

                image_loader.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[*ownership_transfer_barrier],
                );
            });
        image
    }
}

impl Drop for Allocated {
    fn drop(&mut self) {
        log_resource_dropped(
            "AllocatedImage",
            &format!("{} {}x{}", self.name, self.width, self.height),
        );
        unsafe { self.allocator.destroy_image(self.image, &mut self.allocation); }
    }
}

pub struct Texture {
    pub view: Arc<ImageView>,
    pub set: Arc<DescriptorSet>,
    pub sampler: Arc<Sampler>,
}

impl Texture {
    pub fn from_image_view(
        device: &Arc<Device>,
        descriptor_set_manager: &DescriptorSetManager,
        view: Arc<ImageView>,
        sampler: Arc<Sampler>,
    ) -> Self {
        let texture_set_layout = DescriptorSetLayout::new(
            device.clone(),
            vec![descriptor_sets::DescriptorSetLayoutBinding {
                binding: 0,
                count: 1,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                immutable_samplers: None,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
            }],
            None,
        );
        let mut set = descriptor_set_manager.allocate(&texture_set_layout);
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(**view)
            .sampler(**sampler);
        let image_info = &[*image_info];
        let image_write = vk::WriteDescriptorSet::builder()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(0)
            .dst_set(*set)
            .image_info(image_info);
        unsafe { device.update_descriptor_sets(&[*image_write], &[]) };
        set.attach_resources(Box::new(Arc::clone(&view)));
        set.attach_resources(Box::new(Arc::clone(&sampler)));
        let set = Arc::new(set);
        Self { view, set, sampler }
    }
    pub fn new(
        device: &Arc<Device>,
        descriptor_set_manager: &DescriptorSetManager,
        image: Allocated,
        sampler: Arc<Sampler>,
    ) -> Self {
        let image = Arc::new(image);
        let image_view_create_info =
            image.image_view_create_info(vk::Format::R8G8B8A8_SRGB, vk::ImageAspectFlags::COLOR);
        let view = Arc::new(ImageView::new(
            device.clone(),
            &image_view_create_info,
            Some(image.clone()),
            label!("TextureImageView"),
        ));
        Self::from_image_view(device, descriptor_set_manager, view, sampler)
    }
}
