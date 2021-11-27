use crate::{
    buffer::AllocatedBuffer,
    handles::{Allocator, Device, ImageView},
    utils::immediate_submit,
    TransferContext,
};
use erupt::vk::{self};
use std::{path::Path, sync::Arc};

#[derive(Debug)]
pub struct AllocatedImage {
    image: vk::Image,
    allocation: vk_mem_erupt::Allocation,
    allocator: Arc<Allocator>,
}

impl std::ops::Deref for AllocatedImage {
    type Target = vk::Image;
    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[derive(Clone)]
pub struct ImageLoader {
    pub device: Arc<Device>,
    pub transfer_context: Arc<TransferContext>,
    pub allocator: Arc<Allocator>,
}

impl AllocatedImage {
    pub fn image_create_info<'a>(
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
    ) -> vk::ImageCreateInfoBuilder<'a> {
        vk::ImageCreateInfoBuilder::new()
            .image_type(vk::ImageType::_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlagBits::_1)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage_flags)
    }
    pub fn image_view_create_info<'a>(
        format: vk::Format,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
    ) -> vk::ImageViewCreateInfoBuilder<'a> {
        let subresource_range = vk::ImageSubresourceRangeBuilder::new()
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .aspect_mask(aspect);
        vk::ImageViewCreateInfoBuilder::new()
            .view_type(vk::ImageViewType::_2D)
            .image(image)
            .format(format)
            .subresource_range(*subresource_range)
    }
    pub fn new(
        allocator: Arc<Allocator>,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        tiling: vk::ImageTiling,
    ) -> Self {
        let info = Self::image_create_info(format, usage, extent, tiling);
        let allocation_info = vk_mem_erupt::AllocationCreateInfo {
            usage: vk_mem_erupt::MemoryUsage::GpuOnly,
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        let (image, allocation, _) = allocator.create_image(&info, &allocation_info).unwrap();
        Self {
            image,
            allocator,
            allocation,
        }
    }
    pub fn open<P: AsRef<Path>>(image_loader: &ImageLoader, path: P) -> Self {
        dbg!("OPENING IMAGE");
        let image = image::open(path).unwrap();
        let image = image.into_rgba8();
        let width = image.width();
        let height = image.height();
        let image = image.into_raw();
        dbg!(width, height);
        dbg!(image.len());
        Self::load(image_loader, &image, width, height)
    }
    pub fn load(image_loader: &ImageLoader, data: &[u8], width: u32, height: u32) -> Self {
        let image_size = width * height * 4;
        let image_format = vk::Format::R8G8B8A8_SRGB;
        assert_eq!(data.len() as u32, image_size);
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(image_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let staging_buffer = AllocatedBuffer::new(
            image_loader.allocator.clone(),
            *buffer_info,
            vk_mem_erupt::MemoryUsage::CpuOnly,
            Default::default(),
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
        );

        immediate_submit(
            &image_loader.device,
            &image_loader.transfer_context,
            |cmd| unsafe {
                let range = vk::ImageSubresourceRangeBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let image_barrier_transfer = vk::ImageMemoryBarrierBuilder::new()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(*image)
                    .subresource_range(*range)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                image_loader.device.cmd_pipeline_barrier(
                    cmd,
                    Some(vk::PipelineStageFlags::TOP_OF_PIPE),
                    Some(vk::PipelineStageFlags::TRANSFER),
                    None,
                    &[],
                    &[],
                    &[image_barrier_transfer],
                );
                let image_subresource = vk::ImageSubresourceLayersBuilder::new()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1);
                let copy_region = vk::BufferImageCopyBuilder::new()
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
                    &[copy_region],
                );

                let image_barrier_to_read_optimal = image_barrier_transfer
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);

                image_loader.device.cmd_pipeline_barrier(
                    cmd,
                    Some(vk::PipelineStageFlags::TRANSFER),
                    Some(vk::PipelineStageFlags::COMPUTE_SHADER),
                    None,
                    &[],
                    &[],
                    &[image_barrier_to_read_optimal],
                );
            },
        );
        image
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        println!("DROPPED AllocatedImage!");
        self.allocator.destroy_image(self.image, &self.allocation);
    }
}

pub struct Texture {
    pub image: AllocatedImage,
    pub view: ImageView,
}

impl Texture {
    pub fn new(device: Arc<Device>, image: AllocatedImage) -> Self {
        let image_view_create_info = AllocatedImage::image_view_create_info(
            vk::Format::R8G8B8A8_SRGB,
            *image,
            vk::ImageAspectFlags::COLOR,
        );
        let view = ImageView::new(device.clone(), &image_view_create_info);
        /* let set = descriptor_set_manager.allocate(&texture_set_layout);
        let image_info = vk::DescriptorImageInfoBuilder::new()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(*view)
            .sampler(**sampler);
        let image_info = &[image_info];
        let image_write = vk::WriteDescriptorSetBuilder::new()
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_binding(0)
            .dst_set(*set)
            .image_info(image_info);
        unsafe { device.update_descriptor_sets(&[image_write], &[]) }; */
        Self { image, view }
    }
}
