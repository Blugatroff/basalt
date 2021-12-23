use crate::handles::Device;
use crate::utils::{immediate_submit, round_to};
use crate::{handles::Allocator, AllocatedBuffer};
use crate::{GpuDataRenderable, TransferContext};
use erupt::vk;
use std::sync::Arc;

pub struct VertexInfoDescription {
    pub bindings: Vec<vk::VertexInputBindingDescriptionBuilder<'static>>,
    pub attributes: Vec<vk::VertexInputAttributeDescriptionBuilder<'static>>,
    #[allow(dead_code)]
    pub flags: vk::PipelineVertexInputStateCreateFlags,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct Vertex {
    pub position: cgmath::Vector3<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub uv: cgmath::Vector2<f32>,
}

impl Vertex {
    pub fn get_vertex_description() -> VertexInfoDescription {
        let bindings = vec![
            vk::VertexInputBindingDescriptionBuilder::new()
                .binding(0)
                .stride(std::mem::size_of::<Self>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX),
            vk::VertexInputBindingDescriptionBuilder::new()
                .binding(1)
                .stride(std::mem::size_of::<GpuDataRenderable>() as u32)
                .input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let attributes = vec![
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 3]>() as u32),
            vk::VertexInputAttributeDescriptionBuilder::new()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 6]>() as u32),
        ];
        let flags = vk::PipelineVertexInputStateCreateFlags::default();
        VertexInfoDescription {
            bindings,
            attributes,
            flags,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshBounds {
    pub max: cgmath::Vector3<f32>,
    pub min: cgmath::Vector3<f32>,
}

#[derive(Debug)]
pub struct Mesh {
    pub bounds: MeshBounds,
    pub vertex_start: u32,
    pub index_start: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub buffer: Arc<AllocatedBuffer>,
    vertex_type_size: usize,
}

impl Mesh {
    pub fn new(
        vertices: &[Vertex],
        indices: &[u32],
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        device: Arc<Device>,
        host_visible: bool,
    ) -> Self {
        let bounds = Self::calculate_bounds(vertices);
        Self::new_with_bounds(
            vertices,
            indices,
            bounds,
            allocator,
            transfer_context,
            device,
            host_visible,
        )
    }
    #[rustfmt::skip]
    fn calculate_bounds(vertices: &[Vertex]) -> MeshBounds {
        let mut bounds = MeshBounds {
            max: cgmath::Vector3::new(0.0, 0.0, 0.0),
            min: cgmath::Vector3::new(0.0, 0.0, 0.0),
        };
        for vertex in vertices {
            let p = &vertex.position;
            if p.x > bounds.max.x { bounds.max.x = p.x }
            if p.y > bounds.max.y { bounds.max.y = p.y }
            if p.z > bounds.max.z { bounds.max.z = p.z }
            if p.x < bounds.min.x { bounds.min.x = p.x }
            if p.y < bounds.min.y { bounds.min.y = p.y }
            if p.z < bounds.min.z { bounds.min.z = p.z }
        }
        bounds
    }
    pub fn new_into_buffer<V>(
        vertices: &[V],
        indices: &[u32],
        bounds: MeshBounds,
        buffer: Arc<AllocatedBuffer>,
        offset: usize,
    ) -> Option<(Self, usize)> {
        assert_eq!(offset % std::mem::size_of::<V>(), 0);
        if (buffer.size as usize)
            < offset
                + vertices.len() * std::mem::size_of::<V>()
                + indices.len() * std::mem::size_of::<u32>()
        {
            return None;
        }
        let ptr = unsafe { buffer.map().add(offset) };
        unsafe {
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), ptr as *mut V, vertices.len());
        }
        unsafe {
            let ptr = ptr.add(vertices.len() * std::mem::size_of::<V>()) as *mut u32;
            std::ptr::copy_nonoverlapping(indices.as_ptr(), ptr, indices.len());
        }
        buffer.unmap();

        Some((
            Self {
                bounds,
                vertex_start: (offset / std::mem::size_of::<V>()) as u32,
                index_start: ((offset + std::mem::size_of::<V>() * vertices.len())
                    / std::mem::size_of::<u32>()) as u32,
                vertex_count: vertices.len() as u32,
                index_count: indices.len() as u32,
                buffer,
                vertex_type_size: std::mem::size_of::<V>(),
            },
            vertices.len() * std::mem::size_of::<V>() + indices.len() * std::mem::size_of::<u32>(),
        ))
    }
    pub fn override_buffer(&mut self, buffer: Arc<AllocatedBuffer>) {
        self.buffer = buffer;
    }
    pub fn new_with_bounds<V>(
        vertices: &[V],
        indices: &[u32],
        bounds: MeshBounds,
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        device: Arc<Device>,
        host_visible: bool,
    ) -> Self {
        let size =
            std::mem::size_of::<V>() * vertices.len() + indices.len() * std::mem::size_of::<u32>();
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .size(size as u64);
        let buffer = Arc::new(AllocatedBuffer::new(
            allocator,
            *buffer_info,
            vk_mem_erupt::MemoryUsage::CpuToGpu,
            Default::default(),
            label!("MeshStagingBuffer"),
        ));
        let (mut mesh, _) = Self::new_into_buffer(vertices, indices, bounds, buffer, 0).unwrap();
        if host_visible {
            return mesh;
        }
        mesh.buffer = Arc::new(mesh.buffer.copy_to_device_local(
            device,
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        ));
        mesh
    }
    pub fn combine_meshes<'a>(
        meshes: impl IntoIterator<Item = &'a mut Self>,
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        device: Arc<Device>,
    ) {
        let mut meshes = meshes.into_iter().collect::<Vec<&mut Self>>();
        if meshes.is_empty() {
            return;
        }
        let bptr = Arc::as_ptr(&meshes[0].buffer);
        let mut already_combined = true;
        for m in &meshes {
            if Arc::as_ptr(&m.buffer) != bptr {
                already_combined = false;
                break;
            }
        }
        if already_combined {
            return;
        }
        let staging_buffer_size = meshes.iter().fold(0, |size, m| {
            round_to(size, m.vertex_type_size as u64) + m.buffer.size
        });
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(staging_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let staging_buffer = Arc::new(AllocatedBuffer::new(
            allocator,
            *buffer_info,
            vk_mem_erupt::MemoryUsage::CpuToGpu,
            Default::default(),
            label!("CombineMeshesStagingBuffer"),
        ));
        let mut offset = 0;
        immediate_submit(&device, transfer_context, |cmd| unsafe {
            for mesh in meshes.iter_mut() {
                offset = round_to(offset, mesh.vertex_type_size as u64);
                let region = vk::BufferCopyBuilder::new()
                    .src_offset(0)
                    .dst_offset(offset)
                    .size(mesh.buffer.size);

                let regions = &[region];
                device.cmd_copy_buffer(cmd, **mesh.buffer, **staging_buffer, regions);
                mesh.index_start += offset as u32 / std::mem::size_of::<u32>() as u32;
                mesh.vertex_start += offset as u32 / (mesh.vertex_type_size as u32);
                offset += mesh.buffer.size;
            }
        });
        let device_local_buffer = Arc::new(staging_buffer.copy_to_device_local(
            device,
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC,
        ));
        for mesh in &mut meshes {
            mesh.buffer = Arc::clone(&device_local_buffer);
        }
    }
    pub fn load<P: AsRef<std::path::Path> + std::fmt::Debug>(
        allocator: Arc<Allocator>,
        path: P,
        transfer_context: &TransferContext,
        device: Arc<Device>,
    ) -> Vec<Self> {
        let (models, _) = tobj::load_obj(
            &path,
            &tobj::LoadOptions {
                single_index: true,
                triangulate: true,
                ignore_points: true,
                ignore_lines: true,
            },
        )
        .unwrap();
        let mut meshes = models
            .into_iter()
            .map(|model| {
                let positions = model
                    .mesh
                    .positions
                    .chunks_exact(3)
                    .map(|p| cgmath::Vector3::new(p[0], p[1], p[2]));
                let normals = model
                    .mesh
                    .normals
                    .chunks_exact(3)
                    .map(|p| cgmath::Vector3::new(p[0], p[1], p[2]));
                let uvs = model
                    .mesh
                    .texcoords
                    .chunks_exact(2)
                    .map(|p| cgmath::Vector2::new(p[0], p[1]));
                let vertices = positions
                    .zip(normals)
                    .zip(uvs)
                    .map(|((position, normal), mut uv)| {
                        uv.y = 1.0 - uv.y;
                        Vertex {
                            position,
                            normal,
                            uv,
                        }
                    })
                    .collect::<Vec<Vertex>>();
                let indices = model.mesh.indices;
                Self::new(
                    &vertices,
                    &indices,
                    allocator.clone(),
                    transfer_context,
                    device.clone(),
                    true,
                )
            })
            .collect::<Vec<_>>();
        Self::combine_meshes(meshes.iter_mut(), allocator, transfer_context, device);
        meshes
    }
}
