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
struct Hole {
    /// in bytes
    offset: u64,
    /// in bytes
    size: u64,
}

struct Holes {
    holes: Vec<Hole>,
    end: u64,
}

impl Holes {
    fn new(offset: u64, size: u64) -> Self {
        Self {
            holes: vec![Hole { offset, size }],
            end: size,
        }
    }
    fn allocate(&mut self, size: u64) -> Option<u64> {
        if let Some((i, hole)) = self
            .holes
            .iter()
            .enumerate()
            .filter(|(_, h)| h.size >= size)
            .min_by_key(|(_, h)| h.size)
            .map(|(i, h)| (i, *h))
        {
            let offset = hole.offset;
            self.holes[i].size -= size;
            self.holes[i].offset += size;
            Some(offset)
        } else {
            None
        }
    }
    fn merge_holes(&mut self) {
        let mut to_remove = Vec::new();
        for i in 0..self.holes.len() {
            for j in 0..self.holes.len() {
                if i == j {
                    continue;
                }
                let hi = self.holes[i];
                let hj = self.holes[j];
                if hi.offset + hi.size == hj.offset {
                    self.holes[i].size += hj.size;
                    to_remove.push(j);
                } else if hj.offset + hj.size == hi.offset {
                    self.holes[j].size += hi.size;
                    to_remove.push(i);
                }
                if hi.size == 0 && hi.offset != self.end {
                    //to_remove.push(i);
                }
            }
        }
        to_remove.sort_unstable();
        for i in to_remove.into_iter().rev() {
            if self.holes.len() > i {
                self.holes.swap_remove(i);
            }
        }
    }
    fn expand(&mut self, size: u64) {
        let last_hole = self.holes.iter_mut().max_by_key(|h| h.offset).unwrap();
        if last_hole.offset + last_hole.size == self.end {
            last_hole.size = size - last_hole.offset;
        } else {
            let offset = last_hole.offset + last_hole.size;
            self.holes.push(Hole {
                offset,
                size: size - offset,
            })
        }
        self.end = size;
    }
    fn deallocate(&mut self, offset: u64, size: u64) {
        self.holes.push(Hole { offset, size });
        self.holes.sort_unstable_by_key(|h| h.offset);
        let mut last_end = 0;
        for hole in &self.holes {
            assert!(last_end <= hole.offset);
            last_end = hole.offset + hole.size;
        }
        self.merge_holes()
    }
}

#[test]
fn test_holes() {
    let mut holes = Holes::new(0, 120);
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(25), Some(0));
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(50), Some(25));
    holes.add_hole(Hole {
        offset: 0,
        size: 25,
    });
    assert_eq!(holes.holes.len(), 2);
    assert_eq!(holes.find(50), None);
    holes.expand(125);
    assert_eq!(holes.find(50), Some(75));
    assert_eq!(holes.holes.len(), 1);
    holes.add_hole(Hole {
        offset: 25,
        size: 50,
    });
    assert_eq!(holes.holes.len(), 1);
    holes.add_hole(Hole {
        offset: 75,
        size: 50,
    });
    assert_eq!(holes.holes.len(), 1);
    assert_eq!(holes.find(125), Some(0));
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
    fn calculate_bounds(vertices: &[Vertex]) -> MeshBounds {
        let mut bounds = MeshBounds {
            max: cgmath::Vector3::new(0.0, 0.0, 0.0),
            min: cgmath::Vector3::new(0.0, 0.0, 0.0),
        };
        #[rustfmt::skip]
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
            allocator.clone(),
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
            device.clone(),
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        ));
        mesh
    }
    pub fn combine_meshes(
        meshes: &mut Vec<Self>,
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        device: Arc<Device>,
    ) {
        let staging_buffer_size = meshes.iter().fold(0, |size, m| {
            round_to(size, m.vertex_type_size as u64) + m.buffer.size
        });
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(staging_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let staging_buffer = Arc::new(AllocatedBuffer::new(
            allocator.clone(),
            *buffer_info,
            vk_mem_erupt::MemoryUsage::CpuToGpu,
            Default::default(),
            label!("CombineMeshesStagingBuffer"),
        ));
        let mut offset = 0;
        println!("HALLO");
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
        println!("HALLO");
        let device_local_buffer = Arc::new(staging_buffer.copy_to_device_local(
            device.clone(),
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC,
        ));
        println!("HALLO");

        for mesh in meshes {
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
        Self::combine_meshes(&mut meshes, allocator.clone(), transfer_context, device);
        meshes
    }
}
