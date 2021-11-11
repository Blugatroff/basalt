use crate::handles::Device;
use crate::utils::immediate_submit;
use crate::{handles::Allocator, AllocatedBuffer};
use crate::{GpuDataRenderable, TransferContext};
use erupt::vk;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

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

pub struct SubAllocatedBuffer {
    buffer: Arc<AllocatedBuffer>,
    size: u64,
    transfer_context: Arc<TransferContext>,
    device: Arc<Device>,
    allocator: Arc<Allocator>,
    holes: Arc<Mutex<Holes>>,
    usage: vk::BufferUsageFlags,
}

impl std::ops::Deref for SubAllocatedBuffer {
    type Target = Arc<AllocatedBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

pub struct BufferSubAllocation {
    offset: u64,
    pub aligned_offset: u64,
    /// self.offset / std::mem::size_of::<T>() as u64
    holes: Arc<Mutex<Holes>>,
    pub type_size: usize,
    size: u64,
}

impl std::fmt::Debug for BufferSubAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferSubAllocation")
            .field("offset", &self.offset)
            .field("aligned_offset", &self.aligned_offset)
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for BufferSubAllocation {
    fn drop(&mut self) {
        self.holes
            .lock()
            .unwrap()
            .deallocate(self.offset, self.size)
    }
}

impl SubAllocatedBuffer {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Allocator>,
        transfer_context: Arc<TransferContext>,
        start_size: u64,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let size = start_size;
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = Arc::new(AllocatedBuffer::new(
            allocator.clone(),
            *buffer_info,
            vk_mem_erupt::MemoryUsage::GpuOnly,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ));
        Self {
            device,
            buffer,
            allocator,
            transfer_context,
            size,
            holes: Arc::new(Mutex::new(Holes::new(0, size))),
            usage,
        }
    }
    pub fn resize(&mut self, new_size: u64) {
        let mut new = Self::new(
            self.device.clone(),
            self.allocator.clone(),
            self.transfer_context.clone(),
            new_size,
            self.usage,
        );
        immediate_submit(&self.device, &self.transfer_context, |cmd| unsafe {
            self.device.cmd_copy_buffer(
                cmd,
                **self.buffer,
                **new.buffer,
                &[vk::BufferCopyBuilder::new()
                    .dst_offset(0)
                    .src_offset(0)
                    .size(self.size)],
            );
        });
        self.holes.lock().unwrap().expand(self.buffer.size);
        new.holes = self.holes.clone();
        std::mem::swap(self, &mut new);
    }
    pub fn insert<T>(&mut self, data: &[T]) -> BufferSubAllocation {
        let size = (data.len() + 1) * std::mem::size_of::<T>();
        let size = size as u64;
        let alloc = self.holes.lock().unwrap().allocate(size);
        if let Some(offset) = alloc {
            let alignment_offset =
                std::mem::size_of::<T>() - offset as usize % std::mem::size_of::<T>();
            self.upload_data(offset + alignment_offset as u64, data);
            let aligned_offset = offset + alignment_offset as u64;
            assert_eq!(aligned_offset as usize % std::mem::size_of::<T>(), 0);
            BufferSubAllocation {
                offset,
                aligned_offset,
                holes: self.holes.clone(),
                size,
                type_size: std::mem::size_of::<T>(),
            }
        } else {
            self.resize((self.size + size) / 2 * 3);
            self.insert(data)
        }
    }
    pub fn upload_data<T>(&self, offset: u64, data: &[T]) {
        let data_size = std::mem::size_of_val(data) as u64;
        let buffer_info = vk::BufferCreateInfoBuilder::new()
            .size(data_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let src_buffer = AllocatedBuffer::new(
            self.allocator.clone(),
            *buffer_info,
            vk_mem_erupt::MemoryUsage::CpuToGpu,
            Default::default(),
        );
        let ptr = src_buffer.map();
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, data.len()) }
        src_buffer.unmap();
        immediate_submit(&self.device, &self.transfer_context, |cmd| unsafe {
            self.device.cmd_copy_buffer(
                cmd,
                *src_buffer,
                **self.buffer,
                &[vk::BufferCopyBuilder::new()
                    .dst_offset(offset)
                    .src_offset(0)
                    .size(data_size)],
            );
        });
        drop(src_buffer)
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

pub struct Mesh {
    pub bounds: MeshBounds,
    pub vertex_allocation: BufferSubAllocation,
    pub index_allocation: BufferSubAllocation,
    pub vertex_count: u32,
    pub index_count: u32,
    pub id: u32,
}

impl Mesh {
    pub fn new(
        vertices: &[Vertex],
        indices: &[u32],
        vertex_buffer: &mut SubAllocatedBuffer,
    ) -> Self {
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
        Self::new_with_bounds(vertices, indices, vertex_buffer, bounds)
    }
    pub fn new_with_bounds<V>(
        vertices: &[V],
        indices: &[u32],
        vertex_buffer: &mut SubAllocatedBuffer,
        bounds: MeshBounds,
    ) -> Self {
        let vertex_allocation = vertex_buffer.insert(&vertices);
        let index_allocation = vertex_buffer.insert(&indices);
        Self {
            bounds,
            vertex_allocation,
            index_allocation,
            vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            id: 0,
        }
    }
    pub fn load<P: AsRef<std::path::Path> + std::fmt::Debug>(
        vertex_buffer: &mut SubAllocatedBuffer,
        path: P,
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
        let mut meshes = Vec::new();
        for model in models {
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
            let indices = &model.mesh.indices;
            meshes.push(Self::new(&vertices, indices, vertex_buffer));
        }
        meshes
    }
}
