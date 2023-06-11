use crate::buffer;
use crate::handles::Allocator;
use crate::utils::round_to;
use crate::TransferContext;
use ash::vk;
use cgmath::InnerSpace;
use std::any::TypeId;
use std::sync::Arc;

#[derive(Clone)]
pub struct VertexInfoDescription {
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
}

impl VertexInfoDescription {
    pub fn builder(&self) -> vk::PipelineVertexInputStateCreateInfoBuilder {
        assert!(!self.bindings.is_empty());
        assert!(!self.attributes.is_empty());
        vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&self.bindings)
            .vertex_attribute_descriptions(&self.attributes)
            .flags(Default::default())
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct DefaultVertex {
    pub position: cgmath::Vector3<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub uv: cgmath::Vector2<f32>,
}

pub struct LoadingVertex {
    pub position: cgmath::Vector3<f32>,
    pub normal: cgmath::Vector3<f32>,
    pub uv: Option<cgmath::Vector2<f32>>,
    pub color: Option<[u8; 4]>,
}

pub trait Vertex: 'static {
    fn position(&self) -> cgmath::Vector3<f32>;
    fn description() -> VertexInfoDescription;
    fn new(_: LoadingVertex) -> Self
    where
        Self: Sized,
    {
        panic!("tried to load vertex without a new Function")
    }
}

impl Vertex for DefaultVertex {
    fn position(&self) -> cgmath::Vector3<f32> {
        self.position
    }
    fn description() -> VertexInfoDescription {
        let bindings = vec![
            *vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<Self>().try_into().unwrap())
                .input_rate(vk::VertexInputRate::VERTEX),
            *vk::VertexInputBindingDescription::builder()
                .binding(1)
                .stride(std::mem::size_of::<shaders::Object>().try_into().unwrap())
                .input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let attributes = vec![
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 3]>().try_into().unwrap()),
            *vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(std::mem::size_of::<[f32; 6]>().try_into().unwrap()),
        ];
        VertexInfoDescription {
            bindings,
            attributes,
        }
    }
    fn new(v: LoadingVertex) -> Self {
        Self {
            position: v.position,
            normal: v.normal,
            uv: v.uv.unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub max: cgmath::Vector3<f32>,
    pub min: cgmath::Vector3<f32>,
    pub sphere_bounds: f32,
}

pub struct Mesh {
    pub bounds: Bounds,
    pub vertex_start: u32,
    pub index_start: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub buffer: Arc<buffer::Allocated>,
    vertex_type_size: usize,
    vertex_type: TypeId,
    vertex_name: &'static str,
    name: String,
}

impl Mesh {
    pub fn new<V: Vertex + 'static>(
        vertices: &[V],
        indices: &[u32],
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        host_visible: bool,
        name: String,
    ) -> Self {
        let bounds = Self::calculate_bounds(vertices);
        Self::new_with_bounds(
            vertices,
            indices,
            bounds,
            allocator,
            transfer_context,
            host_visible,
            name,
        )
    }
    pub fn vertex_type(&self) -> TypeId {
        self.vertex_type
    }
    pub const fn vertex_name(&self) -> &'static str {
        self.vertex_name
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    #[rustfmt::skip]
    fn calculate_bounds<V: Vertex>(vertices: &[V]) -> Bounds {
        let mut bounds = Bounds {
            max: cgmath::Vector3::new(0.0, 0.0, 0.0),
            min: cgmath::Vector3::new(0.0, 0.0, 0.0),
            sphere_bounds: 0.0
        };
        for vertex in vertices {
            let p = vertex.position();
            if p.x > bounds.max.x { bounds.max.x = p.x }
            if p.y > bounds.max.y { bounds.max.y = p.y }
            if p.z > bounds.max.z { bounds.max.z = p.z }
            if p.x < bounds.min.x { bounds.min.x = p.x }
            if p.y < bounds.min.y { bounds.min.y = p.y }
            if p.z < bounds.min.z { bounds.min.z = p.z }
        }
        for p in [bounds.min, bounds.max] {
            bounds.sphere_bounds = bounds.sphere_bounds.max(p.magnitude());
        }
        bounds
    }
    pub fn new_into_buffer<V: 'static>(
        vertices: &[V],
        indices: &[u32],
        bounds: Bounds,
        buffer: Arc<buffer::Allocated>,
        offset: usize,
        name: String,
    ) -> Option<(Self, usize)> {
        assert_eq!(offset % std::mem::size_of::<V>(), 0);
        for i in indices {
            assert!(*i < vertices.len() as u32);
        }
        if buffer.size
            < (offset
                + vertices.len() * std::mem::size_of::<V>()
                + indices.len() * std::mem::size_of::<u32>()) as u64
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
        let vertex_type = std::any::TypeId::of::<V>();
        let vertex_name = std::any::type_name::<V>();
        Some((
            Self {
                vertex_name,
                vertex_type,
                bounds,
                vertex_start: (offset / std::mem::size_of::<V>()).try_into().unwrap(),
                index_start: ((offset + std::mem::size_of::<V>() * vertices.len())
                    / std::mem::size_of::<u32>())
                .try_into()
                .unwrap(),
                vertex_count: vertices.len().try_into().unwrap(),
                index_count: indices.len().try_into().unwrap(),
                buffer,
                vertex_type_size: std::mem::size_of::<V>(),
                name,
            },
            vertices.len() * std::mem::size_of::<V>() + indices.len() * std::mem::size_of::<u32>(),
        ))
    }
    pub fn new_with_bounds<V: 'static>(
        vertices: &[V],
        indices: &[u32],
        bounds: Bounds,
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
        host_visible: bool,
        name: String,
    ) -> Self {
        let size =
            std::mem::size_of::<V>() * vertices.len() + indices.len() * std::mem::size_of::<u32>();
        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(
                vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
            )
            .size(size as u64);
        let buffer = Arc::new(buffer::Allocated::new(
            allocator,
            *buffer_info,
            vma::MemoryUsage::AutoPreferDevice,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            label!("MeshStagingBuffer"),
        ));
        let (mut mesh, _) =
            Self::new_into_buffer(vertices, indices, bounds, buffer, 0, name).unwrap();
        if host_visible {
            return mesh;
        }
        mesh.buffer = Arc::new(mesh.buffer.copy_to_device_local(
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        ));
        mesh
    }
    pub fn combine_meshes<'a>(
        meshes: impl IntoIterator<Item = &'a mut Self>,
        allocator: Arc<Allocator>,
        transfer_context: &TransferContext,
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
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(staging_buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST);
        let staging_buffer = Arc::new(buffer::Allocated::new(
            allocator,
            *buffer_info,
            vma::MemoryUsage::AutoPreferDevice,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
            label!("CombineMeshesStagingBuffer"),
        ));
        let mut offset = 0;
        let device = transfer_context.device.clone();
        transfer_context.immediate_submit(|cmd| unsafe {
            for mesh in &mut meshes {
                offset = round_to(offset, mesh.vertex_type_size as u64);
                let region = vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(offset)
                    .size(mesh.buffer.size);

                let regions = &[*region];
                device.cmd_copy_buffer(cmd, **mesh.buffer, **staging_buffer, regions);
                mesh.index_start +=
                    TryInto::<u32>::try_into(offset / std::mem::size_of::<u32>() as u64)
                        .unwrap();
                mesh.vertex_start +=
                    TryInto::<u32>::try_into(offset / (mesh.vertex_type_size as u64))
                        .unwrap();
                offset += mesh.buffer.size;
            }
        });
        let device_local_buffer = Arc::new(staging_buffer.copy_to_device_local(
            transfer_context,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC,
        ));
        for mesh in &mut meshes {
            mesh.buffer = Arc::clone(&device_local_buffer);
        }
    }
    const fn obj_options() -> tobj::LoadOptions {
        tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        }
    }
    pub fn load<V: Vertex>(
        allocator: Arc<Allocator>,
        path: String,
        transfer_context: &TransferContext,
    ) -> Result<Vec<Self>, tobj::LoadError> {
        let (models, _) = tobj::load_obj(&path, &Self::obj_options())?;
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
                        V::new(LoadingVertex {
                            position,
                            normal,
                            uv: Some(uv),
                            color: None,
                        })
                    })
                    .collect::<Vec<V>>();
                let indices = model.mesh.indices;
                let name = format!("{}-{}", path, model.name);
                Self::new(
                    &vertices,
                    &indices,
                    allocator.clone(),
                    transfer_context,
                    true,
                    name,
                )
            })
            .collect::<Vec<_>>();
        Self::combine_meshes(meshes.iter_mut(), allocator, transfer_context);
        Ok(meshes)
    }
}
