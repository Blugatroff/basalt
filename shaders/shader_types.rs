#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct GlobalUniform {
    pub view: ::cgmath::Matrix4<f32>,
    pub proj: ::cgmath::Matrix4<f32>,
    pub view_proj: ::cgmath::Matrix4<f32>,
    pub time: f32,
    pub renderables_count: u32,
    pub screen_width: f32,
    pub screen_height: f32,
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ObjectBuffer {
    pub objects: Vec<Object>,
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Object {
    pub transform: ::cgmath::Matrix4<f32>,
    pub batch: u32,
    pub draw: u32,
    pub first_instance: u32,
    pub uncullable: u32,
    pub unused_3: u32,
    pub custom_set: u32,
    pub mesh: u32,
    pub redirect: u32,
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct IndirectDrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
    pub batch_count: u32,
    pub padding0: u32,
    pub padding1: u32,
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Mesh {
    pub bounds_max: cgmath::Vector3<f32>,
    pub first_index: u32,
    pub bounds_min: cgmath::Vector3<f32>,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub padding_0: u32,
    pub padding_1: u32,
    pub padding_2: u32,
}
