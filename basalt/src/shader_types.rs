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


