use std::sync::Arc;

use basalt::{
    label, vk, ColorBlendAttachment, DefaultVertex, DepthStencilInfo, DescriptorSetLayout, Device,
    InputAssemblyState, MaterialLoadFn, MultiSamplingState, Pipeline, PipelineDesc, PipelineLayout,
    RasterizationState, ShaderModule,
};

pub fn mesh_pipeline(device: &Arc<Device>, transparency: bool) -> MaterialLoadFn {
    let vert_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/mesh.vert.spv"),
        String::from("MeshPipelineVertexShader"),
        vk::ShaderStageFlagBits::VERTEX,
    );
    let frag_shader = ShaderModule::new(
        device.clone(),
        include_bytes!("../../shaders/mesh.frag.spv"),
        String::from("MeshPipelineFragmentShader"),
        vk::ShaderStageFlagBits::FRAGMENT,
    );

    let texture_set_layout = DescriptorSetLayout::from_shader(device, &frag_shader)
        .remove(&1)
        .unwrap();
    let texture_set_layout = Arc::new(texture_set_layout);

    Box::new(move |args| {
        let set_layouts = vec![args.global_set_layout.clone(), texture_set_layout.clone()];
        let pipeline_layout = Arc::new(PipelineLayout::new(
            args.device.clone(),
            set_layouts,
            (),
            &label!("MeshPipelineLayout"),
        ));
        let width = args.width;
        let height = args.height;
        let shader_stages = [&vert_shader, &frag_shader];
        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: width as f32,
            height: height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        Pipeline::new::<DefaultVertex>(
            args.device.clone(),
            **args.render_pass,
            &PipelineDesc {
                view_port,
                scissor: vk::Rect2DBuilder::new()
                    .offset(vk::Offset2D { x: 0, y: 0 })
                    .extent(vk::Extent2D { width, height }),
                color_blend_attachment: if transparency {
                    ColorBlendAttachment::default_transparency()
                } else {
                    ColorBlendAttachment::default()
                },
                shader_stages: &shader_stages,
                input_assembly_state: InputAssemblyState {
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                },
                rasterization_state: RasterizationState {
                    polygon_mode: vk::PolygonMode::FILL,
                    front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                    cull_mode: vk::CullModeFlags::BACK,
                },
                multisample_state: MultiSamplingState {},
                layout: pipeline_layout,
                depth_stencil: DepthStencilInfo {
                    write: true,
                    test: Some(vk::CompareOp::LESS),
                },
            },
            &label!("MeshPipeline"),
        )
    })
}
