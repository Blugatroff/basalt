#!/bin/sh

base=$(dirname "$0")

function compile {
    glslc -I ./shaders/ -g -O -fshader-stage=$2 $1 -o $1.spv
}

compile $base/shaders/rgb_triangle.vert vert
compile $base/shaders/rgb_triangle.frag frag

compile $base/shaders/mesh.vert vert
compile $base/shaders/mesh.frag frag

compile $base/shaders/egui.vert vert
compile $base/shaders/egui.frag frag

compile $base/shaders/line.vert vert
compile $base/shaders/line.frag frag

compile $base/shaders/cull.comp comp

# cargo run --bin extract_structs -- ./shaders/*.spv > ./shaders/shader_types.rs

