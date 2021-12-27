#!/bin/sh

base=$(dirname "$0")

function compile {
    # { echo "#version 460"; cat $base/shaders/types.glsl $base/shaders/defaultSet.glsl $1; } | cat
    { echo "#version 460"; cat $base/shaders/types.glsl $base/shaders/defaultSet.glsl $1; } | glslc -fshader-stage=$2 -g -O - -o $1.spv
}

compile $base/shaders/rgb_triangle.vert vert
compile $base/shaders/rgb_triangle.frag frag

compile $base/shaders/mesh.vert vert
compile $base/shaders/mesh.frag frag

compile $base/shaders/egui.vert vert
compile $base/shaders/egui.frag frag

cd ../
cargo run --bin extract_structs -- \
    ./testing/shaders/*.spv \
    > ./testing/src/shader_types.rs

