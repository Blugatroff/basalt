#!/bin/sh

base=$(dirname "$0")

function compile {
  glslc -g -O $1 -o $1.spv
}

compile $base/shaders/rgb_triangle.vert
compile $base/shaders/rgb_triangle.frag

compile $base/shaders/mesh.vert
compile $base/shaders/mesh.frag

compile $base/shaders/test.comp

compile $base/shaders/egui.vert
compile $base/shaders/egui.frag
