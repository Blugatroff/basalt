use std::error::Error;

use spirq::{EntryPoint, SpirvBinary};

#[derive(Debug, Clone)]
pub struct Shader {
    pub name: String,
    pub inputs: Vec<Location>,
    pub outputs: Vec<Location>,
    pub descriptors: Vec<Descriptor>,
    pub strukts: Vec<Strukt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorLength {
    Two,
    Three,
    Four,
}

impl std::fmt::Display for VectorLength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            VectorLength::Two => "2",
            VectorLength::Three => "3",
            VectorLength::Four => "4",
        })
    }
}

impl From<u32> for VectorLength {
    fn from(n: u32) -> Self {
        match n {
            2 => Self::Two,
            3 => Self::Three,
            4 => Self::Four,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarType {
    Float,
    Uint,
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarType::Float => "f32",
            ScalarType::Uint => "u32",
        })
    }
}

impl From<&spirq::ty::ScalarType> for ScalarType {
    fn from(ty: &spirq::ty::ScalarType) -> Self {
        match ty {
            spirq::ty::ScalarType::Unsigned(4) => Self::Uint,
            spirq::ty::ScalarType::Float(4) => Self::Float,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixSize {
    ThreeByThree,
    FourByFour,
}

impl std::fmt::Display for MatrixSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MatrixSize::ThreeByThree => "3",
            MatrixSize::FourByFour => "4",
        })
    }
}

impl From<u32> for MatrixSize {
    fn from(n: u32) -> Self {
        match n {
            3 => Self::ThreeByThree,
            4 => Self::FourByFour,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strukt {
    pub name: Option<String>,
    pub members: Vec<Member>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Vector(ScalarType, VectorLength),
    Scalar(ScalarType),
    Struct(Strukt),
    Array(Box<Type>),
    Matrix(ScalarType, MatrixSize),
    Sampler,
    SampledImage,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&match self {
            Type::Vector(ty, n) => format!("::cgmath::Vector{}<{}>", n, ty),
            Type::Scalar(ty) => format!("{}", ty),
            Type::Struct(strukt) => strukt
                .name
                .clone()
                .unwrap_or_else(|| String::from("UnNamed")),
            Type::Array(ty) => format!("Vec<{}>", ty),
            Type::Matrix(ty, size) => format!("::cgmath::Matrix{}<{}>", size, ty),
            Type::Sampler => unimplemented!(),
            Type::SampledImage => unimplemented!(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    name: Option<String>,
    ty: Type,
}

impl From<&spirq::Type> for Type {
    fn from(ty: &spirq::Type) -> Self {
        match ty {
            spirq::Type::Scalar(spirq::ty::ScalarType::Float(4)) => Self::Scalar(ScalarType::Float),
            spirq::Type::Scalar(spirq::ty::ScalarType::Unsigned(4)) => {
                Self::Scalar(ScalarType::Uint)
            }
            spirq::Type::Vector(spirq::ty::VectorType { scalar_ty, nscalar }) => {
                Self::Vector(ScalarType::from(scalar_ty), VectorLength::from(*nscalar))
            }
            spirq::Type::Struct(strukt) => {
                let name = strukt.name().map(String::from);
                let mut i = 0;
                let members = std::iter::from_fn(|| {
                    let member = strukt.get_member(i);
                    i += 1;
                    member
                })
                .map(|member| {
                    let name = member.name.clone();
                    let ty = Self::from(&member.ty);
                    Member { name, ty }
                })
                .collect();
                Self::Struct(Strukt { name, members })
            }
            spirq::Type::Array(array) => Self::Array(Box::new(Self::from(array.proto_ty()))),
            spirq::Type::Matrix(matrix) => {
                let (ty, n) = (&matrix.vec_ty.scalar_ty, matrix.vec_ty.nscalar);
                Self::Matrix(ScalarType::from(ty), MatrixSize::from(n))
            }
            spirq::Type::Sampler() => Self::Sampler,
            spirq::Type::SampledImage(_) => Self::SampledImage,
            e => {
                dbg!(e);
                unimplemented!()
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Location {
    pub location: u32,
    pub ty: Type,
}

impl From<spirq::InterfaceVariableResolution<'_>> for Location {
    fn from(input: spirq::InterfaceVariableResolution<'_>) -> Self {
        let location = input.location.loc();
        let ty = Type::from(input.ty);
        Self { location, ty }
    }
}

#[derive(Debug, Clone)]
pub struct Descriptor {
    pub set: u32,
    pub binding: u32,
    pub ty: DescriptorType,
}

#[derive(Debug, Clone)]
pub enum DescriptorType {
    Uniform(Type),
    Storage(Type),
    Sampler(Type),
}

impl DescriptorType {
    fn ty(&self) -> &Type {
        match self {
            DescriptorType::Uniform(ty) => ty,
            DescriptorType::Storage(ty) => ty,
            DescriptorType::Sampler(ty) => ty,
        }
    }
}

impl From<&spirq::DescriptorType> for DescriptorType {
    fn from(ty: &spirq::DescriptorType) -> Self {
        match ty {
            spirq::DescriptorType::UniformBuffer(_, ty) => Self::Uniform(Type::from(ty)),
            spirq::DescriptorType::StorageBuffer(_, ty) => Self::Storage(Type::from(ty)),
            spirq::DescriptorType::SampledImage(_, ty) => Self::Sampler(Type::from(ty)),
            _ => unimplemented!(),
        }
    }
}

impl From<spirq::DescriptorResolution<'_>> for Descriptor {
    fn from(descriptor: spirq::DescriptorResolution<'_>) -> Self {
        let ty = DescriptorType::from(descriptor.desc_ty);
        let set = descriptor.desc_bind.set();
        let binding = descriptor.desc_bind.bind();
        Self { ty, set, binding }
    }
}

impl Type {
    fn strukts(&self) -> Vec<&Strukt> {
        fn find<'a, 'b>(strukts: &mut Vec<&'b Strukt>, ty: &'a Type)
        where
            'a: 'b,
        {
            match ty {
                Type::Struct(strukt) => {
                    strukts.push(strukt);
                    for member in &strukt.members {
                        find(strukts, &member.ty);
                    }
                }
                Type::Array(ty) => find(strukts, ty),
                _ => {}
            }
        }
        let mut strukts = Vec::new();
        find(&mut strukts, self);
        strukts
    }
}

impl From<&EntryPoint> for Shader {
    fn from(entry: &EntryPoint) -> Self {
        let name = entry.name.clone();
        let inputs: Vec<Location> = entry.inputs().map(Location::from).collect();
        let outputs: Vec<Location> = entry.outputs().map(Location::from).collect();
        let descriptors: Vec<Descriptor> = entry.descs().map(Descriptor::from).collect();
        let strukts = inputs
            .iter()
            .map(|i| &i.ty)
            .chain(outputs.iter().map(|o| &o.ty))
            .chain(descriptors.iter().map(|d| d.ty.ty()))
            .flat_map(|ty| ty.strukts())
            .fold(Vec::new(), |mut types, ty| {
                if types.contains(ty) {
                    return types;
                }
                types.push(ty.clone());
                types
            });

        Self {
            name,
            inputs,
            outputs,
            descriptors,
            strukts,
        }
    }
}

impl Shader {
    pub fn from_spirv(spirv: &[u8]) -> Result<Vec<Self>, Box<dyn Error>> {
        let spirv = SpirvBinary::from(spirv);
        let entry_points = spirv.reflect_vec()?;
        dbg!(&entry_points);
        Ok(entry_points.iter().map(Self::from).collect())
    }
}

fn camel_case_to_snake_case(ident: &str) -> String {
    let mut s = String::new();
    for c in ident.chars() {
        if c.is_uppercase() {
            s.push('_');
            s.push_str(&c.to_lowercase().to_string());
        } else {
            s.push(c);
        }
    }
    s
}

impl std::fmt::Display for Strukt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let copy = match self.members.iter().any(|m| matches!(m.ty, Type::Array(_))) {
            true => "",
            false => ", Copy",
        };
        let mut s = format!(
            "#[repr(C)]\n#[allow(dead_code)]\n#[derive(Debug, Clone{})]\npub struct {} {{",
            copy,
            self.name.as_deref().unwrap_or("UnNamed")
        );
        for m in &self.members {
            s.push_str(&format!(
                "\n\tpub {}: {},",
                &m.name
                    .as_deref()
                    .map(camel_case_to_snake_case)
                    .unwrap_or_else(|| String::from("_unnamed")),
                m.ty
            ));
        }
        s.push_str("\n}\n");
        f.write_str(&s)
    }
}
