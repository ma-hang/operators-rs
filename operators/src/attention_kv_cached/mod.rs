﻿mod args;
mod operator;

pub use args::Args;

crate::op_trait!(AttnKVCached);

macro_rules! impl_op {
    ($dev:ident, $proc:ident) => {
        pub type Operator = super::operator::Operator<
            crate::$dev::$proc,
            crate::rearrange::$dev::Operator,
            crate::attention::$dev::Operator,
        >;
    };
}

#[cfg(any(use_cpu, test))]
pub mod common_cpu;
#[cfg(use_infini)]
pub mod infini;
#[cfg(use_gpu)]
pub mod nvidia_gpu;
#[cfg(use_cl)]
pub mod opencl;
