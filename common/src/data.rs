﻿use std::alloc::Layout;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct DataLayout {
    packed: u8,
    signed_nbyte: u8,
    exponent: u8,
    mantissa: u8,
}

impl DataLayout {
    #[inline]
    pub const fn new(packed: usize, signed: bool, exponent: usize, mantissa: usize) -> Self {
        assert!(packed <= u8::MAX as usize);
        assert!(exponent <= u8::MAX as usize);
        assert!(mantissa <= u8::MAX as usize);
        let signed = if signed { 1 } else { 0 };
        let total_bits = packed * (signed + exponent + mantissa);
        assert!(total_bits % u8::BITS as usize == 0);
        let nbyte = total_bits / u8::BITS as usize;
        assert!(nbyte.is_power_of_two());
        assert!(nbyte < (1 << 7));
        let signed_nbyte = (signed << 7) | nbyte;
        Self {
            packed: packed as _,
            signed_nbyte: signed_nbyte as _,
            exponent: exponent as _,
            mantissa: mantissa as _,
        }
    }

    #[inline]
    pub const fn packed(&self) -> usize {
        self.packed as _
    }

    #[inline]
    pub const fn signed(&self) -> bool {
        self.signed_nbyte >> 7 == 1
    }

    #[inline]
    pub const fn exponent(&self) -> usize {
        self.exponent as _
    }

    #[inline]
    pub const fn mantissa(&self) -> usize {
        self.mantissa as _
    }

    #[inline]
    pub const fn bits(&self) -> usize {
        self.nbyte() * u8::BITS as usize
    }

    #[inline]
    pub const fn layout(&self) -> Layout {
        let nbyte = self.nbyte();
        unsafe { Layout::from_size_align_unchecked(nbyte, nbyte) }
    }

    #[inline]
    const fn nbyte(&self) -> usize {
        (self.signed_nbyte & ((1 << 7) - 1)) as _
    }
}

pub(crate) mod types {

    #[macro_export]
    macro_rules! layout {
        ($name:ident i($bits:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, true, 0, $bits - 1);
        };
        ($name:ident u($bits:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            #[rustfmt::skip]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, false, 0, $bits);
        };
        ($name:ident e($exp:expr)m($mant:expr)x($packed:expr)) => {
            #[allow(non_upper_case_globals)]
            pub const $name: $crate::DataLayout =
                $crate::DataLayout::new($packed, true, $exp, $mant);
        };

        ($name:ident i($bits:expr)) => {
            layout!($name i($bits)x(1));
        };
        ($name:ident u($bits:expr)) => {
            layout!($name u($bits)x(1));
        };
        ($name:ident e($exp:expr)m($mant:expr)) => {
            layout!($name e($exp)m($mant)x(1));
        };
    }

    layout!(I8     i( 8)          );
    layout!(I16    i(16)          );
    layout!(I32    i(32)          );
    layout!(I64    i(64)          );
    layout!(U8     u( 8)          );
    layout!(U16    u(16)          );
    layout!(U32    u(32)          );
    layout!(U64    u(64)          );
    layout!(F16    e(10)m( 5)     );
    layout!(BF16   e( 7)m( 8)     );
    layout!(F32    e(23)m( 8)     );
    layout!(F64    e(52)m(11)     );

    layout!(F16x2  e(10)m( 5)x(2) );
    layout!(BF16x2 e( 7)m( 8)x(2) );
}