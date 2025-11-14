use crate::curves::CurveBytes;
use core::fmt;
use core::ops::{Add, Div, DivAssign, Mul, Neg, Shr, ShrAssign, Sub};
use elliptic_curve::hash2curve::ExpandMsgXmd;
use elliptic_curve::{
    ScalarPrimitive,
    bigint::{ArrayEncoding, NonZero, U256, U384, U512},
    generic_array::{
        GenericArray,
        typenum::{U48, U64},
    },
    hash2curve::{ExpandMsg, Expander},
    ops::{Invert, Reduce},
    scalar::{FromUintUnchecked, IsHigh},
    zeroize::DefaultIsZeroes,
};
use ff::{Field, FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use rand::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq, CtOption};

#[cfg(feature = "sqrt-table")]
use lazy_static::lazy_static;

#[cfg(feature = "bits")]
use ff::{FieldBits, PrimeFieldBits};

use crate::arithmetic::{SqrtTableHelpers, adc, decode_hex_into_slice, mac, sbb};

#[cfg(feature = "sqrt-table")]
use crate::arithmetic::SqrtTables;
use crate::pallas::Pallas;

/// This represents an element of $\mathbb{F}_q$ where
///
/// `q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001`
///
/// is the base field of the Vesta curve.
// The internal representation of this type is four 64-bit unsigned
// integers in little-endian order. `Fq` values are always in
// Montgomery form; i.e., Fq(a) = aR mod q, with R = 2^256.
#[derive(Clone, Copy, Eq)]
#[repr(transparent)]
pub struct Fq(pub(crate) [u64; 4]);

impl DefaultIsZeroes for Fq {}

impl fmt::Debug for Fq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}", self)
    }
}

impl fmt::Display for Fq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}", self)
    }
}

impl fmt::LowerHex for Fq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tmp = self.to_be_bytes();
        for &b in tmp.iter() {
            write!(f, "{:02x}", b)?;
        }
        Ok(())
    }
}

impl fmt::UpperHex for Fq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tmp = self.to_be_bytes();
        for &b in tmp.iter() {
            write!(f, "{:02X}", b)?;
        }
        Ok(())
    }
}

impl From<bool> for Fq {
    fn from(bit: bool) -> Fq {
        if bit { Self::ONE } else { Self::ZERO }
    }
}

impl From<u8> for Fq {
    fn from(value: u8) -> Self {
        Self([value as u64, 0, 0, 0]) * R2
    }
}

impl From<u16> for Fq {
    fn from(value: u16) -> Self {
        Self([value as u64, 0, 0, 0]) * R2
    }
}

impl From<u32> for Fq {
    fn from(value: u32) -> Self {
        Self([value as u64, 0, 0, 0]) * R2
    }
}

impl From<u64> for Fq {
    fn from(val: u64) -> Self {
        Self([val, 0, 0, 0]) * R2
    }
}

#[cfg(target_pointer_width = "64")]
impl From<u128> for Fq {
    fn from(val: u128) -> Self {
        Self([val as u64, (val >> 64) as u64, 0, 0]) * R2
    }
}

impl ConstantTimeEq for Fq {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0[0].ct_eq(&other.0[0])
            & self.0[1].ct_eq(&other.0[1])
            & self.0[2].ct_eq(&other.0[2])
            & self.0[3].ct_eq(&other.0[3])
    }
}

impl PartialEq for Fq {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other).unwrap_u8() == 1
    }
}

impl core::cmp::Ord for Fq {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        let left = self.to_repr();
        let right = other.to_repr();
        left.iter()
            .zip(right.iter())
            .rev()
            .find_map(|(left_byte, right_byte)| match left_byte.cmp(right_byte) {
                core::cmp::Ordering::Equal => None,
                res => Some(res),
            })
            .unwrap_or(core::cmp::Ordering::Equal)
    }
}

impl core::cmp::PartialOrd for Fq {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl ConditionallySelectable for Fq {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        Fq([
            u64::conditional_select(&a.0[0], &b.0[0], choice),
            u64::conditional_select(&a.0[1], &b.0[1], choice),
            u64::conditional_select(&a.0[2], &b.0[2], choice),
            u64::conditional_select(&a.0[3], &b.0[3], choice),
        ])
    }
}

const HALF_MODULUS: Fq = Fq([
    0xc623759080000000,
    0x11234c7e04ca546e,
    0x0000000000000000,
    0x2000000000000000,
]);

#[cfg(not(target_pointer_width = "64"))]
const HALF_MODULUS_LIMBS_32: [u32; 8] = [
    0x80000000, 0xc6237590, 0x04ca546e, 0x11234c7e, 0x00000000, 0x00000000, 0x00000000, 0x20000000,
];

/// Constant representing the modulus
/// q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
const MODULUS: Fq = Fq([
    0x8c46eb2100000001,
    0x224698fc0994a8dd,
    0x0,
    0x4000000000000000,
]);

/// The modulus as u32 limbs.
#[cfg(not(target_pointer_width = "64"))]
const MODULUS_LIMBS_32: [u32; 8] = [
    0x0000_0001,
    0x8c46_eb21,
    0x0994_a8dd,
    0x2246_98fc,
    0x0000_0000,
    0x0000_0000,
    0x0000_0000,
    0x4000_0000,
];

impl<'a> Neg for &'a Fq {
    type Output = Fq;

    #[inline]
    fn neg(self) -> Fq {
        self.neg()
    }
}

impl Neg for Fq {
    type Output = Fq;

    #[inline]
    fn neg(self) -> Fq {
        -&self
    }
}

impl<'a, 'b> Sub<&'b Fq> for &'a Fq {
    type Output = Fq;

    #[inline]
    fn sub(self, rhs: &'b Fq) -> Fq {
        self.sub(rhs)
    }
}

impl<'a, 'b> Add<&'b Fq> for &'a Fq {
    type Output = Fq;

    #[inline]
    fn add(self, rhs: &'b Fq) -> Fq {
        self.add(rhs)
    }
}

impl<'a, 'b> Mul<&'b Fq> for &'a Fq {
    type Output = Fq;

    #[inline]
    fn mul(self, rhs: &'b Fq) -> Fq {
        self.mul(rhs)
    }
}

impl_binops_additive!(Fq, Fq);
impl_binops_multiplicative!(Fq, Fq);

impl<'a, 'b> Div<&'b Fq> for &'a Fq {
    type Output = Fq;

    fn div(self, rhs: &'b Fq) -> Fq {
        self * Field::invert(rhs).expect("a non-zero scalar")
    }
}

impl Div<&Fq> for Fq {
    type Output = Fq;

    fn div(self, rhs: &Fq) -> Fq {
        &self / rhs
    }
}

impl Div<Fq> for &Fq {
    type Output = Fq;

    fn div(self, rhs: Fq) -> Self::Output {
        self / &rhs
    }
}

impl Div for Fq {
    type Output = Fq;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl DivAssign<&Fq> for Fq {
    fn div_assign(&mut self, rhs: &Fq) {
        *self = &*self / rhs;
    }
}

impl DivAssign for Fq {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = &*self / rhs;
    }
}

impl<T: core::borrow::Borrow<Fq>> core::iter::Sum<T> for Fq {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, item| acc + item.borrow())
    }
}

impl<T: core::borrow::Borrow<Fq>> core::iter::Product<T> for Fq {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, item| acc * item.borrow())
    }
}

/// INV = -(q^{-1} mod 2^64) mod 2^64
const INV: u64 = 0x8c46eb20ffffffff;

/// R = 2^256 mod q
const R: Fq = Fq([
    0x5b2b3e9cfffffffd,
    0x992c350be3420567,
    0xffffffffffffffff,
    0x3fffffffffffffff,
]);

/// R^2 = 2^512 mod q
const R2: Fq = Fq([
    0xfc9678ff0000000f,
    0x67bb433d891a16e3,
    0x7fae231004ccf590,
    0x096d41af7ccfdaa9,
]);

/// R^3 = 2^768 mod q
const R3: Fq = Fq([
    0x008b421c249dae4c,
    0xe13bda50dba41326,
    0x88fececb8e15cb63,
    0x07dd97a06e6792c8,
]);

/// `GENERATOR = 5 mod q` is a generator of the `q - 1` order multiplicative
/// subgroup, or in other words a primitive root of the field.
const GENERATOR: Fq = Fq::from_raw([
    0x0000_0000_0000_0005,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
    0x0000_0000_0000_0000,
]);

const S: u32 = 32;

/// GENERATOR^t where t * 2^s + 1 = q
/// with t odd. In other words, this
/// is a 2^s root of unity.
const ROOT_OF_UNITY: Fq = Fq::from_raw([
    0xa70e2c1102b6d05f,
    0x9bb97ea3c106f049,
    0x9e5c4dfd492ae26e,
    0x2de6a9b8746d3f58,
]);

/// GENERATOR^{2^s} where t * 2^s + 1 = q
/// with t odd. In other words, this
/// is a t root of unity.
const DELTA: Fq = Fq::from_raw([
    0x8494392472d1683c,
    0xe3ac3376541d1140,
    0x06f0a88e7f7949f8,
    0x2237d54423724166,
]);

/// `(t - 1) // 2` where t * 2^s + 1 = p with t odd.
#[cfg(any(test, not(feature = "sqrt-table")))]
const T_MINUS1_OVER2: [u64; 4] = [
    0x04ca_546e_c623_7590,
    0x0000_0000_1123_4c7e,
    0x0000_0000_0000_0000,
    0x0000_0000_2000_0000,
];

impl Default for Fq {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl Fq {
    /// Returns zero, the additive identity.
    pub const ZERO: Self = Self([0, 0, 0, 0]);

    /// Returns one, the multiplicative identity.
    pub const ONE: Self = R;
    /// The number of bytes that represent this value
    pub const BYTES: usize = 32;

    /// Doubles this field element.
    #[inline]
    pub const fn double(&self) -> Fq {
        // TODO: This can be achieved more efficiently with a bitshift.
        self.add(self)
    }

    fn from_u512(limbs: [u64; 8]) -> Fq {
        // We reduce an arbitrary 512-bit number by decomposing it into two 256-bit digits
        // with the higher bits multiplied by 2^256. Thus, we perform two reductions
        //
        // 1. the lower bits are multiplied by R^2, as normal
        // 2. the upper bits are multiplied by R^2 * 2^256 = R^3
        //
        // and computing their sum in the field. It remains to see that arbitrary 256-bit
        // numbers can be placed into Montgomery form safely using the reduction. The
        // reduction works so long as the product is less than R=2^256 multiplied by
        // the modulus. This holds because for any `c` smaller than the modulus, we have
        // that (2^256 - 1)*c is an acceptable product for the reduction. Therefore, the
        // reduction always works so long as `c` is in the field; in this case it is either the
        // constant `R2` or `R3`.
        let d0 = Fq([limbs[0], limbs[1], limbs[2], limbs[3]]);
        let d1 = Fq([limbs[4], limbs[5], limbs[6], limbs[7]]);
        // Convert to Montgomery form
        d0 * R2 + d1 * R3
    }

    /// Converts from an integer represented in little endian
    /// into its (congruent) `Fq` representation.
    pub const fn from_raw(val: [u64; 4]) -> Fq {
        (&Fq(val)).mul(&R2)
    }

    /// Squares this element.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn square(&self) -> Fq {
        let (r1, carry) = mac(0, self.0[0], self.0[1], 0);
        let (r2, carry) = mac(0, self.0[0], self.0[2], carry);
        let (r3, r4) = mac(0, self.0[0], self.0[3], carry);

        let (r3, carry) = mac(r3, self.0[1], self.0[2], 0);
        let (r4, r5) = mac(r4, self.0[1], self.0[3], carry);

        let (r5, r6) = mac(r5, self.0[2], self.0[3], 0);

        let r7 = r6 >> 63;
        let r6 = (r6 << 1) | (r5 >> 63);
        let r5 = (r5 << 1) | (r4 >> 63);
        let r4 = (r4 << 1) | (r3 >> 63);
        let r3 = (r3 << 1) | (r2 >> 63);
        let r2 = (r2 << 1) | (r1 >> 63);
        let r1 = r1 << 1;

        let (r0, carry) = mac(0, self.0[0], self.0[0], 0);
        let (r1, carry) = adc(0, r1, carry);
        let (r2, carry) = mac(r2, self.0[1], self.0[1], carry);
        let (r3, carry) = adc(0, r3, carry);
        let (r4, carry) = mac(r4, self.0[2], self.0[2], carry);
        let (r5, carry) = adc(0, r5, carry);
        let (r6, carry) = mac(r6, self.0[3], self.0[3], carry);
        let (r7, _) = adc(0, r7, carry);

        Fq::montgomery_reduce(r0, r1, r2, r3, r4, r5, r6, r7)
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(not(feature = "uninline-portable"), inline(always))]
    const fn montgomery_reduce(
        r0: u64,
        r1: u64,
        r2: u64,
        r3: u64,
        r4: u64,
        r5: u64,
        r6: u64,
        r7: u64,
    ) -> Self {
        // The Montgomery reduction here is based on Algorithm 14.32 in
        // Handbook of Applied Cryptography
        // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

        let k = r0.wrapping_mul(INV);
        let (_, carry) = mac(r0, k, MODULUS.0[0], 0);
        let (r1, carry) = mac(r1, k, MODULUS.0[1], carry);
        let (r2, carry) = mac(r2, k, MODULUS.0[2], carry);
        let (r3, carry) = mac(r3, k, MODULUS.0[3], carry);
        let (r4, carry2) = adc(r4, 0, carry);

        let k = r1.wrapping_mul(INV);
        let (_, carry) = mac(r1, k, MODULUS.0[0], 0);
        let (r2, carry) = mac(r2, k, MODULUS.0[1], carry);
        let (r3, carry) = mac(r3, k, MODULUS.0[2], carry);
        let (r4, carry) = mac(r4, k, MODULUS.0[3], carry);
        let (r5, carry2) = adc(r5, carry2, carry);

        let k = r2.wrapping_mul(INV);
        let (_, carry) = mac(r2, k, MODULUS.0[0], 0);
        let (r3, carry) = mac(r3, k, MODULUS.0[1], carry);
        let (r4, carry) = mac(r4, k, MODULUS.0[2], carry);
        let (r5, carry) = mac(r5, k, MODULUS.0[3], carry);
        let (r6, carry2) = adc(r6, carry2, carry);

        let k = r3.wrapping_mul(INV);
        let (_, carry) = mac(r3, k, MODULUS.0[0], 0);
        let (r4, carry) = mac(r4, k, MODULUS.0[1], carry);
        let (r5, carry) = mac(r5, k, MODULUS.0[2], carry);
        let (r6, carry) = mac(r6, k, MODULUS.0[3], carry);
        let (r7, _) = adc(r7, carry2, carry);

        // Result may be within MODULUS of the correct value
        (&Fq([r4, r5, r6, r7])).sub(&MODULUS)
    }

    /// Multiplies `rhs` by `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn mul(&self, rhs: &Self) -> Self {
        // Schoolbook multiplication

        let (r0, carry) = mac(0, self.0[0], rhs.0[0], 0);
        let (r1, carry) = mac(0, self.0[0], rhs.0[1], carry);
        let (r2, carry) = mac(0, self.0[0], rhs.0[2], carry);
        let (r3, r4) = mac(0, self.0[0], rhs.0[3], carry);

        let (r1, carry) = mac(r1, self.0[1], rhs.0[0], 0);
        let (r2, carry) = mac(r2, self.0[1], rhs.0[1], carry);
        let (r3, carry) = mac(r3, self.0[1], rhs.0[2], carry);
        let (r4, r5) = mac(r4, self.0[1], rhs.0[3], carry);

        let (r2, carry) = mac(r2, self.0[2], rhs.0[0], 0);
        let (r3, carry) = mac(r3, self.0[2], rhs.0[1], carry);
        let (r4, carry) = mac(r4, self.0[2], rhs.0[2], carry);
        let (r5, r6) = mac(r5, self.0[2], rhs.0[3], carry);

        let (r3, carry) = mac(r3, self.0[3], rhs.0[0], 0);
        let (r4, carry) = mac(r4, self.0[3], rhs.0[1], carry);
        let (r5, carry) = mac(r5, self.0[3], rhs.0[2], carry);
        let (r6, r7) = mac(r6, self.0[3], rhs.0[3], carry);

        Fq::montgomery_reduce(r0, r1, r2, r3, r4, r5, r6, r7)
    }

    /// Subtracts `rhs` from `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn sub(&self, rhs: &Self) -> Self {
        let (d0, borrow) = sbb(self.0[0], rhs.0[0], 0);
        let (d1, borrow) = sbb(self.0[1], rhs.0[1], borrow);
        let (d2, borrow) = sbb(self.0[2], rhs.0[2], borrow);
        let (d3, borrow) = sbb(self.0[3], rhs.0[3], borrow);

        // If underflow occurred on the final limb, borrow = 0xfff...fff, otherwise
        // borrow = 0x000...000. Thus, we use it as a mask to conditionally add the modulus.
        let (d0, carry) = adc(d0, MODULUS.0[0] & borrow, 0);
        let (d1, carry) = adc(d1, MODULUS.0[1] & borrow, carry);
        let (d2, carry) = adc(d2, MODULUS.0[2] & borrow, carry);
        let (d3, _) = adc(d3, MODULUS.0[3] & borrow, carry);

        Fq([d0, d1, d2, d3])
    }

    /// Adds `rhs` to `self`, returning the result.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn add(&self, rhs: &Self) -> Self {
        let (d0, carry) = adc(self.0[0], rhs.0[0], 0);
        let (d1, carry) = adc(self.0[1], rhs.0[1], carry);
        let (d2, carry) = adc(self.0[2], rhs.0[2], carry);
        let (d3, _) = adc(self.0[3], rhs.0[3], carry);

        // Attempt to subtract the modulus, to ensure the value
        // is smaller than the modulus.
        (&Fq([d0, d1, d2, d3])).sub(&MODULUS)
    }

    /// Negates `self`.
    #[cfg_attr(not(feature = "uninline-portable"), inline)]
    pub const fn neg(&self) -> Self {
        // Subtract `self` from `MODULUS` to negate. Ignore the final
        // borrow because it cannot underflow; self is guaranteed to
        // be in the field.
        let (d0, borrow) = sbb(MODULUS.0[0], self.0[0], 0);
        let (d1, borrow) = sbb(MODULUS.0[1], self.0[1], borrow);
        let (d2, borrow) = sbb(MODULUS.0[2], self.0[2], borrow);
        let (d3, _) = sbb(MODULUS.0[3], self.0[3], borrow);

        // `tmp` could be `MODULUS` if `self` was zero. Create a mask that is
        // zero if `self` was zero, and `u64::max_value()` if self was nonzero.
        let mask = (((self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0) as u64).wrapping_sub(1);

        Fq([d0 & mask, d1 & mask, d2 & mask, d3 & mask])
    }

    /// Attempts to convert a big-endian byte representation of
    /// a fq into a `Fq`, failing if the input is not canonical.
    pub fn from_be_bytes(bytes: &[u8; 32]) -> CtOption<Self> {
        let mut tmp = Self([0, 0, 0, 0]);

        tmp.0[3] = u64::from_be_bytes(bytes[0..8].try_into().unwrap());
        tmp.0[2] = u64::from_be_bytes(bytes[8..16].try_into().unwrap());
        tmp.0[1] = u64::from_be_bytes(bytes[16..24].try_into().unwrap());
        tmp.0[0] = u64::from_be_bytes(bytes[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff.ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        tmp *= R2;
        CtOption::new(tmp, Choice::from(is_some))
    }

    /// Attempts to convert a little-endian byte representation of
    /// a scalar into a `Fq`, failing if the input is not canonical.
    pub fn from_le_bytes(bytes: &[u8; 32]) -> CtOption<Self> {
        let mut tmp = Self([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap());
        tmp.0[1] = u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap());
        tmp.0[2] = u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap());
        tmp.0[3] = u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    /// Converts an element of `Fq` into a byte representation in
    /// little-endian byte order.
    pub fn to_le_bytes(&self) -> [u8; 32] {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Self::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    /// Converts an element of `Fq` into a byte representation in
    /// big-endian byte order.
    pub fn to_be_bytes(&self) -> [u8; 32] {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Self::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        let mut res = [0; 32];
        res[0..8].copy_from_slice(&tmp.0[3].to_be_bytes());
        res[8..16].copy_from_slice(&tmp.0[2].to_be_bytes());
        res[16..24].copy_from_slice(&tmp.0[1].to_be_bytes());
        res[24..32].copy_from_slice(&tmp.0[0].to_be_bytes());

        res
    }

    /// Create a new [`Fq`] from the provided big endian hex string.
    pub fn from_be_hex(hex: &str) -> CtOption<Self> {
        let mut buf = [0u8; Self::BYTES];
        decode_hex_into_slice(&mut buf, hex.as_bytes());
        Self::from_be_bytes(&buf)
    }

    /// Create a new [`Fq`] from the provided little endian hex string.
    pub fn from_le_hex(hex: &str) -> CtOption<Self> {
        let mut buf = [0u8; Self::BYTES];
        decode_hex_into_slice(&mut buf, hex.as_bytes());
        Self::from_le_bytes(&buf)
    }

    /// Converts a 512-bit little endian integer into
    /// a `Fq` by reducing by the modulus.
    pub fn from_bytes_wide(bytes: &[u8; 64]) -> Self {
        Self::from_u512([
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[32..40]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[40..48]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[48..56]).unwrap()),
            u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[56..64]).unwrap()),
        ])
    }

    /// Read from output of a KDF
    pub fn from_okm(bytes: &[u8; 48]) -> Self {
        const F_2_192: Fq = Fq([
            0xcc920bb9994a8dd9,
            0x87a7dcbe1ff6e0d7,
            0x496d41af7ccfdaa9,
            0x0ee4537bfffffffc,
        ]);
        let d0 = Fq([
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[16..24]).unwrap()),
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[8..16]).unwrap()),
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[0..8]).unwrap()),
            0,
        ]);
        let d1 = Fq([
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[40..48]).unwrap()),
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[32..40]).unwrap()),
            u64::from_be_bytes(<[u8; 8]>::try_from(&bytes[24..32]).unwrap()),
            0,
        ]);
        (d0 * R2) * F_2_192 + d1 * R2
    }

    /// Converts from an integer represented in little endian
    /// into its (congruent) `Fq` representation.
    pub const fn from_raw_unchecked(val: [u64; 4]) -> Self {
        (&Self(val)).mul(&R2)
    }

    /// Converts this `Fq` into an integer represented in little endian
    pub const fn to_raw(&self) -> [u64; 4] {
        let tmp = Self::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);
        tmp.0
    }

    /// Hash input to `Self`
    pub fn hash<X>(msg: &[u8], dst: &[u8]) -> Self
    where
        X: for<'a> ExpandMsg<'a>,
    {
        let d = [dst];
        let mut expander = X::expand_message(&[msg], &d, 48).unwrap();
        let mut out = [0u8; 48];
        expander.fill_bytes(&mut out);
        Self::from_okm(&out)
    }
}

impl From<Fq> for [u8; 32] {
    fn from(value: Fq) -> [u8; 32] {
        value.to_repr().into()
    }
}

impl<'a> From<&'a Fq> for [u8; 32] {
    fn from(value: &'a Fq) -> [u8; 32] {
        value.to_repr().into()
    }
}

impl Field for Fq {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    fn random(mut rng: impl RngCore) -> Self {
        Self::from_u512([
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ])
    }

    fn double(&self) -> Self {
        self.double()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        #[cfg(feature = "sqrt-table")]
        {
            FQ_TABLES.sqrt_ratio(num, div)
        }

        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_ratio_generic(num, div)
    }

    #[cfg(feature = "sqrt-table")]
    fn sqrt_alt(&self) -> (Choice, Self) {
        FQ_TABLES.sqrt_alt(self)
    }

    /// Computes the square root of this element, if it exists.
    fn sqrt(&self) -> CtOption<Self> {
        #[cfg(feature = "sqrt-table")]
        {
            let (is_square, res) = FQ_TABLES.sqrt_alt(self);
            CtOption::new(res, is_square)
        }

        #[cfg(not(feature = "sqrt-table"))]
        ff::helpers::sqrt_tonelli_shanks(self, &T_MINUS1_OVER2)
    }

    /// Computes the multiplicative inverse of this element,
    /// failing if the element is zero.
    fn invert(&self) -> CtOption<Self> {
        let tmp = self.pow_vartime([
            0x8c46eb20ffffffff,
            0x224698fc0994a8dd,
            0x0,
            0x4000000000000000,
        ]);

        CtOption::new(tmp, !self.ct_eq(&Self::ZERO))
    }

    fn pow_vartime<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::ONE;
        let mut found_one = false;
        for e in exp.as_ref().iter().rev() {
            for i in (0..64).rev() {
                if found_one {
                    res = res.square();
                }

                if ((*e >> i) & 1) == 1 {
                    found_one = true;
                    res *= self;
                }
            }
        }
        res
    }
}

impl ff::PrimeField for Fq {
    type Repr = CurveBytes;

    const MODULUS: &'static str =
        "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001";
    const NUM_BITS: u32 = 255;
    const CAPACITY: u32 = 254;
    const TWO_INV: Self = Fq::from_raw([
        0xc623759080000001,
        0x11234c7e04ca546e,
        0x0000000000000000,
        0x2000000000000000,
    ]);
    const MULTIPLICATIVE_GENERATOR: Self = GENERATOR;
    const S: u32 = S;
    const ROOT_OF_UNITY: Self = ROOT_OF_UNITY;
    const ROOT_OF_UNITY_INV: Self = Fq::from_raw([
        0x57eecda0a84b6836,
        0x4ad38b9084b8a80c,
        0xf4c8f353124086c1,
        0x2235e1a7415bf936,
    ]);
    const DELTA: Self = DELTA;

    fn from_u128(v: u128) -> Self {
        Fq::from_raw([v as u64, (v >> 64) as u64, 0, 0])
    }

    fn from_repr(repr: Self::Repr) -> CtOption<Self> {
        let mut tmp = Fq([0, 0, 0, 0]);

        tmp.0[0] = u64::from_le_bytes(repr[0..8].try_into().unwrap());
        tmp.0[1] = u64::from_le_bytes(repr[8..16].try_into().unwrap());
        tmp.0[2] = u64::from_le_bytes(repr[16..24].try_into().unwrap());
        tmp.0[3] = u64::from_le_bytes(repr[24..32].try_into().unwrap());

        // Try to subtract the modulus
        let (_, borrow) = sbb(tmp.0[0], MODULUS.0[0], 0);
        let (_, borrow) = sbb(tmp.0[1], MODULUS.0[1], borrow);
        let (_, borrow) = sbb(tmp.0[2], MODULUS.0[2], borrow);
        let (_, borrow) = sbb(tmp.0[3], MODULUS.0[3], borrow);

        // If the element is smaller than MODULUS then the
        // subtraction will underflow, producing a borrow value
        // of 0xffff...ffff. Otherwise, it'll be zero.
        let is_some = (borrow as u8) & 1;

        // Convert to Montgomery form by computing
        // (a.R^0 * R^2) / R = a.R
        tmp *= &R2;

        CtOption::new(tmp, Choice::from(is_some))
    }

    fn to_repr(&self) -> Self::Repr {
        // Turn into canonical form by computing
        // (a.R) / R = a
        let tmp = Fq::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        let mut res = CurveBytes::default();
        res[0..8].copy_from_slice(&tmp.0[0].to_le_bytes());
        res[8..16].copy_from_slice(&tmp.0[1].to_le_bytes());
        res[16..24].copy_from_slice(&tmp.0[2].to_le_bytes());
        res[24..32].copy_from_slice(&tmp.0[3].to_le_bytes());

        res
    }

    fn is_odd(&self) -> Choice {
        Choice::from(self.to_repr()[0] & 1)
    }
}

impl AsRef<Fq> for Fq {
    fn as_ref(&self) -> &Fq {
        self
    }
}

#[cfg(all(feature = "bits", not(target_pointer_width = "64")))]
type ReprBits = [u32; 8];

#[cfg(all(feature = "bits", target_pointer_width = "64"))]
type ReprBits = [u64; 4];

#[cfg(feature = "bits")]
#[cfg_attr(docsrs, doc(cfg(feature = "bits")))]
impl PrimeFieldBits for Fq {
    type ReprBits = ReprBits;

    fn to_le_bits(&self) -> FieldBits<Self::ReprBits> {
        let bytes = self.to_repr();

        #[cfg(not(target_pointer_width = "64"))]
        let limbs = [
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            u32::from_le_bytes(bytes[20..24].try_into().unwrap()),
            u32::from_le_bytes(bytes[24..28].try_into().unwrap()),
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        ];

        #[cfg(target_pointer_width = "64")]
        let limbs = [
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
        ];

        FieldBits::new(limbs)
    }

    fn char_le_bits() -> FieldBits<Self::ReprBits> {
        #[cfg(not(target_pointer_width = "64"))]
        {
            FieldBits::new(MODULUS_LIMBS_32)
        }

        #[cfg(target_pointer_width = "64")]
        FieldBits::new(MODULUS.0)
    }
}

#[cfg(feature = "sqrt-table")]
lazy_static! {
    // The perfect hash parameters are found by `squareroottab.sage` in zcash/pasta.
    #[cfg_attr(docsrs, doc(cfg(feature = "sqrt-table")))]
    static ref FQ_TABLES: SqrtTables<Fq> = SqrtTables::new(0x116A9E, 1206);
}

impl SqrtTableHelpers for Fq {
    fn pow_by_t_minus1_over2(&self) -> Self {
        let sqr = |x: Fq, i: u32| (0..i).fold(x, |x, _| x.square());

        let s10 = self.square();
        let s11 = s10 * self;
        let s111 = s11.square() * self;
        let s1001 = s111 * s10;
        let s1011 = s1001 * s10;
        let s1101 = s1011 * s10;
        let sa = sqr(*self, 129) * self;
        let sb = sqr(sa, 7) * s1001;
        let sc = sqr(sb, 7) * s1101;
        let sd = sqr(sc, 4) * s11;
        let se = sqr(sd, 6) * s111;
        let sf = sqr(se, 3) * s111;
        let sg = sqr(sf, 10) * s1001;
        let sh = sqr(sg, 4) * s1001;
        let si = sqr(sh, 5) * s1001;
        let sj = sqr(si, 5) * s1001;
        let sk = sqr(sj, 3) * s1001;
        let sl = sqr(sk, 4) * s1011;
        let sm = sqr(sl, 4) * s1011;
        let sn = sqr(sm, 5) * s11;
        let so = sqr(sn, 4) * self;
        let sp = sqr(so, 5) * s11;
        let sq = sqr(sp, 4) * s111;
        let sr = sqr(sq, 5) * s1011;
        let ss = sqr(sr, 3) * self;
        sqr(ss, 4) // st
    }

    fn get_lower_32(&self) -> u32 {
        // TODO: don't reduce, just hash the Montgomery form. (Requires rebuilding perfect hash table.)
        let tmp = Fq::montgomery_reduce(self.0[0], self.0[1], self.0[2], self.0[3], 0, 0, 0, 0);

        tmp.0[0] as u32
    }
}

impl WithSmallOrderMulGroup<3> for Fq {
    const ZETA: Self = Fq::from_raw([
        0x2aa9d2e050aa0e4f,
        0x0fed467d47c033af,
        0x511db4d81cf70f5a,
        0x06819a58283e528e,
    ]);
}

impl FromUniformBytes<64> for Fq {
    /// Converts a 512-bit little endian integer into
    /// a `Fq` by reducing by the modulus.
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Fq {
        Fq::from_u512([
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
            u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
            u64::from_le_bytes(bytes[32..40].try_into().unwrap()),
            u64::from_le_bytes(bytes[40..48].try_into().unwrap()),
            u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            u64::from_le_bytes(bytes[56..64].try_into().unwrap()),
        ])
    }
}

#[cfg(feature = "gpu")]
impl ec_gpu::GpuName for Fq {
    fn name() -> alloc::string::String {
        ec_gpu::name!()
    }
}

#[cfg(feature = "gpu")]
impl ec_gpu::GpuField for Fq {
    fn one() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&R.0[..])
    }

    fn r2() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&R2.0[..])
    }

    fn modulus() -> alloc::vec::Vec<u32> {
        crate::fields::u64_to_u32(&MODULUS.0[..])
    }
}

impl From<ScalarPrimitive<Pallas>> for Fq {
    fn from(value: ScalarPrimitive<Pallas>) -> Self {
        Self::from_uint_unchecked(*value.as_uint())
    }
}

impl From<&ScalarPrimitive<Pallas>> for Fq {
    fn from(value: &ScalarPrimitive<Pallas>) -> Self {
        Self::from_uint_unchecked(*value.as_uint())
    }
}

impl From<Fq> for ScalarPrimitive<Pallas> {
    fn from(value: Fq) -> Self {
        ScalarPrimitive::from(&value)
    }
}

impl From<&Fq> for ScalarPrimitive<Pallas> {
    fn from(value: &Fq) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            let mut out = [0u64; 4];
            out[..4].copy_from_slice(&value.to_raw());
            ScalarPrimitive::new(U256::from_words(out)).unwrap()
        }
        #[cfg(target_pointer_width = "32")]
        {
            let mut tmp = [0u64; 4];
            tmp[..4].copy_from_slice(&value.to_raw());
            let mut out = [0u32; 8];
            out[0] = tmp[0] as u32;
            out[1] = (tmp[0] >> 32) as u32;
            out[2] = tmp[1] as u32;
            out[3] = (tmp[1] >> 32) as u32;
            out[4] = tmp[2] as u32;
            out[5] = (tmp[2] >> 32) as u32;
            out[6] = tmp[3] as u32;
            out[7] = (tmp[3] >> 32) as u32;
            elliptic_curve::ScalarPrimitive::new(elliptic_curve::bigint::U256::from_words(out))
                .unwrap()
        }
    }
}

impl From<CurveBytes> for Fq {
    fn from(value: CurveBytes) -> Self {
        Self::from_uint_unchecked(elliptic_curve::bigint::U256::from_le_byte_array(value))
    }
}

impl From<Fq> for CurveBytes {
    fn from(value: Fq) -> Self {
        value.to_repr()
    }
}

impl From<U256> for Fq {
    fn from(value: U256) -> Self {
        Self::reduce(value)
    }
}

impl From<Fq> for U256 {
    fn from(value: Fq) -> Self {
        #[cfg(target_pointer_width = "64")]
        {
            let arr = value.to_raw();
            U256::from_words(arr)
        }
        #[cfg(target_pointer_width = "32")]
        {
            let tmp = value.to_raw();
            let mut out = [0u32; 8];
            out[0] = tmp[0] as u32;
            out[1] = (tmp[0] >> 32) as u32;
            out[2] = tmp[1] as u32;
            out[3] = (tmp[1] >> 32) as u32;
            out[4] = tmp[2] as u32;
            out[5] = (tmp[2] >> 32) as u32;
            out[6] = tmp[3] as u32;
            out[7] = (tmp[3] >> 32) as u32;
            U256::from_words(out)
        }
    }
}

impl From<U384> for Fq {
    fn from(value: U384) -> Self {
        Self::reduce(value)
    }
}

impl From<U512> for Fq {
    fn from(value: U512) -> Self {
        Self::reduce(value)
    }
}

impl FromUintUnchecked for Fq {
    type Uint = U256;

    fn from_uint_unchecked(uint: Self::Uint) -> Self {
        let mut out = [0u64; 4];
        #[cfg(target_pointer_width = "64")]
        {
            out.copy_from_slice(&uint.as_words()[..4]);
            Self::from_raw_unchecked(out)
        }
        #[cfg(target_pointer_width = "32")]
        {
            let words = uint.as_words();
            let mut i = 0;
            for index in out.iter_mut() {
                *index = (words[i + 1] as u64) << 32;
                *index |= words[i] as u64;
                i += 2;
            }
            Self::from_raw_unchecked(out)
        }
    }
}

impl Invert for Fq {
    type Output = CtOption<Self>;

    fn invert(&self) -> Self::Output {
        ff::Field::invert(self)
    }
}

impl IsHigh for Fq {
    fn is_high(&self) -> Choice {
        let mut borrow = 0;
        for i in 0..4 {
            let (_, b) = sbb(HALF_MODULUS.0[i], self.0[i], borrow);
            borrow = b;
        }
        ((borrow == u64::MAX) as u8).into()
    }
}

impl Reduce<U256> for Fq {
    type Bytes = CurveBytes;

    fn reduce(n: U256) -> Self {
        const MODULUS_256: NonZero<U256> = NonZero::<U256>::const_new(U256::from_be_hex(
            "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
        ))
        .0;
        let v = n % MODULUS_256;
        Self::from_uint_unchecked(v)
    }

    fn reduce_bytes(bytes: &Self::Bytes) -> Self {
        Self::reduce(U256::from_be_byte_array(*bytes))
    }
}

impl Reduce<U384> for Fq {
    type Bytes = GenericArray<u8, U48>;

    fn reduce(n: U384) -> Self {
        const MODULUS_384: NonZero<U384> = NonZero::<U384>::const_new(U384::from_be_hex("0000000000000000000000000000000040000000000000000000000000000000224698fc0994a8dd8c46eb2100000001")).0;
        let value = (n % MODULUS_384).resize::<{ U256::LIMBS }>();
        Self::from_uint_unchecked(value)
    }

    fn reduce_bytes(bytes: &Self::Bytes) -> Self {
        Self::reduce(U384::from_be_byte_array(*bytes))
    }
}

impl Reduce<U512> for Fq {
    type Bytes = GenericArray<u8, U64>;

    fn reduce(n: U512) -> Self {
        const MODULUS_512: NonZero<U512> = NonZero::<U512>::const_new(U512::from_be_hex("000000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000224698fc0994a8dd8c46eb2100000001")).0;
        let value = (n % MODULUS_512).resize::<{ U256::LIMBS }>();
        Self::from_uint_unchecked(value)
    }

    fn reduce_bytes(bytes: &Self::Bytes) -> Self {
        Self::reduce(U512::from_be_byte_array(*bytes))
    }
}

impl Shr<usize> for Fq {
    type Output = Self;

    fn shr(self, mut rhs: usize) -> Self::Output {
        // TODO: look for a more efficient method to do this
        let mut tmp = self;
        while rhs > 0 {
            tmp *= Self::TWO_INV;
            rhs -= 1;
        }
        tmp
    }
}

impl Shr<usize> for &Fq {
    type Output = Fq;

    fn shr(self, rhs: usize) -> Self::Output {
        *self >> rhs
    }
}

impl ShrAssign<usize> for Fq {
    fn shr_assign(&mut self, rhs: usize) {
        *self = *self >> rhs;
    }
}

impl frost_dkg::ScalarHash for Fq {
    fn hash_to_scalar(bytes: &[u8]) -> Self {
        const DST: &'static [u8] = b"PALLAS_XMD:BLAKE2B-512_RO_NUL_";
        Self::hash::<ExpandMsgXmd<blake2::Blake2b512>>(bytes, DST)
    }
}

#[test]
fn test_inv() {
    // Compute -(r^{-1} mod 2^64) mod 2^64 by exponentiating
    // by totient(2**64) - 1

    let mut inv = 1u64;
    for _ in 0..63 {
        inv = inv.wrapping_mul(inv);
        inv = inv.wrapping_mul(MODULUS.0[0]);
    }
    inv = inv.wrapping_neg();

    assert_eq!(inv, INV);
}

#[test]
fn test_sqrt() {
    // NB: TWO_INV is standing in as a "random" field element
    let v = (Fq::TWO_INV).square().sqrt().unwrap();
    assert!(v == Fq::TWO_INV || (-v) == Fq::TWO_INV);
}

#[test]
fn test_sqrt_32bit_overflow() {
    assert_eq!((Fq::from(5u32)).sqrt().is_none().unwrap_u8(), 1);
}

#[test]
fn test_pow_by_t_minus1_over2() {
    // NB: TWO_INV is standing in as a "random" field element
    let v = (Fq::TWO_INV).pow_by_t_minus1_over2();
    assert!(v == ff::Field::pow_vartime(&Fq::TWO_INV, &T_MINUS1_OVER2));
}

#[test]
fn test_sqrt_ratio_and_alt() {
    // (true, sqrt(num/div)), if num and div are nonzero and num/div is a square in the field
    let num = Fq::TWO_INV.square();
    let div = Fq::from(25u32);
    let div_inverse = Field::invert(&div).unwrap();
    let expected = Fq::TWO_INV * Field::invert(&Fq::from(5u32)).unwrap();
    let (is_square, v) = Fq::sqrt_ratio(&num, &div);
    assert!(bool::from(is_square));
    assert!(v == expected || (-v) == expected);

    let (is_square_alt, v_alt) = Fq::sqrt_alt(&(num * div_inverse));
    assert!(bool::from(is_square_alt));
    assert_eq!(v_alt, v);

    // (false, sqrt(ROOT_OF_UNITY * num/div)), if num and div are nonzero and num/div is a nonsquare in the field
    let num = num * Fq::ROOT_OF_UNITY;
    let expected = Fq::TWO_INV * Fq::ROOT_OF_UNITY * Field::invert(&Fq::from(5u32)).unwrap();
    let (is_square, v) = Fq::sqrt_ratio(&num, &div);
    assert!(!bool::from(is_square));
    assert!(v == expected || (-v) == expected);

    let (is_square_alt, v_alt) = Fq::sqrt_alt(&(num * div_inverse));
    assert!(!bool::from(is_square_alt));
    assert_eq!(v_alt, v);

    // (true, 0), if num is zero
    let num = Fq::ZERO;
    let expected = Fq::ZERO;
    let (is_square, v) = Fq::sqrt_ratio(&num, &div);
    assert!(bool::from(is_square));
    assert_eq!(v, expected);

    let (is_square_alt, v_alt) = Fq::sqrt_alt(&(num * div_inverse));
    assert!(bool::from(is_square_alt));
    assert_eq!(v_alt, v);

    // (false, 0), if num is nonzero and div is zero
    let num = (Fq::TWO_INV).square();
    let div = Fq::ZERO;
    let expected = Fq::ZERO;
    let (is_square, v) = Fq::sqrt_ratio(&num, &div);
    assert!(!bool::from(is_square));
    assert_eq!(v, expected);
}

#[test]
fn test_zeta() {
    assert_eq!(
        format!("{:?}", Fq::ZETA),
        "0x06819a58283e528e511db4d81cf70f5a0fed467d47c033af2aa9d2e050aa0e4f"
    );
    let a = Fq::ZETA;
    assert_ne!(a, Fq::ONE);
    let b = a * a;
    assert_ne!(b, Fq::ONE);
    let c = b * a;
    assert_eq!(c, Fq::ONE);
}

#[test]
fn test_root_of_unity() {
    assert_eq!(
        Fq::ROOT_OF_UNITY.pow_vartime(&[1 << Fq::S, 0, 0, 0]),
        Fq::ONE
    );
}

#[test]
fn test_inv_root_of_unity() {
    assert_eq!(
        Fq::ROOT_OF_UNITY_INV,
        Field::invert(&Fq::ROOT_OF_UNITY).unwrap()
    );
}

#[test]
fn test_inv_2() {
    assert_eq!(Fq::TWO_INV, Field::invert(&Fq::from(2u32)).unwrap());
}

#[test]
fn test_delta() {
    assert_eq!(Fq::DELTA, GENERATOR.pow(&[1u64 << Fq::S, 0, 0, 0]));
    assert_eq!(
        Fq::DELTA,
        Fq::MULTIPLICATIVE_GENERATOR.pow(&[1u64 << Fq::S, 0, 0, 0])
    );
}

#[cfg(not(target_pointer_width = "64"))]
#[test]
fn consistent_modulus_limbs() {
    for (a, &b) in MODULUS
        .0
        .iter()
        .flat_map(|&limb| {
            Some(limb as u32)
                .into_iter()
                .chain(Some((limb >> 32) as u32))
        })
        .zip(MODULUS_LIMBS_32.iter())
    {
        assert_eq!(a, b);
    }
}

#[test]
fn test_from_u512() {
    assert_eq!(
        Fq::from_raw([
            0xe22bd0d1b22cc43e,
            0x6b84e5b52490a7c8,
            0x264262941ac9e229,
            0x27dcfdf361ce4254
        ]),
        Fq::from_u512([
            0x64a80cce0b5a2369,
            0x84f2ef0501bc783c,
            0x696e5e63c86bbbde,
            0x924072f52dc6cc62,
            0x8288a507c8d61128,
            0x3b2efb1ef697e3fe,
            0x75a4998d06855f27,
            0x52ea589e69712cc0
        ])
    );
}
