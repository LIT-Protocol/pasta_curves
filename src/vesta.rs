//! The Vesta and iso-Vesta elliptic curve groups.

use super::{Eq, EqAffine, Fp, Fq};
use elliptic_curve::bigint::{ArrayEncoding, ByteArray, U256};
use elliptic_curve::consts::U32;
use elliptic_curve::point::PointCompression;
use elliptic_curve::scalar::ScalarBits;
use elliptic_curve::{Curve, CurveArithmetic, FieldBytes, FieldBytesEncoding};

/// The base field of the Vesta and iso-Vesta curves.
pub type Base = Fq;

/// The scalar field of the Vesta and iso-Vesta curves.
pub type Scalar = Fp;

/// A Vesta point in the projective coordinate space.
pub type Point = Eq;

/// A Vesta point in the affine coordinate space (or the point at infinity).
pub type Affine = EqAffine;

/// Vesta curve.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Vesta;

unsafe impl Send for Vesta {}

unsafe impl Sync for Vesta {}

impl Curve for Vesta {
    type FieldBytesSize = U32;
    type Uint = U256;

    const ORDER: Self::Uint =
        U256::from_be_hex("40000000000000000000000000000000224698fc094cf91b992d30ed00000001");
}

impl elliptic_curve::PrimeCurve for Vesta {}

impl PointCompression for Vesta {
    const COMPRESS_POINTS: bool = true;
}

impl CurveArithmetic for Vesta {
    type AffinePoint = EqAffine;
    type ProjectivePoint = Eq;
    type Scalar = Fp;
}

/// Bytes of the Vesta field
pub type VestaFieldBytes = FieldBytes<Vesta>;
/// Scalar bits of the Vesta scalar
pub type VestaScalarBits = ScalarBits<Vesta>;
/// Non-zero scalar of the Vesta scalar
pub type VestaNonZeroScalar = elliptic_curve::NonZeroScalar<Vesta>;
impl FieldBytesEncoding<Vesta> for U256 {
    fn decode_field_bytes(field_bytes: &FieldBytes<Vesta>) -> Self {
        let data = ByteArray::<U256>::from_slice(field_bytes);
        U256::from_le_byte_array(*data)
    }

    fn encode_field_bytes(&self) -> FieldBytes<Vesta> {
        let mut data = VestaFieldBytes::default();
        data.copy_from_slice(&self.to_le_byte_array()[..]);
        data
    }
}

#[cfg(feature = "alloc")]
#[test]
fn test_map_to_curve_simple_swu() {
    use crate::arithmetic::CurveExt;
    use crate::curves::IsoEq;
    use crate::hashtocurve::map_to_curve_simple_swu;

    // The zero input is a special case.
    let p: IsoEq = map_to_curve_simple_swu::<Fq, Eq, IsoEq>(&Fq::ZERO, Eq::THETA, Eq::Z);
    let (x, y, z) = p.jacobian_coordinates();

    assert_eq!(
        format!("{:?}", x),
        "0x2ccc4c6ec2660e5644305bc52527d904d408f92407f599df8f158d50646a2e78"
    );
    assert_eq!(
        format!("{:?}", y),
        "0x29a34381321d13d72d50b6b462bb4ea6a9e47393fa28a47227bf35bc0ee7aa59"
    );
    assert_eq!(
        format!("{:?}", z),
        "0x0b851e9e579403a76df1100f556e1f226e5656bdf38f3bf8601d8a3a9a15890b"
    );

    let p: IsoEq = map_to_curve_simple_swu::<Fq, Eq, IsoEq>(&Fq::ONE, Eq::THETA, Eq::Z);
    let (x, y, z) = p.jacobian_coordinates();

    assert_eq!(
        format!("{:?}", x),
        "0x165f8b71841c5abc3d742ec13fb16f099d596b781e6f5c7d0b6682b1216a8258"
    );
    assert_eq!(
        format!("{:?}", y),
        "0x0dadef21de74ed7337a37dd74f126a92e4df73c3a704da501e36eaf59cf03120"
    );
    assert_eq!(
        format!("{:?}", z),
        "0x0a3d6f6c1af02bd9274cc0b80129759ce77edeef578d7de968d4a47d39026c82"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn test_hash_to_curve() {
    use crate::arithmetic::CurveExt;

    // This test vector is chosen so that the first map_to_curve_simple_swu takes the gx1 non-square
    // "branch" and the second takes the gx1 square "branch" (opposite to the Pallas test vector).
    let hash = Point::hash_to_curve("z.cash:test");
    let p: Point = hash(b"hello");
    let (x, y, z) = p.jacobian_coordinates();

    assert_eq!(
        format!("{:?}", x),
        "0x12763505036e0e1a6684b7a7d8d5afb7378cc2b191a95e34f44824a06fcbd08e"
    );
    assert_eq!(
        format!("{:?}", y),
        "0x0256eafc0188b79bfa7c4b2b393893ddc298e90da500fa4a9aee17c2ea4240e6"
    );
    assert_eq!(
        format!("{:?}", z),
        "0x1b58d4aa4d68c3f4d9916b77c79ff9911597a27f2ee46244e98eb9615172d2ad"
    );
}
