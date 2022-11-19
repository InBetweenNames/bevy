use crate::ThinSimdAlignedSlicePtr;

use core::{
    cell::UnsafeCell,
    char::MAX,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::{Index, IndexMut},
};

use crate::bytemuck;

use elain::{Align, Alignment};

use bevy_math::Vec4;

/*
NOTE: We define this constant here as both [`bevy_ptr`] and [`bevy_ecs`] need to know about it.

If this is a problem, this can be replaced with code that looks something like the following:

    #[cfg(all(any(target_feature = "avx"), not(target_feature = "avx512f")))]
    pub const MAX_SIMD_ALIGNMENT: usize = 32;

    #[cfg(any(target_feature = "avx512f"))]
    pub const MAX_SIMD_ALIGNMENT: usize = 64;

    //All platforms get 16-byte alignment on tables guaranteed.
    #[cfg(not(any(target_feature = "avx512f")))]
    pub const MAX_SIMD_ALIGNMENT: usize = 16;
*/
/// The maximum SIMD alignment for a given target.
/// `MAX_SIMD_ALIGNMENT` is 64 for the following reasons:
///  1. This ensures that table columns are aligned to cache lines on x86
///  2. 64 is the maximum alignment required to use all instructions on all known CPU architectures.
///     This simplifies greatly handling cross platform alignment on a case by case basis; by aligning to the worst case, we align for all cases
///  3. The overhead of aligning columns to 64 bytes is very small as columns will in general be much larger than this

//Must be greater than zero!
pub const MAX_SIMD_ALIGNMENT: usize = 64;

//TODO: AoSoA representations

//TODO: when possible, compute alignments automatically using the GCD (requires generic const expressions) and ensure the
//batch lambda accepts arguments of AlignedBatchTrait<T, N> directly. This will let the alignment be automatically computed
//and allow different query elements to have different alignments.

/// Compute batch alignment for batch of [T; N]
pub const fn compute_alignment<T, const N: usize>() -> usize {
    gcd::binary_usize(N * core::mem::size_of::<T>(), MAX_SIMD_ALIGNMENT)
}

/// Semantically, this type represents an array of `[T; N]` aligned to at least `MIN_ALIGN` bytes.
/// It is intended for writing vectorized queries over the Bevy ECS.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AlignedBatch<T, const N: usize>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    _align: Align<{ compute_alignment::<T, N>() }>,
    batch: [T; N],
}

impl<T, const N: usize> AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    //These only make sense when a component is #[repr(transparent)].
    //For example, if you had a repr(transparent) Position component that contained a Vec3, it
    //would be semantically valid to get a reference to the inner component.
    //The TransparentWrapper unsafe trait from the bytemuck crate is used to make this process usable in user code.

    /// If `T` is `repr(transparent)`, then `as_inner` can be used to get a shared reference to an "inner view" of the batch.
    /// To use this, implement [`bytemuck::TransparentWrapper`]  for your component.
    /// For example, if you had a `repr(transparent)` `Position` component that contained a [`bevy_math::Vec3`], you could treat a batch of `Position` as a batch of [`bevy_math::Vec3`].
    #[inline]
    pub fn as_inner<Inner>(&self) -> &AlignedBatch<Inner, N>
    where
        T: bytemuck::TransparentWrapper<Inner>,
        Align<{ compute_alignment::<Inner, N>() }>: Alignment,
    {
        // SAFETY:
        //
        // * T is repr(transparent), with inner type Inner
        // * $batch_type<T, N> is repr(transparent)
        // * $batch_type<Inner, N> is repr(transparent)
        // * Therefore $batch_type<T, N> and $batch_type<Inner, N>
        // * Since self is a shared reference, creating more shared references to the same memory is OK.
        unsafe { &*(self as *const Self as *const AlignedBatch<Inner, N>) }
    }

    /// If `T` is `repr(transparent)`, then `as_inner_mut` can be used to get a mutable reference to an "inner view" of the batch.
    /// To use this, implement [`bytemuck::TransparentWrapper`] for your component.
    /// For example, if you had a `repr(transparent)` `Position` component that contained a [`bevy_math::Vec3`], you could treat a batch of `Position` as a batch of [`bevy_math::Vec3`].
    #[inline]
    pub fn as_inner_mut<Inner>(&mut self) -> &mut AlignedBatch<Inner, N>
    where
        T: bytemuck::TransparentWrapper<Inner>,
        Align<{ compute_alignment::<Inner, N>() }>: Alignment,
    {
        // SAFETY:
        //
        // * Recommended pattern from the Rust book: https://doc.rust-lang.org/std/mem/fn.transmute.html
        //   * Section: "turning an &mut T into an &mut U"
        // * T is repr(transparent), with inner type Inner
        // * $batch_type<T, N> is repr(transparent)
        // * $batch_type<Inner, N> is repr(transparent)
        // * Therefore $batch_type<T, N> and $batch_type<Inner, N>
        unsafe { &mut *(self as *mut Self as *mut AlignedBatch<Inner, N>) }
    }

    /// Constructs a new batch with the result of `func` mapped over the components of this batch.
    //TODO: this doesn't optimize very well...
    #[inline]
    pub fn map<U, F: Fn(T) -> U>(self, func: F) -> AlignedBatch<U, N>
    where
        Align<{ compute_alignment::<U, N>() }>: Alignment,
    {
        AlignedBatch::<U, N> {
            _align: Align::<{ compute_alignment::<U, N>() }>::NEW,
            batch: self.batch.map(func),
        }
    }

    /// Retrieve a shared reference to this batch as an array of `[T; N]`
    /// You can use this to destructure your batch into elements if needed.
    #[inline]
    pub fn as_array(&self) -> &[T; N] {
        self.as_ref()
    }

    /// Retrieve a mutable reference to this batch as an array of `[T; N]`.
    /// You can use this to modify elements of your batch.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut [T; N] {
        self.as_mut()
    }

    /// Convert this batch into an array of `[T; N]`.
    /// A convenience function, as all batches implement [`From`] and [`Into`] for `[T; N]`.
    #[inline]
    pub fn into_array(self) -> [T; N] {
        self.into()
    }

    //TODO: add support for as_simd()/into_simd() when SIMD is stabilized!
}

impl<T, const N: usize> From<[T; N]> for AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    #[inline]
    fn from(batch: [T; N]) -> Self {
        Self {
            _align: Align::NEW,
            batch,
        }
    }
}

impl<T, const N: usize> From<AlignedBatch<T, N>> for [T; N]
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    #[inline]
    fn from(v: AlignedBatch<T, N>) -> Self {
        v.batch
    }
}

impl<T, const N: usize> Index<usize> for AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &<Self as Index<usize>>::Output {
        &self.batch[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut <Self as Index<usize>>::Output {
        &mut self.batch[i]
    }
}

impl<T, const N: usize> AsRef<[T; N]> for AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    fn as_ref(&self) -> &[T; N] {
        &self.batch
    }
}

impl<T, const N: usize> AsMut<[T; N]> for AlignedBatch<T, N>
where
    Align<{ compute_alignment::<T, N>() }>: Alignment,
{
    fn as_mut(&mut self) -> &mut [T; N] {
        &mut self.batch
    }
}

// TODO: when stable, replace `ALIGN` with an `Alignment` enum
// OR when general const expressions are stable, replace with a trait constraint `Alignment`.
// Do the same for batch sizes. For now, this is the best we can do.

//Convenience impls that can go away when SIMD is stabilized
impl AsRef<Vec4> for AlignedBatch<f32, 4> {
    #[inline]
    fn as_ref(&self) -> &Vec4 {
        // SAFETY:
        // * Alignment of Vec4 is 16
        // * Alignment of Self is 16
        // * Self is repr(C) and therefore can be treated as an [f32; 4]
        // * Vec4 is repr(transparent) and can be treated as an [f32; 4] (it is an __mm128)
        // * Only shared refs exist
        // Therefore this cast is sound.
        unsafe { &*(self as *const Self as *const Vec4) }
    }
}

impl AsMut<Vec4> for AlignedBatch<f32, 4> {
    #[inline]
    fn as_mut(&mut self) -> &mut Vec4 {
        // SAFETY:
        // * Alignment of Vec4 is 16
        // * Alignment of Self is 16
        // * Self is repr(C) and therefore can be treated as an [f32; 4]
        // * Vec4 is repr(transparent) and can be treated as an [f32; 4] (it is an __mm128)
        // * &mut T to &mut U pattern used from the Rust book to ensure soundness when casting mutable refs
        // Therefore this cast is sound.
        unsafe { &mut *(self as *mut Self as *mut Vec4) }
    }
}

impl<'a, T> ThinSimdAlignedSlicePtr<'a, T> {
    /// Indexes the slice without doing bounds checks with a batch size of `N`.
    /// The batch size in bytes must be a multiple of `ALIGN`.
    /// A compile-time error will be given if the alignment requirements cannot be met with the given parameters.
    ///
    /// # Safety
    /// `index` must be in-bounds.
    /// `index` must be a multiple of `N`.
    #[inline]
    unsafe fn get_batch_aligned_raw<const N: usize>(
        self,
        index: usize,
        _len: usize,
    ) -> *const AlignedBatch<T, N>
    where
        Align<{ compute_alignment::<T, N>() }>: Alignment,
    {
        #[cfg(debug_assertions)]
        debug_assert!(index + N < self.len);
        #[cfg(debug_assertions)]
        debug_assert_eq!(_len, self.len);
        #[cfg(debug_assertions)]
        debug_assert_eq!(index % N, 0);

        let off_ptr = self.ptr.as_ptr().add(index);

        //NOTE: ZSTs may cause this "slice" to point into nothingness.
        //This sounds dangerous, but won't cause harm as nothing
        //will actually access anything "in the slice"

        //TODO: when pointer_is_aligned is standardized, we can just use ptr::is_aligned()
        #[cfg(debug_assertions)]
        debug_assert_eq!(
            off_ptr as usize % core::mem::align_of::<AlignedBatch<T, N>>(),
            0
        );

        //SAFETY: off_ptr is not null
        off_ptr as *const AlignedBatch<T, N>
    }

    /// Indexes the slice without doing bounds checks with a batch size of N.
    ///
    /// # Safety
    /// `index` must be in-bounds.
    /// `index` must be suitably aligned.
    #[inline]
    pub unsafe fn get_batch_aligned<const N: usize>(
        self,
        index: usize,
        len: usize,
    ) -> &'a AlignedBatch<T, N>
    where
        Align<{ compute_alignment::<T, N>() }>: Alignment,
    {
        &(*self.get_batch_aligned_raw(index, len))
    }
}

impl<'a, T> ThinSimdAlignedSlicePtr<'a, UnsafeCell<T>> {
    /// Indexes the slice without doing bounds checks with a batch size of `N`.
    /// The semantics are like `UnsafeCell` -- you must ensure the aliasing constraints are met.
    ///
    /// # Safety
    /// `index` must be in-bounds.
    /// `index` must be a multiple of `N`.
    ///  No other references exist to the batch of size `N` at `index`
    #[inline]
    pub unsafe fn get_batch_aligned_deref_mut<const N: usize>(
        self,
        index: usize,
        len: usize,
    ) -> &'a mut AlignedBatch<T, N>
    where
        Align<{ compute_alignment::<T, N>() }>: Alignment,
    {
        &mut *(self.as_deref().get_batch_aligned_raw::<N>(index, len) as *mut AlignedBatch<T, N>)
    }

    /// Indexes the slice without doing bounds checks with a batch size of `N`.
    /// The semantics are like `UnsafeCell` -- you must ensure the aliasing constraints are met.
    ///
    /// # Safety
    /// `index` must be in-bounds.
    /// `index` must be a multiple of `N`.
    /// No mutable references exist to the batch of size `N` at `index`
    #[inline]
    pub unsafe fn get_batch_aligned_deref<const N: usize>(
        self,
        index: usize,
        len: usize,
    ) -> &'a AlignedBatch<T, N>
    where
        Align<{ compute_alignment::<T, N>() }>: Alignment,
    {
        &*(self.as_deref().get_batch_aligned_raw::<N>(index, len))
    }
}

//Inspired from: https://github.com/rust-lang/rust/issues/57775#issuecomment-1098001375
struct Assert<T, const N: usize, const MIN_ALIGN: usize> {
    _marker: PhantomData<T>,
}

impl<T, const N: usize, const MIN_ALIGN: usize> Assert<T, N, MIN_ALIGN> {
    const YOUR_BATCH_SIZE_IS_NOT_A_MULTIPLE_OF_ALIGN: () =
        assert!((N * core::mem::size_of::<T>()) % MIN_ALIGN == 0);
}
