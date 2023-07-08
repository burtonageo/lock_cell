// Copyright (c) 2023, George Burton <burtonageo@gmail.com>
//
// SPDX-License-Identifier: 0BSD

#![cfg_attr(not(feature = "enable_std"), no_std)]
#![warn(
    clippy::cargo,
    clippy::complexity,
    clippy::pedantic,
    clippy::perf,
    clippy::style,
    clippy::suspicious,
    clippy::undocumented_unsafe_blocks
)]

//! This crate provides the [`LockCell<T>`] and other supportings types.
//!
//! A `LockCell` is a cell type which provides dynamic mutation using interior
//! mutability. It is similar to [`RefCell<T>`], except that it only allows
//! a single borrow type (a lock). Locking a `LockCell` allows mutating its
//! contents freely.
//!
//! A `LockCell` can only be used in a single threaded context - it cannot be shared
//! across different threads. Generally, a `LockCell` will be stored in a [`Rc<T>`]
//! so that it can be shared.
//!
//! Whether you use a `LockCell` or a `RefCell` depends on the structure and behavior of
//! your program. Generally, if you have a lot of writers and readers, using a `LockCell`
//! may be better, as it ensures that writers are less likely to be starved.
//!
//! The [`Sync`] equivalent of a `LockCell` is [`Mutex<T>`].
//!
//! # Features
//!
//! * The `enable_std` feature enables the standard library. This provides an implementation of
//!   [`std::error::Error`] for the [`TryLockError`] type. This feature is enabled by default.
//!
//! * The `debug_lockcell` feature tracks the location of each `lock()` call in the `LockCell`,
//!   allowing the developer to compare the first lock location in their file to the panicking
//!   lock location, aiding in debugging.
//!
//! [`LockCell<T>`]: ./struct.LockCell.html
//! [`RefCell<T>`]: http://doc.rust-lang.org/std/cell/struct.RefCell.html
//! [`Rc<T>`]: https://doc.rust-lang.org/std/rc/struct.Rc.html
//! [`Mutex<T>`]: http://doc.rust-lang.org/std/sync/struct.Mutex.html
//! [`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html
//! [`std::error::Error`]: https://doc.rust-lang.org/std/error/trait.Error.html
//! [`TryLockError`]: ./struct.TryLockError.html

#[cfg(feature = "debug_lockcell")]
use core::panic::Location;
use core::{
    borrow::{Borrow, BorrowMut},
    cell::{Cell, UnsafeCell},
    cmp::{Ordering, PartialEq, PartialOrd},
    convert::{AsMut, AsRef, TryFrom},
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut, FnOnce},
    str::FromStr,
};
#[cfg(feature = "enable_std")]
use std::error::Error as StdError;

/// A mutable memory location with dynamically checked borrow rules.
///
/// See the [module level documentation] for more.
///
/// [module level documentation]: ./index.html
pub struct LockCell<T: ?Sized> {
    /// Used to track the lock state of the `LockCell`.
    is_locked: Cell<bool>,
    /// Stores where the `LockCell` was first locked. This is used as
    /// part of the `debug_lockcell` feature to help debug double locks.
    #[cfg(feature = "debug_lockcell")]
    first_locked_at: Cell<Option<&'static Location<'static>>>,
    /// The inner value of the `LockCell`.
    value: UnsafeCell<T>,
}

impl<T> LockCell<T> {
    /// Create a new `LockCell` with the given `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new("I could be anything!".to_string());
    /// # let _ = cell;
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            is_locked: Cell::new(false),
            #[cfg(feature = "debug_lockcell")]
            first_locked_at: Cell::new(None),
            value: UnsafeCell::new(value),
        }
    }

    /// Consumes the `LockCell`, returning the inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    ///
    /// let five = cell.into_inner();
    ///
    /// assert_eq!(five, 5);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }

    /// Swaps the wrapped values of `self` and `rhs`.
    ///
    /// This function corresponds to [`std::mem::swap`].
    ///
    /// # Panics
    ///
    /// Panics if either `LockCell` is locked, or if `self` and `rhs` point to the same
    /// `LockCell`.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell_1 = LockCell::new(3);
    /// let cell_2 = LockCell::new(24);
    ///
    /// cell_1.swap(&cell_2);
    ///
    /// assert_eq!(cell_1.into_inner(), 24);
    /// assert_eq!(cell_2.into_inner(), 3);
    /// # }
    /// ```
    ///
    /// [`std::mem::swap`]: https://doc.rust-lang.org/std/mem/fn.swap.html
    #[track_caller]
    #[inline]
    pub fn swap(&self, rhs: &LockCell<T>) {
        mem::swap(&mut *self.lock(), &mut *rhs.lock());
    }

    /// Sets the value in this `LockCell` to `new_value`, returning the previous value
    /// in the `LockCell`.
    ///
    /// # Panics
    ///
    /// This method will panic if the cell is locked.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    ///
    /// let old_value = cell.replace(6);
    ///
    /// assert_eq!(old_value, 5);
    ///
    /// assert_eq!(cell.into_inner(), 6);
    /// # }
    /// ```
    #[inline]
    #[track_caller]
    pub fn replace(&self, new_value: T) -> T {
        let mut lock = self.lock();
        mem::replace(&mut *lock, new_value)
    }

    /// Replaces the wrapped value with a new value computed from the function `f`,
    /// returning the old value without deinitializing either.
    ///
    /// # Panics
    ///
    /// This method will panic if the `LockCell` is locked.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    /// let old_value = cell.replace_with(|old| {
    ///     *old += 1;
    ///     *old + 1
    /// });
    ///
    /// assert_eq!(old_value, 6);
    ///
    /// assert_eq!(cell.into_inner(), 7);
    /// # }
    /// ```
    #[inline]
    #[track_caller]
    pub fn replace_with<F>(&self, f: F) -> T
    where
        F: FnOnce(&mut T) -> T,
    {
        let mut lock = self.lock();
        let replacement = f(&mut *lock);
        mem::replace(&mut *lock, replacement)
    }

    /// Replaces the value in this `LockCell` with the [`Default::default()`] value,
    /// returning the previous value in the `LockCell`.
    ///
    /// # Panics
    ///
    /// This method will panic if the cell is locked.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    ///
    /// let old_value = cell.take();
    ///
    /// assert_eq!(old_value, 5);
    ///
    /// assert_eq!(cell.into_inner(), 0);
    /// # }
    /// ```
    ///
    /// [`Default::default()`]: https://doc.rust-lang.org/std/default/trait.Default.html
    #[inline]
    #[track_caller]
    pub fn take(&self) -> T
    where
        T: Default,
    {
        self.replace(Default::default())
    }
}

impl<T: ?Sized> LockCell<T> {
    /// Attempt to lock the `LockCell`.
    ///
    /// # Notes
    ///
    /// If this `LockCell` is not locked, the function succeeds and will return a
    /// guard which provides mutable access to the inner value.
    ///
    /// # Errors
    ///
    /// If the `LockCell` is already locked, this function will fail and will
    /// return a [`TryLockError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use lock_cell::{LockCell, TryLockError};
    /// # fn main() -> Result<(), TryLockError> {
    /// let cell = LockCell::new(21);
    ///
    /// let first_access = cell.try_lock();
    /// assert!(first_access.is_ok());
    ///
    /// let first_lock = first_access?;
    /// assert_eq!(*first_lock, 21);
    ///
    /// let second_access = cell.try_lock();
    /// assert!(second_access.is_err());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// [`TryLockError`]: ./struct.TryLockError.html
    #[inline]
    #[track_caller]
    pub fn try_lock(&self) -> Result<LockGuard<'_, T>, TryLockError> {
        if self.is_locked.replace(true) {
            return Err(TryLockError::new(self));
        }

        #[cfg(feature = "debug_lockcell")]
        {
            self.first_locked_at.set(Some(Location::caller()));
        }

        Ok(LockGuard {
            value: self.value.get(),
            is_locked: &self.is_locked,
            #[cfg(feature = "debug_lockcell")]
            locked_at: &self.first_locked_at,
            _boo: PhantomData,
        })
    }

    /// Lock the given `LockCell`, returning a [`LockGuard`] which can be used to access
    /// the value.
    ///
    /// The `LockCell` will be locked until the returned [`LockGuard`] goes out of scope.
    /// The cell can only have a single lock at a time active.
    ///
    /// # Panics
    ///
    /// This method will panic if the `LockCell` is already locked.
    ///
    /// To avoid this, you can use the [`try_lock()`] method to return a `Result` to
    /// check if the lock succeeded, or you can use the [`is_locked()`] method to check
    /// ahead of time if the lock will succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new("Hello".to_string());
    ///
    /// let lock = cell.lock();
    ///
    /// assert_eq!(&*lock, "Hello");
    /// # }
    /// ```
    ///
    /// [`LockGuard`]: ./struct.LockGuard.html
    /// [`try_lock()`]: ./struct.LockCell.html#method.try_lock
    /// [`is_locked()`]: ./struct.LockCell.html#method.is_locked
    #[inline]
    #[track_caller]
    pub fn lock(&self) -> LockGuard<'_, T> {
        self.try_lock().expect("already locked")
    }

    /// Returns whether this `LockCell` is currently locked.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    ///
    /// assert!(!cell.is_locked());
    ///
    /// let lock = cell.lock();
    ///
    /// assert!(cell.is_locked());
    /// # let _ = lock;
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn is_locked(&self) -> bool {
        self.is_locked.get()
    }

    /// Provides mutable access to the inner value.
    ///
    /// As this requires exclusive access to the `LockCell`, no locking is
    /// required to provide exclusive access to the value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let mut cell = LockCell::new(54);
    ///
    /// *cell.get_mut() = 20;
    ///
    /// assert_eq!(cell.into_inner(), 20);
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
    }

    /// Return a raw pointer to the underlying data in this `LockCell`.
    ///
    /// # Notes
    ///
    /// This function does not lock the `LockCell`. Therefore, any mutations made through
    /// the returned pointer must be synchronized in some other way, or undefined behaviour
    /// may occur.
    /// 
    /// # Examples
    ///
    /// ```
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let cell = LockCell::new(5);
    ///
    /// let ptr = cell.as_ptr();
    /// # }
    /// ```
    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.value.get()
    }

    /// Resets the lock state, in case that any [`LockGuard`]s have been leaked.
    ///
    /// This method takes `self` by `&mut` to ensure that there are no other borrows
    /// of the `LockCell` in flight.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::mem;
    /// use lock_cell::LockCell;
    /// # fn main() {
    /// let mut cell = LockCell::new(12);
    ///
    /// let mut lock = cell.lock();
    ///
    /// *lock = 54;
    ///
    /// mem::forget(lock);
    ///
    /// assert!(cell.is_locked());
    ///
    /// cell.reset_lock();
    ///
    /// assert!(!cell.is_locked());
    ///
    /// assert_eq!(cell.into_inner(), 54);
    /// # }
    /// ```
    ///
    /// [`LockGuard`]: ./struct.LockGuard.html
    #[inline]
    pub fn reset_lock(&mut self) -> &mut T {
        self.is_locked.set(false);

        #[cfg(feature = "debug_lockcell")]
        {
            self.first_locked_at.set(None);
        }

        self.get_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for LockCell<T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lock_result = self.try_lock();
        let value: &dyn fmt::Debug = if let Ok(value) = lock_result.as_deref() {
            value
        } else {
            struct LockedPlaceholder;
            impl fmt::Debug for LockedPlaceholder {
                #[inline]
                fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmtr.write_str("<locked>")
                }
            }

            const PLACEHOLDER: LockedPlaceholder = LockedPlaceholder;
            &PLACEHOLDER
        };

        fmtr.debug_struct("LockCell").field("value", value).finish()
    }
}

impl<T: Default> Default for LockCell<T> {
    #[inline]
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<T: FromStr> FromStr for LockCell<T> {
    type Err = T::Err;
    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        T::from_str(s).map(LockCell::new)
    }
}

impl<T> From<T> for LockCell<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

/// A `LockGuard` provides exclusive access to the inner value of a [`LockCell<T>`].
///
/// An instance of this type can be constructed from a `LockCell` using the [`LockCell::try_lock()`]
/// or [`LockCell::lock()`] methods.
///
/// See the [module level documentation] for more.
///
/// [`LockCell<T>`]: ./struct.LockCell.html
/// [`LockCell::try_lock()`]: ./struct.LockCell.html#method.try_lock
/// [`LockCell::lock()`]: ./struct.LockCell.html#method.lock
/// [module level documentation]: ./index.html
#[must_use]
pub struct LockGuard<'lock, T: ?Sized> {
    /// The location of the original value in the `LockCell`.
    value: *mut T,
    /// The lock state of the `LockCell`.
    is_locked: &'lock Cell<bool>,
    /// The location where the original `LockCell` was first locked.
    ///
    /// The `LockGuard` will reset this value when it is dropped.
    #[cfg(feature = "debug_lockcell")]
    locked_at: &'lock Cell<Option<&'static Location<'static>>>,
    /// Phantom data.
    _boo: PhantomData<&'lock UnsafeCell<T>>,
}

impl<'lock, T: ?Sized> LockGuard<'lock, T> {
    /// Applies the given `func` to the contents `LockGuard` to return a new `LockGuard` which
    /// points to a sub-part of the original data.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::{LockCell, LockGuard};
    /// # fn main() {
    /// let cell = LockCell::<(i32, i32)>::default();
    /// let lock = cell.lock();
    ///
    /// let mut value = LockGuard::map(lock, |(_, ref mut val)| val);
    /// *value = 21;
    /// drop(value);
    ///
    /// let tuple = cell.into_inner();
    /// assert_eq!(tuple.1, 21);
    /// # }
    /// ```
    #[inline]
    pub fn map<F, U: ?Sized>(this: Self, func: F) -> LockGuard<'lock, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let mut this = ManuallyDrop::new(this);

        LockGuard {
            // SAFETY:
            // The `value` ptr has been created from a valid `LockCell`, so it always valid.
            value: unsafe { func(&mut *this.value) } as *mut _,
            #[cfg(feature = "debug_lockcell")]
            locked_at: this.locked_at,
            is_locked: this.is_locked,
            _boo: PhantomData,
        }
    }

    /// Applies the given `func` to the contents of `LockGuard` to return an optional reference
    /// to a part of the original data.
    ///
    /// # Errors
    ///
    /// If `func` returns `None`, then the original guard will be returned in the `Err` variant
    /// of the return value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lock_cell::{LockCell, LockGuard};
    /// # fn main() {
    /// let cell = LockCell::new(Some(0));
    /// let lock = cell.lock();
    ///
    /// let mut value = match LockGuard::filter_map(lock, |value| value.as_mut()) {
    ///     Ok(inner) => inner,
    ///     Err(old_lock) => panic!("Unexpectedly empty value: {:?}", old_lock),
    /// };
    /// *value = 5;
    /// drop(value);
    ///
    /// let old_value = cell.replace(None);
    /// assert_eq!(old_value, Some(5));
    ///
    /// let lock = cell.lock();
    /// let value = match LockGuard::filter_map(lock, |value| value.as_mut()) {
    ///     Ok(inner) => panic!("Unexpected value is present: {:?}", inner),
    ///     Err(old_lock) => old_lock,
    /// };
    ///
    /// assert_eq!(*value, None);
    /// # }
    /// ```
    #[inline]
    pub fn filter_map<F, U: ?Sized>(this: Self, func: F) -> Result<LockGuard<'lock, U>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
    {
        let mut this = ManuallyDrop::new(this);

        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        let value = match unsafe { func(&mut *this.value) } {
            Some(value) => value as *mut _,
            _ => return Err(ManuallyDrop::into_inner(this)),
        };

        Ok(LockGuard {
            // SAFETY: value has been created from a reference so it is always valid.
            value: unsafe { &mut *value },
            #[cfg(feature = "debug_lockcell")]
            locked_at: this.locked_at,
            is_locked: this.is_locked,
            _boo: PhantomData,
        })
    }
}

impl<'lock, T> TryFrom<&'lock LockCell<T>> for LockGuard<'lock, T> {
    type Error = TryLockError;
    #[inline]
    #[track_caller]
    fn try_from(lock_cell: &'lock LockCell<T>) -> Result<Self, Self::Error> {
        lock_cell.try_lock()
    }
}

impl<'lock, T: ?Sized> Drop for LockGuard<'lock, T> {
    #[inline]
    fn drop(&mut self) {
        self.is_locked.set(false);
        #[cfg(feature = "debug_lockcell")]
        {
            self.locked_at.set(None);
        }
    }
}

impl<'lock, T: ?Sized> AsRef<T> for LockGuard<'lock, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &*self.value }
    }
}

impl<'lock, T: ?Sized> AsMut<T> for LockGuard<'lock, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &mut *self.value }
    }
}

impl<'lock, T: ?Sized> Borrow<T> for LockGuard<'lock, T> {
    #[inline]
    fn borrow(&self) -> &T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &*self.value }
    }
}

impl<'lock, T: ?Sized> BorrowMut<T> for LockGuard<'lock, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &mut *self.value }
    }
}

impl<'lock, T: ?Sized> Deref for LockGuard<'lock, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &*self.value }
    }
}

impl<'lock, T: ?Sized> DerefMut for LockGuard<'lock, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY:
        // The `value` ptr has been created from a valid `LockCell`, so it always valid.
        unsafe { &mut *self.value }
    }
}

impl<'lock, T: fmt::Debug + ?Sized> fmt::Debug for LockGuard<'lock, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("LockGuard").field("value", self).finish()
    }
}

impl<'lock, T: fmt::Display + ?Sized> fmt::Display for LockGuard<'lock, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as fmt::Display>::fmt(self, fmtr)
    }
}

impl<'lock, T: ?Sized + Hash> Hash for LockGuard<'lock, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        <T as Hash>::hash(self, state);
    }
}

impl<'lock, T: ?Sized + PartialEq<U>, U: ?Sized> PartialEq<U> for LockGuard<'lock, T> {
    #[inline]
    fn eq(&self, other: &U) -> bool {
        <T as PartialEq<U>>::eq(self, other)
    }
}

impl<'lock, T: ?Sized + PartialOrd<U>, U: ?Sized> PartialOrd<U> for LockGuard<'lock, T> {
    #[inline]
    fn partial_cmp(&self, other: &U) -> Option<Ordering> {
        <T as PartialOrd<U>>::partial_cmp(self, other)
    }
}

/// An error returned from the [`LockCell::try_lock()`] method to indicate
/// that the `LockCell` could not be locked.
///
/// [`LockCell::try_lock()`]: ./struct.LockCell.html#method.try_lock
#[non_exhaustive]
pub struct TryLockError {
    /// The location where the `LockCell` was first locked.
    #[cfg(feature = "debug_lockcell")]
    first_lock_location: &'static Location<'static>,
    /// The latest location where the `LockCell` was locked. This should provide
    /// the location of the erroneous lock.
    #[cfg(feature = "debug_lockcell")]
    latest_lock_location: &'static Location<'static>,
    _priv: (),
}

impl TryLockError {
    /// Create a new `TryLockError` from the given caller location.
    #[cfg_attr(not(feature = "debug_lockcell"), allow(unused_variables))]
    #[track_caller]
    #[inline]
    fn new<T: ?Sized>(cell: &LockCell<T>) -> Self {
        TryLockError {
            #[cfg(feature = "debug_lockcell")]
            first_lock_location: cell
                .first_locked_at
                .get()
                .expect("Cell must be already locked"),
            #[cfg(feature = "debug_lockcell")]
            latest_lock_location: Location::caller(),
            _priv: (),
        }
    }
}

impl fmt::Debug for TryLockError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmtr.debug_struct("TryLockError");

        #[cfg(feature = "debug_lockcell")]
        {
            builder.field("first_locked_at", &self.first_lock_location);
            builder.field("last_locked_at", &self.latest_lock_location);
        }

        builder.finish_non_exhaustive()
    }
}

impl fmt::Display for TryLockError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        #[cfg(feature = "debug_lockcell")]
        {
            write!(
                fmtr,
                "first lock at {} conflicts with lock at {}",
                self.first_lock_location, self.latest_lock_location,
            )
        }

        #[cfg(not(feature = "debug_lockcell"))]
        {
            fmtr.write_str("already locked")
        }
    }
}

#[cfg(feature = "enable_std")]
impl StdError for TryLockError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locking() {
        let mut mtx = LockCell::new(23);

        let mut lk = mtx.lock();
        assert_eq!(*lk, 23);
        assert!(mtx.is_locked());

        *lk = 32;
        assert_eq!(*lk, 32);

        assert!(mtx.try_lock().is_err());
        drop(lk);
        assert_eq!(*mtx.get_mut(), 32);
    }

    #[test]
    fn lock_map() {
        #[derive(Default, Debug)]
        struct TestData {
            x: i32,
            y: i32,
        }

        let mtx = LockCell::<TestData>::default();
        let mut lk = LockGuard::map(mtx.lock(), |test_data| &mut test_data.y);
        *lk = 42;
        drop(lk);

        let lk = mtx.lock();
        let mut lk = match LockGuard::filter_map(lk, |data| Some(&mut data.x)) {
            Ok(new_lk) => new_lk,
            Err(old_lk) => panic!("{:?}", old_lk),
        };
        assert!(mtx.is_locked());
        *lk = 21;
        assert_eq!(*lk, 21);
        match LockGuard::filter_map(lk, |_| -> Option<&mut i32> { None }) {
            Ok(new_lk) => panic!("Unexpected lock guard found: {:?}", new_lk),
            Err(_) => {}
        }

        let data = mtx.into_inner();
        assert_eq!(data.x, 21);
        assert_eq!(data.y, 42);
    }
}
