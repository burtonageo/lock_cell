// Copyright (c) 2023, George Burton <burtonageo@gmail.com>
//
// SPDX-License-Identifier: 0BSD

#![cfg_attr(not(feature = "enable_std"), no_std)]

//! This crate provides the [`LockCell<T>`] and other supportings types.
//!
//! A `LockCell` is a cell type which provides dynamic mutation using interior
//! mutability. It is similar to [`RefCell<T>`], except that it only allows
//! a single borrow type (a lock). Locking a `LockCell` allows mutating its
//! contents freely.
//!
//! A `LockCell` can only be used in a single threaded context - it cannot be sent or
//! shared across different threads. Generally, a `LockCell` will be stored in a [`Rc<T>`]
//! so that it can be shared.
//!
//! Whether you use a `LockCell` or a `RefCell` depends on the structure and behavior of
//! your program. Generally, if you have a lot of writers and readers, using a `LockCell`
//! may be better, as it ensures that writers are less likely to be starved.
//!
//! The `Sync` equivalent of a `LockCell` is [`Mutex<T>`].
//! 
//! [`LockCell<T>`]: ./struct.LockCell.html
//! [`RefCell<T>`]: http://doc.rust-lang.org/stable/std/cell/struct.RefCell.html
//! [`Rc<T>`]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html
//! [`Mutex<T>`]: http://doc.rust-lang.org/stable/std/sync/struct.Mutex.html

use core::{
    borrow::{Borrow, BorrowMut},
    convert::{AsRef, AsMut, TryFrom},
    cell::{Cell, UnsafeCell},
    fmt,
    marker::PhantomData,
    mem::{self, ManuallyDrop},
    ops::{Deref, DerefMut, FnOnce},
    panic::Location,
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
    #[inline]
    pub const fn new(value: T) -> Self {
        Self {
            value: UnsafeCell::new(value),
            is_locked: Cell::new(false),
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
    #[inline]
    pub fn into_inner(self) -> T {
        self.value.into_inner()
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
    /// [`Default::default()`]: https://doc.rust-lang.org/stable/std/default/trait.Default.html
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
    /// If the `LockCell` is already locked, this function will fail and will
    /// return a [`TryLockError`].
    ///
    /// [`TryLockError`]: ./struct.TryLockError.html
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
    ///
    /// # Ok(())
    /// # }
    #[inline]
    #[track_caller]
    pub fn try_lock<'a>(&'a self) -> Result<LockGuard<'a, T>, TryLockError> {
        if self.is_locked.replace(true) {
            return Err(TryLockError::new(Location::caller()))
        }

        Ok(LockGuard {
            value: self.value.get(),
            is_locked: &self.is_locked,
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
    /// [`LockGuard`]: ./struct.LockGuard.html
    /// [`try_lock()`]: ./struct.LockGuard.html#method.try_lock
    /// [`is_locked()`]: ./struct.LockGuard.html#method.is_locked
    #[inline]
    #[track_caller]
    pub fn lock<'a>(&'a self) -> LockGuard<'a, T> {
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
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        self.value.get_mut()
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
        self.get_mut()
    }
}

impl<T: fmt::Debug> fmt::Debug for LockCell<T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lock_result = self.try_lock();
        let value: &dyn fmt::Debug = match lock_result {
            Ok(ref value) => &*value,
            Err(_) => {
                struct LockedPlaceholder;
                impl fmt::Debug for LockedPlaceholder {
                    #[inline]
                    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
                        fmtr.write_str("<locked>")
                    }
                }

                const PLACEHOLDER: LockedPlaceholder = LockedPlaceholder;
                &PLACEHOLDER
            }
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
pub struct LockGuard<'a, T: ?Sized> {
    /// The location of the original value in the `LockCell`.
    value: *mut T,
    /// The lock state of the `LockCell`.
    is_locked: &'a Cell<bool>,
    /// Phantom data.
    _boo: PhantomData<&'a UnsafeCell<T>>,
}

impl<'a, T: ?Sized> LockGuard<'a, T> {
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
    #[track_caller]
    pub fn map<F, U: ?Sized>(this: Self, func: F) -> LockGuard<'a, U>
    where
        F: FnOnce(&mut T) -> &mut U,
    {
        let mut this = mem::ManuallyDrop::new(this);

        LockGuard {
            value: unsafe { func(&mut *this.value) as *mut _ },
            is_locked: this.is_locked,
            _boo: PhantomData,
        }
    }

    /// Applies the given `func` to the contents of `LockGuard` to return an optional reference
    /// to a part of the original data.
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
    #[track_caller]
    pub fn filter_map<F, U: ?Sized>(this: Self, func: F) -> Result<LockGuard<'a, U>, Self>
    where
        F: FnOnce(&mut T) -> Option<&mut U>,
    {
        let mut this = mem::ManuallyDrop::new(this);

        let value = match unsafe { func(&mut *this.value) } {
            Some(value) => value as *mut _,
            _ => return Err(ManuallyDrop::into_inner(this)),
        };

        Ok(LockGuard {
            value: unsafe { &mut *value },
            is_locked: this.is_locked,
            _boo: PhantomData,
        })
    }
}

impl<'a, T> TryFrom<&'a LockCell<T>> for LockGuard<'a, T> {
    type Error = TryLockError;
    #[inline]
    #[track_caller]
    fn try_from(lock_cell: &'a LockCell<T>) -> Result<Self, Self::Error> {
        lock_cell.try_lock()
    }
}

impl<'a, T: ?Sized> Drop for LockGuard<'a, T> {
    #[inline]
    fn drop(&mut self) {
        self.is_locked.set(false);
    }
}

impl<'a, T: ?Sized> AsRef<T> for LockGuard<'a, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        unsafe { &*self.value }
    }
}

impl<'a, T: ?Sized> AsMut<T> for LockGuard<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        unsafe { &mut *self.value }
    }
}

impl<'a, T: ?Sized> Borrow<T> for LockGuard<'a, T> {
    #[inline]
    fn borrow(&self) -> &T {
        unsafe { &*self.value }
    }
}

impl<'a, T: ?Sized> BorrowMut<T> for LockGuard<'a, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        unsafe { &mut *self.value }
    }
}

impl<'a, T: ?Sized> Deref for LockGuard<'a, T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.value }
    }
}

impl<'a, T: ?Sized> DerefMut for LockGuard<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.value }
    }
}

impl<'a, T: fmt::Debug + ?Sized> fmt::Debug for LockGuard<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("LockGuard").field("value", &self.deref()).finish()
    }
}

impl<'a, T: fmt::Display + ?Sized> fmt::Display for LockGuard<'a, T> {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.deref(), fmtr)
    }
}

/// An error returned from the [`LockCell::try_lock()`] method to indicate
/// that the `LockCell` could not be locked.
///
/// [`LockCell::try_lock()`]: ./struct.LockGuard.html#method.try_lock
#[non_exhaustive]
pub struct TryLockError {
    location: &'static Location<'static>,
}

impl TryLockError {
    /// Create a new `TryLockError` from the given caller location.
    #[inline]
    const fn new(location: &'static Location<'static>) -> Self {
        TryLockError {
            location,
        }
    }
}

impl fmt::Debug for TryLockError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.debug_struct("TryLockError").field("location", &self.location).finish()
    }
}

impl fmt::Display for TryLockError {
    #[inline]
    fn fmt(&self, fmtr: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmtr.write_str("already locked")
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
