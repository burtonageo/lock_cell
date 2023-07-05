# lock_cell

This crate provides the [`LockCell<T>`] and other supportings types.

A `LockCell` is a cell type which provides dynamic mutation using interior
mutability. It is similar to [`RefCell<T>`], except that it only allows
a single borrow type (a lock). Locking a `LockCell` allows mutating its
contents freely.

A `LockCell` can only be used in a single threaded context - it cannot be shared
across different threads. Generally, a `LockCell` will be stored in a [`Rc<T>`]
so that it can be shared.

Whether you use a `LockCell` or a `RefCell` depends on the structure and behavior of
your program. Generally, if you have a lot of writers and readers, using a `LockCell`
may be better, as it ensures that writers are less likely to be starved.

The [`Sync`] equivalent of a `LockCell` is [`Mutex<T>`].

# Features

* The `enable_std` feature enables the standard library. This provides an implementation of
  [`std::error::Error`] for the [`TryLockError`] type. This feature is enabled by default.

* The `debug_lockcell` feature tracks the location of each `lock()` call in the `LockCell`,
  allowing the developer to compare the first lock location in their file to the panicking
  lock location, aiding in debugging.

[`LockCell<T>`]: https://docs.rs/lock_cell/latest/lock_cell/struct.LockCell.html
[`RefCell<T>`]: http://doc.rust-lang.org/std/cell/struct.RefCell.html
[`Rc<T>`]: https://doc.rust-lang.org/std/rc/struct.Rc.html
[`Mutex<T>`]: http://doc.rust-lang.org/std/sync/struct.Mutex.html
[`Sync`]: https://doc.rust-lang.org/std/marker/trait.Sync.html
[`std::error::Error`]: https://doc.rust-lang.org/std/error/trait.Error.html
[`TryLockError`]: https://docs.rs/lock_cell/latest/lock_cell/struct.TryLockError.html
