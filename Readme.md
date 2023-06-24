# lock_cell

This crate provides the [`LockCell<T>`] and other supportings types.

A `LockCell` is a cell type which provides dynamic mutation using interior
mutability. It is similar to [`RefCell<T>`], except that it only allows
a single borrow type (a lock). Locking a `LockCell` allows mutating its
contents freely.

A `LockCell` can only be used in a single threaded context - it cannot be sent or
shared across different threads. Generally, a `LockCell` will be stored in a [`Rc<T>`]
so that it can be shared.

Whether you use a `LockCell` or a `RefCell` depends on the structure and behavior of
your program. Generally, if you have a lot of writers and readers, using a `LockCell`
may be better, as it ensures that writers are less likely to be starved.

The `Sync` equivalent of a `LockCell` is [`Mutex<T>`].

[`LockCell<T>`]: ./struct.LockCell.html
[`RefCell<T>`]: http://doc.rust-lang.org/stable/std/cell/struct.RefCell.html
[`Rc<T>`]: https://doc.rust-lang.org/stable/std/rc/struct.Rc.html
[`Mutex<T>`]: http://doc.rust-lang.org/stable/std/sync/struct.Mutex.html