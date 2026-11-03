# C++: knowledge check (questions only)

Twenty-four items for C++ roles on trading desks and in low-latency
infrastructure, written fresh for this repository. They come in three
kinds:

1. **What does it print (C01-C11).** Read the linked snippet, write down
   the exact output, then check the key. Each snippet is a complete
   program; `out(x)` prints `x` followed by a space. These test the rules
   that cause real bugs: object lifetimes, special member functions,
   dynamic dispatch, conversions, overload resolution, layout.
2. **Write it (C12-C20).** Implement the component from scratch in about
   30 minutes, with the stated guarantees, then compare with the
   reference in [`solutions/`](solutions/). The references are exercised
   by [`checks/`](checks/) under the address, undefined-behaviour and
   thread sanitisers.
3. **Explain it (C21-C24).** Talk it through as you would at a whiteboard.

The answers are in [`keys/answers.md`](keys/answers.md). How it is checked:
`career-prep/tests/test_cpp_pack.py` compiles and runs every snippet and
requires its output to equal the key, and builds and runs both check
programs (GCC, `-std=c++20`; the reference solutions also with
`-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`).

Further reading: Meyers, *Effective Modern C++* (O'Reilly, 2014);
Williams, *C++ Concurrency in Action* (2nd ed., Manning, 2019); the C++
standard draft at <https://eel.is/c++draft/>.

## What does it print

- **C01** [snippets/C01.cpp](snippets/C01.cpp): a type that logs its constructors and destructor is pushed and emplaced into a vector with reserved space, then copied. Write the whole log.
- **C02** [snippets/C02.cpp](snippets/C02.cpp): a virtual function called from the base and derived constructors and destructors.
- **C03** [snippets/C03.cpp](snippets/C03.cpp): a derived object passed by value, by reference, assigned to a base, and put in a `std::vector<Base>`.
- **C04** [snippets/C04.cpp](snippets/C04.cpp): a signed/unsigned comparison, `unsigned char` wrap-around, the type of `'a' + 1`, and `uint32_t(0) - 1`.
- **C05** [snippets/C05.cpp](snippets/C05.cpp): members initialised in a different order from the constructor's initialiser list, then a local, then everything destroyed.
- **C06** [snippets/C06.cpp](snippets/C06.cpp): overloads for `int`, `long` and `double` plus a function template, called with `1`, `1L`, `1.0f`, `'a'` and `1.0`.
- **C07** [snippets/C07.cpp](snippets/C07.cpp): `std::move` of a `const` object, of a non-const object, and a nest of temporaries.
- **C08** [snippets/C08.cpp](snippets/C08.cpp): `sizeof` of five structs on x86-64 Linux (GCC): padding, reordering, `alignas`, an empty class, and `[[no_unique_address]]`.
- **C09** [snippets/C09.cpp](snippets/C09.cpp): `shared_ptr` use counts, `weak_ptr::expired`, and converting a `unique_ptr` to a `shared_ptr`.
- **C10** [snippets/C10.cpp](snippets/C10.cpp): fold expressions in both directions and a `constexpr` function.
- **C11** [snippets/C11.cpp](snippets/C11.cpp): `std::remove` without `erase`, then the erase-remove idiom, then C++20 `std::erase`.

## Write it

- **C12** A move-only owning pointer: move construct and assign, `release`, `reset`, `get`, `operator*`, `operator->`, explicit `bool`, and a `make_unique`. It must be no larger than a raw pointer, and self-move-assignment must not delete the object.
- **C13** A single-producer single-consumer lock-free ring buffer with `try_push` and `try_pop`. Choose the memory orderings and justify each one. Use every slot (no "one empty slot" trick).
- **C14** A multi-producer multi-consumer blocking queue with `push`, a blocking `pop` and `close()`. After `close`, `push` fails and `pop` drains what is left, then returns empty.
- **C15** A fixed-capacity object pool that never touches the heap after construction: `create(args...)` returns `nullptr` when full, `destroy(p)` runs the destructor and recycles the slot. It must stay consistent if a constructor throws.
- **C16** A test-and-test-and-set spinlock usable with `std::lock_guard`, plus `try_lock`.
- **C17** A fixed-size thread pool whose `submit(f)` returns a `std::future` of `f`'s result, carries exceptions to the caller, and whose destructor finishes queued work and joins.
- **C18** A fixed-point price type on a 0.0001 tick: parse a decimal string exactly, reject anything finer than a tick or malformed, print it back, and add and multiply with overflow detection. Show why `double` is the wrong type for a price.
- **C19** A sequence lock for one writer and many readers of a small trivially copyable struct. Readers never block the writer and never see a torn value. It must be free of data races in the C++ memory model, not just "work on x86".
- **C20** A concept `SummableRange` and a function `sum_all` constrained by it; and a per-thread counter type that cannot share a cache line with its neighbour in an array.

## Explain it

- **C21** Name the six memory orders. For each atomic operation in C13, say which order you used and what goes wrong with `relaxed`.
- **C22** A colleague's market-data handler has a p99.9 latency 50 times its median. List the usual causes on the hot path and how you would find each one.
- **C23** What is false sharing? How would you show it exists in a program, and how does C20's padded counter fix it?
- **C24** Virtual functions versus CRTP versus `std::variant` with `std::visit` for an order-handler hierarchy on a hot path: what does each cost, and when would you pick each?
