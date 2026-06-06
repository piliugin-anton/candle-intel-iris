# AGENT.md — Rust 2026 Best Practices

This file contains coding standards and best practices that all AI agents and contributors
must follow when working on this Rust project. These guidelines reflect the state of
idiomatic, production-grade Rust as of 2026.

---

## 1. Edition & Language Features

- **Always target the Rust 2024 Edition** (`edition = "2024"` in `Cargo.toml`). The 2024
  edition is fully stabilized and changes fundamental patterns around async closures and
  lifetime capturing.
- **Use async closures** (`async || {}`) for local, short-lived async units of work. Avoid
  the old `Box<dyn Future>` workaround — it introduces unnecessary heap allocations.
- **Use Precise Capturing syntax** (`use<'a, T>`) when returning opaque `Future` or
  `Iterator` types that borrow from input arguments. This prevents the compiler from making
  overly conservative lifetime assumptions and reduces borrowing conflicts.
- **Use `const` trait implementations** where applicable to enable more powerful
  compile-time computations.
- **Use improved try blocks** and custom `?` operator conversions to reduce boilerplate in
  error-propagating code.

---

## 2. Project Structure: Module-First, Crate-Last

- **Default to modules**, not crates. Keep code in a deeply nested module tree within a
  single library crate for as long as possible. Every crate boundary is a compilation wall
  that slows incremental builds.
- **Split into a separate crate only when:**
  1. You need a **procedural macro** (required by language design).
  2. You need **strict visibility enforcement** between subsystems at the compiler level.
  3. Two large independent chunks of code can benefit from **parallel compilation**.
- Use `mod.rs` or the file-per-module style consistently across the project — do not mix.
- Avoid workspace fragmentation driven by organizational impulse rather than technical need.

---

## 3. Error Handling

- **Libraries → `thiserror`**: Define structured, machine-readable error enums with
  `#[derive(thiserror::Error)]`. Always provide variants that callers can `match` on for
  retry logic or specific recovery paths.

  ```rust
  #[derive(thiserror::Error, Debug)]
  pub enum DataError {
      #[error("IO error: {0}")]
      Io(#[from] std::io::Error),
      #[error("Parse error at line {line}, col {col}")]
      Parse { line: usize, col: usize },
  }
  ```

- **Applications → `anyhow`**: In `main.rs` or top-level handlers, use `anyhow::Error`
  with the `.context()` method to build human-readable error chains.

  ```rust
  // Always attach context — never use bare `?` in application code
  db.fetch_user(id).await.context("failed to fetch user for profile update")?;
  ```

- **Never use `unwrap()` or `expect()` in production paths.** These cause panics and make
  debugging harder. Reserve them only for tests or cases where the invariant is proven
  by construction and documented.
- Replace `null` patterns with `Option` and `Result` types. Use the `?` operator to
  propagate errors idiomatically.

---

## 4. Async & Tokio

- **Tokio is the standard async runtime.** Do not introduce alternative runtimes without
  a strong, documented reason.
- **Never perform blocking operations inside async functions.** This blocks Tokio's worker
  threads and degrades throughput for all concurrent tasks.

  ```rust
  // Wrong
  let data = std::fs::read_to_string("file.txt")?;

  // Correct
  let data = tokio::task::spawn_blocking(|| {
      std::fs::read_to_string("file.txt")
  }).await??;
  ```

- **Design for cancellation safety.** When using `tokio::select!`, the dropped branch may
  be mid-operation (socket write, DB update). Always verify that the crates you depend on
  (e.g., `sqlx`, `tokio-util`) are cancellation-safe. Wrap non-cancellation-safe operations
  in a spawned task that runs to completion.
- **Prefer async closures** over boxed futures for callback-style APIs. Update traits to
  support `IntoFuture` in public library interfaces.

---

## 5. Performance & Memory

### Zero-Copy Serialization

- For high-throughput data paths (IPC, caching, hot-path parsing), prefer **zero-copy
  deserialization** over standard serde JSON. Evaluate `rkyv` which maps a byte buffer
  directly onto a Rust struct with no allocation overhead.
- `serde` / `serde_json` is acceptable for low-frequency paths (config files, API payloads
  not in the hot path).

### Data-Oriented Design

- For performance-critical loops, prefer **struct-of-arrays (SoA)** layouts over
  array-of-structs (AoS). SoA keeps relevant fields contiguous in CPU cache, reducing
  cache misses and improving throughput.
- Audit hot paths for unnecessary `clone()` calls. Prefer borrowing and lifetimes or
  `Arc`/`Rc` sharing over gratuitous cloning.

### Linker

- Use **LLD** as the default linker on Linux for faster link times. Keep the toolchain
  updated via `rustup update`.
- Profile builds with `cargo build --timings` to identify bottlenecks before optimizing.

---

## 6. Testing

- **Unit tests are the bare minimum.** Every module must have unit tests, but go further.
- **Property-based testing** with `proptest`: for any non-trivial logic, write property
  tests that assert invariants across randomly generated inputs (including edge cases like
  `NaN`, empty strings, max integers).

  ```rust
  use proptest::prelude::*;
  proptest! {
      #[test]
      fn add_is_commutative(a: i32, b: i32) {
          prop_assert_eq!(add(a, b), add(b, a));
      }
  }
  ```

- **Snapshot testing** with `insta`: for APIs, CLIs, or any component with complex string
  or JSON output, use `insta` to snapshot-test outputs. Review diffs on changes and
  explicitly accept or reject them.
- **Run tests with `cargo-nextest`**, not `cargo test`. Nextest is faster, isolates each
  test in its own process, and produces clearer failure output.

  ```bash
  cargo nextest run
  ```

---

## 7. Macros

- Prefer `macro_rules!` for simple, repetitive code generation patterns.
- Use procedural macros only when declarative macros are insufficient.
- **Document every macro** with clear examples showing inputs and outputs. Undocumented
  macros are a maintenance liability.
- When debugging unexpected macro behavior, use `cargo-expand` to inspect the generated code:

  ```bash
  cargo expand my::module
  ```

---

## 8. WebAssembly (Wasm) Targets

- With Rust 2026's enhanced Wasm support, the linker now **detects and reports undefined
  symbols at build time** rather than causing silent runtime failures. Treat Wasm linker
  warnings as errors in CI.
- Minimize Wasm binary size by compiling with `opt-level = "z"` and enabling LTO in
  `Cargo.toml` for `wasm32` targets.
- Use `wasm-bindgen` and `wasm-pack` for browser-targeting workflows.

---

## 9. Security & Dependency Management

- Run `cargo audit` in CI on every PR to catch known CVEs in the dependency tree.

  ```bash
  cargo audit
  ```

- Keep all dependencies up to date. Prefer well-maintained crates with clear security
  disclosure policies.
- Review crates.io security advisories regularly.
- Never disable `deny(unsafe_code)` at the crate level without a documented, reviewed
  exception. If `unsafe` is required, isolate it in a dedicated module with a clear
  safety comment on every `unsafe` block.

---

## 10. Tooling Checklist

Every pull request must pass the following before merge:

| Tool | Command | Purpose |
|---|---|---|
| `rustfmt` | `cargo fmt --check` | Consistent formatting |
| `clippy` | `cargo clippy -- -D warnings` | Lints & idiomatic patterns |
| `nextest` | `cargo nextest run` | Test runner |
| `audit` | `cargo audit` | Dependency CVE check |
| `expand` | (as needed) | Macro debugging |
| `timings` | `cargo build --timings` | Build performance profiling |

---

## 11. Anti-Patterns: What NOT to Do

This section catalogs the most common bad habits that silently degrade correctness,
performance, or maintainability in Rust codebases. Treat violations as review blockers.

---

### 11.1 Cloning to Satisfy the Borrow Checker

**Problem:** Using `.clone()` as a quick fix whenever the borrow checker complains.
Cloning heap types like `String` or `Vec<T>` creates new allocations, increases memory
pressure, and introduces cache misses. It also masks underlying design problems with
ownership.

```rust
// ❌ Anti-pattern
fn process(data: String, map: &HashMap<String, String>) {
    let key = data.clone(); // unnecessary heap allocation
    map.get(&key);
}

// ✅ Correct
fn process(data: &str, map: &HashMap<String, String>) {
    map.get(data); // HashMap<String,_> accepts &str via Borrow
}
```

**Rules:**
- Question every `.clone()`. Can you borrow (`&T`, `&str`), move, or share with
  `Arc`/`Rc` instead?
- `Arc::clone` is fine — it only copies a pointer and bumps an atomic counter.
- `Clone` on large structures (`Vec<T>`, nested structs) in a hot loop is always a bug.
- Run `cargo clippy`; it detects many unnecessary clones automatically.

---

### 11.2 Blocking I/O Inside Async Functions

**Problem:** Calling blocking operations (`std::fs`, `std::net`, CPU-intensive loops)
inside `async fn` stalls the Tokio worker thread and starves all other concurrent tasks.

```rust
// ❌ Anti-pattern — blocks the async runtime thread
async fn handler() -> String {
    std::fs::read_to_string("data.txt").unwrap()
}

// ✅ Correct
async fn handler() -> String {
    tokio::task::spawn_blocking(|| {
        std::fs::read_to_string("data.txt").unwrap()
    })
    .await
    .unwrap()
}

// ✅ Even better — use the async API directly
async fn handler() -> String {
    tokio::fs::read_to_string("data.txt").await.unwrap()
}
```

**Rules:**
- Always use `tokio::fs`, `tokio::net`, and other async-native APIs inside `async fn`.
- For unavoidable blocking work (legacy sync libs, CPU-bound tasks), use
  `tokio::task::spawn_blocking`.
- Use `tokio-console` in development to spot blocked worker threads.

---

### 11.3 Holding a Lock Guard Across an `.await` Point

**Problem:** A `MutexGuard` held across an `.await` keeps the lock locked while the
runtime switches to another task — guaranteed deadlock in any realistic workload. Clippy
flags this for `std::sync::Mutex`; the pattern is equally dangerous with other guard
types (connection pool handles, `RwLockWriteGuard`, etc.).

```rust
// ❌ Anti-pattern — guard lives across .await
async fn bad(state: Arc<Mutex<State>>) {
    let mut guard = state.lock().unwrap();
    do_async_work().await; // deadlock waiting to happen
    guard.field = 42;
}

// ✅ Correct — drop guard before awaiting
async fn good(state: Arc<Mutex<State>>) {
    {
        let mut guard = state.lock().unwrap();
        guard.field = 42;
    } // guard dropped here
    do_async_work().await;
}
```

**Rules:**
- Drop all `MutexGuard`s before any `.await` expression.
- If you genuinely need to hold a lock across an await (rare), use
  `tokio::sync::Mutex`, which is designed for this.
- Prefer channels or actor patterns over shared mutable state guarded by a mutex.
- Define a fixed lock acquisition order across the codebase to prevent ordering
  deadlocks when multiple locks must be held simultaneously.

---

### 11.4 Overusing `unwrap()` and `expect()` in Production Code

**Problem:** `.unwrap()` and `.expect()` panic on `None`/`Err`. In production paths,
this terminates the thread or the whole process, loses error context, and makes stack
traces hard to trace back to the root cause.

```rust
// ❌ Anti-pattern
let port: u16 = env::var("PORT").unwrap().parse().unwrap();

// ✅ Correct
let port: u16 = env::var("PORT")
    .context("PORT env var missing")?
    .parse()
    .context("PORT must be a valid u16")?;
```

**Rules:**
- `unwrap()` / `expect()` are permitted only in: tests, examples, and provably
  unreachable branches (document the invariant with a comment).
- In all other code, propagate with `?` and attach `.context()` via `anyhow`.
- Never use `unwrap()` in library code — it forces panics on downstream consumers.

---

### 11.5 Unnecessary or Excessive `mut`

**Problem:** Marking variables `mut` by default, rather than by necessity, makes code
harder to reason about — any subsequent line could legally mutate the value.

```rust
// ❌ Anti-pattern
let mut result = compute();
println!("{result}"); // result is never actually mutated

// ✅ Correct
let result = compute();
println!("{result}");
```

**Rules:**
- Declare variables immutable by default. Add `mut` only when the value genuinely
  changes after its initial binding.
- Confine mutation to the smallest possible scope.
- Clippy will warn on unused `mut` — never suppress this warning.

---

### 11.6 Wrong Collection for the Job

**Problem:** Using `Vec<T>` for lookup-heavy workloads (O(n) scan), or using `HashMap`
when ordered iteration is required, or string-building with `+=` in a loop.

| Situation | Anti-pattern | Correct choice |
|---|---|---|
| Keyed lookup | `Vec::iter().find(...)` — O(n) | `HashMap` — O(1) amortized |
| Sorted key iteration | `HashMap` | `BTreeMap` |
| Building a string in a loop | `result += &piece` — O(n²) | `Vec<String>` then `.join("")` |
| Fixed-size stack buffer | `Vec::new()` | array `[T; N]` |
| Read-only set membership | `Vec::contains(...)` | `HashSet` |

```rust
// ❌ Anti-pattern — O(n²) string building
let mut s = String::new();
for piece in pieces {
    s += &piece;
}

// ✅ Correct — single allocation
let s: String = pieces.join("");
// or for dynamic cases:
let mut buf = String::with_capacity(estimated_len);
for piece in &pieces { buf.push_str(piece); }
```

---

### 11.7 Ignoring Iterator Adapters — Manual Loops Instead

**Problem:** Writing explicit `for` loops with `push` / `if` when iterator chains
(`map`, `filter`, `flat_map`, `fold`, `chain`) would be cleaner, more composable, and
often compile to the same or better machine code.

```rust
// ❌ Anti-pattern
let mut evens = Vec::new();
for x in data {
    if x % 2 == 0 { evens.push(x * 2); }
}

// ✅ Correct
let evens: Vec<_> = data.iter()
    .filter(|&&x| x % 2 == 0)
    .map(|&x| x * 2)
    .collect();
```

**Rules:**
- Prefer iterator chains for transformations and filters over index loops.
- Use `.collect()` with an explicit type to avoid guessing.
- Avoid calling `.iter().cloned().collect()` when `.to_vec()` or direct `.clone()` on the
  slice is clearer.

---

### 11.8 Reference Cycles with `Rc`/`Arc`

**Problem:** Graphs, parent-child trees, and observer patterns implemented with
`Rc<RefCell<T>>` or `Arc<Mutex<T>>` can form reference cycles that prevent the
reference count from ever reaching zero — a memory leak that the borrow checker does
not catch.

```rust
// ❌ Can cause a cycle:
// parent holds Arc<Child>, child holds Arc<Parent> → neither drops

// ✅ Break cycles with Weak
use std::sync::{Arc, Weak};
struct Child { parent: Weak<Parent> } // Weak does not keep parent alive
```

**Rules:**
- Use `Weak<T>` (from `std::rc::Weak` or `std::sync::Weak`) for back-references in
  parent-child or observer relationships.
- Do not wrap everything in `Arc` by default; it adds atomic overhead and hides
  ownership intent. Use `Arc` only when data genuinely needs to be shared across
  thread boundaries.
- Prefer `Rc` over `Arc` in single-threaded code. The compiler will tell you when `Arc`
  is actually required.

---

### 11.9 Misusing Macros — Using Them Where Functions Suffice

**Problem:** Writing a macro for logic that a plain function would express more clearly.
Macros are harder to read, harder to document, harder to debug (errors point into
expansion sites), and do not benefit from IDE tooling the same way functions do.

```rust
// ❌ Anti-pattern — overkill macro for simple logic
macro_rules! double { ($x:expr) => { $x * 2 }; }

// ✅ Just a function
fn double(x: i32) -> i32 { x * 2 }
```

**Rules:**
- Use a function unless the task genuinely requires: variadic arguments, compile-time
  code generation, or syntax not expressible as a function (e.g., `vec![]`).
- Never use a macro to work around a type system problem — fix the types.
- Every macro must have a `/// # Examples` doc-comment showing inputs and outputs.

---

### 11.10 Ignoring Compiler Warnings and Clippy Lints

**Problem:** Suppressing or ignoring warnings via `#[allow(...)]` or `// TODO: fix
later` comments allows known issues to accumulate and rot.

**Rules:**
- CI must run `cargo clippy -- -D warnings`. Warnings are errors in CI.
- `#[allow(...)]` at the file or crate level is **banned** without a tracked issue and
  a comment explaining the exception.
- `#[allow(...)]` at the individual item level must have a one-line comment justifying
  the suppression directly above it.
- Never add `#![allow(dead_code)]` across a whole crate — remove or `pub`-use the dead
  code, or document it as intentionally unused.

---

### 11.11 Unbounded Concurrency — Spawning Tasks Without Back-Pressure

**Problem:** Spawning a `tokio::spawn` for every incoming item without limiting
concurrency can exhaust file descriptors, memory, or downstream connection pools.

```rust
// ❌ Anti-pattern — unbounded spawning
for item in items {
    tokio::spawn(process(item));
}

// ✅ Correct — bounded with a semaphore
let sem = Arc::new(Semaphore::new(MAX_CONCURRENT));
for item in items {
    let permit = sem.clone().acquire_owned().await?;
    tokio::spawn(async move {
        let _permit = permit; // dropped at end of task
        process(item).await;
    });
}
```

**Rules:**
- Always bound the maximum number of concurrent tasks for workloads driven by external
  input (network requests, file lists, queue messages).
- Use `tokio::sync::Semaphore` or a buffered stream (`futures::stream::iter(...).buffer_unordered(N)`).
- Use `tokio-console` to observe task counts in development.

---

## 12. Summary

| Category | Rule |
|---|---|
| **Edition** | Rust 2024; async closures; precise capturing |
| **Structure** | Module-first; crate only for macros, strict boundaries, parallelism |
| **Errors** | `thiserror` in libs; `anyhow` + `.context()` in apps; no bare `unwrap` |
| **Async** | `spawn_blocking` for blocking work; cancellation-safe design |
| **Performance** | Zero-copy (`rkyv`) for hot paths; SoA layouts; no gratuitous clones |
| **Testing** | `proptest` + `insta` + `cargo-nextest` |
| **Macros** | Document every macro; use `cargo-expand` to debug |
| **Wasm** | Treat linker warnings as errors; minimize binary size |
| **Security** | `cargo audit` in CI; isolate and document all `unsafe` |
| **Anti-patterns** | No borrow-checker clones; no blocking in async; no lock across `.await`; no bare `unwrap`; prefer iterators; right collection for the job; `Weak` for cycles; bound concurrency |
