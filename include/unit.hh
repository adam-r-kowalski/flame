#pragma once

namespace flame {

inline namespace v0 {

template <class T> auto zero() -> T;

// IMPLEMENTATION

template <class T> auto zero() -> T { return 0; }

} // namespace v0

} // namespace flame
