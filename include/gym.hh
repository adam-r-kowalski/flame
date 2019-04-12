#pragma once

#include <string_view>

namespace flame {
namespace gym {
inline namespace v0 {

struct Environment {};

auto make(const std::string_view &name) -> Environment;

} // namespace v0
} // namespace gym
} // namespace flame
