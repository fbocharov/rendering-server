#pragma once

#include <functional>

template<typename T>
struct scoped_guard
{
    scoped_guard(std::function<void(T&)> const & ctor, std::function<void(T&)> const & dtor)
        : dtor_(dtor)
    {
        ctor(resource_);
    }

    operator T&()
    {
        return resource_;
    }

    ~scoped_guard()
    {
        dtor_(resource_);
    }

private:
    T resource_;
    std::function<void(T&)> dtor_;
};
