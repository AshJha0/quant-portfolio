#include "out.hpp"
struct B {
    B() { hello(); }
    virtual ~B() { hello(); }
    virtual void hello() { out("B"); }
};
struct D : B {
    D() { hello(); }
    ~D() override { hello(); }
    void hello() override { out("D"); }
};
int main() { D d; }
