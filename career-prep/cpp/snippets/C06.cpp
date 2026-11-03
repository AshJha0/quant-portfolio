#include "out.hpp"
void f(int) { out("int"); }
void f(long) { out("long"); }
void f(double) { out("double"); }
template <class T>
void f(T) { out("T"); }
int main() {
    f(1);
    f(1L);
    f(1.0f);
    f('a');
    f(1.0);
}
