/* GenMC model-checking harness for the SPSC ring in ../spsc_ring.hpp.
 *
 * GenMC (Kokologiannakis et al., PLDI 2019; https://github.com/MPI-SWS/genmc)
 * explores every execution of a C11 program allowed by the RC11 memory model.
 * It takes C, so this file restates the ring's push/pop with exactly the same
 * memory orders as the C++ header: relaxed load of the own index, acquire load
 * of the other side's index, release store of the own index. Keep the two in
 * step; a change of memory order in the header must be mirrored here.
 *
 * Run:   genmc -- -DN=3 -DCAP=2 cpp/ull/ring/genmc/spsc_genmc.c
 * Claim: no execution violates an assert (each value read once, in order, and
 * never before it was written). Weakening the release store of `tail` to
 * relaxed makes GenMC report the data race on `slots`; see LATENCY_METHOD.md.
 */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>

#ifndef N
#define N 3 /* values transferred */
#endif
#ifndef CAP
#define CAP 2 /* ring capacity (power of two), < N forces wrap-around */
#endif

static int slots[CAP];
static atomic_ulong head; /* consumer index */
static atomic_ulong tail; /* producer index */

static int try_push(int v)
{
	unsigned long t = atomic_load_explicit(&tail, memory_order_relaxed);
	unsigned long h = atomic_load_explicit(&head, memory_order_acquire);
	if (t - h >= CAP)
		return 0;
	slots[t % CAP] = v;
	atomic_store_explicit(&tail, t + 1, memory_order_release);
	return 1;
}

static int try_pop(int *out)
{
	unsigned long h = atomic_load_explicit(&head, memory_order_relaxed);
	unsigned long t = atomic_load_explicit(&tail, memory_order_acquire);
	if (h == t)
		return 0;
	*out = slots[h % CAP];
	atomic_store_explicit(&head, h + 1, memory_order_release);
	return 1;
}

static void *producer(void *arg)
{
	(void)arg;
	for (int i = 1; i <= N; ++i)
		while (!try_push(i))
			;
	return NULL;
}

static void *consumer(void *arg)
{
	(void)arg;
	int v;
	for (int want = 1; want <= N; ++want) {
		while (!try_pop(&v))
			;
		assert(v == want);
	}
	return NULL;
}

int main(void)
{
	pthread_t p, c;
	pthread_create(&p, NULL, producer, NULL);
	pthread_create(&c, NULL, consumer, NULL);
	pthread_join(p, NULL);
	pthread_join(c, NULL);
	return 0;
}
