# Mock round 5: Low-latency developer (40 minutes)

Sit this cold, with a timer, on paper. Keys are in a separate file;
see [the debrief](README.md#debrief) once the time is up.

## Graded

- **M5.1** On x86-64 with the usual alignment rules, what is sizeof(struct { char a; double b; int c; })?
- **M5.2** Reorder the members as { double b; int c; char a; }. What is the size now?
- **M5.3** How many 64-byte cache lines does a loop over 1,000 contiguous 8-byte integers touch, if the array starts on a line boundary?
- **M5.4** A ring buffer of capacity 1024 indexes slots with head & (capacity - 1). Which slot does head = 5000 use?

## Discussion

- **M5.5** What is false sharing and how do you prevent it in an SPSC queue?
- **M5.6** Why acquire/release rather than sequentially consistent atomics on the queue indices?
