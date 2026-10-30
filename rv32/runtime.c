/* Minimal freestanding support for compiler-generated zero initialization. */
#include <stddef.h>
/* Fill a memory region with the low byte of value and return its starting address. */
void *memset(void *destination, int value, size_t count) {
    unsigned char *p = destination;
    for (size_t i = 0; i < count; ++i) p[i] = (unsigned char)value;
    return destination;
}
