// Stubs for SV-exported DPI functions referenced by dpi_memutil.o but unused by the Ara testbench
// (which loads memories through VerilatorMemUtil). They satisfy the linker with no behaviour.
#include <cstdint>
extern "C" {
void simutil_memload(const char *file) { (void)file; }
int simutil_set_mem(int index, const uint32_t *val) { (void)index; (void)val; return 0; }
}
