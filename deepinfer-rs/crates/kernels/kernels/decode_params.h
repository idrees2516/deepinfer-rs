// decode_params.h — layout of the device-resident decode parameter block
// shared by the `decode.cu` kernel family and the Rust launchers.
//
// The host writes two ints per step into `p_pos` (`{pos, start}`: the
// seq-local position being decoded and the sequence's start row inside the
// shared cache arenas); `di_step_params_kernel` fans them out into this
// block, and every `_dev` kernel reads the fields it needs by address.
// Field order is ABI: Rust mirrors it in `deepinfer-kernels/src/decode.rs`.
//
// Seq-local fields address a sequence's slices (thunder k cache, feat for
// scoring, attention indices); arena-global fields address the shared
// per-layer arenas (KV rows, indexer k appends, conv features).

#define DP_POS 0          // seq-local position of the token being decoded
#define DP_START 1        // arena start row of this sequence
#define DP_LEN 2          // new total sequence length (seq-local)
#define DP_ROW 3          // arena row of the new token (START + POS)
#define DP_S_KV 4         // DSA / attention key count (= new total)
#define DP_K_NBLOCKS 5    // complete lightning-quant blocks (seq-local)
#define DP_LAST_BLOCK 6   // last (possibly partial) block index (seq-local)
#define DP_CROSS_LO 7     // cross window: first ARENA row
#define DP_CROSS_NKEYS 8  // cross window: key count
#define DP_CROSS_QOFF 9   // cross window dense-causal q_offset
#define DP_SWA_N 10       // SWA ring: key count
#define DP_SWA_QOFF 11    // SWA dense-causal q_offset
#define DP_TOPK 12        // fixed selection width (cfg topk_tokens)
#define DP_SWA_START 13    // SWA ring: slot of the oldest visible row
#define DP_COUNT 14       // number of fields
