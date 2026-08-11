//! Pack this crate's `.tiktoken` text into the binary form splintr's
//! loader reads. The text is what ships; the binary is derived, so the
//! two cannot disagree.

fn main() {
    splintr_vocab_pack::pack_into_out_dir("o200k_base");
}
