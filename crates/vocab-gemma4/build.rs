//! Pack this crate's `.mbpe` into the two binary forms splintr's loader reads —
//! the vocabulary and the merge order. The text is what ships, both binaries are
//! derived, so none of the three can disagree.

fn main() {
    splintr_vocab_pack::pack_mbpe_into_out_dir("gemma4");
}
