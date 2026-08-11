# splintr-vocab-pack

Build-time packer for splintr's bundled vocabularies.

Each `splintr-vocab-*` crate ships the `.tiktoken` text it was given and calls
this from its build script to produce the binary form splintr's loader reads.
Nobody needs to depend on this directly.
