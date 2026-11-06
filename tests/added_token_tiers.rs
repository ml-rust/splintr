//! Each bundled vocabulary against the `added_tokens` of its reference
//! `tokenizer.json`, token by token.
//!
//! `from_pretrained` states tiers in a table and a `.spm` type column, and
//! `from_json` reads them. These tests check that the two routes agree.
//!
//! The documents are model files, so this test reads them from
//! `BOOSTR_MODELS_DIR` at run time. With the variable unset it prints a line and
//! passes. The unit tests in `src/core/pretrained.rs` pin the same tiers on
//! named tokens with no model directory.

use std::path::PathBuf;

use serde_json::Value;
use splintr::{
    from_json_path, from_pretrained, AddedToken, AddedTokenSet, FxHashSet, PolicyError,
    SpecialMode, SpmTokenizer,
};

/// A downstream crate writes `AddedToken` out in full, and states the tier
/// through `AddedTokenSet`.
#[test]
fn a_downstream_struct_literal_and_insert_vocabulary_set_the_tier() {
    let pieces: Vec<String> = ["<unk>", "<c>", "<n>", "a"]
        .iter()
        .map(|piece| piece.to_string())
        .collect();
    let scores = vec![0.0; pieces.len()];
    let mut set = AddedTokenSet::new();
    set.insert_plain("<c>", 1);
    set.insert_vocabulary(
        "<n>",
        AddedToken {
            id: 2,
            lstrip: false,
            rstrip: false,
        },
    );
    let tok = SpmTokenizer::new(pieces, scores, None, None)
        .expect("the vocabulary builds")
        .with_prefix_space(false)
        .with_added_tokens(set)
        .expect("the matcher builds");

    let none = FxHashSet::default();
    let allow = SpecialMode::Allow(&none);
    assert_eq!(tok.encode_with("a<n>", &allow).unwrap(), [3, 2]);
    assert_eq!(
        tok.encode_with("a<n>", &SpecialMode::Ordinary).unwrap(),
        [3, 2]
    );
    assert!(matches!(
        tok.encode_with("<c>", &allow),
        Err(PolicyError::DisallowedSpecial { .. })
    ));
}

/// `(vocabulary, tokenizer.json under the models directory)`.
const REFERENCES: &[(&str, &str)] = &[
    ("gemma2", "gemma-2-2b/tokenizer.json"),
    ("gemma3", "embeddinggemma-300m-tokenizer/tokenizer.json"),
    ("gemma4", "tokenizer-survey/google_gemma-4-12B-it.json"),
    ("qwen3", "qwen3-8b-tokenizer/tokenizer.json"),
    ("glm4", "glm-4.5-tokenizer/tokenizer.json"),
    ("deepseek_v3", "deepseek-v3-tokenizer/tokenizer.json"),
    ("gpt_oss", "gpt-oss-tokenizer/tokenizer.json"),
    ("llama3", "llama-3.2-1b/tokenizer.json"),
    ("llama2", "tokenizer-survey/NousResearch_Llama-2-7b-hf.json"),
    (
        "codellama",
        "tokenizer-survey/codellama_CodeLlama-7b-hf.json",
    ),
    ("mistral_v2", "mistral-7b-v0.3/tokenizer.json"),
    (
        "modernbert",
        "tokenizer-survey/answerdotai_ModernBERT-base.json",
    ),
    ("olmo2", "tokenizer-survey/allenai_OLMo-2-1124-7B.json"),
    ("phi4", "tokenizer-survey/microsoft_phi-4.json"),
    (
        "whisper_v3",
        "tokenizer-survey/openai_whisper-large-v3.json",
    ),
];

fn models_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("BOOSTR_MODELS_DIR").map(PathBuf::from);
    if dir.is_none() {
        eprintln!("BOOSTR_MODELS_DIR is unset: no reference document read.");
    }
    dir
}

/// Every added token the bundled vocabulary declares takes the document's tier.
///
/// A token the bundled table leaves out does not end in its id under
/// `encode_raw` and is not compared. A standalone prefix can precede the id, as
/// on Mistral. The count of compared tokens guards against a document that
/// compares nothing.
#[test]
fn bundled_tiers_match_the_reference_document() {
    let Some(dir) = models_dir() else { return };
    let none = FxHashSet::default();
    for (name, relative) in REFERENCES {
        let path = dir.join(relative);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}. Set BOOSTR_MODELS_DIR.", path.display()));
        let document: Value = serde_json::from_slice(&bytes).expect("the document parses");
        let tok = from_pretrained(name).expect("the vocabulary loads");
        let mut compared = 0usize;
        let mut wrong = Vec::new();
        for entry in document["added_tokens"].as_array().expect("added_tokens") {
            let content = entry["content"].as_str().expect("content");
            let id = entry["id"].as_u64().expect("id") as u32;
            let special = entry["special"].as_bool().expect("special");
            if tok.encode_raw(content).last() != Some(&id) {
                continue;
            }
            compared += 1;
            let refused = matches!(
                tok.encode_with(content, &SpecialMode::Allow(&none)),
                Err(PolicyError::DisallowedSpecial { .. })
            );
            if refused != special {
                wrong.push(content.to_string());
            }
        }
        eprintln!("{name}: compared {compared} added tokens against {relative}");
        assert!(compared > 0, "{name}: {relative} compared no token");
        assert!(
            wrong.is_empty(),
            "{name}: {} of {compared} tokens take the wrong tier: {:?}",
            wrong.len(),
            &wrong[..wrong.len().min(12)]
        );
    }
}

/// Bundled Gemma 2 and Gemma 2 from its own `tokenizer.json` give one answer.
#[test]
fn gemma2_routes_agree_under_ordinary() {
    let Some(dir) = models_dir() else { return };
    let bundled = from_pretrained("gemma2").expect("the vocabulary loads");
    let from_file =
        from_json_path(dir.join("gemma-2-2b/tokenizer.json")).expect("the document loads");
    for text in [
        "<blockquote>",
        "a<blockquote>b",
        "<blockquote>x",
        " <blockquote>",
        "a\nb",
        "<bos>x",
        "<start_of_turn>user\nhi<end_of_turn>",
        "<pad><blockquote>",
    ] {
        // The file's template opens every sequence with `<bos>`. The bundled
        // vocabulary states no template.
        let templated = from_file.encode_with(text, &SpecialMode::Ordinary).unwrap();
        assert_eq!(templated[0], 2, "{text:?}");
        assert_eq!(
            bundled.encode_with(text, &SpecialMode::Ordinary).unwrap(),
            templated[1..],
            "{text:?}"
        );
    }
}
