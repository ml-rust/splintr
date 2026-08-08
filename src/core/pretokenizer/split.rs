use super::spec::Behavior;

/// A starting capacity for a piece's worth of pre-tokens.
///
/// Sizing the output from the *input* piece count instead — which is 1 for the
/// first stage, since it is handed the whole text — meant growing from one
/// element to a few dozen on every call, and that reallocation was a tenth of
/// `from_json` encode time. Four bytes per pre-token is about right for prose
/// and code across the bundled vocabularies; being wrong only costs the growth
/// this avoids.
#[inline]
pub(crate) fn estimated_pieces(piece: &str) -> usize {
    piece.len() / 4 + 8
}

/// Split `piece` by `matcher`. The matched spans are the delimiters and
/// everything between them is content; with `invert`, the matches are the
/// content and the spans between them are the delimiters. The delimiters are
/// combined with the surrounding content according to `behavior`.
pub(super) fn split_regex<'p>(
    piece: &'p str,
    matcher: &super::stage::SplitMatcher,
    behavior: Behavior,
    invert: bool,
    out: &mut Vec<&'p str>,
) {
    let mut matches: Vec<(usize, usize)> = Vec::with_capacity(estimated_pieces(piece));
    matcher.matches(piece, &mut matches);

    // Two shapes cover every bundled `tokenizer.json`, and neither needs a
    // `Segment` built and classified only to be emitted or dropped wholesale.
    match behavior {
        // Keeps every segment, delimiter or not, so the result is just the piece
        // partitioned at the match edges. Which side is the delimiter never
        // comes up, `invert` included: it only swaps a label nothing reads.
        // Qwen's file is this shape.
        Behavior::Isolated => {
            let mut last = 0;
            for (s, e) in matches {
                if s > last {
                    out.push(&piece[last..s]);
                }
                if e > s {
                    out.push(&piece[s..e]);
                }
                last = e;
            }
            if last < piece.len() {
                out.push(&piece[last..]);
            }
            return;
        }
        // Drops the delimiters, so exactly one side survives — the matches when
        // inverted, the gaps between them otherwise. Both OpenAI files are the
        // inverted form, which makes this the hot path for GPT-4 and GPT-4o:
        // the output is the match list and nothing else.
        Behavior::Removed => {
            if invert {
                for (s, e) in matches {
                    if e > s {
                        out.push(&piece[s..e]);
                    }
                }
            } else {
                let mut last = 0;
                for (s, e) in matches {
                    if s > last {
                        out.push(&piece[last..s]);
                    }
                    last = e;
                }
                if last < piece.len() {
                    out.push(&piece[last..]);
                }
            }
            return;
        }
        _ => {}
    }

    // At most one gap segment before each match, plus a trailing one.
    let mut segs: Vec<Segment> = Vec::with_capacity(matches.len() * 2 + 1);

    // Flatten into an ordered list of (range, is_delimiter) segments.
    //
    // `invert` only swaps which side is the delimiter: normally the matches are
    // the delimiters and the gaps between them are content, and inverted it is
    // the other way round. It must not be implemented by deriving the gaps and
    // re-complementing them — the boundary between two *adjacent* matches is not
    // a gap, so complementing merges them into one span. These pre-tokenizer
    // patterns match contiguously across the whole input (` ?\p{L}+|\s+|…`),
    // which is exactly the case with no gaps at all, and merging there means no
    // splitting whatsoever.
    let mut last = 0;
    for (s, e) in matches {
        if s > last {
            segs.push(Segment {
                start: last,
                end: s,
                delimiter: invert,
            });
        }
        if e > s {
            segs.push(Segment {
                start: s,
                end: e,
                delimiter: !invert,
            });
        }
        last = e;
    }
    if last < piece.len() {
        segs.push(Segment {
            start: last,
            end: piece.len(),
            delimiter: invert,
        });
    }

    emit_segments(piece, &segs, behavior, out);
}

/// One span of a piece, and whether the split matched it as a delimiter.
///
/// Carried as a range rather than a `&str` because every delimiter behavior
/// combines *adjacent* segments, and adjacent ranges merge into one wider range
/// — so combining never has to concatenate, and every piece the pre-tokenizer
/// emits stays a subslice of its input.
#[derive(Debug, Clone, Copy)]
struct Segment {
    start: usize,
    end: usize,
    delimiter: bool,
}

/// Combine ordered (text, is_delimiter) segments per the HF delimiter behavior,
/// appending the resulting pieces to `out`.
fn emit_segments<'p>(piece: &'p str, segs: &[Segment], behavior: Behavior, out: &mut Vec<&'p str>) {
    // Segments partition `piece` in order, so any two that are combined are
    // adjacent and their union is the single range from the first's start to
    // the last's end. Every behavior below is therefore a range merge, and
    // `push` is always a subslice — never a concatenation.
    let mut push = |start: usize, end: usize| out.push(&piece[start..end]);

    match behavior {
        Behavior::Isolated => {
            for seg in segs {
                push(seg.start, seg.end);
            }
        }
        Behavior::Removed => {
            for seg in segs.iter().filter(|seg| !seg.delimiter) {
                push(seg.start, seg.end);
            }
        }
        Behavior::MergedWithPrevious => {
            // A delimiter attaches to the preceding emitted piece (from this
            // split only); a leading delimiter with no predecessor stands alone.
            let mut open: Option<(usize, usize)> = None;
            for seg in segs {
                match (&mut open, seg.delimiter) {
                    (Some((_, end)), true) => *end = seg.end,
                    _ => {
                        if let Some((start, end)) = open.replace((seg.start, seg.end)) {
                            push(start, end);
                        }
                    }
                }
            }
            if let Some((start, end)) = open {
                push(start, end);
            }
        }
        Behavior::MergedWithNext => {
            // A delimiter attaches to the following piece; trailing delimiters
            // with no successor stand alone.
            let mut pending: Option<usize> = None;
            for seg in segs {
                if seg.delimiter {
                    pending.get_or_insert(seg.start);
                } else {
                    push(pending.take().unwrap_or(seg.start), seg.end);
                }
            }
            if let Some(start) = pending {
                push(start, segs[segs.len() - 1].end);
            }
        }
        Behavior::Contiguous => {
            // Runs of adjacent delimiters merge into one piece; content stays
            // split. (Adjacent delimiters arise from back-to-back matches.)
            let mut run: Option<(usize, usize)> = None;
            for seg in segs {
                if seg.delimiter {
                    match &mut run {
                        Some((_, end)) => *end = seg.end,
                        None => run = Some((seg.start, seg.end)),
                    }
                } else {
                    if let Some((start, end)) = run.take() {
                        push(start, end);
                    }
                    push(seg.start, seg.end);
                }
            }
            if let Some((start, end)) = run {
                push(start, end);
            }
        }
    }
}

pub(super) fn split_digits<'p>(piece: &'p str, individual: bool, out: &mut Vec<&'p str>) {
    // Tracks the open run as a range instead of accumulating a String, so each
    // emitted piece is a subslice of `piece`.
    let mut start: Option<usize> = None;
    let mut run_is_digit = false;
    for (i, c) in piece.char_indices() {
        // HF's Digits pre-tokenizer uses Unicode numericity (`char::is_numeric`,
        // categories Nd/Nl/No), so superscripts/fractions like ²/³/½ count as
        // digits — not just ASCII 0-9.
        let d = c.is_numeric();
        if let Some(open) = start {
            if d != run_is_digit || (d && individual) {
                out.push(&piece[open..i]);
                start = None;
            }
        }
        let open = *start.get_or_insert(i);
        run_is_digit = d;
        if d && individual {
            out.push(&piece[open..i + c.len_utf8()]);
            start = None;
        }
    }
    if let Some(open) = start {
        out.push(&piece[open..]);
    }
}

pub(super) fn split_punctuation<'p>(piece: &'p str, behavior: Behavior, out: &mut Vec<&'p str>) {
    // Each punctuation char is a delimiter segment; consecutive non-punctuation
    // chars form a content segment. The behavior then combines them.
    let mut segs: Vec<Segment> = Vec::new();
    let mut content_start = 0;
    let mut i = 0;
    for c in piece.chars() {
        let len = c.len_utf8();
        if is_punctuation(c) {
            if i > content_start {
                segs.push(Segment {
                    start: content_start,
                    end: i,
                    delimiter: false,
                });
            }
            segs.push(Segment {
                start: i,
                end: i + len,
                delimiter: true,
            });
            content_start = i + len;
        }
        i += len;
    }
    if i > content_start {
        segs.push(Segment {
            start: content_start,
            end: i,
            delimiter: false,
        });
    }
    emit_segments(piece, &segs, behavior, out);
}

/// HF punctuation definition (ASCII punctuation + Unicode P* categories).
fn is_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    use unicode_general_category::{get_general_category, GeneralCategory::*};
    matches!(
        get_general_category(c),
        ConnectorPunctuation
            | DashPunctuation
            | ClosePunctuation
            | FinalPunctuation
            | InitialPunctuation
            | OtherPunctuation
            | OpenPunctuation
    )
}
