//! Static document-frequency priors for Sieve's windowed scorer.
//!
//! These are heuristic priors, not corpus-measured truth. The English list is seeded from a
//! public 500-word common-English list and normalized for embedding in Rust; the identifier
//! list is hand-curated for mixed code / logs / config corpora. Use the lookup function below
//! as a smoothed prior, then blend it with on-the-fly DF from the scanned fresh set.
//!
//! Cargo.toml:
//! phf = { version = "0.11", features = ["macros"] }

#[rustfmt::skip]
pub static ENGLISH_DF_PRIOR: phf::Map<&'static str, f32> = phf::phf_map! {
    "the" => 0.3600f32,
    "of" => 0.3356f32,
    "to" => 0.3111f32,
    "and" => 0.2867f32,
    "a" => 0.2622f32,
    "in" => 0.2378f32,
    "is" => 0.2133f32,
    "it" => 0.1889f32,
    "you" => 0.1644f32,
    "that" => 0.1400f32,
    "he" => 0.1300f32,
    "was" => 0.1264f32,
    "for" => 0.1229f32,
    "on" => 0.1193f32,
    "are" => 0.1157f32,
    "with" => 0.1121f32,
    "as" => 0.1086f32,
    "i" => 0.1050f32,
    "his" => 0.1014f32,
    "they" => 0.0979f32,
    "be" => 0.0943f32,
    "at" => 0.0907f32,
    "one" => 0.0871f32,
    "have" => 0.0836f32,
    "this" => 0.0800f32,
    "from" => 0.0780f32,
    "or" => 0.0766f32,
    "had" => 0.0752f32,
    "by" => 0.0739f32,
    "hot" => 0.0725f32,
    "but" => 0.0711f32,
    "some" => 0.0698f32,
    "what" => 0.0684f32,
    "there" => 0.0670f32,
    "we" => 0.0656f32,
    "can" => 0.0643f32,
    "out" => 0.0629f32,
    "other" => 0.0615f32,
    "were" => 0.0601f32,
    "all" => 0.0587f32,
    "your" => 0.0574f32,
    "when" => 0.0560f32,
    "up" => 0.0546f32,
    "use" => 0.0532f32,
    "word" => 0.0519f32,
    "how" => 0.0505f32,
    "said" => 0.0491f32,
    "an" => 0.0478f32,
    "each" => 0.0464f32,
    "she" => 0.0450f32,
    "which" => 0.0440f32,
    "do" => 0.0436f32,
    "their" => 0.0431f32,
    "time" => 0.0427f32,
    "if" => 0.0422f32,
    "will" => 0.0418f32,
    "way" => 0.0413f32,
    "about" => 0.0409f32,
    "many" => 0.0404f32,
    "then" => 0.0400f32,
    "them" => 0.0395f32,
    "would" => 0.0391f32,
    "write" => 0.0386f32,
    "like" => 0.0382f32,
    "so" => 0.0377f32,
    "these" => 0.0373f32,
    "her" => 0.0368f32,
    "long" => 0.0364f32,
    "make" => 0.0359f32,
    "thing" => 0.0355f32,
    "see" => 0.0350f32,
    "him" => 0.0346f32,
    "two" => 0.0341f32,
    "has" => 0.0337f32,
    "look" => 0.0332f32,
    "more" => 0.0328f32,
    "day" => 0.0323f32,
    "could" => 0.0319f32,
    "go" => 0.0314f32,
    "come" => 0.0310f32,
    "did" => 0.0305f32,
    "my" => 0.0301f32,
    "sound" => 0.0296f32,
    "no" => 0.0292f32,
    "most" => 0.0287f32,
    "number" => 0.0283f32,
    "who" => 0.0278f32,
    "over" => 0.0274f32,
    "know" => 0.0269f32,
    "water" => 0.0265f32,
    "than" => 0.0260f32,
    "call" => 0.0256f32,
    "first" => 0.0251f32,
    "people" => 0.0247f32,
    "may" => 0.0242f32,
    "down" => 0.0238f32,
    "side" => 0.0233f32,
    "been" => 0.0229f32,
    "now" => 0.0224f32,
    "find" => 0.0220f32,
    "any" => 0.0210f32,
    "new" => 0.0209f32,
    "work" => 0.0208f32,
    "part" => 0.0207f32,
    "take" => 0.0206f32,
    "get" => 0.0204f32,
    "place" => 0.0203f32,
    "made" => 0.0202f32,
    "live" => 0.0201f32,
    "where" => 0.0200f32,
    "after" => 0.0199f32,
    "back" => 0.0198f32,
    "little" => 0.0197f32,
    "only" => 0.0196f32,
    "round" => 0.0194f32,
    "man" => 0.0193f32,
    "year" => 0.0192f32,
    "came" => 0.0191f32,
    "show" => 0.0190f32,
    "every" => 0.0189f32,
    "good" => 0.0188f32,
    "me" => 0.0187f32,
    "give" => 0.0186f32,
    "our" => 0.0184f32,
    "under" => 0.0183f32,
    "name" => 0.0182f32,
    "very" => 0.0181f32,
    "through" => 0.0180f32,
    "just" => 0.0179f32,
    "form" => 0.0178f32,
    "much" => 0.0177f32,
    "great" => 0.0176f32,
    "think" => 0.0174f32,
    "say" => 0.0173f32,
    "help" => 0.0172f32,
    "low" => 0.0171f32,
    "line" => 0.0170f32,
    "before" => 0.0169f32,
    "turn" => 0.0168f32,
    "cause" => 0.0167f32,
    "same" => 0.0166f32,
    "mean" => 0.0164f32,
    "differ" => 0.0163f32,
    "move" => 0.0162f32,
    "right" => 0.0161f32,
    "boy" => 0.0160f32,
    "old" => 0.0159f32,
    "too" => 0.0158f32,
    "does" => 0.0157f32,
    "tell" => 0.0156f32,
    "sentence" => 0.0154f32,
    "set" => 0.0153f32,
    "three" => 0.0152f32,
    "want" => 0.0151f32,
    "air" => 0.0150f32,
    "well" => 0.0149f32,
    "also" => 0.0148f32,
    "play" => 0.0147f32,
    "small" => 0.0146f32,
    "end" => 0.0144f32,
    "put" => 0.0143f32,
    "home" => 0.0142f32,
    "read" => 0.0141f32,
    "hand" => 0.0140f32,
    "port" => 0.0139f32,
    "large" => 0.0138f32,
    "spell" => 0.0137f32,
    "add" => 0.0136f32,
    "even" => 0.0134f32,
    "land" => 0.0133f32,
    "here" => 0.0132f32,
    "must" => 0.0131f32,
    "big" => 0.0130f32,
    "high" => 0.0129f32,
    "such" => 0.0128f32,
    "follow" => 0.0127f32,
    "act" => 0.0126f32,
    "why" => 0.0124f32,
    "ask" => 0.0123f32,
    "men" => 0.0122f32,
    "change" => 0.0121f32,
    "went" => 0.0120f32,
    "light" => 0.0119f32,
    "kind" => 0.0118f32,
    "off" => 0.0117f32,
    "need" => 0.0116f32,
    "house" => 0.0114f32,
    "picture" => 0.0113f32,
    "try" => 0.0112f32,
    "us" => 0.0111f32,
    "again" => 0.0110f32,
    "animal" => 0.0109f32,
    "point" => 0.0108f32,
    "mother" => 0.0107f32,
    "world" => 0.0106f32,
    "near" => 0.0104f32,
    "build" => 0.0103f32,
    "self" => 0.0102f32,
    "earth" => 0.0101f32,
    "father" => 0.0100f32,
    "head" => 0.0098f32,
    "stand" => 0.0098f32,
    "own" => 0.0097f32,
    "page" => 0.0097f32,
    "should" => 0.0096f32,
    "country" => 0.0096f32,
    "found" => 0.0096f32,
    "answer" => 0.0095f32,
    "school" => 0.0095f32,
    "grow" => 0.0095f32,
    "study" => 0.0094f32,
    "still" => 0.0094f32,
    "learn" => 0.0093f32,
    "plant" => 0.0093f32,
    "cover" => 0.0093f32,
    "food" => 0.0092f32,
    "sun" => 0.0092f32,
    "four" => 0.0091f32,
    "thought" => 0.0091f32,
    "let" => 0.0091f32,
    "keep" => 0.0090f32,
    "eye" => 0.0090f32,
    "never" => 0.0090f32,
    "last" => 0.0089f32,
    "door" => 0.0089f32,
    "between" => 0.0088f32,
    "city" => 0.0088f32,
    "tree" => 0.0088f32,
    "cross" => 0.0087f32,
    "since" => 0.0087f32,
    "hard" => 0.0086f32,
    "start" => 0.0086f32,
    "might" => 0.0086f32,
    "story" => 0.0085f32,
    "saw" => 0.0085f32,
    "far" => 0.0085f32,
    "sea" => 0.0084f32,
    "draw" => 0.0084f32,
    "left" => 0.0083f32,
    "late" => 0.0083f32,
    "run" => 0.0083f32,
    "don't" => 0.0082f32,
    "while" => 0.0082f32,
    "press" => 0.0081f32,
    "close" => 0.0081f32,
    "night" => 0.0081f32,
    "real" => 0.0080f32,
    "life" => 0.0080f32,
    "few" => 0.0080f32,
    "stop" => 0.0079f32,
    "open" => 0.0079f32,
    "seem" => 0.0078f32,
    "together" => 0.0078f32,
    "next" => 0.0078f32,
    "white" => 0.0077f32,
    "children" => 0.0077f32,
    "begin" => 0.0077f32,
    "got" => 0.0076f32,
    "walk" => 0.0076f32,
    "example" => 0.0075f32,
    "ease" => 0.0075f32,
    "paper" => 0.0075f32,
    "often" => 0.0074f32,
    "always" => 0.0074f32,
    "music" => 0.0073f32,
    "those" => 0.0073f32,
    "both" => 0.0073f32,
    "mark" => 0.0072f32,
    "book" => 0.0072f32,
    "letter" => 0.0072f32,
    "until" => 0.0071f32,
    "mile" => 0.0071f32,
    "river" => 0.0070f32,
    "car" => 0.0070f32,
    "feet" => 0.0070f32,
    "care" => 0.0069f32,
    "second" => 0.0069f32,
    "group" => 0.0068f32,
    "carry" => 0.0068f32,
    "took" => 0.0068f32,
    "rain" => 0.0067f32,
    "eat" => 0.0067f32,
    "room" => 0.0067f32,
    "friend" => 0.0066f32,
    "began" => 0.0066f32,
    "idea" => 0.0065f32,
    "fish" => 0.0065f32,
    "mountain" => 0.0065f32,
    "north" => 0.0064f32,
    "once" => 0.0064f32,
    "base" => 0.0063f32,
    "hear" => 0.0063f32,
    "horse" => 0.0063f32,
    "cut" => 0.0062f32,
    "sure" => 0.0062f32,
    "watch" => 0.0062f32,
    "color" => 0.0061f32,
    "face" => 0.0061f32,
    "wood" => 0.0060f32,
    "main" => 0.0060f32,
    "enough" => 0.0058f32,
    "plain" => 0.0058f32,
    "girl" => 0.0058f32,
    "usual" => 0.0057f32,
    "young" => 0.0057f32,
    "ready" => 0.0057f32,
    "above" => 0.0057f32,
    "ever" => 0.0057f32,
    "red" => 0.0057f32,
    "list" => 0.0056f32,
    "though" => 0.0056f32,
    "feel" => 0.0056f32,
    "talk" => 0.0056f32,
    "bird" => 0.0056f32,
    "soon" => 0.0055f32,
    "body" => 0.0055f32,
    "dog" => 0.0055f32,
    "family" => 0.0055f32,
    "direct" => 0.0055f32,
    "pose" => 0.0055f32,
    "leave" => 0.0054f32,
    "song" => 0.0054f32,
    "measure" => 0.0054f32,
    "state" => 0.0054f32,
    "product" => 0.0054f32,
    "black" => 0.0053f32,
    "short" => 0.0053f32,
    "numeral" => 0.0053f32,
    "class" => 0.0053f32,
    "wind" => 0.0053f32,
    "question" => 0.0053f32,
    "happen" => 0.0052f32,
    "complete" => 0.0052f32,
    "ship" => 0.0052f32,
    "area" => 0.0052f32,
    "half" => 0.0052f32,
    "rock" => 0.0051f32,
    "order" => 0.0051f32,
    "fire" => 0.0051f32,
    "south" => 0.0051f32,
    "problem" => 0.0051f32,
    "piece" => 0.0051f32,
    "told" => 0.0050f32,
    "knew" => 0.0050f32,
    "pass" => 0.0050f32,
    "farm" => 0.0050f32,
    "top" => 0.0050f32,
    "whole" => 0.0049f32,
    "king" => 0.0049f32,
    "size" => 0.0049f32,
    "heard" => 0.0049f32,
    "best" => 0.0049f32,
    "hour" => 0.0049f32,
    "better" => 0.0048f32,
    "true" => 0.0048f32,
    "during" => 0.0048f32,
    "hundred" => 0.0048f32,
    "am" => 0.0048f32,
    "remember" => 0.0047f32,
    "step" => 0.0047f32,
    "early" => 0.0047f32,
    "hold" => 0.0047f32,
    "west" => 0.0047f32,
    "ground" => 0.0047f32,
    "interest" => 0.0046f32,
    "reach" => 0.0046f32,
    "fast" => 0.0046f32,
    "five" => 0.0046f32,
    "sing" => 0.0046f32,
    "listen" => 0.0045f32,
    "six" => 0.0045f32,
    "table" => 0.0045f32,
    "travel" => 0.0045f32,
    "less" => 0.0045f32,
    "morning" => 0.0045f32,
    "ten" => 0.0044f32,
    "simple" => 0.0044f32,
    "several" => 0.0044f32,
    "vowel" => 0.0044f32,
    "toward" => 0.0044f32,
    "war" => 0.0043f32,
    "lay" => 0.0043f32,
    "against" => 0.0043f32,
    "pattern" => 0.0043f32,
    "slow" => 0.0043f32,
    "center" => 0.0043f32,
    "love" => 0.0042f32,
    "person" => 0.0042f32,
    "money" => 0.0042f32,
    "serve" => 0.0042f32,
    "appear" => 0.0042f32,
    "road" => 0.0041f32,
    "map" => 0.0041f32,
    "science" => 0.0041f32,
    "rule" => 0.0041f32,
    "govern" => 0.0041f32,
    "pull" => 0.0041f32,
    "cold" => 0.0040f32,
    "notice" => 0.0040f32,
    "voice" => 0.0040f32,
    "fall" => 0.0039f32,
    "power" => 0.0039f32,
    "town" => 0.0039f32,
    "fine" => 0.0039f32,
    "certain" => 0.0038f32,
    "fly" => 0.0038f32,
    "unit" => 0.0038f32,
    "lead" => 0.0038f32,
    "cry" => 0.0038f32,
    "dark" => 0.0038f32,
    "machine" => 0.0038f32,
    "note" => 0.0037f32,
    "wait" => 0.0037f32,
    "plan" => 0.0037f32,
    "figure" => 0.0037f32,
    "star" => 0.0037f32,
    "box" => 0.0037f32,
    "noun" => 0.0037f32,
    "field" => 0.0036f32,
    "rest" => 0.0036f32,
    "correct" => 0.0036f32,
    "able" => 0.0036f32,
    "pound" => 0.0036f32,
    "done" => 0.0036f32,
    "beauty" => 0.0036f32,
    "drive" => 0.0035f32,
    "stood" => 0.0035f32,
    "contain" => 0.0035f32,
    "front" => 0.0035f32,
    "teach" => 0.0035f32,
    "week" => 0.0035f32,
    "final" => 0.0035f32,
    "gave" => 0.0034f32,
    "green" => 0.0034f32,
    "oh" => 0.0034f32,
    "quick" => 0.0034f32,
    "develop" => 0.0034f32,
    "sleep" => 0.0034f32,
    "warm" => 0.0034f32,
    "free" => 0.0033f32,
    "minute" => 0.0033f32,
    "strong" => 0.0033f32,
    "special" => 0.0033f32,
    "mind" => 0.0033f32,
    "behind" => 0.0033f32,
    "clear" => 0.0033f32,
    "tail" => 0.0032f32,
    "produce" => 0.0032f32,
    "fact" => 0.0032f32,
    "street" => 0.0032f32,
    "inch" => 0.0032f32,
    "lot" => 0.0032f32,
    "nothing" => 0.0032f32,
    "course" => 0.0032f32,
    "stay" => 0.0031f32,
    "wheel" => 0.0031f32,
    "full" => 0.0031f32,
    "force" => 0.0031f32,
    "blue" => 0.0031f32,
    "object" => 0.0031f32,
    "decide" => 0.0031f32,
    "surface" => 0.0030f32,
    "deep" => 0.0030f32,
    "moon" => 0.0030f32,
    "island" => 0.0030f32,
    "foot" => 0.0030f32,
    "yet" => 0.0030f32,
    "busy" => 0.0030f32,
    "test" => 0.0029f32,
    "record" => 0.0029f32,
    "boat" => 0.0029f32,
    "common" => 0.0029f32,
    "gold" => 0.0029f32,
    "possible" => 0.0029f32,
    "plane" => 0.0029f32,
    "age" => 0.0028f32,
    "dry" => 0.0028f32,
    "wonder" => 0.0028f32,
    "laugh" => 0.0028f32,
    "thousand" => 0.0028f32,
    "ago" => 0.0028f32,
    "ran" => 0.0028f32,
    "check" => 0.0027f32,
    "game" => 0.0027f32,
    "shape" => 0.0027f32,
    "yes" => 0.0027f32,
    "miss" => 0.0027f32,
    "brought" => 0.0027f32,
    "heat" => 0.0027f32,
    "snow" => 0.0026f32,
    "bed" => 0.0026f32,
    "bring" => 0.0026f32,
    "sit" => 0.0026f32,
    "perhaps" => 0.0026f32,
    "fill" => 0.0026f32,
    "east" => 0.0026f32,
    "weight" => 0.0025f32,
    "language" => 0.0025f32,
    "among" => 0.0025f32,
    "because" => 0.0025f32,
};

#[rustfmt::skip]
pub static IDENTIFIER_DF_PRIOR: phf::Map<&'static str, f32> = phf::phf_map! {
    "data" => 0.0950f32,
    "value" => 0.0926f32,
    "values" => 0.0903f32,
    "name" => 0.0879f32,
    "id" => 0.0855f32,
    "key" => 0.0832f32,
    "keys" => 0.0808f32,
    "type" => 0.0784f32,
    "types" => 0.0761f32,
    "error" => 0.0737f32,
    "errors" => 0.0713f32,
    "result" => 0.0689f32,
    "results" => 0.0666f32,
    "config" => 0.0642f32,
    "request" => 0.0618f32,
    "response" => 0.0595f32,
    "status" => 0.0571f32,
    "state" => 0.0547f32,
    "handler" => 0.0524f32,
    "handlers" => 0.0500f32,
    "client" => 0.0490f32,
    "server" => 0.0483f32,
    "user" => 0.0476f32,
    "users" => 0.0468f32,
    "file" => 0.0461f32,
    "files" => 0.0454f32,
    "path" => 0.0447f32,
    "list" => 0.0439f32,
    "map" => 0.0432f32,
    "item" => 0.0425f32,
    "items" => 0.0418f32,
    "count" => 0.0410f32,
    "size" => 0.0403f32,
    "index" => 0.0396f32,
    "test" => 0.0389f32,
    "tests" => 0.0381f32,
    "message" => 0.0374f32,
    "messages" => 0.0367f32,
    "event" => 0.0360f32,
    "events" => 0.0352f32,
    "option" => 0.0345f32,
    "options" => 0.0338f32,
    "param" => 0.0331f32,
    "params" => 0.0323f32,
    "query" => 0.0316f32,
    "queries" => 0.0309f32,
    "header" => 0.0302f32,
    "headers" => 0.0294f32,
    "body" => 0.0287f32,
    "cache" => 0.0280f32,
    "store" => 0.0270f32,
    "storage" => 0.0267f32,
    "service" => 0.0265f32,
    "services" => 0.0262f32,
    "controller" => 0.0259f32,
    "manager" => 0.0257f32,
    "model" => 0.0254f32,
    "entity" => 0.0251f32,
    "record" => 0.0249f32,
    "repo" => 0.0246f32,
    "repository" => 0.0243f32,
    "db" => 0.0241f32,
    "database" => 0.0238f32,
    "sql" => 0.0236f32,
    "table" => 0.0233f32,
    "schema" => 0.0230f32,
    "row" => 0.0228f32,
    "column" => 0.0225f32,
    "parser" => 0.0222f32,
    "encoder" => 0.0220f32,
    "decoder" => 0.0217f32,
    "serializer" => 0.0214f32,
    "deserializer" => 0.0212f32,
    "format" => 0.0209f32,
    "formatter" => 0.0206f32,
    "reader" => 0.0204f32,
    "writer" => 0.0201f32,
    "stream" => 0.0198f32,
    "buffer" => 0.0196f32,
    "input" => 0.0193f32,
    "output" => 0.0190f32,
    "ctx" => 0.0188f32,
    "context" => 0.0185f32,
    "env" => 0.0182f32,
    "auth" => 0.0180f32,
    "token" => 0.0177f32,
    "session" => 0.0174f32,
    "cookie" => 0.0172f32,
    "api" => 0.0169f32,
    "router" => 0.0167f32,
    "route" => 0.0164f32,
    "endpoint" => 0.0161f32,
    "middleware" => 0.0159f32,
    "callback" => 0.0156f32,
    "job" => 0.0153f32,
    "task" => 0.0151f32,
    "tasks" => 0.0148f32,
    "worker" => 0.0145f32,
    "process" => 0.0143f32,
    "thread" => 0.0140f32,
    "pool" => 0.0135f32,
    "lock" => 0.0134f32,
    "mutex" => 0.0133f32,
    "queue" => 0.0132f32,
    "stack" => 0.0131f32,
    "tree" => 0.0130f32,
    "node" => 0.0129f32,
    "graph" => 0.0128f32,
    "module" => 0.0127f32,
    "modules" => 0.0126f32,
    "package" => 0.0125f32,
    "packages" => 0.0124f32,
    "crate" => 0.0123f32,
    "lib" => 0.0122f32,
    "library" => 0.0121f32,
    "app" => 0.0120f32,
    "main" => 0.0119f32,
    "init" => 0.0118f32,
    "start" => 0.0117f32,
    "stop" => 0.0116f32,
    "create" => 0.0115f32,
    "update" => 0.0114f32,
    "delete" => 0.0113f32,
    "remove" => 0.0112f32,
    "add" => 0.0111f32,
    "get" => 0.0109f32,
    "set" => 0.0108f32,
    "build" => 0.0107f32,
    "run" => 0.0106f32,
    "exec" => 0.0105f32,
    "check" => 0.0104f32,
    "validate" => 0.0103f32,
    "validator" => 0.0102f32,
    "spec" => 0.0101f32,
    "fixture" => 0.0100f32,
    "mock" => 0.0099f32,
    "stub" => 0.0098f32,
    "util" => 0.0097f32,
    "utils" => 0.0096f32,
    "helper" => 0.0095f32,
    "helpers" => 0.0094f32,
    "common" => 0.0093f32,
    "core" => 0.0092f32,
    "base" => 0.0091f32,
    "object" => 0.0090f32,
    "objects" => 0.0089f32,
    "class" => 0.0088f32,
    "interface" => 0.0087f32,
    "impl" => 0.0086f32,
    "implementation" => 0.0085f32,
    "raw" => 0.0082f32,
    "default" => 0.0081f32,
    "active" => 0.0081f32,
    "enabled" => 0.0080f32,
    "disabled" => 0.0079f32,
    "temp" => 0.0079f32,
    "current" => 0.0078f32,
    "next" => 0.0077f32,
    "prev" => 0.0077f32,
    "first" => 0.0076f32,
    "last" => 0.0075f32,
    "min" => 0.0075f32,
    "max" => 0.0074f32,
    "total" => 0.0074f32,
    "time" => 0.0073f32,
    "date" => 0.0072f32,
    "timestamp" => 0.0072f32,
    "version" => 0.0071f32,
    "meta" => 0.0070f32,
    "metadata" => 0.0070f32,
    "debug" => 0.0069f32,
    "trace" => 0.0068f32,
    "log" => 0.0068f32,
    "logger" => 0.0067f32,
    "exception" => 0.0066f32,
    "fail" => 0.0066f32,
    "failure" => 0.0065f32,
    "retry" => 0.0064f32,
    "timeout" => 0.0064f32,
    "fallback" => 0.0063f32,
    "reason" => 0.0062f32,
    "cause" => 0.0062f32,
    "code" => 0.0061f32,
    "flag" => 0.0060f32,
    "feature" => 0.0060f32,
    "cfg" => 0.0059f32,
    "msg" => 0.0058f32,
    "req" => 0.0058f32,
    "resp" => 0.0057f32,
    "svc" => 0.0057f32,
    "dto" => 0.0056f32,
    "uid" => 0.0055f32,
    "uuid" => 0.0055f32,
    "url" => 0.0054f32,
    "uri" => 0.0053f32,
    "json" => 0.0053f32,
    "yaml" => 0.0052f32,
    "xml" => 0.0051f32,
    "http" => 0.0051f32,
    "https" => 0.0050f32,
};

/// Lookup a token's static DF prior in the range [0.0, 1.0].
///
/// Rules:
/// - direct exact hit on the normalized token wins
/// - otherwise, if the token is a composite identifier (snake/camel/kebab), split it and
///   derive a prior from the parts
/// - otherwise, fall back to a shape-based default
pub fn static_df_frac(token: &str) -> f32 {
    let raw = trim_df_token(token);
    if raw.is_empty() {
        return 0.20;
    }

    let norm = raw.to_ascii_lowercase();

    if let Some(v) = direct_df_lookup(&norm) {
        return v;
    }

    let parts = split_identifier_parts(raw);
    if parts.len() > 1 {
        let mut max_v = 0.0f32;
        let mut sum_v = 0.0f32;
        let mut n = 0usize;

        for part in parts {
            if part.is_empty() {
                continue;
            }
            let v = direct_df_lookup(&part).unwrap_or_else(|| default_df_frac(&part));
            max_v = max_v.max(v);
            sum_v += v;
            n += 1;
        }

        if n > 0 {
            let mean_v = sum_v / n as f32;
            return (0.60 * max_v + 0.40 * mean_v).clamp(0.0015, 0.20);
        }
    }

    default_df_frac(raw)
}

#[inline]
fn direct_df_lookup(token: &str) -> Option<f32> {
    let english = ENGLISH_DF_PRIOR.get(token).copied();
    let ident = IDENTIFIER_DF_PRIOR.get(token).copied();
    match (english, ident) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => {
            if token.contains('\'') {
                let squashed = token.replace('\'', "");
                let english2 = ENGLISH_DF_PRIOR.get(squashed.as_str()).copied();
                let ident2 = IDENTIFIER_DF_PRIOR.get(squashed.as_str()).copied();
                match (english2, ident2) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            } else {
                None
            }
        }
    }
}

/// Fallback for unknown tokens.
///
/// Intuition:
/// - very short alphabetic tokens are usually common
/// - mixed-case / separated identifiers are rarer than plain words
/// - long hex-like / numeric strings are extremely rare
/// - unknown long alphabetic words are uncommon but not impossible
pub fn default_df_frac(token: &str) -> f32 {
    let t = token.trim();
    if t.is_empty() {
        return 0.20;
    }

    if t.bytes().all(|b| b.is_ascii_digit()) {
        return 0.0015;
    }

    if looks_like_hex(t) {
        return 0.0008;
    }

    let has_sep = t.contains('_') || t.contains('-');
    let camel = is_camel_like(t);
    let upper_acronym = is_upper_acronym(t);

    if has_sep || camel {
        return match t.len() {
            0..=3 => 0.0100,
            4..=7 => 0.0055,
            8..=15 => 0.0035,
            _ => 0.0025,
        };
    }

    if upper_acronym {
        return 0.0030;
    }

    match t.len() {
        0..=2 => 0.0500,
        3 => 0.0250,
        4 => 0.0120,
        5..=7 => 0.0060,
        8..=12 => 0.0030,
        _ => 0.0015,
    }
}

#[inline]
pub fn trim_df_token(token: &str) -> &str {
    token.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '\''))
}

#[inline]
pub fn normalize_df_token(token: &str) -> String {
    trim_df_token(token).to_ascii_lowercase()
}

fn looks_like_hex(s: &str) -> bool {
    s.len() >= 8 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn is_upper_acronym(s: &str) -> bool {
    let bytes = s.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 6
        && bytes.iter().any(|b| b.is_ascii_uppercase())
        && bytes
            .iter()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

fn is_camel_like(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut has_lower = false;
    let mut has_upper = false;
    for &b in bytes {
        has_lower |= b.is_ascii_lowercase();
        has_upper |= b.is_ascii_uppercase();
    }
    has_lower && has_upper
}

fn split_identifier_parts(s: &str) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }

    let mut parts = Vec::new();
    let mut buf = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = if i + 1 < chars.len() {
            Some(chars[i + 1])
        } else {
            None
        };

        let boundary = match prev {
            None => false,
            Some(p) => {
                (c == '_' || c == '-')
                    || (p.is_ascii_lowercase() && c.is_ascii_uppercase())
                    || (p.is_ascii_alphabetic() && c.is_ascii_digit())
                    || (p.is_ascii_digit() && c.is_ascii_alphabetic())
                    || (p.is_ascii_uppercase()
                        && c.is_ascii_uppercase()
                        && next.map(|n| n.is_ascii_lowercase()).unwrap_or(false))
            }
        };

        if c == '_' || c == '-' {
            if !buf.is_empty() {
                parts.push(std::mem::take(&mut buf));
            }
            continue;
        }

        if boundary && !buf.is_empty() {
            parts.push(std::mem::take(&mut buf));
        }

        buf.push(c.to_ascii_lowercase());
    }

    if !buf.is_empty() {
        parts.push(buf);
    }

    parts
}
