import json
import os
import re
from copy import deepcopy


DEFAULT_CATEGORY_ALIASES = {
    "dress": "dress",
    "dresses": "dress",
    "robe": "dress",
    "robes": "dress",
    "skirt": "skirt",
    "skirts": "skirt",
    "miniskirt": "skirt",
    "jacket": "jacket",
    "jackets": "jacket",
    "blazer": "jacket",
    "blazers": "jacket",
    "coat": "jacket",
    "coats": "jacket",
    "vest": "jacket",
    "vests": "jacket",
    "suit": "jacket",
    "suits": "jacket",
    "parka": "jacket",
    "parkas": "jacket",
    "pants": "pants",
    "leggings": "pants",
    "jeans": "pants",
    "shorts": "pants",
    "sweatpants": "pants",
    "trousers": "pants",
    "top": "top",
    "tops": "top",
    "shirt": "top",
    "shirts": "top",
    "blouse": "top",
    "blouses": "top",
    "t-shirt": "top",
    "t-shirts": "top",
    "tank": "top",
    "sweater": "top",
    "sweaters": "top",
    "turtleneck": "top",
    "turtlenecks": "top",
    "sweatshirt": "top",
    "sweatshirts": "top",
    "hoodie": "top",
    "hoodies": "top",
    "jumpsuit": "dress",
    "jumpsuits": "dress",
    "romper": "dress",
    "rompers": "dress",
    "cardigan": "top",
    "cardigans": "top",
    "tunic": "top",
    "tunics": "top",
}

DEFAULT_PRIORITY_WHEN_AMBIGUOUS = [
    "dress",
    "jacket",
    "skirt",
    "pants",
    "top",
]

DEFAULT_ATTRIBUTE_KEYWORDS = {
    "color": [
        "black",
        "white",
        "blue",
        "red",
        "green",
        "silver",
        "brown",
        "beige",
        "pink",
        "gray",
        "purple",
        "gold",
        "orange",
        "yellow",
    ],
    "material": [
        "denim",
        "cotton",
        "leather",
        "lace",
        "polyester",
        "spandex",
        "velvet",
        "fur",
        "faux",
        "silk",
        "stretchy material",
        "lightweight material",
        "shiny material",
        "soft material",
        "knit",
        "quilted",
        "tulle",
        "stretchy",
        "lightweight",
        "shiny",
        "soft",
        "blend",
    ],
    "pattern": [
        "polka dot",
        "floral",
        "striped",
        "plain",
        "check",
        "print",
        "pattern",
    ],
    "fit": [
        "loose fit",
        "fitted",
        "relaxed fit",
        "wide",
        "loose",
        "relaxed",
        "cropped",
        "straight",
        "pencil",
    ],
    "sleeve": [
        "long sleeves",
        "short sleeves",
        "long sleeve",
        "short sleeve",
        "sleeveless",
        "sleeves",
        "sleeve",
    ],
    "neckline": [
        "halter neckline",
        "v-neckline",
        "crew neckline",
        "strapless",
        "neckline",
        "collar",
        "halter",
        "round neckline",
        "rounded",
    ],
    "waist": [
        "high waist",
        "low waist",
        "fitted waist",
        "drawstring",
        "waistband",
        "waistline",
    ],
    "length": [
        "knee-length",
        "mini",
        "midi",
        "maxi",
    ],
    "closure": [
        "zipper",
        "button-down",
        "button",
        "buckle",
        "belt",
        "closure",
        "hood",
    ],
    "detail": [
        "pocket",
        "pockets",
        "bodice",
        "flowy",
        "flowing",
        "distressed",
        "pleated",
        "sheer",
        "slit",
        "ruffled",
        "overlay",
    ],
    "style": [
        "fashionable",
        "casual",
        "formal",
        "feminine",
        "silhouette",
    ],
}

DEFAULT_GLOBAL_ATTRIBUTES = ["color", "material", "pattern"]

MULTI_VALUE_GLOBAL_ATTRIBUTES = {"color", "pattern"}

DEFAULT_TYPE_SPECIFIC_ATTRIBUTES = {
    "dress": ["neckline", "sleeve", "length", "waist", "closure", "detail", "style"],
    "top": ["neckline", "sleeve", "fit", "closure", "detail", "style"],
    "skirt": ["length", "waist", "fit", "closure", "detail", "style"],
    "pants": ["waist", "fit", "length", "closure", "detail", "style"],
    "jacket": ["neckline", "sleeve", "fit", "closure", "detail", "style"],
}

DEFAULT_COMPARISON_PATTERNS = [
    r"\b(?:styled|looks?|dressed)\s+like\s+(?:a|an)\s+\w+",
    r"\blike\s+(?:a|an)\s+\w+",
    r"\bsimilar\s+to\s+(?:a|an)\s+\w+",
    r"\bresembl(?:es|ing)\s+(?:a|an)\s+\w+",
    r"\bas\s+if\s+it\s+(?:were|was)\s+(?:a|an)\s+\w+",
]

DEFAULT_VERB_DRESS_PATTERNS = [
    r"\bto\s+dress\b",
    r"\bdress(?:es|ed|ing)?\s+(?:up|down|casually|formally|smartly|well|nicely)\b",
    r"\bdress\s+code\b",
]

DEFAULT_VAGUE_MATERIAL_WORDS = {"fabric", "material", "type of fabric"}
DEFAULT_VAGUE_PATTERN_WORDS = {"pattern", "print"}
DEFAULT_VAGUE_STYLE_WORDS = {"silhouette"}

DEFAULT_GENERIC_MATERIAL_DESCRIPTORS = {
    "soft",
    "stretchy",
    "lightweight",
    "shiny",
    "blend",
    "soft material",
    "stretchy material",
    "lightweight material",
    "shiny material",
}

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CATALOG_DICT_PATH = os.path.join(_CURRENT_DIR, "catalog_dict.json")


def _load_catalog_dict():
    if not os.path.exists(_CATALOG_DICT_PATH):
        return {}

    with open(_CATALOG_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_keywords(attribute_keywords):
    return {
        attr: sorted(keywords, key=len, reverse=True)
        for attr, keywords in attribute_keywords.items()
    }


def _merge_type_attributes(global_attributes, type_specific_attributes):
    return {
        item_type: list(global_attributes) + list(attributes)
        for item_type, attributes in type_specific_attributes.items()
    }


def _split_type_attributes(type_attributes, global_attributes):
    global_set = set(global_attributes)
    return {
        item_type: [attr for attr in attributes if attr not in global_set]
        for item_type, attributes in type_attributes.items()
    }


_catalog_dict = _load_catalog_dict()

CATEGORY_ALIASES = dict(DEFAULT_CATEGORY_ALIASES)
CATEGORY_ALIASES.update(_catalog_dict.get("category_aliases", {}))

PRIORITY_WHEN_AMBIGUOUS = (
    _catalog_dict.get("priority_when_ambiguous")
    or _catalog_dict.get("priority_rules")
    or list(DEFAULT_PRIORITY_WHEN_AMBIGUOUS)
)

ATTRIBUTE_KEYWORDS = _sorted_keywords(
    deepcopy(_catalog_dict.get("attribute_keywords", DEFAULT_ATTRIBUTE_KEYWORDS))
)

GLOBAL_ATTRIBUTES = list(
    _catalog_dict.get("global_attributes", DEFAULT_GLOBAL_ATTRIBUTES)
)

TYPE_SPECIFIC_ATTRIBUTES = deepcopy(
    _catalog_dict.get("type_specific_attributes", {})
)
if not TYPE_SPECIFIC_ATTRIBUTES:
    if "type_attributes" in _catalog_dict:
        TYPE_SPECIFIC_ATTRIBUTES = _split_type_attributes(
            _catalog_dict["type_attributes"],
            GLOBAL_ATTRIBUTES,
        )
    else:
        TYPE_SPECIFIC_ATTRIBUTES = deepcopy(DEFAULT_TYPE_SPECIFIC_ATTRIBUTES)

TYPE_ATTRIBUTES = deepcopy(
    _catalog_dict.get(
        "type_attributes",
        _merge_type_attributes(GLOBAL_ATTRIBUTES, TYPE_SPECIFIC_ATTRIBUTES),
    )
)

COMPARISON_PATTERNS = list(
    _catalog_dict.get("comparison_patterns", DEFAULT_COMPARISON_PATTERNS)
)
VERB_DRESS_PATTERNS = list(
    _catalog_dict.get("verb_dress_patterns", DEFAULT_VERB_DRESS_PATTERNS)
)

VAGUE_MATERIAL_WORDS = set(
    _catalog_dict.get("vague_material_words", DEFAULT_VAGUE_MATERIAL_WORDS)
)
VAGUE_PATTERN_WORDS = set(
    _catalog_dict.get("vague_pattern_words", DEFAULT_VAGUE_PATTERN_WORDS)
)
VAGUE_STYLE_WORDS = set(
    _catalog_dict.get("vague_style_words", DEFAULT_VAGUE_STYLE_WORDS)
)
GENERIC_MATERIAL_DESCRIPTORS = set(
    _catalog_dict.get(
        "generic_material_descriptors",
        DEFAULT_GENERIC_MATERIAL_DESCRIPTORS,
    )
)

_CATEGORY_ALIAS_ITEMS = sorted(
    CATEGORY_ALIASES.items(),
    key=lambda item: len(item[0]),
    reverse=True,
)

MULTI_ITEM_CONNECTOR_PATTERNS = [
    r"worn with",
    r"paired with",
    r"underneath",
    r"matching",
    r"along with",
    r"with a pair of (?:pants|jeans|leggings|shorts|trousers)",
    r"and a pair of (?:pants|jeans|leggings|shorts|trousers)",
]


def normalize_category(cat: str):
    """Normalize plural/alias category text into one of the main DB categories."""
    if not cat:
        return "unknown"
    return CATEGORY_ALIASES.get(cat.lower().strip(), cat.lower().strip())


def normalize_caption(caption: str):
    """Light cleanup before parsing model output."""
    caption = caption or ""
    caption = caption.lower().strip()
    return re.sub(r"\s*-\s*", "-", caption)


def _remove_ignored_category_context(segment: str):
    cleaned = segment
    for pattern in COMPARISON_PATTERNS + VERB_DRESS_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _contains_keyword(text: str, keyword: str):
    return re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE) is not None


def detect_type(segment: str, return_diagnostic=False):
    """
    Detect the item category from one caption segment.
    return_diagnostic=True returns (result_type, ambiguous).
    """
    segment = normalize_caption(segment)
    searchable_segment = _remove_ignored_category_context(segment)

    found = []
    for keyword, main_type in _CATEGORY_ALIAS_ITEMS:
        match = re.search(
            rf"\b{re.escape(keyword)}\b",
            searchable_segment,
            flags=re.IGNORECASE,
        )
        if match:
            found.append((match.start(), main_type))

    if not found:
        result_type, ambiguous = "unknown", False
    else:
        found_types = {item_type for _, item_type in found}
        if len(found_types) > 1:
            picked = next(
                (item_type for item_type in PRIORITY_WHEN_AMBIGUOUS if item_type in found_types),
                None,
            )
            result_type, ambiguous = picked or sorted(found, key=lambda item: item[0])[0][1], True
        else:
            result_type, ambiguous = sorted(found, key=lambda item: item[0])[0][1], False

    return (result_type, ambiguous) if return_diagnostic else result_type


def split_items(caption: str):
    caption = normalize_caption(caption)
    pattern = r"\b(?:" + "|".join(MULTI_ITEM_CONNECTOR_PATTERNS) + r")\b"
    segments = re.split(pattern, caption, flags=re.IGNORECASE)
    segments = [segment.strip(" ,.;") for segment in segments if segment.strip(" ,.;")]
    return segments if segments else [caption]


def extract_attributes(segment: str, item_type: str):
    """Extract global and type-specific attribute keywords for optional downstream use."""
    segment = normalize_caption(segment)
    allowed_attributes = set(GLOBAL_ATTRIBUTES)
    allowed_attributes.update(TYPE_SPECIFIC_ATTRIBUTES.get(item_type, []))

    result = {}
    for attr in allowed_attributes:
        values = [
            keyword
            for keyword in ATTRIBUTE_KEYWORDS.get(attr, [])
            if _contains_keyword(segment, keyword)
        ]
        if values:
            result[attr] = values

    return result


def parse_caption(
    image_id: str,
    caption: str,
    return_diagnostic: bool = False,
    return_attributes: bool = False,
):
    """Single parser used by both catalog building and evaluation scripts."""
    caption = normalize_caption(caption)
    segments = split_items(caption)
    first_segment = segments[0]
    predicted_type, ambiguous = detect_type(first_segment, return_diagnostic=True)

    result = {"image_id": image_id, "caption": caption, "type": predicted_type}

    if return_attributes:
        attributes = extract_attributes(first_segment, predicted_type)
        result["attributes"] = attributes
        result["global_attributes"] = {
            attr: values for attr, values in attributes.items() if attr in GLOBAL_ATTRIBUTES
        }
        result["type_specific_attributes"] = {
            attr: values for attr, values in attributes.items() if attr not in GLOBAL_ATTRIBUTES
        }

    if return_diagnostic:
        result["ambiguous"] = ambiguous
        result["n_segments"] = len(segments)

    return result
