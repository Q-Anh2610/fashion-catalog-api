import json
import os
import re


_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_CATALOG_DICT_PATH = os.path.join(_CURRENT_DIR, "catalog_dict.json")


def _load_catalog_dict():
    if not os.path.exists(_CATALOG_DICT_PATH):
        raise FileNotFoundError(
            f"catalog_dict.json not found at {_CATALOG_DICT_PATH}. "
            "classify_utils.py does not hardcode fallback data, "
            "so this file is required."
        )
    with open(_CATALOG_DICT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _require(catalog: dict, key: str):
    if key not in catalog:
        raise KeyError(
            f"catalog_dict.json is missing required key '{key}'. "
            "Add it to catalog_dict.json (classify_utils.py does not hardcode fallbacks)."
        )
    return catalog[key]


# ============ Dữ liệu load từ catalog_dict.json (nguồn duy nhất) ============
_catalog_dict = _load_catalog_dict()

CATEGORY_ALIASES = _require(_catalog_dict, "category_aliases")
PRIORITY_WHEN_AMBIGUOUS = _require(_catalog_dict, "priority_rules")

GLOBAL_ATTRIBUTES = _require(_catalog_dict, "global_attributes")
# catalog_dict.json hiện dùng không lưu key này -> mặc định color/pattern là
# multi-value, đúng giá trị gốc trong notebook. Nếu sau này catalog_dict.json
# có thêm key "multi_value_global_attributes", nó sẽ được ưu tiên dùng.
MULTI_VALUE_GLOBAL_ATTRIBUTES = set(
    _catalog_dict.get("multi_value_global_attributes", ["color", "pattern"])
)
TYPE_SPECIFIC_ATTRIBUTES = _require(_catalog_dict, "type_specific_attributes")

_RAW_ATTRIBUTE_KEYWORDS = _require(_catalog_dict, "attribute_keywords")
# Sắp xếp keyword dài -> ngắn trong từng attribute để match theo kiểu "greedy"
# (ưu tiên cụm dài hơn trước, vd "short sleeves" match trước "sleeve").
ATTRIBUTE_KEYWORDS = {
    attr: sorted(keywords, key=len, reverse=True)
    for attr, keywords in _RAW_ATTRIBUTE_KEYWORDS.items()
}

# ============ Regex logic xử lý (không phải từ điển -> không nằm trong
# catalog_dict.json, giữ hardcode trong code như notebook gốc) ============
# Tránh match nhầm "dress" khi nó xuất hiện ở dạng so sánh ("like a dress")
# hoặc động từ ("to dress up").
COMPARISON_PATTERNS = [
    r"\b(?:styled|looks?|dressed)\s+like\s+(?:a|an)\s+\w+",
    r"\blike\s+(?:a|an)\s+\w+",
    r"\bsimilar\s+to\s+(?:a|an)\s+\w+",
    r"\bresembl(?:es|ing)\s+(?:a|an)\s+\w+",
    r"\bas\s+if\s+it\s+(?:were|was)\s+(?:a|an)\s+\w+",
]
VERB_DRESS_PATTERNS = [
    r"\bto\s+dress\b",
    r"\bdress(?:es|ed|ing)?\s+(?:up|down|casually|formally|smartly|well|nicely)\b",
    r"\bdress\s+code\b",
]

# Nếu caption dùng 2 từ khóa category khác nhau (vd "dress ... with a flowy
# skirt"), danh từ xuất hiện SỚM trong câu (nằm trong cụm mở đầu
# "color + material + TYPE ...") mới là item chính; từ khóa xuất hiện muộn
# hơn thường chỉ mô tả 1 chi tiết phụ. LEADING_NOUN_WINDOW_CHARS định nghĩa
# "sớm" là trong khoảng bao nhiêu ký tự đầu câu.
LEADING_NOUN_WINDOW_CHARS = 30


# ============ Xử lý ============
def normalize_category(cat: str):
    mapping = {
        "dresses": "dress",
        "tops": "top",
        "skirts": "skirt",
        "jackets": "jacket",
        "pants": "pants",
    }
    return mapping.get(cat, cat)


def normalize_caption(caption: str):
    caption = (caption or "").lower().strip()
    return re.sub(r"\s*-\s*", "-", caption)


def _remove_ignored_category_context(text: str):
    """Loại bỏ các cụm dạng so sánh ('like a dress') hoặc động từ ('to dress up')
    TRƯỚC khi tìm keyword category, để tránh match nhầm category giả."""
    cleaned = text
    for pattern in COMPARISON_PATTERNS + VERB_DRESS_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def detect_type(caption: str, return_diagnostic=False):
    caption = normalize_caption(caption)
    searchable = _remove_ignored_category_context(caption)

    found = []
    for keyword, main_type in CATEGORY_ALIASES.items():
        m = re.search(rf"\b{re.escape(keyword)}\b", searchable, flags=re.IGNORECASE)
        if m:
            found.append((m.start(), main_type))

    if not found:
        result_type, ambiguous = "unknown", False
    else:
        found.sort(key=lambda x: x[0])
        found_types = set(t for _, t in found)

        if len(found_types) > 1:
            # Ưu tiên 1: nếu từ khóa xuất hiện sớm nhất nằm trong "cửa sổ danh
            # từ chính" (đầu câu) -> coi đó là item chính, bỏ qua
            # PRIORITY_WHEN_AMBIGUOUS.
            earliest_pos, earliest_type = found[0]
            if earliest_pos <= LEADING_NOUN_WINDOW_CHARS:
                result_type, ambiguous = earliest_type, True
            else:
                # Ưu tiên 2 (fallback): dùng PRIORITY_WHEN_AMBIGUOUS khi không
                # có danh từ nào rõ ràng nằm ở đầu câu.
                picked = next((p for p in PRIORITY_WHEN_AMBIGUOUS if p in found_types), None)
                result_type, ambiguous = (picked, True) if picked else (found[0][1], True)
        else:
            result_type, ambiguous = found[0][1], False

    return (result_type, ambiguous) if return_diagnostic else result_type


def extract_attributes(caption: str, item_type: str):
    """Trích attribute keyword hợp lệ theo type (global + type-specific).
    color/pattern cho phép multi-value; các attribute còn lại chỉ lấy 1 giá
    trị khớp đầu tiên. Giá trị trả về luôn là list (kể cả single-value) —
    parse_caption() sẽ rút gọn xuống scalar khi build key "attributes"."""
    caption = normalize_caption(caption)
    allowed = set(GLOBAL_ATTRIBUTES) | set(TYPE_SPECIFIC_ATTRIBUTES.get(item_type, []))

    result = {}
    for attr in allowed:
        matches = [
            kw
            for kw in ATTRIBUTE_KEYWORDS.get(attr, [])
            if re.search(rf"\b{re.escape(kw)}\b", caption, flags=re.IGNORECASE)
        ]
        if not matches:
            continue
        result[attr] = matches if attr in MULTI_VALUE_GLOBAL_ATTRIBUTES else matches[:1]

    return result


def parse_caption(image_id: str, caption: str, return_diagnostic: bool = False, return_attributes: bool = True):
    """Hàm DUY NHẤT — dùng cho cả build catalog thật lẫn phân tích/đánh giá.
    Mỗi ảnh chỉ có 1 type (không còn multi-item), nên parse trực tiếp trên
    toàn bộ caption."""
    caption = normalize_caption(caption)
    predicted_type, ambiguous = detect_type(caption, return_diagnostic=True)

    result = {"image_id": image_id, "caption": caption, "type": predicted_type}

    if return_attributes:
        attributes = extract_attributes(caption, predicted_type)
        global_attrs = {a: v for a, v in attributes.items() if a in GLOBAL_ATTRIBUTES}
        type_attrs = {a: v for a, v in attributes.items() if a not in GLOBAL_ATTRIBUTES}

        result["global_attributes"] = global_attrs
        result["type_specific_attributes"] = type_attrs

        # Bản rút gọn giá trị (list -> scalar, trừ color/pattern vẫn là list)
        # để db_utils.py (map_codes_only, classify_and_update_product,
        # search_products_by_image...) dùng thẳng qua
        # parser_result["attributes"]["global"/"type_specific"] mà không bắt
        # buộc phải đi qua bước format riêng ở main.py.
        result["attributes"] = {
            "global": {
                a: (v if a in MULTI_VALUE_GLOBAL_ATTRIBUTES else v[0])
                for a, v in global_attrs.items()
            },
            "type_specific": {a: v[0] for a, v in type_attrs.items()},
        }

    if return_diagnostic:
        result["ambiguous"] = ambiguous

    return result
