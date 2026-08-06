import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)  # client mặc định, dùng cho các thao tác public (đọc, search)


# ======================================================================
# PHẦN DÙNG CHUNG — Mã hoá sản phẩm (product_code)
# ======================================================================

_codebook_cache: dict[str, dict[str, str]] = {}

def load_codebook_cache():
    """Gọi 1 lần lúc app khởi động. Load toàn bộ attribute_code vào RAM,
    nhóm theo code_type để tra cứu O(1) thay vì query DB mỗi lần."""
    res = supabase.table("attribute_code").select("code_type, code, name").execute()
    cache: dict[str, dict[str, str]] = {"type": {}, "color": {}, "material": {}, "pattern": {}}
    for row in res.data:
        cache[row["code_type"]][row["name"]] = row["code"]
    _codebook_cache.clear()
    _codebook_cache.update(cache)

def map_value_to_code(value: str, code_type: str) -> str:
    """
    Tra codebook (đã cache trong RAM) theo code_type ('type'/'color'/'material'/'pattern'),
    map giá trị text (VD: "black") -> mã 1 ký tự (VD: "B").
    Trả về "X" nếu value rỗng hoặc không tìm thấy.
    """
    if not value:
        return "X"
    return _codebook_cache.get(code_type, {}).get(value, "X")


def get_next_sequence_number(type_code: str, material_code: str) -> int:
    """
    Đếm số sản phẩm đã có ĐÚNG tổ hợp type+material (kể cả khi có X),
    trả về số thứ tự tiếp theo.
    """
    res = supabase.table("classification_results") \
        .select("result_id", count="exact") \
        .eq("type_code", type_code) \
        .eq("material_code", material_code) \
        .execute()
    return (res.count or 0) + 1


def map_codes_only(parser_result: dict) -> dict:
    global_attrs = parser_result["attributes"]["global"]

    type_code = map_value_to_code(parser_result["type"], "type")
    material_code = map_value_to_code(global_attrs.get("material"), "material")

    color_codes = sorted({
        c for c in (map_value_to_code(v, "color") for v in global_attrs.get("color", []))
        if c and c != "X"
    })
    pattern_codes = sorted({
        c for c in (map_value_to_code(v, "pattern") for v in global_attrs.get("pattern", []))
        if c and c != "X"
    })

    return {
        "type_code": type_code,
        "material_code": material_code,
        "color_codes": color_codes,
        "pattern_codes": pattern_codes,
    }


def build_product_code(parser_result: dict) -> tuple[str, dict]:
    codes = map_codes_only(parser_result)
    seq = get_next_sequence_number(codes["type_code"], codes["material_code"])
    # [TYPE:1][MATERIAL:1][SEQ:4] = 6 ký tự, không còn color/pattern trong code
    product_code = f"{codes['type_code']}{codes['material_code']}{seq:04d}"
    return product_code, codes

# ======================================================================
# LUỒNG NGƯỜI BÁN (cần seller_client đã xác thực OTP, seller_id từ auth_utils.py)
# ======================================================================

def create_pending_product(seller_client, seller_id: str, image_local_path: str,
                            price: float) -> int:
    remote_filename = f"{uuid.uuid4()}.jpg"
    with open(image_local_path, "rb") as f:
        seller_client.storage.from_("product-images").upload(remote_filename, f)

    try:
        res = seller_client.table("product").insert({
            "seller_id": seller_id,
            "image_path": remote_filename,
            "price": price,
            "status": "pending",
        }).execute()
    except Exception:
        seller_client.storage.from_("product-images").remove([remote_filename])
        raise

    return res.data[0]["product_id"]


def create_pending_products_batch(seller_client, seller_id: str, image_local_paths: list[str],
                                   prices: list[float]) -> list[int]:
    if len(image_local_paths) > 20:
        raise ValueError("Chỉ được upload tối đa 20 ảnh cùng lúc")
    if len(prices) != len(image_local_paths):
        raise ValueError("Số lượng price phải khớp với số lượng ảnh")

    product_ids = []
    for img_path, price in zip(image_local_paths, prices):
        pid = create_pending_product(seller_client, seller_id, img_path, price)
        product_ids.append(pid)
    return product_ids


def save_caption(seller_client, product_id: int, caption_text: str, caption_source: str = "manual") -> int:
    """Lưu caption do người bán tự nhập (hoặc gợi ý AI đã chỉnh sửa), gắn với product_id đã tạo."""
    res = seller_client.table("caption").insert({
        "product_id": product_id,
        "caption_text": caption_text,
        "caption_source": caption_source,
    }).execute()
    return res.data[0]["caption_id"]


def classify_and_update_product(seller_client, product_id: int, caption_id: int, parser_result: dict) -> str:
    """
    Bước 3-5: build product_code -> insert classification_results
              -> insert classification_result_attribute (color/pattern, multi-value)
              -> insert attribute_<type> -> UPDATE product.
    Trả về product_code đã sinh.
    """
    product_code, codes = build_product_code(parser_result)

    # Insert classification_results — không còn color_code/pattern_code
    result = seller_client.table("classification_results").insert({
        "product_id": product_id,
        "caption_id": caption_id,
        "type_code": codes["type_code"],
        "material_code": codes["material_code"],
        "is_ambiguous": parser_result["ambiguous"],
    }).execute()
    result_id = result.data[0]["result_id"]

    # Insert color/pattern (multi-value) vào bảng junction
    attribute_rows = [
        {"result_id": result_id, "code_type": "color", "code": c}
        for c in codes["color_codes"]
    ] + [
        {"result_id": result_id, "code_type": "pattern", "code": c}
        for c in codes["pattern_codes"]
    ]
    if attribute_rows:
        seller_client.table("classification_result_attribute").insert(attribute_rows).execute()

    # Insert vào đúng bảng attribute_<type>, nếu type xác định được
    item_type = parser_result["type"]
    if item_type != "unknown":
        table_name = f"attribute_{item_type}"
        specific_attrs = parser_result["attributes"]["type_specific"]
        seller_client.table(table_name).upsert(
            {"product_id": product_id, **specific_attrs},
            on_conflict="product_id",
        ).execute()

    # UPDATE product với product_code, đổi status
    seller_client.table("product").update({
        "product_code": product_code,
        "status": "done",
    }).eq("product_id", product_id).execute()

    return product_code


def get_seller_products(seller_client, seller_id: str):
    """Lấy danh sách sản phẩm của 1 người bán, kèm caption + kết quả phân loại + color/pattern."""
    res = seller_client.table("product") \
        .select("*, caption(*), classification_results(*, classification_result_attribute(code_type, code))") \
        .eq("seller_id", seller_id) \
        .order("uploaded_at", desc=True) \
        .execute()
    return res.data


# ======================================================================
# LUỒNG NGƯỜI MUA (không cần login, dùng device_id)
# ======================================================================

def ensure_device_exists(device_id: str, platform: str = "gradio-local"):
    now = datetime.now(timezone.utc).isoformat()
    existing = supabase.table("user_device").select("device_id").eq("device_id", device_id).limit(1).execute()

    if existing.data:
        supabase.table("user_device").update({"last_seen_at": now, "platform": platform}) \
            .eq("device_id", device_id).execute()
        return

    supabase.table("user_device").insert({
        "device_id": device_id,
        "first_seen_at": now,
        "last_seen_at": now,
        "platform": platform,
    }).execute()


def search_products_by_text(device_id: str, query_text: str, filters: dict = None):
    ensure_device_exists(device_id)

    query = supabase.table("caption") \
        .select("caption_text, product(*, classification_results(*, classification_result_attribute(code_type, code)))") \
        .ilike("caption_text", f"%{query_text}%")
    res = query.execute()

    rows = []
    for r in res.data:
        product = r.get("product")
        if not product:
            continue
        product["caption"] = [{"caption_text": r["caption_text"]}]
        rows.append(product)

    if filters:
        def matches(row):
            cls_list = row.get("classification_results") or []
            if not cls_list:
                return False
            c = cls_list[0]
            attr_rows = c.get("classification_result_attribute") or []
            colors = {a["code"] for a in attr_rows if a["code_type"] == "color"}
            patterns = {a["code"] for a in attr_rows if a["code_type"] == "pattern"}

            for key, value in filters.items():
                if key == "color_code":
                    if value not in colors:
                        return False
                elif key == "pattern_code":
                    if value not in patterns:
                        return False
                else:
                    if c.get(key) != value:
                        return False
            return True

        rows = [r for r in rows if matches(r)]

    result_product_ids = [r["product_id"] for r in rows]
    supabase.table("search_by_text").insert({
        "device_id": device_id,
        "query_text": query_text,
        "result_product_ids": result_product_ids,
    }).execute()

    return rows

def search_products_by_image(device_id: str, query_image_local_path: str,
                              generated_caption: str, parser_result: dict,
                              similarity_dimensions: list[str] = None):
    ensure_device_exists(device_id)
    similarity_dimensions = similarity_dimensions or []

    remote_filename = f"{uuid.uuid4()}.jpg"
    with open(query_image_local_path, "rb") as f:
        supabase.storage.from_("buyers-uploads").upload(remote_filename, f)

    codes = map_codes_only(parser_result)

    query = supabase.table("classification_results") \
        .select("product_id, product(*), classification_result_attribute(code_type, code)")

    if "type" in similarity_dimensions:
        query = query.eq("type_code", codes["type_code"])
    if "material" in similarity_dimensions:
        query = query.eq("material_code", codes["material_code"])

    res = query.execute()

    rows = res.data or []
    if "color" in similarity_dimensions:
        targets = set(codes["color_codes"])
        rows = [
            r for r in rows
            if targets & {a["code"] for a in (r.get("classification_result_attribute") or []) if a["code_type"] == "color"}
        ]
    if "pattern" in similarity_dimensions:
        targets = set(codes["pattern_codes"])
        rows = [
            r for r in rows
            if targets & {a["code"] for a in (r.get("classification_result_attribute") or []) if a["code_type"] == "pattern"}
        ]

    result_product_ids = [row["product_id"] for row in rows]

    supabase.table("search_by_image").insert({
        "device_id": device_id,
        "query_image_path": remote_filename,
        "generated_caption": generated_caption,
        "similarity_level": len(similarity_dimensions),
        "result_product_ids": result_product_ids,
    }).execute()

    return [row["product"] for row in rows]
