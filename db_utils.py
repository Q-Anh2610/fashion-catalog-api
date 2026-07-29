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

def map_value_to_code(value: str, codebook_table: str, name_column: str, code_column: str) -> str:
    """
    Tra bảng codebook (type_code/color_code/material_code/pattern_code),
    map giá trị text (VD: "black") -> mã 1 ký tự (VD: "B").
    Trả về "X" nếu value rỗng hoặc không tìm thấy trong codebook.
    """
    if not value:
        return "X"
    res = supabase.table(codebook_table).select(code_column).eq(name_column, value).limit(1).execute()
    return res.data[0][code_column] if res.data else "X"


def get_next_sequence_number(type_code: str, color_code: str, material_code: str, pattern_code: str) -> int:
    """
    Đếm số sản phẩm đã có ĐÚNG tổ hợp type+color+material+pattern (kể cả khi có X),
    trả về số thứ tự tiếp theo. Query qua classification_results vì đó là nơi lưu 4 mã này.
    """
    res = supabase.table("classification_results") \
        .select("result_id", count="exact") \
        .eq("type_code", type_code) \
        .eq("color_code", color_code) \
        .eq("material_code", material_code) \
        .eq("pattern_code", pattern_code) \
        .execute()
    return (res.count or 0) + 1


def map_codes_only(parser_result: dict) -> dict:
    """Chỉ map 4 giá trị sang code, không tính sequence number."""
    global_attrs = parser_result["attributes"]["global"]
    return {
        "type_code": map_value_to_code(parser_result["type"], "type_code", "type_name", "type_code"),
        "color_code": map_value_to_code(global_attrs.get("color"), "color_code", "color_name", "color_code"),
        "material_code": map_value_to_code(global_attrs.get("material"), "material_code", "material_name", "material_code"),
        "pattern_code": map_value_to_code(global_attrs.get("pattern"), "pattern_code", "pattern_name", "pattern_code"),
    }


def build_product_code(parser_result: dict) -> tuple[str, dict]:
    codes = map_codes_only(parser_result)
    seq = get_next_sequence_number(**codes)
    product_code = f"{codes['type_code']}{codes['color_code']}{codes['material_code']}{codes['pattern_code']}{seq:03d}"
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


def save_caption(seller_client, product_id: int, caption_text: str) -> int:
    """Bước 2: lưu caption model sinh ra, gắn với product_id đã tạo."""
    res = seller_client.table("caption").insert({
        "product_id": product_id,
        "caption_text": caption_text,
    }).execute()
    return res.data[0]["caption_id"]


def classify_and_update_product(seller_client, product_id: int, caption_id: int, parser_result: dict) -> str:
    """
    Bước 3-5: build product_code -> insert classification_results
              -> insert attribute_<type> -> UPDATE product.
    Trả về product_code đã sinh.
    """
    product_code, codes = build_product_code(parser_result)

    # Insert classification_results
    seller_client.table("classification_results").insert({
        "product_id": product_id,
        "caption_id": caption_id,
        "type_code": codes["type_code"],
        "color_code": codes["color_code"],
        "material_code": codes["material_code"],
        "pattern_code": codes["pattern_code"],
        "is_ambiguous": parser_result["ambiguous"],
    }).execute()

    # Insert vào đúng bảng attribute_<type>, nếu type xác định được
    item_type = parser_result["type"]
    if item_type != "unknown":
        table_name = f"attribute_{item_type}"
        specific_attrs = parser_result["attributes"]["type_specific"]
        seller_client.table(table_name).insert({
            "product_id": product_id,
            **specific_attrs,
        }).execute()

    # UPDATE product với product_code, đổi status
    seller_client.table("product").update({
        "product_code": product_code,
        "status": "done",
    }).eq("product_id", product_id).execute()

    return product_code


def get_seller_products(seller_client, seller_id: str):
    """Lấy danh sách sản phẩm của 1 người bán, kèm caption + kết quả phân loại."""
    res = seller_client.table("product") \
        .select("*, caption(*), classification_results(*)") \
        .eq("seller_id", seller_id) \
        .order("uploaded_at", desc=True) \
        .execute()
    return res.data


def submit_seller_correction(seller_client, result_id: int, corrected_category: str):
    """Người bán tự sửa category nếu parser gán sai."""
    seller_client.table("seller_correction").insert({
        "result_id": result_id,
        "corrected_category": corrected_category,
    }).execute()


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
        .select("caption_text, product(*, classification_results(*))") \
        .ilike("caption_text", f"%{query_text}%")
    res = query.execute()

    rows = []
    for r in res.data:
        product = r.get("product")
        if not product:
            continue
        product["caption"] = [{"caption_text": r["caption_text"]}]  # gắn để khớp _product_result_rows
        rows.append(product)

    if filters:
        def matches(row):
            cls = row.get("classification_results") or []
            if not cls:
                return False
            c = cls[0]
            return all(c.get(k) == v for k, v in filters.items())
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
    """
    Tìm sản phẩm tương đồng theo ảnh. similarity_dimensions là subset của
    ["type", "color", "material", "pattern"] — có thể chọn 0 đến 4 mức,
    không cần liên tục/theo thứ tự (VD chỉ chọn ["material", "pattern"] vẫn được).
    Nếu để trống -> trả về tất cả (không lọc theo attribute nào).
    """
    ensure_device_exists(device_id)
    similarity_dimensions = similarity_dimensions or []

    # Upload ảnh tìm kiếm vào bucket riêng buyers-uploads
    remote_filename = f"{uuid.uuid4()}.jpg"
    with open(query_image_local_path, "rb") as f:
        supabase.storage.from_("buyers-uploads").upload(remote_filename, f)

    # Build mã từ ảnh vừa upload (dùng chung hàm build_product_code)
    codes = map_codes_only(parser_result)

    dimension_to_code_column = {
        "type": "type_code",
        "color": "color_code",
        "material": "material_code",
        "pattern": "pattern_code",
    }

    query = supabase.table("classification_results").select("product_id, product(*)")
    for dim in similarity_dimensions:
        col = dimension_to_code_column[dim]
        query = query.eq(col, codes[col])
    res = query.execute()

    result_product_ids = [row["product_id"] for row in res.data]

    supabase.table("search_by_image").insert({
        "device_id": device_id,
        "query_image_path": remote_filename,
        "generated_caption": generated_caption,
        "similarity_level": len(similarity_dimensions),
        "result_product_ids": result_product_ids,
    }).execute()

    return [row["product"] for row in res.data]