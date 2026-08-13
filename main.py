"""
main.py — FastAPI layer gom các module (auth_utils, classify_utils, db_utils, model_utils_2)
thành 1 API duy nhất, dùng lại đúng luồng nghiệp vụ đang có trong app.py (bản Gradio demo).

Chạy local:
    uvicorn main:app --reload --port 8000
Swagger docs tự sinh tại: http://localhost:8000/docs
"""

import io
import os
import shutil
import tempfile
import uuid
import torch
import json as json_lib
import pandas as pd
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

USE_MOCK_AUTH = os.getenv("USE_MOCK_AUTH", "false").lower() == "true"

if USE_MOCK_AUTH:
    from auth_utils_2 import get_authenticated_client, send_otp, verify_otp
else:
    from auth_utils import get_authenticated_client, send_otp, verify_otp
#from auth_utils import get_authenticated_client, send_otp, verify_otp

from classify_utils import (
    ATTRIBUTE_KEYWORDS,
    GLOBAL_ATTRIBUTES,
    TYPE_SPECIFIC_ATTRIBUTES,
    MULTI_VALUE_GLOBAL_ATTRIBUTES,
    parse_caption,
)
from db_utils import (
    load_codebook_cache,
    classify_and_update_product,
    create_pending_product,
    ensure_device_exists,
    get_seller_products,
    map_value_to_code,
    save_caption,
    search_products_by_text,
    supabase,
)
from model_utils_2 import generate_caption


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fashion Catalog API")

USE_MOCK_CLIP = os.getenv("USE_MOCK_CLIP", "false").lower() == "true"

_clip_model = None
_clip_processor = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DEV ONLY — khi deploy production nên giới hạn domain Flutter Web thật
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_codebook_cache()
    if not USE_MOCK_CLIP:
        global _clip_model, _clip_processor
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()

def get_image_embedding(image_path: str) -> list[float]:
    if USE_MOCK_CLIP:
        # Vector giả nhưng deterministic theo nội dung file -> cùng ảnh luôn ra cùng vector,
        # đủ để test logic sort/cosine mà không cần load model thật (tránh OOM trên Render free tier).
        import hashlib, random
        with open(image_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        rng = random.Random(file_hash)
        vec = [rng.uniform(-1, 1) for _ in range(512)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = _clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = _clip_model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)  # normalize để cosine = dot product
    return features[0].tolist()

TYPE_CHOICES = ["dress", "jacket", "skirt", "pants", "top"]
SIMILARITY_DIMENSIONS = ["type", "color", "material", "pattern"]

# Tên bucket Supabase Storage nơi ảnh sản phẩm được lưu (theo đúng thông báo
# "Đã lưu sản phẩm vào storage `product-images`" ở bản Gradio demo).
PRODUCT_IMAGE_BUCKET = "product-images"


# ---------------------------------------------------------------------------
# Helpers — port nguyên logic từ app.py, chỉ đổi input từ gr.Image/gr.File
# sang UploadFile của FastAPI
# ---------------------------------------------------------------------------

def _save_upload_to_tmp(upload: UploadFile) -> str:
    """Lưu UploadFile ra file tạm trên đĩa, trả về path (model_utils cần path, không nhận bytes)."""
    suffix = os.path.splitext(upload.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(upload.file, f)
    return tmp.name


def _first_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value or None

def _all_values(value):
    if isinstance(value, list):
        return [v for v in value if v]
    return [value] if value else []

def _format_parser_for_db(parsed: dict[str, Any]) -> dict[str, Any]:
    item_type = parsed.get("type", "unknown")
    global_values = {}
    for attr in GLOBAL_ATTRIBUTES:
        value = parsed.get("global_attributes", {}).get(attr)
        if value is None:
            value = parsed.get("attributes", {}).get(attr)
        if attr in MULTI_VALUE_GLOBAL_ATTRIBUTES:
            global_values[attr] = _all_values(value)      # color, pattern -> list
        else:
            global_values[attr] = _first_value(value)      # material -> 1 giá trị

    type_values = {}
    for attr in TYPE_SPECIFIC_ATTRIBUTES.get(item_type, []):
        value = parsed.get("type_specific_attributes", {}).get(attr)
        if value is None:
            value = parsed.get("attributes", {}).get(attr)
        value = _first_value(value)
        if value is not None:
            type_values[attr] = value

    return {
        "image_id": parsed.get("image_id"),
        "caption": parsed.get("caption"),
        "type": item_type,
        "ambiguous": bool(parsed.get("ambiguous", False)),
        "attributes": {
            "global": global_values,
            "type_specific": type_values,
        },
    }


def _display_attributes(db_parser: dict[str, Any]) -> str:
    attrs = {}
    attrs.update(db_parser["attributes"]["global"])
    attrs.update(db_parser["attributes"]["type_specific"])

    parts = []
    for key, value in attrs.items():
        if isinstance(value, list):
            if value:
                parts.append(f"{key}: {', '.join(value)}")
        elif value:
            parts.append(f"{key}: {value}")
    return ", ".join(parts)


def _parse_image(image_path: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    caption = generate_caption(image_path)
    parsed = parse_caption(
        image_id=os.path.basename(image_path),
        caption=caption,
        return_diagnostic=True,
        return_attributes=True,
    )
    db_parser = _format_parser_for_db(parsed)
    return caption, parsed, db_parser

def _parse_caption(image_id: str, caption_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Chỉ chạy parser trên caption đã có sẵn (không gọi model) — dùng cho luồng seller."""
    parsed = parse_caption(
        image_id=image_id,
        caption=caption_text,
        return_diagnostic=True,
        return_attributes=True,
    )
    db_parser = _format_parser_for_db(parsed)
    return parsed, db_parser


MIN_CAPTION_WORDS = 5

def _validate_caption(caption: Optional[str]) -> str:
    caption = (caption or "").strip()
    if len(caption.split()) < MIN_CAPTION_WORDS:
        raise HTTPException(
            400, f"Caption is too short (minimum {MIN_CAPTION_WORDS} words) for accurate classification."
        )
    return caption

def _code_filters(type_filter, color_filter, material_filter, pattern_filter):
    filters = {}
    if type_filter:
        filters["type_code"] = map_value_to_code(type_filter, "type")
    if color_filter:
        filters["color_code"] = map_value_to_code(color_filter, "color")
    if material_filter:
        filters["material_code"] = map_value_to_code(material_filter, "material")
    if pattern_filter:
        filters["pattern_code"] = map_value_to_code(pattern_filter, "pattern")
    return {key: value for key, value in filters.items() if value and value != "X"}

def _target_codes(db_parser: dict[str, Any]) -> dict[str, Any]:
    global_attrs = db_parser["attributes"]["global"]
    return {
        "type_code": map_value_to_code(db_parser["type"], "type"),
        "material_code": map_value_to_code(global_attrs.get("material"), "material"),
        "color_codes": [map_value_to_code(v, "color") for v in global_attrs.get("color", [])],
        "pattern_codes": [map_value_to_code(v, "pattern") for v in global_attrs.get("pattern", [])],
    }

def _product_to_json(product: dict, score=None) -> dict:
    cls_rows = product.get("classification_results") or []
    cls = cls_rows[0] if isinstance(cls_rows, list) and cls_rows else {}
    attr_rows = cls.get("classification_result_attribute") or []
    color_codes = sorted({a["code"] for a in attr_rows if a["code_type"] == "color"})
    pattern_codes = sorted({a["code"] for a in attr_rows if a["code_type"] == "pattern"})

    caption_rows = product.get("caption") or []
    caption = ""
    if isinstance(caption_rows, list) and caption_rows:
        caption = caption_rows[0].get("caption_text", "")

    return {
        "match_score": score,
        "product_code": product.get("product_code", ""),
        "product_id": product.get("product_id"),
        "price": product.get("price"),
        "caption": caption,
        "type_code": cls.get("type_code", ""),
        "material_code": cls.get("material_code", ""),
        "color_codes": color_codes,
        "pattern_codes": pattern_codes,
        "image_path": _public_image_url(product.get("image_path", "")),
    }

def _random_price() -> int:
    """Random giá trong khoảng 100_000 - 500_000, luôn là bội số của 1000 (3 số cuối = 000)."""
    import random
    return random.randrange(100_000, 500_001, 1_000)


def _public_image_url(image_path: Optional[str]) -> str:
    """
    SỬA #1: `product.image_path` lưu trong DB là path tương đối bên trong bucket
    Supabase Storage (VD "sellers/abc123.jpg"), Flutter không thể Image.network()
    trực tiếp path đó. Hàm này build ra full public URL để frontend chỉ việc
    hiển thị thẳng, không cần tự ghép domain nữa.

    Yêu cầu: bucket PRODUCT_IMAGE_BUCKET ("product-images") phải để ở chế độ
    Public trên Supabase Storage (Storage > product-images > Settings > Public bucket).
    Nếu bucket đang Private, cần đổi sang create_signed_url(...) thay vì
    get_public_url(...) — báo mình biết nếu đúng trường hợp này để đổi lại.
    """
    if not image_path:
        return ""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path  # đã là URL đầy đủ sẵn, không cần xử lý thêm
    try:
        result = supabase.storage.from_(PRODUCT_IMAGE_BUCKET).get_public_url(image_path)
        # supabase-py có thể trả str hoặc dict tùy version, chuẩn hóa lại
        if isinstance(result, dict):
            return result.get("publicUrl") or result.get("public_url") or ""
        return result or ""
    except Exception:
        return ""


def _get_result_id(client, caption_id) -> Optional[int]:
    """
    Query lại classification_results theo caption_id vừa tạo để lấy result_id
    của dòng phân loại tương ứng, trả về cho frontend hiển thị/debug.
    """
    try:
        res = (
            client.table("classification_results")
            .select("result_id")
            .eq("caption_id", caption_id)
            .order("result_id", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0]["result_id"] if rows else None
    except Exception:
        return None


def _search_products_by_image_soft(image_path, generated_caption, db_parser, dimensions, device_id):
    ensure_device_exists(device_id, platform="flutter-web")
    dimensions = dimensions or []
    remote_filename = ""
    try:
        remote_filename = f"buyer-search/{uuid.uuid4()}.jpg"
        with open(image_path, "rb") as f:
            supabase.storage.from_("buyers-uploads").upload(remote_filename, f)
    except Exception:
        remote_filename = ""

    query_embedding = get_image_embedding(image_path)

    codes = _target_codes(db_parser)

    res = supabase.table("classification_results") \
        .select("*, product(*), classification_result_attribute(code_type, code)") \
        .execute()

    scored = []
    for row in res.data or []:
        product = row.get("product")
        if not product:
            continue

        attr_rows = row.get("classification_result_attribute") or []
        row_colors = {a["code"] for a in attr_rows if a["code_type"] == "color"}
        row_patterns = {a["code"] for a in attr_rows if a["code_type"] == "pattern"}

        if dimensions:
            matched = []
            for dim in dimensions:
                if dim == "type":
                    target = codes.get("type_code")
                    if target and target != "X" and row.get("type_code") == target:
                        matched.append(dim)
                elif dim == "material":
                    target = codes.get("material_code")
                    if target and target != "X" and row.get("material_code") == target:
                        matched.append(dim)
                elif dim == "color":
                    targets = {c for c in codes.get("color_codes", []) if c and c != "X"}
                    if targets and (targets & row_colors):
                        matched.append(dim)
                elif dim == "pattern":
                    targets = {c for c in codes.get("pattern_codes", []) if c and c != "X"}
                    if targets and (targets & row_patterns):
                        matched.append(dim)
            if not matched:
                continue  # không khớp dimension nào -> loại khỏi tập ứng viên
        else:
            matched = []

        prod_embedding = product.get("image_embedding")
        if prod_embedding:
            similarity = sum(a * b for a, b in zip(query_embedding, prod_embedding))
            similarity = max(0.0, min(1.0, similarity))  # kẹp về [0, 1] cho an toàn hiển thị %
        else:
            similarity = 0.0

        product["classification_results"] = [row]
        scored.append((score, matched, product))

    scored.sort(key=lambda item: item[0], reverse=True)
    result_product_ids = [product.get("product_id") for _, _, product in scored]

    try:
        supabase.table("search_by_image").insert({
            "device_id": device_id,
            "query_image_path": remote_filename,
            "generated_caption": generated_caption,
            "similarity_level": len(dimensions),
            "result_product_ids": result_product_ids,
        }).execute()
    except Exception:
        pass

    products, scores = [], {}
    for score, matched, product in scored:
        pid = product.get("product_id")
        scores[pid] = f"{score:.0%}"
        products.append(product)
    return products, scores

# ---------------------------------------------------------------------------
# Auth dependency — Flutter gửi lại access_token + seller_id sau khi verify OTP
# ---------------------------------------------------------------------------

def get_seller_client(
    authorization: str = Header(..., description="Bearer <access_token>"),
    x_seller_id: str = Header(..., alias="X-Seller-Id"),
):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Thiếu Bearer token trong header Authorization.")
    access_token = authorization.removeprefix("Bearer ").strip()
    try:
        client = get_authenticated_client(access_token)
    except Exception as exc:
        raise HTTPException(401, f"Token không hợp lệ: {exc}")
    return client, x_seller_id


# ---------------------------------------------------------------------------
# Pydantic models cho request/response
# ---------------------------------------------------------------------------

class SendOtpRequest(BaseModel):
    email: str


class VerifyOtpRequest(BaseModel):
    email: str
    code: str


class BuyerTextSearchRequest(BaseModel):
    query: str = ""
    type: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    pattern: Optional[str] = None
    device_id: Optional[str] = None  # Flutter tự sinh 1 uuid/local, giữ ổn định giữa các lần gọi
    # SỬA #3: khoảng giá — filter được áp dụng ở tầng main.py (sau khi lấy rows
    # từ search_products_by_text), vì hàm đó hiện chỉ nhận filter theo code
    # (type/color/material/pattern), không có tham số giá.
    price_min: Optional[int] = None
    price_max: Optional[int] = None

class SellerRecaptionRequest(BaseModel):
    caption: str

def _apply_price_filter(rows: list[dict], price_min: Optional[int], price_max: Optional[int]) -> list[dict]:
    if price_min is None and price_max is None:
        return rows
    filtered = []
    for row in rows:
        price = row.get("price")
        if price is None:
            filtered.append(row)  # không rõ giá thì không loại, tránh mất sản phẩm hợp lệ
            continue
        if price_min is not None and price < price_min:
            continue
        if price_max is not None and price > price_max:
            continue
        filtered.append(row)
    return filtered


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/send-otp")
def api_send_otp(payload: SendOtpRequest):
    if not payload.email:
        raise HTTPException(400, "Thiếu email.")
    result = send_otp(payload.email.strip())
    return {"message": result}


@app.post("/auth/verify-otp")
def api_verify_otp(payload: VerifyOtpRequest):
    if not payload.email or not payload.code:
        raise HTTPException(400, "Thiếu email hoặc mã xác thực.")
    auth_state = verify_otp(payload.email.strip(), payload.code.strip())
    if not auth_state:
        raise HTTPException(401, "Xác thực thất bại. Kiểm tra lại mã trong Gmail.")
    # auth_state kỳ vọng có access_token, seller_id, email — Flutter lưu lại 3 giá trị này
    # và gửi kèm access_token (header Authorization: Bearer ...) + seller_id (header X-Seller-Id)
    # ở các request seller phía sau.
    return auth_state


# ---------------------------------------------------------------------------
# Seller endpoints
# ---------------------------------------------------------------------------

@app.post("/seller/products")
def api_seller_upload_single(
    file: UploadFile = File(...),
    caption: str = Form(...),
    price: Optional[int] = Form(None),
    seller=Depends(get_seller_client),
):
    client, seller_id = seller
    image_path = _save_upload_to_tmp(file)
    try:
        final_price = price if price is not None else _random_price()
        caption_text = _validate_caption(caption)
        _, db_parser = _parse_caption(os.path.basename(image_path), caption_text)

        product_id = create_pending_product(
            seller_client=client,
            seller_id=seller_id,
            image_local_path=image_path,
            price=final_price,
        )

        embedding = get_image_embedding(image_path)
        supabase.table("product").update({"image_embedding": embedding}).eq("product_id", product_id).execute()

        caption_id = save_caption(client, product_id, caption_text, caption_source="manual")
        product_code = classify_and_update_product(client, product_id, caption_id, db_parser)
        result_id = _get_result_id(client, caption_id)

        return {
            "product_code": product_code,
            "product_id": product_id,
            "result_id": result_id,
            "caption": caption_text,
            "type": db_parser["type"],
            "attributes": _display_attributes(db_parser),
            "status": "done",
        }
    finally:
        os.remove(image_path)

@app.put("/seller/products/{product_id}/caption")
def api_seller_update_caption(
    product_id: int,
    payload: SellerRecaptionRequest,
    seller=Depends(get_seller_client),
):
    client, seller_id = seller
    caption_text = _validate_caption(payload.caption)
    _, db_parser = _parse_caption(f"product-{product_id}", caption_text)

    caption_id = save_caption(client, product_id, caption_text, caption_source="manual")
    product_code = classify_and_update_product(client, product_id, caption_id, db_parser)
    result_id = _get_result_id(client, caption_id)

    return {
        "product_code": product_code,
        "result_id": result_id,
        "caption": caption_text,
        "type": db_parser["type"],
        "attributes": _display_attributes(db_parser),
    }

def _load_caption_map(caption_file: UploadFile) -> dict[str, dict]:
    filename = caption_file.filename or ""
    content = caption_file.file.read()

    if filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
        if not {"filename", "caption"}.issubset(df.columns):
            raise HTTPException(400, "Excel file must have 'filename' and 'caption' columns.")
        result = {}
        for _, row in df.iterrows():
            result[str(row["filename"]).strip()] = {
                "caption": str(row["caption"]).strip(),
                "price": int(row["price"]) if "price" in df.columns and pd.notna(row.get("price")) else None,
            }
        return result

    elif filename.endswith(".json"):
        data = json_lib.loads(content)
        result = {}
        for item in data:
            if "filename" not in item or "caption" not in item:
                raise HTTPException(400, "Each JSON entry must have 'filename' and 'caption'.")
            result[item["filename"].strip()] = {
                "caption": item["caption"].strip(),
                "price": item.get("price"),
            }
        return result

    raise HTTPException(400, "Only .xlsx or .json files are supported.")


@app.post("/seller/products/batch")
def api_seller_upload_batch(
    files: list[UploadFile] = File(...),
    caption_file: UploadFile = File(...),
    seller=Depends(get_seller_client),
):
    client, seller_id = seller
    if len(files) > 20:
        raise HTTPException(400, "Maximum 20 images allowed per upload.")

    caption_map = _load_caption_map(caption_file)
    caption_source = "batch_excel" if caption_file.filename.endswith((".xlsx", ".xls")) else "batch_json"

    missing = [f.filename for f in files if f.filename not in caption_map]
    if missing:
        raise HTTPException(400, f"No caption found for: {', '.join(missing)}")

    results = []
    ok_count = 0
    for upload in files:
        image_path = _save_upload_to_tmp(upload)
        try:
            entry = caption_map[upload.filename]
            caption_text = _validate_caption(entry["caption"])
            price = entry["price"] if entry["price"] is not None else _random_price()

            _, db_parser = _parse_caption(os.path.basename(image_path), caption_text)
            product_id = create_pending_product(
                seller_client=client,
                seller_id=seller_id,
                image_local_path=image_path,
                price=price,
            )

            embedding = get_image_embedding(image_path)
            supabase.table("product").update({"image_embedding": embedding}).eq("product_id", product_id).execute()

            caption_id = save_caption(client, product_id, caption_text, caption_source=caption_source)
            product_code = classify_and_update_product(client, product_id, caption_id, db_parser)
            result_id = _get_result_id(client, caption_id)
            results.append({
                "filename": upload.filename,
                "product_code": product_code,
                "product_id": product_id,
                "result_id": result_id,
                "caption": caption_text,
                "type": db_parser["type"],
                "attributes": _display_attributes(db_parser),
                "status": "done",
            })
            ok_count += 1
        except Exception as exc:
            results.append({
                "filename": upload.filename,
                "product_code": None, "product_id": None, "result_id": None, "caption": None,
                "type": None, "attributes": None, "status": f"failed: {exc}",
            })
        finally:
            os.remove(image_path)

    return {"processed": ok_count, "total": len(files), "results": results}


@app.get("/seller/products")
def api_seller_list_products(seller=Depends(get_seller_client)):
    client, seller_id = seller
    products = get_seller_products(client, seller_id)  # đã filter đúng theo seller_id, không lộ sản phẩm seller khác
    return {"count": len(products or []), "products": [_product_to_json(p) for p in (products or [])]}

@app.post("/seller/suggest-caption")
def api_seller_suggest_caption(
    file: UploadFile = File(...),
    seller=Depends(get_seller_client),
):
    """Model sinh caption gợi ý — seller có thể dùng làm điểm khởi đầu rồi tự sửa."""
    image_path = _save_upload_to_tmp(file)
    try:
        caption = generate_caption(image_path)
        return {"suggested_caption": caption}
    finally:
        os.remove(image_path)

# ---------------------------------------------------------------------------
# Buyer endpoints
# ---------------------------------------------------------------------------

@app.post("/buyer/search/text")
def api_buyer_search_text(payload: BuyerTextSearchRequest):
    device_id = payload.device_id or str(uuid.uuid4())
    filters = _code_filters(payload.type, payload.color, payload.material, payload.pattern)
    rows = search_products_by_text(
        device_id=device_id,
        query_text=(payload.query or "").strip(),
        filters=filters,
    )
    rows = _apply_price_filter(rows, payload.price_min, payload.price_max)
    return {
        "device_id": device_id,
        "count": len(rows),
        "products": [_product_to_json(p) for p in rows],
    }


@app.post("/buyer/search/image")
def api_buyer_search_image(
    file: UploadFile = File(...),
    dimensions: list[str] = Form(default=SIMILARITY_DIMENSIONS),
    device_id: Optional[str] = Form(None),
    price_min: Optional[int] = Form(None),  # SỬA #3
    price_max: Optional[int] = Form(None),  # SỬA #3
):
    if len(dimensions) == 1 and "," in dimensions[0]:
        dimensions = [d.strip() for d in dimensions[0].split(",") if d.strip()]
    device_id = device_id or str(uuid.uuid4())
    image_path = _save_upload_to_tmp(file)
    try:
        caption, _, db_parser = _parse_image(image_path)
        products, scores = _search_products_by_image_soft(
            image_path=image_path,
            generated_caption=caption,
            db_parser=db_parser,
            dimensions=dimensions,
            device_id=device_id,
        )
        products = _apply_price_filter(products, price_min, price_max)
        return {
            "device_id": device_id,
            "query_caption": caption,
            "query_type": db_parser["type"],
            "query_attributes": _display_attributes(db_parser),
            "count": len(products),
            "products": [_product_to_json(p, scores.get(p.get("product_id"))) for p in products],
        }
    finally:
        os.remove(image_path)


# ---------------------------------------------------------------------------
# Metadata endpoint tiện cho Flutter build dropdown filter (type/color/material/pattern)
# ---------------------------------------------------------------------------

@app.get("/meta/attributes")
def api_meta_attributes():
    return {
        "types": TYPE_CHOICES,
        "colors": sorted(set(ATTRIBUTE_KEYWORDS["color"])),
        "materials": sorted(set(ATTRIBUTE_KEYWORDS["material"])),
        "patterns": sorted(set(ATTRIBUTE_KEYWORDS["pattern"])),
        "similarity_dimensions": SIMILARITY_DIMENSIONS,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
