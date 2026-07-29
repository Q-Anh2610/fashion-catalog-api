"""
main.py — FastAPI layer gom các module (auth_utils, classify_utils, db_utils, model_utils_2)
thành 1 API duy nhất, dùng lại đúng luồng nghiệp vụ đang có trong app.py (bản Gradio demo).

Chạy local:
    uvicorn main:app --reload --port 8000
Swagger docs tự sinh tại: http://localhost:8000/docs
"""

import os
import shutil
import tempfile
import uuid
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from auth_utils import get_authenticated_client, send_otp, verify_otp
from classify_utils import (
    ATTRIBUTE_KEYWORDS,
    GLOBAL_ATTRIBUTES,
    TYPE_SPECIFIC_ATTRIBUTES,
    parse_caption,
)
from db_utils import (
    classify_and_update_product,
    create_pending_product,
    ensure_device_exists,
    get_seller_products,
    map_value_to_code,
    save_caption,
    search_products_by_text,
    submit_seller_correction,
    supabase,
)
from model_utils_2 import generate_caption


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fashion Catalog API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # DEV ONLY — khi deploy production nên giới hạn domain Flutter Web thật
    allow_methods=["*"],
    allow_headers=["*"],
)

TYPE_CHOICES = ["dress", "jacket", "skirt", "pants", "top"]
SIMILARITY_DIMENSIONS = ["type", "color", "material", "pattern"]


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


def _format_parser_for_db(parsed: dict[str, Any]) -> dict[str, Any]:
    item_type = parsed.get("type", "unknown")
    global_values = {}
    for attr in GLOBAL_ATTRIBUTES:
        value = parsed.get("global_attributes", {}).get(attr)
        if value is None:
            value = parsed.get("attributes", {}).get(attr)
        global_values[attr] = _first_value(value)

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
    visible = {key: value for key, value in attrs.items() if value}
    if not visible:
        return ""
    return ", ".join(f"{key}: {value}" for key, value in visible.items())


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


def _code_filters(type_filter, color_filter, material_filter, pattern_filter):
    filters = {}
    if type_filter:
        filters["type_code"] = map_value_to_code(type_filter, "type_code", "type_name", "type_code")
    if color_filter:
        filters["color_code"] = map_value_to_code(color_filter, "color_code", "color_name", "color_code")
    if material_filter:
        filters["material_code"] = map_value_to_code(
            material_filter, "material_code", "material_name", "material_code",
        )
    if pattern_filter:
        filters["pattern_code"] = map_value_to_code(
            pattern_filter, "pattern_code", "pattern_name", "pattern_code",
        )
    return {key: value for key, value in filters.items() if value and value != "X"}


def _target_codes(db_parser: dict[str, Any]) -> dict[str, str]:
    global_attrs = db_parser["attributes"]["global"]
    return {
        "type_code": map_value_to_code(db_parser["type"], "type_code", "type_name", "type_code"),
        "color_code": map_value_to_code(global_attrs.get("color"), "color_code", "color_name", "color_code"),
        "material_code": map_value_to_code(
            global_attrs.get("material"), "material_code", "material_name", "material_code",
        ),
        "pattern_code": map_value_to_code(global_attrs.get("pattern"), "pattern_code", "pattern_name", "pattern_code"),
    }


def _product_to_json(product: dict, score=None) -> dict:
    cls_rows = product.get("classification_results") or []
    cls = cls_rows[0] if isinstance(cls_rows, list) and cls_rows else {}
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
        "color_code": cls.get("color_code", ""),
        "material_code": cls.get("material_code", ""),
        "pattern_code": cls.get("pattern_code", ""),
        "image_path": product.get("image_path", ""),
    }


def _random_price() -> int:
    """Random giá trong khoảng 100_000 - 500_000, luôn là bội số của 1000 (3 số cuối = 000)."""
    import random
    return random.randrange(100_000, 500_001, 1_000)


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

    codes = _target_codes(db_parser)
    dimension_to_code_column = {
        "type": "type_code", "color": "color_code", "material": "material_code", "pattern": "pattern_code",
    }

    res = supabase.table("classification_results").select("*, product(*)").execute()
    scored = []
    for row in res.data or []:
        product = row.get("product")
        if not product:
            continue

        if not dimensions:
            score, matched = 0, []
        else:
            matched = []
            for dim in dimensions:
                col = dimension_to_code_column[dim]
                target = codes.get(col)
                if target and target != "X" and row.get(col) == target:
                    matched.append(dim)
            score = len(matched) / len(dimensions)
            if score == 0:
                continue

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
        scores[pid] = "all" if not dimensions else f"{score:.0%} ({', '.join(matched)})"
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
    price: Optional[int] = Form(None),
    seller=Depends(get_seller_client),
):
    client, seller_id = seller
    image_path = _save_upload_to_tmp(file)
    try:
        final_price = price if price is not None else _random_price()
        caption, _, db_parser = _parse_image(image_path)

        product_id = create_pending_product(
            seller_client=client,
            seller_id=seller_id,
            image_local_path=image_path,
            price=final_price,
        )
        caption_id = save_caption(client, product_id, caption)
        product_code = classify_and_update_product(client, product_id, caption_id, db_parser)

        return {
            "product_code": product_code,
            "product_id": product_id,
            "caption": caption,
            "type": db_parser["type"],
            "attributes": _display_attributes(db_parser),
            "status": "done",
        }
    finally:
        os.remove(image_path)


@app.post("/seller/products/batch")
def api_seller_upload_batch(
    files: list[UploadFile] = File(...),
    seller=Depends(get_seller_client),
):
    client, seller_id = seller
    if len(files) > 20:
        raise HTTPException(400, "Chỉ được upload tối đa 20 ảnh mỗi lần.")

    results = []
    ok_count = 0
    for index, upload in enumerate(files, start=1):
        image_path = _save_upload_to_tmp(upload)
        try:
            price = _random_price()
            caption, _, db_parser = _parse_image(image_path)
            product_id = create_pending_product(
                seller_client=client,
                seller_id=seller_id,
                image_local_path=image_path,
                price=price,
            )
            caption_id = save_caption(client, product_id, caption)
            product_code = classify_and_update_product(client, product_id, caption_id, db_parser)
            results.append({
                "product_code": product_code,
                "product_id": product_id,
                "caption": caption,
                "type": db_parser["type"],
                "attributes": _display_attributes(db_parser),
                "status": "done",
            })
            ok_count += 1
        except Exception as exc:
            results.append({
                "product_code": None, "product_id": None, "caption": None,
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


class SellerCorrectionRequest(BaseModel):
    result_id: int
    corrected_category: str


@app.post("/seller/corrections")
def api_seller_submit_correction(payload: SellerCorrectionRequest, seller=Depends(get_seller_client)):
    client, _seller_id = seller
    submit_seller_correction(client, payload.result_id, payload.corrected_category)
    return {"status": "ok"}


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
):
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