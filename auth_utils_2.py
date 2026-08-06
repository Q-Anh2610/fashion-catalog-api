import os
import uuid
from dotenv import load_dotenv
from supabase import create_client
from supabase.client import ClientOptions

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
# LƯU Ý: đây là SERVICE ROLE KEY (Project Settings > API > service_role),
# KHÁC với SUPABASE_KEY (anon key) đang dùng ở db_utils.py.
# Service role bypass toàn bộ RLS -> CHỈ dùng khi test local, KHÔNG bao giờ
# đưa key này vào Flutter/frontend hoặc commit lên git.
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

_admin_client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY,
    options=ClientOptions(postgrest_client_timeout=30, storage_client_timeout=30),
)

MOCK_TOKEN_PREFIX = "MOCK-DEV-TOKEN::"


def send_otp(email: str):
    """MOCK — không gọi Supabase Auth thật, không gửi email, trả về ngay lập tức."""
    return f"[MOCK] Bỏ qua gửi OTP thật cho {email}. Gọi /auth/verify-otp với code bất kỳ (VD '000000') để đăng nhập ngay."


def _get_or_create_user(email: str):
    """Tìm user theo email trong auth.users; nếu chưa có thì tạo mới (service role, không cần OTP)."""
    page = _admin_client.auth.admin.list_users()
    for u in page:
        if u.email and u.email.lower() == email.lower():
            return u

    created = _admin_client.auth.admin.create_user({
        "email": email,
        "email_confirm": True,
        "password": str(uuid.uuid4()),  # password ngẫu nhiên, không dùng tới vì không login bằng password
    })
    return created.user


def verify_otp(email: str, code: str):
    """
    MOCK — bỏ qua kiểm tra code (chấp nhận bất kỳ giá trị nào).
    Tạo/lấy user thật trong auth.users (để FK product.seller_id hợp lệ),
    trả về access_token đánh dấu là mock token.
    """
    try:
        user = _get_or_create_user(email.strip())
    except Exception as e:
        print(f"[MOCK AUTH] Lỗi tạo/lấy user: {e}")
        return None

    return {
        "seller_id": user.id,
        "access_token": f"{MOCK_TOKEN_PREFIX}{user.id}",
        "email": user.email,
    }


def get_authenticated_client(access_token: str):
    """
    Nếu là mock token -> trả về client service-role, bypass RLS hoàn toàn.
    File này CHỈ dùng để test local.
    """
    if access_token.startswith(MOCK_TOKEN_PREFIX):
        return _admin_client
    raise ValueError("Token không phải mock token. File này chỉ dùng để test local.")