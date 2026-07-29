import os
from dotenv import load_dotenv
from supabase import create_client
from supabase.client import ClientOptions

load_dotenv()
supabase = create_client(
    os.environ["SUPABASE_URL"], 
    os.environ["SUPABASE_KEY"],
    options=ClientOptions(postgrest_client_timeout=30, storage_client_timeout=30),
)


def send_otp(email: str):
    """Bước 1: gửi mã 6 số về email người bán"""
    supabase.auth.sign_in_with_otp({"email": email})
    return f"Đã gửi mã xác thực tới {email}, kiểm tra hộp thư."


def verify_otp(email: str, code: str):
    """
    Bước 2: người bán nhập mã 6 số nhận được -> trả về session
    Trả về seller_id (UUID) nếu thành công, None nếu sai mã
    """
    try:
        res = supabase.auth.verify_otp({
            "email": email,
            "token": code,
            "type": "email"
        })
        session = res.session
        user = res.user
        return {
            "seller_id": user.id,
            "access_token": session.access_token,
            "email": user.email
        }
    except Exception as e:
        print(f"Xác thực thất bại: {e}")
        return None


def get_authenticated_client(access_token: str):
    """
    Tạo 1 client Supabase riêng, gắn access_token của người bán
    -> mọi request qua client này sẽ được RLS nhận diện đúng auth.uid()
    """
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    client.auth.set_session(access_token, "")
    client.postgrest.auth(access_token)
    client.storage._client.headers["Authorization"] = f"Bearer {access_token}"
    return client