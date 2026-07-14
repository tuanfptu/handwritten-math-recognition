import requests
import base64

def image_to_hex(image_path: str) -> str:
    """Đọc ảnh và chuyển sang chuỗi hex."""
    with open(image_path, "rb") as f:
        return f.read().hex()

def send_request(image_path: str, prompt: str, server_url="https://8000-01k9kneq7qchd1wbgcy1698mf9.cloudspaces.litng.ai/predict"):
    # Chuyển ảnh sang hex
    image_hex = image_to_hex(image_path)

    # Payload JSON gửi đến API
    payload = {
        "image_bytes": image_hex,
        "prompt": prompt,
        "Type": "Latex"
    }

    # Gửi POST request
    response = requests.post(server_url, json=payload)

    # Kiểm tra kết quả
    if response.status_code == 200:
        print("✅ Response from server:")
        print(response.text)
    else:
        print(f"❌ Error {response.status_code}: {response.text}")

if __name__ == "__main__":
    # --- ví dụ sử dụng ---
    image_path = "D:\\projectDAT\\image-computer\\new_process\\Data\\crohme2019\\test\\ISICal19_1201_em_750.png"  # thay bằng đường dẫn ảnh của bạn
    prompt = "Describe the content of this image in detail."

    send_request(image_path, prompt)
