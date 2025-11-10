# generate_qr.py
import qrcode

url = "https://predictive-analytics-for-mortality-risk-ifter.streamlit.app/"

# konfigurasi QR
qr = qrcode.QRCode(
    version=1,  # 1..40 (size) ; 1 cukup untuk short URL
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

qr.add_data(url)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")

# simpan file
output_path = "mortality_app_qr.png"
img.save(output_path)
print(f"QR code saved to: {output_path}")