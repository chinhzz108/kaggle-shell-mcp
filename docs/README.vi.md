# kaggle-shell-mcp (`kgsh`) ⚡ — Bản Rust (Tài liệu Tiếng Việt)

**Biến phiên GPU Kaggle 12h miễn phí (Dual NVIDIA T4 × 2 = 30 GB VRAM) thành máy trạm điều khiển trực tiếp qua Terminal & MCP cho AI Agent — đóng gói thành 1 file binary Rust duy nhất, không cần cài Python hay Kaggle SDK ở máy local.**

> Fork và viết lại bằng **Rust** từ [`NguyenQuocDat06-AI/kaggle-shell-mcp`](https://github.com/NguyenQuocDat06-AI/kaggle-shell-mcp).

---

## Điểm khác biệt của `kaggle-shell-mcp` (Rust)

1. **1 file thực thi Rust duy nhất, khởi động tức thì**: Tích hợp trực tiếp Kaggle API client (`https://api.kaggle.com/v1/...`), đóng gói `.tar.gz`, và MCP Server chính chủ `rmcp` hoàn toàn bằng Rust — không cần `.venv`, `pip`, hay Python ở máy local.
2. **Không cần cài VPN hay tài khoản trung gian**: Chỉ cần duy nhất `KAGGLE_USERNAME` và `KAGGLE_KEY` (`KGAT_...` hoặc key hex cũ) trong `.env`. Tự động mở đường truyền qua Cloudflare Quick Tunnel và đồng bộ địa chỉ qua Private Dataset `<username>/kaggle-mcp-state` (vẫn hỗ trợ `TS_AUTHKEY` nếu muốn chạy qua mạng riêng Tailscale).
3. **Giữ nguyên 100% môi trường CUDA/PyTorch trên Kaggle**: Nhúng sẵn `worker_template.py` vào binary Rust qua `include_str!` để chạy trực tiếp bên trong tiến trình Kaggle Kernel mà không cần cài `openssh-server`.
4. **Tách biệt lệnh ngắn & tiến trình train dài**:
   - `term_exec`: Bash PTY giữ trạng thái (`cd`, `export`, `conda`/`venv`) cho các thao tác kiểm tra nhanh.
   - `job_spawn` / `job_logs`: Chạy ngầm độc lập với kết nối mạng, tự gọt ký tự `\r` của thanh `tqdm` để không làm tràn context của Agent.
5. **Bảo vệ Quota 30h/tuần & Chống mất dữ liệu ở mốc 12h**:
   - Tự gửi `SIGTERM` lúc `11h35m` và thoát sạch `exit(0)` lúc `11h40m` để Kaggle lưu trọn vẹn 20 GB `/kaggle/working` thay vì bị kill lỗi timeout ở `12h00m`.
   - Tự tắt sạch khi rỗi mặc định sau **10 phút** không có lệnh, không có job nền và GPU `< 3%` (`idle_timeout_min=10`; truyền `--idle 0` nếu muốn tắt cơ chế này).

---

## Cài đặt nhanh (Rust / Cargo)

```powershell
# Biên dịch bản release
cargo build --release

# Hoặc cài trực tiếp lệnh `kgsh` vào PATH (~/.cargo/bin)
cargo install --path .
```

Tạo file `.env` ở thư mục gốc dự án (hoặc `~/.kaggle_mcp/.env`):
```ini
KAGGLE_USERNAME=ten_tai_khoan_kaggle
KAGGLE_KEY=KGAT_token_cua_ban
```

## Cấu hình MCP (`mcp_config.json`)

```json
{
  "mcpServers": {
    "kgsh": {
      "command": "C:/duong/dan/toi/target/release/kgsh.exe",
      "args": []
    }
  }
}
```
*(Nếu đã chạy `cargo install --path .`, có thể điền `"command": "kgsh"`.)*

## Sử dụng nhanh qua dòng lệnh (`kgsh`)

```powershell
# Kiểm tra trạng thái (không tốn quota)
kgsh status

# Bật máy Dual T4 (~50 giây; mặc định tự tắt sau 10 phút rỗi, thêm --idle 0 nếu muốn giữ liên tục)
kgsh up

# Chạy lệnh trực tiếp trên máy Kaggle
kgsh exec "nvidia-smi"

# Đẩy thư mục code từ máy lên /kaggle/working/my_app
kgsh push .\my_app /kaggle/working

# Chạy train ngầm trên cả 2 card T4
kgsh spawn "torchrun --nproc_per_node=2 my_app/train.py" --name run1
kgsh logs <job_id> --tail 50

# Kéo kết quả về & tắt máy ngay để tiết kiệm giờ GPU
kgsh pull /kaggle/working/outputs .\outputs
kgsh down
```
