#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localdream_mcp_server.py — Local Dream MCP 桥接服务器（Termux 运行）

把手机上的 Local Dream（本地 Stable Diffusion，8081 后端）包装成 MCP 服务器，
让 RikkaHub 直接通过 MCP 调用生成图片。Rikka 与 Local Dream 同机 → 完全离线闭环。

架构：
  [Rikka] 设置→MCP→添加服务器 → http://127.0.0.1:8000/mcp
     ↓ MCP 工具调用
  [本脚本] localdream_mcp_server.py（Termux 跑，0.0.0.0:8000）
     ↓ HTTP
  [Local Dream] 127.0.0.1:8081/generate（需先加载模型 + 开"允许局域网访问"）

用法（Termux）：
  pkg install -y python
  termux-setup-storage
  python3 localdream_mcp_server.py --port 8000 [--token xxx]
"""
import argparse
import base64
import datetime
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"
LD_BACKEND = os.environ.get("LD_BACKEND", "http://127.0.0.1:8081")
SAVE_DIR = os.path.expanduser(os.environ.get("LD_SAVE_DIR", "~/storage/pictures/LocalDreamMCP"))


def ld_health():
    try:
        with urllib.request.urlopen(f"{LD_BACKEND}/health", timeout=5) as r:
            return {"ok": True, "status": r.status}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def ld_generate(prompt, negative="", steps=25, cfg=7.0, denoise=0.6,
                width=512, height=512, seed=None, image_b64=None,
                output_format="jpeg", timeout=600):
    """调用 Local Dream /generate（SSE），返回 (图片字节, 格式)"""
    if not seed:
        seed = int(time.time() * 1000) % (2 ** 32)
    body = {
        "prompt": prompt, "negative_prompt": negative, "steps": int(steps),
        "cfg": float(cfg), "denoise_strength": float(denoise),
        "width": int(width), "height": int(height), "seed": int(seed),
        "scheduler": "dpm", "output_format": output_format, "preview_format": "raw",
    }
    if image_b64:
        body["image"] = image_b64
    req = urllib.request.Request(f"{LD_BACKEND}/generate", data=json.dumps(body).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "ld-mcp"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        final = None
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except Exception:
                continue
            if evt.get("image"):
                final = evt
        if not final:
            raise RuntimeError("未收到生成结果事件")
        img_bytes = base64.b64decode(final["image"])
        fmt = final.get("format", output_format)
        if fmt == "raw":
            img_bytes = _raw_to_png(img_bytes, int(width), int(height))
            fmt = "png"
        return img_bytes, fmt


def _raw_to_png(data, w, h):
    try:
        from PIL import Image
    except ImportError:
        row_size = (w * 3 + 3) & ~3
        bmp_size = 54 + row_size * h
        out = bytearray(b"BM" + bmp_size.to_bytes(4, "little") + b"\x00\x00\x00\x00" +
                        (54).to_bytes(4, "little") + (40).to_bytes(4, "little") +
                        w.to_bytes(4, "little") + h.to_bytes(4, "little") +
                        (1).to_bytes(2, "little") + (24).to_bytes(2, "little") +
                        b"\x00\x00\x00\x00" + (row_size * h).to_bytes(4, "little") +
                        (2835).to_bytes(4, "little") + (2835).to_bytes(4, "little") +
                        b"\x00\x00\x00\x00\x00\x00\x00\x00")
        for y in range(h):
            row = data[y * w * 3:(y + 1) * w * 3]
            out += row + b"\x00" * (row_size - w * 3)
        return bytes(out)
    img = Image.frombytes("RGB", (w, h), data)
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def save_image(img_bytes, ext, prompt):
    os.makedirs(SAVE_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"ld_{stamp}.{ext}"
    fpath = os.path.join(SAVE_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(img_bytes)
    return fpath


def _ensure_valid_image(img_bytes, width, height):
    """校验图片字节：返回 (字节, 正确扩展名)。损坏/无文件头 → 按 raw RGB(A) 恢复"""
    if img_bytes[:2] == b"\xff\xd8":
        return img_bytes, "jpg"
    if img_bytes[:4] == b"\x89PNG":
        return img_bytes, "png"
    if img_bytes[:6] in (b"GIF89a", b"GIF87a"):
        return img_bytes, "gif"
    if len(img_bytes) == width * height * 3:
        return _raw_to_png(img_bytes, width, height), "png"
    if len(img_bytes) == width * height * 4:
        try:
            from PIL import Image
            import io
            img = Image.frombytes("RGBA", (width, height), img_bytes).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return buf.getvalue(), "png"
        except Exception:
            pass
    return img_bytes, "bin"


def tool_generate(args):
    """生成图片（txt2img / img2img）并保存到手机相册目录"""
    prompt = (args or {}).get("prompt", "")
    if not prompt:
        return {"error": "prompt 必填"}
    try:
        width = int((args or {}).get("width", 512))
        height = int((args or {}).get("height", 512))
        img_bytes, fmt = ld_generate(
            prompt=prompt,
            negative=(args or {}).get("negative_prompt", ""),
            steps=int((args or {}).get("steps", 25)),
            cfg=float((args or {}).get("cfg", 7.0)),
            denoise=float((args or {}).get("denoise", 0.6)),
            width=width, height=height,
            seed=int((args or {}).get("seed", 0)) or None,
            image_b64=(args or {}).get("image_base64", "") or None,
        )
        img_bytes, fmt = _ensure_valid_image(img_bytes, width, height)
        fpath = save_image(img_bytes, fmt, prompt)
        return {"status": "ok", "saved_to": fpath, "size": f"{len(img_bytes)/1024:.0f}KB",
                "hint": f"图片已保存到 {fpath}，可去相册/文件管理器查看"}
    except Exception as e:
        return {"error": str(e)[:300]}


def tool_status(_):
    return ld_health()


TOOLS = [
    {"name": "localdream_generate",
     "description": "用手机上的 Local Dream 生成图片（txt2img 或 img2img）。支持参考图（image_base64 传 base64 图片则 img2img，denoise 越低越保留原图）。生成后保存到手机相册目录并返回路径。",
     "inputSchema": {"type": "object", "properties": {
         "prompt": {"type": "string"}, "negative_prompt": {"type": "string"},
         "image_base64": {"type": "string"}, "denoise": {"type": "number"},
         "steps": {"type": "integer"}, "cfg": {"type": "number"},
         "width": {"type": "integer"}, "height": {"type": "integer"},
         "seed": {"type": "integer"}}, "required": ["prompt"]}},
    {"name": "localdream_status",
     "description": "检查 Local Dream 后端是否在线（模型是否加载）。",
     "inputSchema": {"type": "object", "properties": {}}},
]

HANDLERS = {"localdream_generate": tool_generate, "localdream_status": tool_status}


def handle_request(body, token):
    try:
        msg = json.loads(body)
    except Exception:
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, False
    method = msg.get("method", "")
    msg_id = msg.get("id")
    is_notification = "id" not in msg
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "localdream-mcp", "version": "1.0.0"}}
    elif method == "notifications/initialized":
        return None, True
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name", "")
        args = params.get("arguments", {}) or {}
        if name not in HANDLERS:
            result = {"content": [{"type": "text", "text": json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)}], "isError": True}
        else:
            try:
                out = HANDLERS[name](args)
                result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]}
            except Exception as e:
                result = {"content": [{"type": "text", "text": json.dumps({"error": str(e)}, ensure_ascii=False)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}, False
    if is_notification:
        return None, True
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}, False


class MCPHandler(BaseHTTPRequestHandler):
    server_version = "localdream-mcp/1.0"

    def _check_auth(self):
        token = self.server.token
        if not token:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {token}"

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False)
        payload = f"event: message\ndata: {data}\n\n".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path.rstrip("/") != "/mcp":
            self._send_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32004, "message": "Not found"}}, 404)
            return
        if not self._check_auth():
            self._send_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32001, "message": "Unauthorized"}}, 401)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode()
        except Exception:
            self._send_json({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}, 400)
            return
        result, is_notification = handle_request(body, self.server.token)
        if is_notification:
            self.send_response(202)
            self.send_header("MCP-Protocol-Version", PROTOCOL_VERSION)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        accept = self.headers.get("Accept", "")
        if "text/event-stream" in accept:
            self._send_sse(result)
        else:
            self._send_json(result)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            body = json.dumps(ld_health()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = b"localdream-mcp is running. POST /mcp for MCP protocol.\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("[mcp] %s\n" % (fmt % args))


def main():
    global LD_BACKEND, SAVE_DIR
    ap = argparse.ArgumentParser(description="Local Dream MCP 桥接服务器（Termux）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--token", default=os.environ.get("LD_MCP_TOKEN", ""))
    ap.add_argument("--backend", default=LD_BACKEND)
    args = ap.parse_args()
    LD_BACKEND = args.backend
    if not SAVE_DIR.startswith("/"):
        SAVE_DIR = os.path.expanduser(SAVE_DIR)
    print(f"✅ localdream-mcp 启动: http://{args.host}:{args.port}/mcp")
    print(f"   后端: {LD_BACKEND}   保存目录: {SAVE_DIR}")
    if args.token:
        print("   Bearer token 已启用")
    server = ThreadingHTTPServer((args.host, args.port), MCPHandler)
    server.cfg = None
    server.token = args.token
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 已停止")


if __name__ == "__main__":
    main()