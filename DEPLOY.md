# Research-Copilot 服务器部署指南

## 架构说明

与 `canyu-portfolio` 共用一台服务器，通过 Nginx 区分：

| 路径 | 服务 |
|------|------|
| `http://IP/` | canyu-portfolio 博客 |
| `http://IP/api/` | Research-Copilot API |

---

## 前置检查

部署前请确认：

1. **LLM API Key 有效** — `.env` 中的 `OPENAI_API_KEY` 必须可用（当前配置用的是 DeepSeek）
2. **服务器配置** — 建议 2核4G 以上（Embedding 模型会占用内存）
3. **磁盘空间** — BGE-M3 模型约 1~2GB，预留 5GB 以上

---

## 部署步骤

```bash
# 1. 安装 Docker（如果还没装）
curl -fsSL https://get.docker.com | sudo sh -
sudo usermod -aG docker $USER
newgrp docker

# 2. 安装 Docker Compose
sudo apt install docker-compose-plugin -y

# 3. 上传代码到服务器
sudo mkdir -p /var/www/research-copilot
cd /var/www/research-copilot
sudo git clone https://github.com/chenxiyou-1314/Research-Copilot.git .

# 4. 配置环境变量
cp .env.example .env
nano .env   # 修改 OPENAI_API_KEY 为你的有效 key

# 5. 构建并启动
sudo docker compose up -d --build

# 6. 查看日志
sudo docker compose logs -f

# 7. 测试 API
curl http://localhost:8000/docs   # 应返回 Swagger UI
```

---

## 常用命令

```bash
# 查看状态
sudo docker compose ps

# 重启服务
sudo docker compose restart

# 更新代码后重建
sudo docker compose down
sudo git pull
sudo docker compose up -d --build

# 查看日志
sudo docker compose logs -f research-copilot
sudo docker compose logs -f redis
```

---

## 与博客共存时的 Nginx 配置

确保 `canyu-portfolio/nginx.conf` 中包含 `/api/` 的反向代理：

```nginx
location /api/ {
    proxy_pass http://127.0.0.1:8000/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_buffering off;
    proxy_read_timeout 3600s;
}
```

配置后重载 Nginx：
```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 访问方式

| 服务 | 地址 |
|------|------|
| API 文档 (Swagger) | `http://你的IP/api/docs` |
| API 接口 | `http://你的IP/api/research` 等 |
| 博客 | `http://你的IP/` |
