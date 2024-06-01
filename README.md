# 本项目的改进

### 功能

处理来自 Gemini-OpenAI-Proxy 的响应的方式与处理来自 OpenAI 的响应的方式相同。

现在，您的应用程序可以通过 Gemini-OpenAI-Proxy 利用 OpenAI 功能，从而弥合 OpenAI 与使用 Google Gemini Pro 协议的应用程序之间的差距。


### 模型映射

| GPT Model            | Gemini Model                 |
|----------------------|------------------------------|
| gpt-4o               | gemini-1.0-pro-vision-latest |
| gpt-4-vision-preview | gemini-1.0-pro-vision-latest |
| gpt-4*               | gemini-1.5-pro-latest        |
| gpt-3.5*             | gemini-1.0-pro-latest        |


### 特殊修改

+ 如果要将 `gpt-4o` 映射到 gemini-1.5 flash-latest，可以配置环境变量 `GPT_4o=gemini-1.5-flash-latest`

+ 如果要将 `gpt-4-vision-preview` 映射到 gemini-1.5 pro-latest，可以配置环境变量 `GPT_4_VISION_PREVIEW=gemini-1.5-pro-latest`


# 部署和实例(来自[zhu327](https://github.com/zhu327/gemini-openai-proxy))

### 部署

使用 Docker 部署 Gemini-OpenAI-Proxy，以便轻松完成设置。请按照以下步骤使用 Docker 进行部署：

```bash
docker run --restart=always -it -d -p 8080:8080 --name gemini yilee01/gemini-openai-proxy:latest
```

根据需要调整端口映射（例如 `-p 8080:8080`），并确保 Docker 镜像版本（`yilee01/gemini-openai-proxy:latest`）符合您的要求。


### 代理集成

+ 示例 API 请求(假设代理部署在 `http://localhost:8080`):
   ```bash
   curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $YOUR_GOOGLE_AI_STUDIO_API_KEY" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'
   ```

+ 使用 Gemini Pro Vision:

   ```bash
   curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $YOUR_GOOGLE_AI_STUDIO_API_KEY" \
    -d '{
        "model": "gpt-4-vision-preview",
        "messages": [{"role": "user", "content": [
           {"type": "text", "text": "What’s in this image?"},
           {
             "type": "image_url",
             "image_url": {
               "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
             }
           }
        ]}],
        "temperature": 0.7
    }'
   ```

+ 使用 Gemini 1.5 Pro:

   ```bash
   curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $YOUR_GOOGLE_AI_STUDIO_API_KEY" \
    -d '{
        "model": "gpt-4-turbo-preview",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "temperature": 0.7
    }'
   ```


### 普通构建

构建 Gemini-OpenAI-Proxy：

```bash
go build -o gemini main.go
```


## License

Gemini-OpenAI-Proxy is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.