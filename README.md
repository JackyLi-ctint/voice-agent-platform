# Voice Bot Platform

一个基于 Pipecat 的实时语音对话系统，同时支持**网页**和**电话呼入（PSTN）**两种入口，可在浏览器
里以"监听者"身份实时观察任意通话的 ASR / LLM / TTS 事件流。

## 功能一览

- **多语言**：粤语 / 普通话 / 英文 / 日文
- **多 LLM**：Nova 2 Lite / Nova Lite / Nova Micro / Claude Haiku 4.5 / Claude Sonnet 4.6（Bedrock）
- **多 TTS**：MiniMax（多种音色，含粤语）+ AWS Polly（neural / generative / long-form）
- **End-to-end S2S**：可选 Nova Sonic v2，单模型完成 ASR + LLM + TTS，端到端延迟更低
- **场景预设**：默认 / 销售客服 / IT 支持 / 模拟面试 / 语言陪练 / 设备技术支持（KB grounded）
- **总结**：按需对当前会话生成多语言总结（Markdown 渲染）
- **电话呼入**：Chime SDK Voice Connector → SIP/RTP → `voice-server`（Node, TS） → bot
- **Web Monitor**：浏览器以只读模式订阅任意活跃通话的事件流

## 架构

```
┌─────────────┐     WSS /ws (16 kHz PCM)        ┌──────────────────┐
│  浏览器 Web │ ─────────────────────────────► │                  │
└─────────────┘                                  │                  │
                                                 │   bot.py         │
┌─────────────┐  SIP+RTP   ┌────────────────┐    │   (Pipecat)      │
│ Chime Voice │ ─────────► │  voice-server  │    │                  │
│  Connector  │ (UDP 5060, │   Node SIP UAS │    │  /ws  /phone/ws  │
│             │  10000-    │                │    │  /monitor/ws     │
│             │  10999)    │  G.711 μ-law ↔ │    │  /api/...        │
│             │            │  PCM 16 kHz    │ ─► │                  │
└─────────────┘            └────────────────┘    │ Pipeline:        │
                                                 │ STT + LLM + TTS  │
                                                 │   或             │
                                                 │ Nova Sonic v2    │
                                                 │ (S2S)            │
                                                 │                  │
┌─────────────┐  WSS /monitor/ws (events only)  │                  │
│ Web Monitor │ ◄────────────────────────────── │                  │
└─────────────┘                                  └──────────────────┘
                                                          │
                                            ┌─────────────┼─────────────┐
                                            ▼             ▼             ▼
                                       Bedrock      Transcribe        Polly
                                       (LLM)        (STT)             (TTS)
                                                                    + MiniMax
```

## 目录结构

```
.
├── bot.py                  主服务（FastAPI + Pipecat）
├── static/index.html       Web UI（Talk + Monitor 双模式）
├── voice-server/           Node SIP/RTP 桥（电话路径）
│   ├── src/server.ts        SIP UAS + 来电分发
│   ├── src/sip/             SIP 信令、SDP、RTP session
│   ├── src/audio-utils.ts   8 ↔ 16 ↔ 24 kHz 重采样、μ-law 编解码
│   └── src/pipecat-client.ts Pipecat WS 客户端
├── data/
│   └── acme-security-support/kb.md  设备技术支持 KB 场景使用的示例文档
├── deploy/
│   ├── deploy.sh            一键部署脚本
│   ├── cloudformation.yaml  CFN 模板（EC2 + CloudFront + IAM）
│   └── README.md            部署细节
├── smoke_*.py               单服务连通性测试（Bedrock / Transcribe / Polly / MiniMax）
└── pipecat/                 Pipecat 上游源码 vendor 副本（仅参考，运行用 .venv 里的版本）
```

## 本地运行（开发模式）

适合在已有 IAM 权限的 EC2 上跑（Bedrock / Transcribe / Polly 都需要 AWS 凭据）。

```bash
# 1. 准备 Python 3.12 + Node 20
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install \
  'pipecat-ai[aws,aws-nova-sonic,silero,websocket]' \
  boto3 aiohttp python-dotenv fastapi uvicorn

# 2. 写 .env
cat > .env <<EOF
AWS_REGION=us-east-1
SITE_PASSWORD=               # 留空 = 不要密码
MINIMAX_API_KEY=sk-...       # 不用 MiniMax 留空，UI 切到 Polly/Nova Sonic
MINIMAX_BASE_URL=https://api.minimax.chat/v1/t2a_v2
MINIMAX_MODEL=speech-2.8-turbo
EOF

# 3. 启动 bot.py
.venv/bin/python bot.py --host 0.0.0.0 --port 7860

# 4. （可选）启动电话桥
cd voice-server
npm install
npx tsc
PUBLIC_IP=<EC2_PUBLIC_IP> RTP_PORT_BASE=10000 RTP_PORT_COUNT=1000 \
  PIPECAT_WS_URL=ws://127.0.0.1:7860/phone/ws \
  node dist/server.js
```

打开浏览器访问 `http://<host>:7860/`。如果设置了 `SITE_PASSWORD`，会弹出 Basic Auth 提示。

## Docker Compose 开发 / Docker Compose development

当 Kiro 不能使用 Dev Containers 插件时，用 Docker Compose 运行 Linux + Python 3.12 测试环境：

```powershell
docker compose build app
docker compose run --rm app python -m voice_agent.dev.smoke
docker compose run --rm app python -m pytest -q tests
```

完整前置条件、针对性回归命令、交互式 shell 和安全注意事项见 [`docs/docker-compose-development.md`](docs/docker-compose-development.md)。

## 云端部署（推荐）

详细见 [`deploy/README.md`](deploy/README.md)。要点：

```bash
cd deploy
MINIMAX_API_KEY=sk-... SITE_PASSWORD=your-pwd ./deploy.sh
```

部署完成后会输出一个 `https://*.cloudfront.net/` 地址，浏览器打开即可。

栈名默认 `voicebot`，Region 必须 `us-east-1`（Nova Sonic 可用区 + CloudFront 前缀列表 ID 写死）。

### 首次设置管理员 / First-run admin setup

首次部署后没有管理员账号，**不再使用 `ADMIN_PASSWORD`**（旧机制因 CFN heredoc
转义 bug 会损坏含特殊字符的密码，已废弃）。打开 CloudFront 地址会**自动跳转到
`/setup`** 设置向导：设置管理员用户名 + 密码 → 提交后**自动登录**进入后台。
之后向导永久关闭（再访问 `/setup` 反跳，`POST /api/auth/setup` 恒返回 `409`）。
`deploy.sh` 里的 “Admin UI password” 提示已无作用，可留空。

> After the first deploy there is no admin account yet, and `ADMIN_PASSWORD` is
> **no longer used** to seed one. Open the CloudFront URL → you are sent to
> **`/setup`** → create the admin username + password → you are **logged in
> automatically**. The wizard then closes permanently (`/setup` redirects away,
> `POST /api/auth/setup` returns `409`). The `deploy.sh` "Admin UI password"
> prompt is now a no-op; leave it blank.

详见 / see [`docs/first-run-setup.md`](docs/first-run-setup.md)（含锁死后的恢复方法 /
lockout recovery）。

## API 参考

bot.py 暴露的端点：

| 路由 | 类型 | 用途 |
| --- | --- | --- |
| `GET /` | HTTP | 返回 `static/index.html`（Web UI） |
| `GET /api/config` | HTTP | UI 启动时拉取的所有可选项（语言、模型、TTS 提供商、音色、场景、引擎） |
| `POST /api/summary` | HTTP | 对一次会话历史做 LLM 总结（请求体含 `turns`、`lang`） |
| `GET /api/calls` | HTTP | 当前活跃电话/Web 会话列表（Monitor UI 用） |
| `WS  /ws` | WebSocket | Web 客户端音频通道；二进制帧为 16 kHz PCM，文本帧为事件 JSON |
| `WS  /phone/ws` | WebSocket | voice-server → bot 的电话音频通道（**不**校验 SitePassword，因 Chime 不发 Basic Auth） |
| `POST /twilio/incoming-call` | HTTP | Twilio 语音 webhook：校验 `X-Twilio-Signature`，返回 Connect/Stream TwiML（见下「Twilio 接入」）。未配 Twilio env → 503 |
| `WS  /twilio/media-stream` | WebSocket | Twilio Media Streams 音频通道；读到 `start` 后用 pipecat `TwilioFrameSerializer` 接入同一会话内核。未配 Twilio env → 拒接 |
| `WS  /monitor/ws?call_id=...` | WebSocket | 只读事件订阅，多路并行，受 SitePassword 保护 |

`WS /ws` query params：
- `engine` — `pipeline` 或 `nova-sonic`
- `lang`、`model`、`provider`、`voice`、`minimax_model`、`scenario`
- `system` / `greeting` — 自定义系统提示和开场（可选）

## 配置 / 环境变量

bot.py 与 voice-server 共享同一份 `.env`。

### 必需

| 名称 | 含义 |
| --- | --- |
| `AWS_REGION` | Bedrock / Transcribe / Polly 调用区域，必须 `us-east-1` |

### Web 与默认值

| 名称 | 默认值 | 含义 |
| --- | --- | --- |
| `SITE_PASSWORD` | (空) | 设了之后整站启用 Basic Auth |
| `MINIMAX_API_KEY` | (空) | 留空时 UI 选 MiniMax 会失败，请改用 Polly/Nova Sonic |
| `MINIMAX_BASE_URL` | `https://api.minimax.chat/v1/t2a_v2` | 海外版端点 |
| `MINIMAX_MODEL` / `MINIMAX_MODEL_DEFAULT` | `speech-2.8-turbo` | MiniMax TTS 模型 |
| `MINIMAX_GROUP_ID` | (空) | 海外版账户通常不需要；留空别留空字符串否则 MiniMax 会按"未知账户"计费失败 |

### 电话呼入默认

代码里 `bot.py` 顶部读以下 env 决定每个 PSTN 通话默认走哪条路；UI 不影响电话。

| 名称 | 默认（云端部署） | 含义 |
| --- | --- | --- |
| `PHONE_ENGINE` | `nova-sonic` | `pipeline` 或 `nova-sonic` |
| `PHONE_LANG` | `en-US` | 同 LANGUAGES key |
| `PHONE_SCENARIO` | `acme-security-support` | 同 SCENARIOS / KB_SCENARIOS key |
| `PHONE_VOICE` | `tiffany` | 引擎对应音色 ID |
| `PHONE_PROVIDER` | `minimax` | 仅 pipeline 模式生效 |
| `PHONE_MODEL` | `nova-2-lite` | 仅 pipeline 模式生效 |
| `PHONE_MINIMAX_MODEL` | `speech-2.8-turbo` | 仅 pipeline 模式生效 |

### Twilio 接入（与 Chime 并存）

bot 现在支持**第二条 PSTN 入口：Twilio Media Streams**，与现有 Chime（SIP/RTP）
**永久并存**——按拨入的号码切换，两条路复用同一个 bot、同一套场景/语言/历史。

- **Chime（不变）**：`PSTN → Chime VC → SIP/RTP → voice-server(Node) → WS /phone/ws → bot`
- **Twilio（新增）**：`PSTN → Twilio → POST /twilio/incoming-call (TwiML) → WSS /twilio/media-stream → bot`

两条路最终都进同一个会话内核 `_run_phone_session`，复用 `PHONE_*` 电话默认值
（引擎 / 语言 / 场景 / 音色等，见上）。区别只在 transport 的 **serializer**：
Chime 用 `RawPCMSerializer`（逐字节零回归），Twilio 用 pipecat 内置的
`TwilioFrameSerializer`（μ-law ↔ PCM、8 kHz ↔ pipeline 速率重采样、打断 `clear`
事件全部由 pipecat 处理，我们不写自有音频代码）。

#### Twilio 控制台配置

在 Twilio 号码的 Voice 配置里，把 **A CALL COMES IN** 设为：

```
Webhook (HTTP POST) → https://<CloudFront-或自定义域名>/twilio/incoming-call
```

`<CloudFront-或自定义域名>` 就是 Twilio 从公网能访问到的入口（现有 CloudFront 分发
已代理 EC2:7860 + WebSocket + TLS，`/twilio/*` 直接走它，无需新开端口/安全组）。
该域名必须与 `TWILIO_PUBLIC_BASE_URL` 一致——TwiML 里回给 Twilio 的 `wss://` Stream
host 与签名校验串都用它（CloudFront 会改写 Host，所以不能信任入站请求的 Host）。

#### 所需环境变量

| 名称 | 必需 | 含义 |
| --- | --- | --- |
| `TWILIO_AUTH_TOKEN` | 是 | Twilio 账户 Auth Token，用于校验 `X-Twilio-Signature`（HMAC-SHA1，stdlib 实现，不依赖 twilio SDK） |
| `TWILIO_PUBLIC_BASE_URL` | 是 | Twilio 公网看到的 https 入口（如 `https://<dist>.cloudfront.net`）。用于签名串的 full_url **以及** TwiML 的 `wss://` host |
| `TWILIO_ACCOUNT_SID` | 否 | 仅 `TwilioFrameSerializer` 的 auto-hangup（挂断时回调 Twilio REST）需要；缺省则 auto-hangup 关闭 |

**开关**：`TWILIO_ENABLED = bool(TWILIO_AUTH_TOKEN and TWILIO_PUBLIC_BASE_URL)`。
**未配齐这两个 env 时 Twilio 端点优雅禁用**：`POST /twilio/incoming-call` 返回
`503`，`WS /twilio/media-stream` accept 后立即关闭。**对现有 Chime/Web 链路零影响**
——所以即使代码已部署，没注入 Twilio 凭据也不会改变现网行为。

#### 安全

- `incoming-call` 强制校验 `X-Twilio-Signature`：缺/错签名 → `403`。签名串用
  `TWILIO_PUBLIC_BASE_URL`+path+query 算（外部 URL，**不是** CloudFront 内部 Host）。
- `media-stream` 是 WS upgrade，Twilio 不签 → 靠 incoming-call 已签 + 必须先收到
  合法 `start` 帧（含不可猜的 `streamSid`）才进会话；空闲/非法连接快速关闭。

#### 范围

- 本期**只做入站接听**（inbound）。**出站拨号（outbound）暂不在范围内**，但
  `TwilioFrameSerializer` 的 auto-hangup 已为将来留口。
- 没提供 Twilio 号码/凭据时无法做真机端到端拨测；当前验证基于 fake-websocket 帧序列
  集成测试（`tests/test_twilio_integration.py`）+ 签名向量（`tests/test_twilio_sig.py`、
  `tests/test_twilio_endpoints.py`）。Twilio 号码与 webhook 的实际配置、是否上线，交由用户决定。

### voice-server

| 名称 | 默认 | 含义 |
| --- | --- | --- |
| `PUBLIC_IP` | `0.0.0.0` | SDP 里通告的媒体地址，必须是 Chime 能访问的 EC2 公网 IP |
| `RTP_PORT_BASE` | `10000` | RTP 起始端口（与 SG 的 `10000-10999` 对应） |
| `RTP_PORT_COUNT` | `1000` | RTP 端口池大小 |
| `PIPECAT_WS_URL` | `ws://127.0.0.1:7860/phone/ws` | bot.py 的 WS endpoint |

## 选项清单

### 语言（4）

`zh-HK`（粤语）、`zh-CN`（普通话）、`en-US`（英语）、`ja-JP`（日语）。

### LLM 模型（5，pipeline 模式）

| key | Bedrock id |
| --- | --- |
| `nova-2-lite` | `us.amazon.nova-2-lite-v1:0` |
| `nova-lite` | `us.amazon.nova-lite-v1:0` |
| `nova-micro` | `us.amazon.nova-micro-v1:0` |
| `claude-haiku-4.5` | `anthropic.claude-haiku-4-5-20251001-v1:0` |
| `claude-sonnet-4.6` | `anthropic.claude-sonnet-4-6` |

### Nova Sonic 音色（10，S2S 模式）

`tiffany` / `matthew` / `amy` / `carlos` / `sofia` / `beatrice` / `lorenzo` /
`marie` / `lennart` / `ana`。

### TTS Provider（pipeline 模式）

- **MiniMax**：海外版账户的全部 305 个系统音色（按语言筛选展示）
- **AWS Polly**：generative / neural / long-form / standard 多种引擎，
  按 4 种语言列出可用 voice id

### 场景

| key | 来源 | 说明 |
| --- | --- | --- |
| `default` | SCENARIOS | 友善通用助手 |
| `sales` | SCENARIOS | 销售客服 |
| `it-support` | SCENARIOS | IT 支持工程师 |
| `interviewer` | SCENARIOS | 模拟面试官 |
| `language-tutor` | SCENARIOS | 语言陪练 |
| `acme-security-support` | KB_SCENARIOS | 安防设备技术支持 Tina（示例 KB，基于 `data/acme-security-support/kb.md`） |

KB 场景把文档以"用户先发一条 message"的形式注入会话上下文，而不是塞进
system prompt——这样 Nova Sonic v2 的对话历史长度限制（~22 k 字符）才能装下大文档。

## 监控（Web Monitor）

任意一通进行中的通话都可以被订阅：

1. 打开 Web UI，顶部切到 **Monitor** 模式
2. 下拉里出现所有活跃 `call_id` + 主叫号码
3. 选一通通话即可看到 ASR partial / final、LLM token 流、TTS 起止、用户/机器人
   说话状态
4. 监听者只接收事件，不接收音频——可以同时挂多个浏览器

实现层面：每个通话有个 `ACTIVE_SESSIONS[call_id]`，pipeline 的
`EventBroadcaster` 把每个事件 fan-out 到通话主 emit + 所有监听 emit。

## 电话呼入接入（Chime SDK Voice Connector）

CFN 模板自动处理 EC2 / SG / IAM。Chime VC 的设置需要在 AWS 控制台手工做一次：

1. **Chime SDK → Voice Connectors → Create Voice Connector**（us-east-1）
2. **Origination** 加一条路由：
   - Host：CFN 输出的 `PublicIP`
   - Port：`5060`，UDP，Priority `1`，Weight `100`
3. **Phone numbers** 申请或绑定一个号码，inbound 路由指向这个 VC

成功后拨打该号码就会进入 bot 的电话路径，Web UI 的 Monitor 模式可以实时观看。

切换电话默认行为（语言、场景、音色、引擎），编辑 EC2 上 `/opt/voicebot/.env`
的 `PHONE_*` 段然后 `systemctl restart voicebot voiceserver`，或修改
`cloudformation.yaml` 后 `./deploy.sh` 重新部署。

## 常见问题

- **MiniMax TTS 没声音 / 401 / "insufficient balance"**：90% 是
  `MINIMAX_GROUP_ID` 设了空字符串，导致 MiniMax 把请求归到一个不存在的账户上。
  解决：直接不写 `MINIMAX_GROUP_ID`（不是写空，是不写）。
- **Polly 没声音**：Pipecat 自带的 Polly service 在我们的 pipeline 模式下有 bug，
  这个项目里写了一个 `SimplePollyTTSService`（直接调 `polly.synthesize_speech()`）
  绕过去，已默认启用，不需要额外配置。
- **Nova Sonic 报 `Timed out waiting for audio bytes`**：Nova Sonic v2 必须 16 kHz
  入站。voice-server 已默认把 8 kHz μ-law 解码后上采样到 16 kHz 再转发。
- **电话打不断**：voice-server 的 RTP 出栈队列在 `user_speaking=true` /
  `asr_partial` / `asr_final` 时会进入 mute 模式（直接丢新 audio），等
  `llm_start` / `tts_start` / `user_speaking=false` 任一事件再解 mute。
- **CloudFormation Update 后 EC2 公网 IP 变了**：CFN `aws cloudformation deploy`
  只在 instance 属性变化时才 replace，user-data 不会自动重跑。可以用 SSM
  `aws ssm send-command` 直接拉新 tarball 覆盖 `/opt/voicebot/`，或加 EIP 让 IP 固定。

## 相关命令

```bash
# 看实时日志
aws ssm start-session --target <InstanceId>
sudo tail -f /var/log/voicebot.log /var/log/voiceserver.log

# 重启
sudo systemctl restart voicebot voiceserver

# 拉最新代码不重启栈
aws s3 cp <new-tarball.tar.gz> /tmp/code.tar.gz
sudo -u ubuntu tar -xzf /tmp/code.tar.gz -C /opt/voicebot
cd /opt/voicebot/voice-server && sudo -u ubuntu npx tsc
sudo systemctl restart voicebot voiceserver

# 销毁
aws cloudformation delete-stack --stack-name voicebot --region us-east-1
aws s3 rb s3://voicebot-deploy-<account>-us-east-1 --force
```

## 已知限制

- 仅支持 Inbound（呼入），不做 Outbound
- 通话内容不持久化，全部在内存
- 没有 IVR 菜单 / 主叫鉴权
- Chime VC 是按分钟计费，长跑别忘了关
