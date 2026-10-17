# Planned Application Structure

No implementation has been copied yet. Use this directory for the clean Connect application rather than carrying forward the legacy monolithic transport coupling.

Proposed modules:

```text
src/
  app/                 service startup, health, configuration
  domain/              session identity, lifecycle, outcomes
  transports/
    base.py             transport-independent contracts
    connect_a2a.py      Amazon Connect A2A protocol adapter
  pipelines/
    nova_sonic.py       speech-to-speech engine
    staged.py           Transcribe -> Bedrock -> TTS engine
  sessions/             orchestration, registry, reconnect handling
  tools/                end, escalate, and business/MCP tools
  history/              DynamoDB records and summaries
  observability/        events, metrics, traces, redaction
```

Rules for migration:

- The pipeline must not import Connect protocol types.
- The A2A adapter must not contain scenario or model-selection logic.
- Completion and escalation are lifecycle callbacks, not socket-close side effects.
- Sample rates are negotiated session properties, not global telephony constants.
- Required Connect trace events are emitted through an explicit observer.
- Do not add SIP, RTP, G.711, Chime SDK, fixed-IP, or UDP-port dependencies.
- Keep KVS outside `transports/`; any future KVS consumer belongs to a separate observation subsystem.
