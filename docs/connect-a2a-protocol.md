# Amazon Connect A2A Adapter Milestone

## Scope

`voice_agent.transports.connect_a2a` implements the authenticated, transport-independent A2A session controller and documented JSON-RPC envelope codec for an Amazon Connect external audio collaborator.

It deliberately does **not** add a public `/connect/a2a` WebSocket route yet. The current Pipecat pipeline is still constructed around `FastAPIWebsocketTransport`; routing a real Connect call into the new adapter before that core is migrated would create a nonfunctional second call path.

## Implemented A2A behavior

- Constant-time `Authorization: Bearer <apiKey>` validation.
- Required `A2A-Extensions` activation header validation and response echo data.
- `INIT_SESSION` parsing and `INIT_SESSION_RESPONSE` generation.
- Fixed PCM negotiation for LINEAR_PCM, 16-bit, mono audio at 8/16/24 kHz; the adapter counter-proposes 16 kHz input and 24 kHz output.
- `CHANNEL_STATE READY` output gating.
- Continuous base64 `audio/lpcm` input mapped to `AudioChunk` values.
- One active output artifact/task with independently playable base64 PCM chunks.
- Interruption signaling and cancellation of the active task without `lastChunk`.
- `COMPLETE`, `ESCALATE`, and failed terminal status updates; unsupported finish types fail the task rather than emitting an invalid FINISH.
- Connect-style OTLP trace artifact generation from session spans.
- Stable context/contact-derived identity with exactly one allowed unexpected reconnect; terminal records cannot reopen and registry retention is automatically bounded, with an explicit purge hook for host lifecycle integration.

## Deferred integration

Before exposing a live WebSocket endpoint, complete these work items:

1. Adapt the Pipecat engine into the Phase 1 `SessionEngine`/`SessionPorts` core.
2. Add the authenticated ASGI upgrade route, header echo, JSON frame loop, and supported close-code handling.
3. Implement fragment reassembly, DTMF, silence input, request-level JSON-RPC ErrorInfo envelopes, supported close-code handling, and full per-turn tool/LLM trace events.
4. Validate a real session with an Amazon Connect test instance and the official A2A sample-event contract.
5. Add contact-flow error and escalation handling.

## Sources

- [A2A protocol contract for external AI agents](https://docs.aws.amazon.com/connect/latest/devguide/a2a-developer-guide.html)
- [A2A sample events](https://docs.aws.amazon.com/connect/latest/devguide/a2a-sample-events.html)
- [Build an external AI agent server](https://docs.aws.amazon.com/connect/latest/adminguide/a2a-voice.html)

Content derived from AWS documentation was rephrased for compliance with licensing restrictions.
