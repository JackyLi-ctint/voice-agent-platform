# Target Architecture

## Summary

Amazon Connect terminates the PSTN call and hands it immediately to this application as an external AI-agent collaborator. Connect and the application exchange bidirectional PCM audio over a persistent secure WebSocket using the A2A protocol with Amazon Connect extensions.

```text
PSTN
  -> Amazon Connect
     -> inbound contact flow
     -> Connect AI agent / Lex routing
     -> immediate external-agent handoff
        -> wss://<public-host>/connect/a2a
           -> A2A protocol adapter
           -> transport-independent session core
              -> Nova Sonic speech-to-speech
                 or
              -> Transcribe -> Bedrock -> Polly/MiniMax
        <- PCM response audio
     <- COMPLETE or ESCALATE
  -> disconnect or human queue
```

## Component responsibilities

### Amazon Connect

- Own the phone number, PSTN termination, contact flow, and caller lifecycle.
- Establish the external A2A collaboration.
- Play application audio to the caller.
- Flush buffered output when the application signals interruption.
- Route `COMPLETE`, `ESCALATE`, and error outcomes.
- Transfer escalated callers to a real human queue.

### A2A adapter

- Expose a public TLS WebSocket endpoint.
- Authenticate Connect's bearer token.
- Implement session initialization and READY-state gating.
- Negotiate PCM16 mono at 8, 16, or 24 kHz.
- Translate A2A audio chunks to and from the session core.
- Enforce one open response artifact at a time.
- Translate local VAD/barge-in into A2A interruption events.
- Emit required per-turn trace spans.
- Handle reconnect, errors, `COMPLETE`, and `ESCALATE` correctly.

### Voice-agent core

- Resolve runtime configuration and scenarios per call.
- Run Nova Sonic or the staged Transcribe/Bedrock/TTS pipeline.
- Own VAD, turn detection, model/tool execution, and response generation.
- Record transcripts, outcomes, summaries, metrics, and activity events.
- Remain independent of Connect, Twilio, browser, or any other transport.

### Runtime platform

For proof of concept, a single process may expose the endpoint. Production should use a multi-AZ runtime behind a public TLS load balancer with WebSocket support, connection draining, autoscaling, centralized telemetry, and isolated administration access.

## Audio strategy

Preferred A2A negotiation:

- Nova Sonic input: PCM16 mono, 16 kHz.
- Application output: PCM16 mono, 24 kHz.
- Staged pipeline input: compare 8 and 16 kHz for recognition quality and latency.

The selected Nova Sonic rates eliminate the legacy 8-to-16 kHz interpolation and all G.711/RTP processing.

## Call-control mapping

| Application intent | A2A result | Connect behavior |
|---|---|---|
| Task complete or caller ends | `FINISH: COMPLETE` | Complete flow and disconnect cleanly |
| Caller requests human | `FINISH: ESCALATE` | Transfer to human queue |
| Adapter/model failure | Failed task/protocol error | Follow contact-flow error fallback |
| Caller hangs up | Remote session close/contact event | Cancel processing and finalize history |

The application must no longer depend on WebSocket closure causing a SIP BYE.

## KVS decision

Kinesis Video Streams has no required role in this architecture. It is not the A2A transport and is not a supported return-audio mechanism for the custom bot.

Do not deploy KVS initially. Consider it later only when a named consumer needs live raw audio or separate caller/output tracks and native Connect recording or analytics cannot meet the requirement.

If approved, KVS is a parallel observation plane:

```text
Connect call -> Start media streaming -> Connect-managed KVS stream -> read-only consumer
```

KVS consumer failure must not affect the A2A conversation. Enabling it requires a separate decision covering retention, KMS, IAM, stream quotas, consumer ownership, cost, and sensitive-audio handling.

## Legacy boundary

Reusable concepts from `../../voice-agent-platform`:

- scenarios, prompts, knowledge bases, voices, and MCP tools;
- Nova Sonic and staged voice pipelines;
- session registry, history, summaries, metrics, audit, and admin behavior;
- browser and optional Twilio channels.

Explicitly excluded:

- Chime Voice Connector provisioning and administration;
- SIP, SDP, RTP, G.711, UDP 5060, and RTP port ranges;
- Node `voice-server` and fixed-IP/EIP origination design.

## AWS constraints to validate before production

- Connect Customer tier and target-region availability.
- Lex V2 bot with `AMAZON.QInConnectIntent` and required Nova Sonic locale configuration.
- External collaborator configured for audio streaming and immediate handoff.
- External audio collaborator is the sole collaborator for the orchestrating agent.
- Public `wss://` application registered as `A2A_SERVER`.
- API-key secret encrypted with a customer-managed KMS key.
- Security profile and contact-context permissions.
- Required trace events and supported reconnect behavior.
- Pricing, quotas, phone-number cutover, and human fallback.

## References

- [Amazon Connect agent-to-agent collaboration](https://docs.aws.amazon.com/connect/latest/adminguide/a2a-collaboration.html)
- [Build an external AI agent server](https://docs.aws.amazon.com/connect/latest/adminguide/a2a-voice.html)
- [Set up an external AI agent integration](https://docs.aws.amazon.com/connect/latest/adminguide/a2a-setup-external.html)
- [A2A quotas and limitations](https://docs.aws.amazon.com/connect/latest/adminguide/a2a-quotas.html)

AWS documentation content was rephrased for compliance with licensing restrictions.
