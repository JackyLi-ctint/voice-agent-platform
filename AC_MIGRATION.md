# Voice Agent for Amazon Connect

Clean target workspace for migrating the existing voice-agent platform from Amazon Chime SDK Voice Connector/SIP/RTP to Amazon Connect external Agent-to-Agent (A2A) bidirectional audio.

## Target path

```text
PSTN caller
  -> Amazon Connect phone number and contact flow
  -> Connect AI agent, immediate handoff
  -> external A2A audio over secure WebSocket
  -> Pipecat voice-agent core
  -> A2A audio back through Connect
  -> caller
```

## Decisions

- A2A WebSocket is the real-time conversational transport.
- Kinesis Video Streams is not part of the base architecture.
- KVS may be added later as an independent observation plane only for an approved raw-audio analytics or archival requirement.
- Reuse the existing scenarios, tools, Nova Sonic and staged pipelines, history, monitoring, and admin concepts.
- Do not migrate Chime provisioning, SIP/SDP/RTP, G.711 handling, fixed UDP ports, or the Node voice server.
- Keep the existing `../voice-agent-platform` project unchanged as the comparison and rollback baseline during migration.

## Workspace map

- [`docs/architecture.md`](docs/architecture.md) — target boundaries and KVS decision.
- [`docs/implementation-plan.md`](docs/implementation-plan.md) — phased delivery and acceptance gates.
- [`voice_agent/README.md`](voice_agent/README.md) — implemented Phase 1 application boundaries.
- [`infra/README.md`](infra/README.md) — intended Connect and runtime infrastructure.

## Current status

Phase 1 is implemented and validated in Docker Compose. The Amazon Connect A2A controller/codec is implemented and contract-tested, but no public A2A WebSocket route is mounted until the Pipecat engine is adapted to the transport-independent session ports and a real Connect test instance validates the integration.

## Next implementation milestone

Adapt one Pipecat engine (Nova Sonic first) to the transport-independent `SessionEngine`/`SessionPorts` core, then mount the authenticated A2A WebSocket bridge and validate it with an Amazon Connect test number.
