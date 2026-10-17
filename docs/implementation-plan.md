# Implementation Plan

## Delivery principles

- Build the Connect path beside the legacy project; do not mutate the rollback baseline.
- Separate transport protocol from conversation behavior before migrating features.
- Start with one engine and one test number.
- Treat completion, escalation, interruption, tracing, and reconnect as core functionality.
- Keep KVS out of scope unless its separate decision gate is approved.
- Remove Chime only after canary acceptance and rollback approval.

## Phase 0 — validate AWS prerequisites

- [ ] Confirm Connect Customer tier, region, pricing, quotas, and A2A audio access.
- [ ] Confirm test phone-number strategy.
- [ ] Define Complete, Escalate, after-hours, and error flow behavior.
- [ ] Review external audio latency and quality expectations with AWS.
- [ ] Confirm KVS is excluded from the initial scope.

Exit gate: a test number reaches a basic flow and the target account supports the required external A2A configuration.

## Phase 1 — establish the application skeleton

- [x] Select and pin direct Python/runtime dependency versions. `requirements-dev.txt` and the tested Linux/Python 3.12 transitive lock in `requirements-dev.lock` are validated by the Compose suite; production deployment migration remains a later infrastructure task.
- [x] Define configuration and secret-loading conventions.
- [x] Define transport-independent audio, event, lifecycle, and tracing interfaces.
- [x] Define stable session identity using Connect context/contact identifiers.
- [x] Add health, readiness, structured logging, and local development entry points.

Exit gate: the deterministic in-memory session and the complete Python suite pass in Docker Compose. Full service startup with live AWS credentials remains a separate integration check.

## Phase 2 — migrate reusable voice-agent behavior

- [ ] Migrate scenario/demo loading without Chime assumptions.
- [ ] Migrate Nova Sonic as the first engine.
- [ ] Migrate VAD, interruption detection, tools, and event broadcasting.
- [ ] Migrate history, outcome, summary, metric, and audit contracts.
- [ ] Preserve staged Transcribe/Bedrock/TTS as a later adapter to the same core.
- [ ] Keep browser/Twilio migration optional and isolated.

Exit gate: a local PCM fixture can drive a complete Nova Sonic session and produce output audio, events, and history through the generic core.

## Phase 3 — implement the Connect A2A adapter

- [x] Validate bearer credentials using constant-time comparison.
- [x] Implement initialization, response, and READY-state handling in the transport controller.
- [x] Implement PCM negotiation, initially 16/24 kHz preference with documented 8/16/24 kHz support.
- [x] Map continuous base64 PCM input and one output artifact/task in the protocol controller.
- [x] Close output artifacts and task state in the protocol controller.
- [x] Emit interruption, terminal outcome, and session trace protocol frames in the controller.
- [x] Implement `FINISH: COMPLETE` and `FINISH: ESCALATE` protocol frames.
- [x] Handle one eligible reconnect idempotently by context ID in the controller.
- [x] Map invalid controller state to typed protocol errors.
- [ ] Map request failures to standard JSON-RPC ErrorInfo envelopes and WebSocket close codes in the future ASGI bridge.

Deferred: mount the adapter as a public authenticated WebSocket endpoint only after the Pipecat engine is adapted to the transport-independent `SessionPorts` core. Fragmentation, DTMF, silence input, full per-turn tool/LLM OTLP propagation, request-error envelopes, and real Connect contract validation remain required before production use.

Exit gate: a deterministic A2A protocol harness completes, interrupts, escalates, reconnects, and rejects invalid state without leaking task/artifact state.

## Phase 4 — configure Amazon Connect

- [ ] Create or import the Connect instance and test number.
- [ ] Create the inbound contact flow and human fallback queue.
- [ ] Create the assistant, Lex V2 bot, and orchestrating AI agent.
- [ ] Register the public application as `A2A_SERVER`.
- [ ] Configure immediate external audio handoff.
- [ ] Create the API-key secret and customer-managed KMS policy.
- [ ] Configure the AI-agent security profile and contact-context access.
- [ ] Publish a versioned AI-agent configuration.
- [ ] Subscribe to Connect contact events for lifecycle reconciliation.
- [ ] Make API-only configuration reproducible through deployment automation.

Exit gate: one PSTN test call reaches the application's A2A endpoint and can return audio.

## Phase 5 — end-to-end proof of concept

Validate:

- [ ] multi-turn conversation;
- [ ] clear audio in both directions;
- [ ] prompt interruption/barge-in;
- [ ] normal completion;
- [ ] live human escalation;
- [ ] caller hangup cleanup;
- [ ] first reconnect recovery;
- [ ] second-drop/error fallback;
- [ ] transcript/history finalization;
- [ ] trace acceptance;
- [ ] comparison with legacy Chime latency and audio quality.

Exit gate: all agreed functional cases pass and measured latency/quality is acceptable to stakeholders.

## Phase 6 — production hardening

- [ ] Deploy multiple tasks/instances across Availability Zones.
- [ ] Add TLS load balancing, WebSocket draining, and autoscaling.
- [ ] Add readiness checks that cover critical dependencies.
- [ ] Load-test expected concurrent long-lived sessions.
- [ ] Add alarms for connection failure, protocol error, underrun, reconnect, model failure, and escalation failure.
- [ ] Validate least-privilege IAM, secret rotation, redaction, and retention.
- [ ] Document operational response and contact-flow fallback.

Exit gate: reliability, security, observability, and load gates pass in a production-like environment.

## Phase 7 — canary, cutover, and cleanup

- [ ] Route a test number or controlled traffic slice to Connect.
- [ ] Compare completion, escalation, error, latency, interruption, and transcript-quality metrics.
- [ ] Increase traffic gradually with explicit rollback thresholds.
- [ ] Preserve the legacy path through the rollback window.
- [ ] After approval, remove Chime/SIP/RTP resources from the old deployment.
- [ ] Update runbooks, ownership, costs, and architecture records.

Exit gate: Connect owns production PSTN traffic and no operational process depends on the Chime path.

## Optional KVS phase

Do not start this phase without a documented business requirement and owner.

- [ ] Document why Connect recording and native analytics are insufficient.
- [ ] Define required tracks, latency, retention, encryption, and deletion.
- [ ] Estimate peak streams and obtain sufficient quotas.
- [ ] Enable Start/Stop media streaming in the flow.
- [ ] Build a read-only, independently monitored consumer.
- [ ] Ensure consumer failure cannot affect A2A calls.

## First work item

Create the application skeleton and generic session interfaces, followed by a protocol-focused A2A adapter. Do not begin by copying `bot.py` wholesale: first isolate the reusable session core so the new code does not inherit FastAPI WebSocket, Chime, or SIP lifecycle coupling.
