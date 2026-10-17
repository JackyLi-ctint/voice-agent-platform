# Nova Sonic SessionEngine Milestone

`voice_agent.engines.nova_sonic.NovaSonicSessionEngine` is the first Pipecat engine that can run over the transport-independent `SessionContext` contract.

It uses `voice_agent.pipecat.SessionPortsTransport` to map:

```text
SessionPorts inbound PCM16 16 kHz
  -> Pipecat InputAudioRawFrame
  -> VAD / Nova Sonic / context aggregation
  -> Pipecat OutputAudioRawFrame 24 kHz
  -> SessionPorts outbound PCM
```

The adapter preserves Nova’s native 16 kHz input and 24 kHz output assumptions, supports optional resolved tool schemas, seed messages, VAD configuration, MCP function bindings, greeting startup, event translation, and MCP-client cleanup.

## Intentional boundary

This module is not mounted in `bot.py` and does not alter browser, Chime, or Twilio routes. The future A2A ASGI bridge must resolve runtime defaults, prompts, voices, tools, and MCP clients using the existing application configuration, then create this immutable engine config after the A2A `READY` event.

The A2A controller’s port interruption is non-terminal: Pipecat `InterruptionFrame` flushes current Connect output while the continuous Nova pipeline remains alive for the next caller turn.

Before live Connect traffic, complete the public ASGI bridge, Connect ErrorInfo and close-code mapping, fragment/DTMF/silence handling, full trace propagation, legacy-config extraction, and real Amazon Connect test-instance validation.
