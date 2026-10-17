# Phase 1 Application Foundation

This package introduces transport-independent contracts without moving or duplicating the legacy `bot.py` implementation.

```text
voice_agent/
  app/
    settings.py       side-effect-free settings and redacted secret values
    readiness.py      cached critical/optional startup checks
  session/
    contracts.py      audio, identity, event, lifecycle, trace, and port types
    runner.py         deterministic lifecycle and guaranteed cleanup
  transports/
    memory.py         finite in-memory development transport
  observability/
    logging.py        structured session correlation for new modules
  dev/
    smoke.py          no-AWS deterministic local session entry point
```

Run the foundation smoke check from the repository root:

```text
python -m voice_agent.dev.smoke
```

Migration rules:

- Pipelines must not import Amazon Connect protocol types.
- A2A adapters must not contain scenario or model-selection logic.
- Completion and escalation are lifecycle outcomes, not socket-close effects.
- Sample rates belong to each `AudioFormat`, not a global telephony constant.
- Connect-required traces will be mapped through the explicit `TraceSink` port.
- Do not add SIP, RTP, G.711, Chime SDK, fixed-IP, or UDP dependencies here.
- KVS is not a conversational transport; any future consumer is an independent observation subsystem.

The current Chime and Twilio implementations continue to use their existing Pipecat transports. A later focused phase will adapt those paths to the generic session core after this foundation is stable.
