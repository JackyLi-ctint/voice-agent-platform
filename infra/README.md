# Planned Infrastructure

The target deployment supports Amazon Connect external A2A audio and does not include KVS by default.

## Connect control-plane resources

- Connect Customer-tier instance or references to an existing managed instance.
- Claimed/ported phone number and inbound contact flow.
- Human fallback queue and routing configuration.
- Connect assistant, Lex V2 bot, orchestrating AI agent, and published version.
- External AppIntegrations application of type `A2A_SERVER`.
- Integration association and AI-agent security profile.
- Secrets Manager API key protected by a customer-managed KMS key.
- EventBridge rules for contact lifecycle reconciliation.

Some external-agent collaboration setup is API-only. Keep it reproducible with deployment scripts or custom resources where native infrastructure-as-code coverage is unavailable.

## Application runtime

Proof of concept may use one instance. Production target:

```text
Internet / Amazon Connect
  -> public TLS Application Load Balancer
  -> multi-AZ ECS/Fargate service
  -> Bedrock, Transcribe, Polly, DynamoDB, Secrets Manager
```

Required properties:

- persistent WebSocket support;
- health and readiness checks;
- connection draining and reconnect-aware deployments;
- autoscaling for concurrent sessions;
- centralized logs, metrics, alarms, and trace correlation;
- least-privilege IAM;
- no public SIP/RTP ports;
- administration separated from the public A2A endpoint.

## KVS

No KVS resources, permissions, retention configuration, or consumer should be created initially. A future KVS observation stack must be independently deployable and must not be on the conversational availability path.
