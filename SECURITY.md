# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in APEX-1, please report it responsibly:

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email: [security@aarambhdevhub.org](mailto:security@aarambhdevhub.org)
3. Include a detailed description of the vulnerability
4. Include steps to reproduce if possible
5. We will acknowledge receipt within 48 hours
6. We will provide a fix timeline within 7 days

## Scope

Security concerns for APEX-1 include:
- Model weight tampering or injection
- Checkpoint deserialization vulnerabilities
- Tokenizer exploits (adversarial inputs)
- Training data poisoning vectors
- Inference-time prompt injection
- Denial of service via crafted inputs

## Responsible Disclosure

We follow a 90-day responsible disclosure policy. We ask that you give us
reasonable time to address the vulnerability before public disclosure.
