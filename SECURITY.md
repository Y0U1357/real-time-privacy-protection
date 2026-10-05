# Security Policy

## Scope

This repository is an engineering and portfolio demo for browser-local person
detection, tracking, and privacy masking. It is not a formal anonymity or
production security guarantee.

The intended privacy boundary is that webcam and local-video frames remain in
the current browser tab. The application does not intentionally upload raw
frames. Runtime diagnostics expose an image/frame upload counter to help verify
that boundary during testing.

## Reporting a vulnerability

Please avoid posting credentials, private media, or other sensitive data in a
public issue. Report only the minimum information needed to reproduce a problem.
If private vulnerability reporting is available for this repository, use that
channel for security-sensitive reports.

## Maintainer checklist

- Do not commit secrets, private keys, local media, or private datasets.
- Keep `.env*`, keys, checkpoints, and local benchmark video files ignored.
- Keep GitHub Actions permissions at least privilege.
- Treat PyTorch checkpoints as executable/untrusted input because `torch.load`
  deserializes Python objects.
- Preserve third-party license and attribution files when redistributing.
- Do not describe detector/tracker coverage as a formal privacy guarantee.
