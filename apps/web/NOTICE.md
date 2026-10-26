# Third-party components

The following components are adapted from [ElevenLabs UI](https://github.com/elevenlabs/ui),
commit [`3a28ebab197f8cfa599934f52db12ad1396e54f4`](https://github.com/elevenlabs/ui/tree/3a28ebab197f8cfa599934f52db12ad1396e54f4):

- `src/components/elevenlabs/audio-player.tsx` from `apps/www/registry/elevenlabs-ui/ui/audio-player.tsx`.
- `src/components/elevenlabs/live-waveform.tsx` from `apps/www/registry/elevenlabs-ui/ui/live-waveform.tsx`.

The audio player imports the project's local shadcn-compatible primitives and omits
upstream demo tracks. The live waveform accepts an optional external `MediaStream`
to share the recorder microphone without requesting a second stream. The component
never stops externally owned tracks; its own microphone setup also handles cancellation
while microphone permission is pending.

Copyright (c) 2025 Eleven Labs Inc. Licensed under the MIT License, reproduced in
[`src/components/elevenlabs/LICENSE.md`](src/components/elevenlabs/LICENSE.md).
No ElevenLabs API or account is required for these components.

## Fonts

Instrument Sans Regular and Bold are bundled in `public/fonts/` and served locally.
Copyright 2022 The Instrument Sans Project Authors
([project source](https://github.com/Instrument/instrument-sans)). Licensed under
the SIL Open Font License 1.1, reproduced in
[`public/fonts/InstrumentSans-OFL.txt`](public/fonts/InstrumentSans-OFL.txt).
