I've been running a fully local voice pipeline for a couple of months (openWakeWord, Wyoming-onnx-asr for STT, Qwen3-8B as the local LLM, KokoroFastAPI for TTS) and it works great, except when the TV is on. The satellite picks up everything - my voice, the TV, whoever else is in the room. The result is garbage transcripts like "What time is it? People look at you like some kind of service freak" (actual transcript, the second half was from the TV).

I tried adjusting VAD sensitivity, microphone gain, different wake words, placement tweaks... nothing really solved it. The fundamental problem is that a microphone can't tell whose voice it's hearing.

So I built Wyoming Voice Match - a Wyoming protocol proxy that sits between HA and your ASR service. It uses ECAPA-TDNN to compare incoming audio against enrolled voiceprints and does two things:

Rejects audio that doesn't match any enrolled speaker (prevents TV dialogue from triggering commands)

Extracts only your voice from the audio buffer before sending it to ASR, so the TV dialogue never reaches Whisper at all

Beyond the TV noise problem, this also means your voice assistant only responds to people you've explicitly enrolled. Guests, kids' friends, random visitors - none of them can control your smart home unless you add their voiceprint. It's basically access control for your voice pipeline.

The extraction part is what I'm most happy with. It splits the audio into speech regions using energy analysis, then runs each region through the speaker model to check if it sounds like you. Your voice scores 0.3-0.5 similarity against your voiceprint. TV speakers score -0.04 to 0.07. The separation is night and day - there's no overlap at all. So even if the TV is blasting while you give a command, only your voice gets forwarded to your STT service and you get a clean transcript.

How it works in practice:

You enroll by recording ~30 five-second samples of yourself speaking naturally

The proxy generates a voiceprint (averaged speaker embedding)

When a wake word triggers, audio streams to the proxy

It runs speaker verification while audio is still streaming in

Once verified, it waits for the satellite to finish streaming, extracts your voice segments, and forwards them to ASR

If nobody enrolled is speaking, it returns an empty transcript and HA ignores the trigger

Setup is a single Docker container that you point to your existing ASR service. You change HA's STT provider to point at the proxy instead of directly at your ASR service. No changes to wake word, satellites, or anything else.

Works with any Wyoming-compatible ASR backend - wyoming-onnx-asr, Faster Whisper, etc.

Performance on GPU is good - speaker verification takes about 15ms and the extraction adds another 20-30ms. The main latency comes from waiting for the satellite to finish streaming, which is 8-15 seconds when the TV is on (because the satellite's VAD can't find silence). In a quiet room it's much faster.

The project is at https://github.com/jxlarrea/wyoming-voice-match. GPU and CPU Docker images are available. MIT licensed.

Happy to answer questions about the approach or implementation.
