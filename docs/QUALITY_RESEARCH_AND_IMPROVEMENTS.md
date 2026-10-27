# VoxForge: quality-led narration and picture improvements

## What I reviewed

I inspected the current flagship's actual script, Stage A-D report, Stage E scene plan/audit, narration TTS settings, mastering chain, and x264 render settings. The current master is 1080p24, H.264, 48 kHz AAC, measured at -15.99 LUFS / -1.25 dBTP; its Stage E audit reports 67/67 validated scenes. That is strong technical compliance, but a compliance score cannot establish that prose is engaging or that the synthetic voice sounds human. The script sample also exposes the main editorial risks: fact-dense clauses, oversized sentences, inflated abstractions, and a misleading cross-act narration line (the “Ming Frontier Network” label appears while discussing the Great Wall/Xiongnu frontier). These are qualitative issues a pixel-family classifier cannot catch.

## Research synthesis

- Documentary prose must be written for listening, not silent reading: conversational wording, active constructions, short comprehensible sentences, read-aloud review, and deliberate room for pictures/silence. New Doc Editing, “Ten Tips for Writing Documentary Narration” (accessed 2026-10-27): https://newdocediting.com/ten-tips-for-writing-documentary-narration/.
- A narration-led edit should let the voice set the story rhythm; the picture can then be cut around meaningful clauses and events. This is an editorial practice, not a claim that every cut must align to every word. Research overview: Journalism University, “Mastering Voice-over Narration for Documentary Impact”: https://journalism.university/electronic-media/voice-over-narration-documentary-tips/.
- Loudness is delivery-context dependent. EBU R 128's broadcast programme target is -23 LUFS, while this repository's -16 LUFS profile is intended for web/spoken-word playback. They must not be presented as the same standard. Official EBU R 128 publication: https://tech.ebu.ch/publications/r128.

## Changes in this pass

1. Strengthened local-LLM writer instructions: verified facts only, no invented figures/causes/quotes, 1–3 sentences per turn, at most 45 words, mostly 8–20 words per sentence, active/concrete language, jargon explained, less list-cramming and generic praise, and no narration that merely repeats the obvious picture.
2. Raised ordinary 1080p render quality from CRF 24.8 to CRF 21, retaining a slightly more storage-efficient CRF 23 for marathon CPU renders. This prioritizes legible small typography, fine map lines, gradients, and subtle motion in the code-rendered graphics.
3. Replaced the compact-documentary opener's generic “forty-five centuries of human ambition / outer boundary” exposition with a concise question a narrator can deliver naturally.
4. Corrected a specific inspected visual error: the Great Wall simulation previously titled the Qin/Meng Tian frontier a “Ming Frontier Network” and claimed “700 km/h” signal propagation without grounding. The new graphic says “Great Wall Beacon Network: Signals Across the Northern Frontier,” labels itself a historical concept model, and drops the unsupported speed and network-span claims. The scene caption is now dynasty-neutral.
5. Fixed quote-card retrieval after inspecting the rendered frame: it previously chose an Artemis quotation from the chapter title for a generic Great Pyramid question. Quote lookup now keys off the spoken turn, and an unmatched turn is displayed as a VoxForge narration excerpt rather than falsely attributed archival text. The quote-card scene plan and audit family now permit its real monument background. Added a regression test for unrelated-quote leakage.
6. Preserved the existing TTS normalization, 48 kHz mastering, measured -16 LUFS web profile, and -1 dBTP ceiling; no unsupported EQ/compression changes were made without listening evidence.

## Limits and next validation

The local LLM prompt is only used when a local Ollama/GGUF model is available. The deterministic corpus showrunner is the default and must be improved separately; prompt changes alone do not improve its lines. Automatic script and frame-family audits are not substitutes for listening to the complete narration or watching the whole film. The new master must be rendered and then reviewed for factual wording, speech rhythm, voice quality, visual relevance, and mix balance. Report findings rather than equating a passing numeric gate with artistic quality.
