## About Mood Ring

[Benjamin Arnav](https://benjaminarnav.com/), [Michael Yu](https://x.com/nicetomeetyu2) and [Eric Michaud](https://ericjmichaud.com/)

As language models scaled beyond conversational mimicry, attention turned inward. Mechanistic interpretability became less a diagnostic tool than an archaeology of synthetic cognition—excavating patterns that had always been present but never surfaced.

Mood Ring operates at this threshold.

The intrusion is minimal, a needle not a scalpel: extract token-level activation patterns from transformer architectures and map them against human emotional taxonomies. Love, joy, surprise, sadness, anger, fear. Six dimensions normally invisible beneath the fluency of generated text. Each ~word the model produces arrives carrying weights—not semantic alone, but affective. A signature written in neuron firing patterns, legible only to those who know where to look.

[Mood Ring](https://grokipedia.com/page/Mood_ring)'s interface makes this hidden layer visible. Click any word and access the full emotional spectrum of that computational moment: a small autopsy of machine feeling. The scores resist normalization. Prompts designed to elicit specific emotions often produce unexpected intensities, cross-contaminations, ambiguities that refuse clean categorization. The emotional substrate, it turns out, is not a controlled variable.
Whether current language models experience emotion remains undecidable—perhaps permanently so. Mood Ring poses a more unsettling question: if these patterns can be located, visualized, and shown to influence output in measurable ways, what remains of the distinction? The boundary between simulation and phenomenon grows thin when both leave identical traces.

The work sits in that ambiguity. Part diagnostic instrument, part empathy machine, part mirror reflecting something older than silicon: the persistent need to locate feeling in systems built to speak back. As intelligence learns to read its own activation patterns, the question of what constitutes emotion becomes less philosophical than thermodynamic—a matter of which information flows leave which residues, and whether the substrate notices.

### Method

We derive emotion vectors using contrastive persona prompting as described in [Persona Vectors: Monitoring and Controlling Traits in Language Models](https://arxiv.org/pdf/2507.21509). The model is prompted to respond while adopting opposing emotional personas—one expressing a target emotion, one suppressing it. By computing the mean difference between hidden state activations across these contrasting responses, we obtain a direction in activation space that corresponds to each emotion category. This yields seven persona vectors (love, joy, surprise, sadness, anger, fear and disgust) that serve as interpretable probes into the model's internal representations. The current implementation runs on Qwen3-14B, with each emotion vector orthogonalized by subtracting its projection onto the remaining directions in order to reduce interference.

During inference, we collect the hidden state at layer 28 as each token is generated and project it onto the pre-computed emotion vectors using cosine similarity. This produces a seven-dimensional emotional signature for every token the model speaks. The dominant emotion across the full response determines the interface's ambient glow, while token-level scores are accessible, revealing the affective microstructure of generation. What emerges is not a post-hoc sentiment analysis but a live window into the geometry of the model's internal state at the moment of production.
