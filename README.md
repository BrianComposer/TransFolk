<h1 align="center">TransFolk</h1>

<h2 align="center">Transformer-Based Generation of Folk Melodies</h2>

<p align="center">
  <b>Symbolic AI for structured, style-aware folk melody generation</b>
</p>

<h2 align="center">
  <a href="https://transfolk.netlify.app/">Open the TransFolk Web App</a>
</h2>

<p align="center">
  <a href="https://doi.org/10.1007/978-3-032-27827-2_38"><b>Paper DOI: 10.1007/978-3-032-27827-2_38</b></a>
</p>

---

## Overview

**TransFolk** is a transformer-based system for generating monophonic folk melodies from a curated Iberian symbolic music corpus. The project combines symbolic music preprocessing, tokenisation, neural sequence modelling, generation, evaluation, API deployment, and an interactive web interface.

The current version of the project is deployed as a web application and exposes trained symbolic music models through a FastAPI backend. Users can inspect available models, configure generation parameters, condition generation from symbolic prompts, and render generated melodies as musical scores.

---

## Live Project

The interactive frontend is available here:

<h2>
  <a href="https://transfolk.netlify.app/">https://transfolk.netlify.app/</a>
</h2>

The deployed application provides:

* Model selection and model metadata inspection.
* Parameter-controlled melody generation.
* Prompt-conditioned generation from symbolic input.
* MusicXML-based score rendering.
* Playback-ready generated outputs.
* Integration with a deployed FastAPI backend.

---

## Product Preview

### Generated Melodies

Outputs are rendered as structured musical scores:

<p align="center">
  <img src="doc/images/melody1.jpg" width="85%">
</p>

<p align="center">
  <img src="doc/images/melody2.jpg" width="85%">
</p>

---

### Interactive Model Inspection

Each model exposes full metadata for reproducibility and analysis:

<p align="center">
  <img src="doc/images/modelinfo.jpg" width="60%">
</p>

---

### Generation Control Interface

Fine-grained control over sampling parameters:

<p align="center">
  <img src="doc/images/parameters.jpg" width="60%">
</p>

---

### Prompt Conditioning

The system allows generation conditioned on real symbolic input:

<p align="center">
  <img src="doc/images/melody_prompt.jpg" width="85%">
</p>

---

## Current Project Status

The latest version of TransFolk includes the following updates:

* Public frontend deployed on **Netlify**.
* FastAPI backend deployed externally for model inference.
* Runtime model discovery through a backend model registry.
* Support for released model weights and model metadata stored outside the frontend.
* Multiple released models can be exposed by the backend when their `.pt` weights and `.json` configuration files are available.
* Generated outputs are served through backend output endpoints and consumed by the frontend.
* The frontend now integrates model listing, model details, generation controls, score rendering, and playback-oriented output handling.
* The system is structured as a deployable research prototype rather than only a local training repository.

---

## Main Features

* Decoder-only Transformer architectures implemented with **PyTorch**.
* Symbolic melody generation from learned folk-style representations.
* Modular design prepared for future architectures, including encoder-decoder and hierarchical approaches.
* Multiple tokenisation strategies:
  * Event-based tokenisation.
  * Metric-aware tokenisation.
  * Pattern-aware tokenisation.
* Conditioning support for:
  * Time signatures.
  * Tonality or modal context.
  * Symbolic melodic prompts.
* Generation of symbolic music outputs suitable for:
  * MusicXML rendering.
  * MIDI-oriented playback workflows.
  * Web-based score visualisation.
* Evaluation framework including:
  * Token entropy.
  * Conditional entropy.
  * Modal stability.
  * Pattern retention metrics.
  * One-class style classification.
* Interactive frontend for non-technical use.
* API-based architecture for separating model inference from the web client.

---

## Released and Experimental Models

Released models are handled as deployable artifacts. A released model normally includes:

* `.pt` — trained PyTorch weights.
* `.json` — model configuration and experiment metadata.
* Vocabulary/tokenisation metadata required to reconstruct the model interface.

The backend model registry reads available released models and exposes them to the frontend. This makes it possible to add new trained models without changing the frontend code, provided that the backend can access the corresponding weights and metadata.

The current model family explores several decoder-only Transformer variants. The main differences are model scale and positional/attention mechanism.


### GPT-style `mick` models

The `mick` models use a standard decoder-only GPT-style architecture with learned positional embeddings and causal self-attention. They differ mainly in scale:

* `mick000` is the smallest model. It is useful for lightweight deployment, debugging, and low-memory inference.
* `mick003` increases the embedding size, number of heads, and number of layers, providing a medium-capacity configuration.
* `mick006` is a larger configuration with 768-dimensional embeddings, 12 attention heads, and 10 transformer layers.
* `mick010` is the largest listed GPT-style model, with 1024-dimensional embeddings, 16 heads, 12 layers, and a 4096-dimensional feed-forward block.

These models follow an autoregressive next-token prediction setup: each generated symbolic token is conditioned on the previously generated musical context.

### RoPE-based `robb` model

`robb006` keeps the same global scale as `mick006` but replaces learned positional embeddings with **Rotary Positional Embeddings**. RoPE injects positional information into the attention computation itself, which can improve the modelling of relative position and longer-range dependencies.

This makes `robb006` useful for comparing standard learned-position GPT behaviour against a rotary-position alternative at the same model scale.

### Relative-attention `john` model

`john006` also keeps the same global scale as `mick006`, but uses **relative attention** instead of standard causal self-attention with learned positional embeddings. This architecture models pairwise token distances more explicitly, which is especially relevant for symbolic music because rhythmic patterns, motivic repetitions, and phrase-level structures often depend on relative positions rather than absolute sequence indices.

This makes `john006` an experimental architecture for evaluating whether relative positional modelling improves folk melody generation.

---

## Research Context

TransFolk investigates the use of transformer-based symbolic generation for folk melody modelling. The project focuses on the relationship between musical representation and generative behaviour, with special attention to:

* Structural coherence.
* Stylistic consistency.
* Tokenisation strategy.
* Prompt conditioning.
* Entropy–structure trade-offs.
* Reproducibility of symbolic music generation experiments.

The system has been designed both as a research prototype and as a usable web-based demonstrator for generated folk melodies.

---

## Publication

Martínez-Rodríguez, B. (2026). *TransFolk: Transformer-Based Generation of Folk Melodies*. Proceedings of the 10th International Conference of Mathematics and Music. Springer.

Published: **2026-06-13**

DOI: <a href="https://doi.org/10.1007/978-3-032-27827-2_38">https://doi.org/10.1007/978-3-032-27827-2_38</a>

---

## Author

**Brian Martínez-Rodríguez**

GitHub: <a href="https://github.com/BrianComposer">https://github.com/BrianComposer</a>

Email: <a href="mailto:info@brianmartinez.music">info@brianmartinez.music</a>

Web: <a href="https://www.brianmartinez.music">www.brianmartinez.music</a>

---

## License

MIT License
