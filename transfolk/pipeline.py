import json
import torch
from transfolk_config.entities.model import Model as Model_cfg
from transfolk_tokenization.tokenizer import (
    process_musicxml_file
)

from .model.transformer import MusicTransformer
from .generation.generator import (
    generate_sequence_from_prompt
)
from transfolk_tokenization.decoder import tokens_to_music21_stream


class TransFolkPipeline:

    def __init__(
        self,
        model_file,
        vocab_file,
        model_config: Model_cfg,
        device="cpu"
    ):
        if vocab_file is None:
            raise RuntimeError("vocab_file cannot be None")

        if model_config is None:
            raise RuntimeError("model_config cannot be None")
        self.model_cfg = model_config

        with open(vocab_file, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.device = torch.device(device)

        # Acepta tanto un contenedor serializable como una colección simple
        # if isinstance(allowed_durations, (list, set, tuple)):
        #     self.allowed_durations = set(allowed_durations)
        # else:
        #     self.allowed_durations = set(allowed_durations.values)
        #self.allowed_durations = model_config.experiment.allowed_durations.durations
        exp = model_config.experiment
        if exp is None:
            raise RuntimeError("model_config.experiment cannot be None")

        if exp.tokenizer is None or exp.music_context is None:
            raise RuntimeError("Model experiment is incomplete")

        self.algorithm = exp.tokenizer.name
        self.time_signature = exp.music_context.time_signature
        self.mode = exp.music_context.tonality

        if not all([self.algorithm, self.time_signature, self.mode]):
            raise RuntimeError("Model experiment metadata is incomplete")

        self.model = MusicTransformer(
            vocab_size=len(self.vocab),
            d_model=model_config.architecture.d_model,
            nhead=model_config.architecture.n_heads,
            num_layers=model_config.architecture.n_layers,
            dim_feedforward=model_config.architecture.d_ff,
            dropout=model_config.architecture.dropout,
            max_seq_len=model_config.architecture.max_seq_len
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device)
        )

        self.model.eval()

    def generate(
            self,
          temperature: float = 1.2,
          max_len: int = 256,
          penalty: float = 1.0,
          topK: int = 25,
          topP: float = 1.0
          ):
            prompt_tokens = []
            # 1. Generar tokens iniciales
            for t in ["TS_2/4", "MODE_major", "BAR"]:
                if t in self.vocab:
                    prompt_tokens.append(t)
                else:
                    print(f"Token not in vocabulary: {t}")


            # 3. Convertir tokens del prompt a IDs
            try:
                prompt_ids = [self.vocab[t] for t in prompt_tokens]
                prompt_ids = [self.vocab["START"]] + prompt_ids
            except KeyError as e:
                raise RuntimeError(f"Token not in vocabulary: {e}, {self.vocab.items}")

            # 4. Cargamos el modelo
            with torch.no_grad():
                generated_tokens = generate_sequence_from_prompt(
                    model=self.model,
                    start_token_id_list=prompt_ids,
                    max_len=max_len,
                    vocab=self.vocab,
                    inv_vocab=self.inv_vocab,
                    device=self.device,
                    temperature=temperature,
                    penalty=penalty,
                    top_k=topK,
                    top_p=topP
                )

                return tokens_to_music21_stream(generated_tokens, self.model_cfg.experiment.allowed_durations.durations)

    def generate_from_xml(
        self,
        xml_path: str,
        temperature: float = 1.2,
        max_len: int = 256,
        penalty: float = 1.1,
        topK: int = 25,
        topP: float = 0.9
    ):
        # 1. Leer el archivo XML y tokenizarlo
        errors = {}
        prompt_tokens = []
        # tokenizacion del promp xml
        prompt_tokens = process_musicxml_file(xml_path,
                                              self.model_cfg.experiment.tokenizer.name,
                                              self.model_cfg.experiment.music_context.time_signature,
                                              self.model_cfg.experiment.music_context.tonality,
                                              self.model_cfg.experiment.allowed_durations.durations,
                                              errors)

        if not prompt_tokens:
            raise RuntimeError("Prompt tokenization produced no tokens.")


        if not prompt_tokens:
            raise RuntimeError("Empty prompt")

        try:
            prompt_ids = [self.vocab["START"]] + [
                self.vocab[t] for t in prompt_tokens
            ]
        except KeyError as e:
            raise RuntimeError(f"Token not in vocabulary: {e}") from e

        with torch.no_grad():
            generated_tokens = generate_sequence_from_prompt(
                model=self.model,
                start_token_id_list=prompt_ids,
                max_len=max_len,
                vocab=self.vocab,
                inv_vocab=self.inv_vocab,
                device=self.device,
                temperature=temperature,
                penalty=penalty,
                top_k=topK,
                top_p=topP
            )

        return tokens_to_music21_stream(generated_tokens, self.model_cfg.experiment.allowed_durations.durations)

    def generate_from_TS_tonality(self,
                                  time_signature: str="2/4",
                                  tonality: str = "major",
                                  temperature: float = 1.2,
                                  max_len: int = 256,
                                  penalty: float = 1.0,
                                  topK: int = 25,
                                  topP: float = 1.0
                                  ):
        # 1. Generar tokens iniciales
        prompt_tokens = []
        prompt_tokens.append(f"TS_{time_signature}")
        prompt_tokens.append(f"MODE_{tonality.lower()}")
        prompt_tokens.append(f"BAR")

        # 3. Convertir tokens del prompt a IDs
        try:
            prompt_ids = [self.vocab[t] for t in prompt_tokens]
            prompt_ids = [self.vocab["START"]] + prompt_ids
        except KeyError as e:
            raise RuntimeError(f"Token not in vocabulary: {e}")

        # 4. Cargamos el modelo
        with torch.no_grad():
            generated_tokens = generate_sequence_from_prompt(
                model=self.model,
                start_token_id_list=prompt_ids,
                max_len=max_len,
                vocab=self.vocab,
                inv_vocab=self.inv_vocab,
                device=self.device,
                temperature=temperature,
                penalty=penalty,
                top_k=topK,
                top_p=topP
            )

            return tokens_to_music21_stream(generated_tokens, self.model_cfg.experiment.allowed_durations.durations)