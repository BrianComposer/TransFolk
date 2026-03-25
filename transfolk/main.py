import json
import os
from datetime import datetime
import time
from logging import exception

import torch
from torch.utils.data import DataLoader

from transfolk_config import *
from .model.dataset import MusicDataset
from .model.transformer import MusicTransformer
from .training.trainer import train
from .generation.generator import generate_sequence, generate_sequence_from_prompt
from .utils.training_logger import save_loss_to_json
from transfolk_tokenization.tokenizer import process_musicxml_file
from transfolk_tokenization.decoder import tokens_to_music21_stream, tokens_to_music21_stream_with_ts


def run_train(
    model: Model):

    # -----------------------------
    # ⏱️ INICIO
    # -----------------------------
    start_time = datetime.now()
    start_perf = time.perf_counter()

    print(f"🎼 TRAINING MODE START: {model.experiment.corpus.name}, {model.experiment.tokenizer.name}, {model.experiment.music_context.time_signature}, {model.experiment.music_context.tonality} at {start_time}")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    sequences_file = resolver.sequences_file(model.architecture, model.experiment)
    vocab_file = resolver.vocab_file(model.architecture, model.experiment)

    #Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and log files
    model_file = resolver.model_file(model)
    model_json = resolver.model_cfg_file(model)
    log_file = resolver.loss_log_file(model)
    model.save_json(str(model_json))
    # Load vocab and sequences
    with open(sequences_file, "r") as f:
        sequences = json.load(f)
    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    dataset = MusicDataset(sequences, max_seq_len=model.architecture.max_seq_len, pad_token_id=0)
    dataloader = DataLoader(dataset, batch_size=model.runtime_train.batch_size, shuffle=True)

    #####################################################################################################
    #TO DO: en funcion del architecture, se instanciara distintos tipos de modelos de transformer
    # if model.architecture.type=="...."
    #####################################################################################################

    trans_model = MusicTransformer(vocab_size=len(vocab),
                             d_model=model.architecture.d_model,
                             nhead=model.architecture.n_heads,
                             num_layers=model.architecture.n_layers,
                             dim_feedforward=model.architecture.d_ff,
                             dropout = model.architecture.dropout,
                             max_seq_len=model.architecture.max_seq_len).to(device)
    optimizer = torch.optim.Adam(trans_model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(model.runtime_train.epochs):
        loss = train(trans_model, dataloader, optimizer, criterion, len(vocab), device)
        print(f"✅ Epoch {epoch+1} completed — Average Loss: {loss:.4f}")
        save_loss_to_json(log_file, epoch + 1, loss)
    torch.save(trans_model.state_dict(), model_file)

    # -----------------------------
    # ⏱️ FIN
    # -----------------------------
    end_time = datetime.now()
    end_perf = time.perf_counter()
    total_time = end_perf - start_perf

    # -----------------------------
    # 💾 GUARDAR EN EL OBJETO
    # -----------------------------
    model.train_start_time = start_time.isoformat()
    model.train_end_time = end_time.isoformat()
    model.train_total_time = total_time
    model.train_date = start_time.date().isoformat()
    model.vocab_file=vocab_file.name #muy importante guardar el vocal_file en el model
    model.save_json(str(model_json))

    print(f"✅ MODEL TRAINING FINISHED at {end_time.isoformat()}")
    return model


def run_generate(
        model_cng: Model,
        runtime: RuntimeGenerate):

    print(f"🎶 GENERATION MODE: {model_cng.name}, {model_cng.experiment.corpus.name}, {model_cng.experiment.tokenizer.name}, {model_cng.experiment.music_context.time_signature}, {model_cng.experiment.music_context.tonality}, {runtime.temperature}")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cng.architecture, model_cng.experiment)
    model_file = resolver.model_file(model_cng)
    prod_dir = resolver.production_dir(model_cng, runtime)


    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans_model = MusicTransformer(vocab_size=len(vocab),
                             d_model=model_cng.architecture.d_model,
                             nhead=model_cng.architecture.n_heads,
                             num_layers=model_cng.architecture.n_layers,
                             dim_feedforward=model_cng.architecture.d_ff,
                             dropout=model_cng.architecture.dropout,
                             max_seq_len=model_cng.architecture.max_seq_len).to(device)
    trans_model.load_state_dict(torch.load(model_file, map_location=device))

    for i in range(runtime.num_productions):
        try:
            start_token_id = vocab["START"]
            generated_tokens = generate_sequence(
                model=trans_model,
                start_token_id=start_token_id,
                max_len=runtime.max_len,
                vocab=vocab,
                inv_vocab=inv_vocab,
                device=device,
                temperature=runtime.temperature,
                top_k=runtime.top_k,
                top_p=runtime.top_p,
                penalty=runtime.repetition_penalty
            )

            print("🎼 Tokens generados:", generated_tokens)

            # Crear carpeta "productions" si no existe
            os.makedirs(prod_dir, exist_ok=True)
            filepath = resolver.generated_new_file(model_cng, runtime)

            # Guardar el archivo
            music_stream = tokens_to_music21_stream(generated_tokens,
                                                    model_cng.experiment.allowed_durations.durations
                                                    )
            music_stream.write("musicxml", fp=filepath)
            print(f"✅ Archivo generado: {filepath}")
        except Exception as e:
            print(e)


def run_generate_from_musicxml_prompt(
        model_cng: Model,
        runtime: RuntimeGenerate,
        file_xml_prompt_path:str):
    # 1. Leer el archivo XML y tokenizarlo
    errors = {}
    prompt_tokens = []
    #tokenizacion del promp xml
    prompt_tokens = process_musicxml_file(file_xml_prompt_path,
                                          model_cng.experiment.tokenizer.name,
                                          model_cng.experiment.music_context.time_signature,
                                          model_cng.experiment.music_context.tonality,
                                          model_cng.experiment.allowed_durations.durations,
                                          errors)

    if not prompt_tokens:
        raise RuntimeError("Prompt tokenization produced no tokens.")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cng.architecture, model_cng.experiment)
    model_file = resolver.model_file(model_cng)
    prod_dir = resolver.production_dir(model_cng, runtime)

    # 2. Cargar vocabulario
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}


    # 3. Convertir tokens del prompt a IDs
    try:
        prompt_token_ids = [vocab[t] for t in prompt_tokens]
        prompt_token_ids = [vocab["START"]] + prompt_token_ids
    except KeyError as e:
        raise RuntimeError(f"Token not in vocabulary: {e}")

    # 4. Cargamos el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans_model = MusicTransformer(vocab_size=len(vocab),
                                   d_model=model_cng.architecture.d_model,
                                   nhead=model_cng.architecture.n_heads,
                                   num_layers=model_cng.architecture.n_layers,
                                   dim_feedforward=model_cng.architecture.d_ff,
                                   dropout=model_cng.architecture.dropout,
                                   max_seq_len=2*model_cng.architecture.max_seq_len).to(device)
    trans_model.load_state_dict(torch.load(model_file, map_location=device))


    try:
        start_token_id = vocab["START"]
        generated_tokens = generate_sequence_from_prompt(
            model=trans_model,
            start_token_id_list=prompt_token_ids,
            max_len=runtime.max_len,
            vocab=vocab,
            inv_vocab=inv_vocab,
            device=device,
            temperature=runtime.temperature,
            top_k=runtime.top_k,
            top_p=runtime.top_p,
            penalty=runtime.repetition_penalty
        )


        # Guardar el archivo
        print("🎼 Tokens generados:", generated_tokens)

        # Crear carpeta "productions" si no existe
        os.makedirs(prod_dir, exist_ok=True)
        filepath = resolver.generated_new_file(model_cng, runtime)

        # Guardar el archivo
        music_stream = tokens_to_music21_stream(generated_tokens,
                                                model_cng.experiment.allowed_durations.durations
                                                )
        music_stream.write("musicxml", fp=filepath)
        print(f"✅ Archivo generado: {filepath}")

    except Exception as e:
        print(e)


def run_generate_from_TS_tonality(
        model_cng: Model,
        runtime: RuntimeGenerate,
        time_signature="2/4",
        tonality="major"):

    # 1. Generar tokens iniciales
    prompt_tokens = []
    if not time_signature or not tonality:
        raise exception("No Time Signature or Tonality was given.")
    if tonality.lower() not in ["minor", "major"]:
        raise exception("Tonality not allowed.")

    #prompt_tokens.append(f"START")
    prompt_tokens.append(f"TS_{time_signature}")
    prompt_tokens.append(f"MODE_{tonality.lower()}")
    prompt_tokens.append(f"BAR")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cng.architecture, model_cng.experiment)
    model_file = resolver.model_file(model_cng)
    prod_dir = resolver.production_dir(model_cng, runtime)

    # 2. Cargar vocabulario
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}


    # 3. Convertir tokens del prompt a IDs
    try:
        prompt_token_ids = [vocab[t] for t in prompt_tokens]
        prompt_token_ids = [vocab["START"]] + prompt_token_ids
    except KeyError as e:
        raise RuntimeError(f"Token not in vocabulary: {e}")

    # 4. Cargamos el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans_model = MusicTransformer(vocab_size=len(vocab),
                                   d_model=model_cng.architecture.d_model,
                                   nhead=model_cng.architecture.n_heads,
                                   num_layers=model_cng.architecture.n_layers,
                                   dim_feedforward=model_cng.architecture.d_ff,
                                   dropout=model_cng.architecture.dropout,
                                   max_seq_len=2*model_cng.architecture.max_seq_len).to(device)
    trans_model.load_state_dict(torch.load(model_file, map_location=device))


    try:
        start_token_id = vocab["START"]
        generated_tokens = generate_sequence_from_prompt(
            model=trans_model,
            start_token_id_list=prompt_token_ids,
            max_len=runtime.max_len,
            vocab=vocab,
            inv_vocab=inv_vocab,
            device=device,
            temperature=runtime.temperature,
            top_k=runtime.top_k,
            top_p=runtime.top_p,
            penalty=runtime.repetition_penalty
        )


        # Guardar el archivo
        print("🎼 Tokens generados:", generated_tokens)

        # Crear carpeta "productions" si no existe
        os.makedirs(prod_dir, exist_ok=True)
        filepath = resolver.generated_new_file(model_cng, runtime)

        # Guardar el archivo
        music_stream = tokens_to_music21_stream(generated_tokens,
                                                model_cng.experiment.allowed_durations.durations
                                                )
        music_stream.write("musicxml", fp=filepath)
        print(f"✅ Archivo generado: {filepath}")

    except Exception as e:
        print(e)


