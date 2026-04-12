import json
import os
from datetime import datetime
import time
from logging import exception

import torch
from torch.utils.data import DataLoader

from transfolk_config import *
from .model.dataset import MusicDataset
from .model.music_transformer import MusicTransformer
from .training.trainer import train
from .generation.generator import generate_sequence, generate_sequence_from_prompt
from .utils.training_logger import save_loss_to_json
from transfolk_tokenization.tokenizer import process_musicxml_file
from transfolk_tokenization.decoder import tokens_to_music21_stream, tokens_to_music21_stream_with_ts
from transfolk.model.model_factory import ModelFactory
from transfolk.training.optimizer_factory import OptimizerFactory
from transfolk.training.loss_factory import LossFactory
from transfolk.training.scheduler_factory import SchedulerFactory
from .model.architecture_test import test_architecture
from pathlib import Path
from transfolk_config.settings import Settings

def run_train(
    model_cfg: Model,
    root_path: str = None,
    save_each_epoch = False):

    # -----------------------------
    # ⏱️ INICIO
    # -----------------------------
    start_time = datetime.now()
    start_perf = time.perf_counter()

    print(f"🎼 TRAINING MODE START: \nModel: {model_cfg.name}, Architecture: {model_cfg.architecture.name} ({model_cfg.architecture.type}, d_model:{model_cfg.architecture.d_model}, n_heads:{model_cfg.architecture.n_heads}, n_layers:{model_cfg.architecture.n_layers}), Runtime:  ({model_cfg.runtime_train.optimizer}, {model_cfg.runtime_train.scheduler}, {model_cfg.runtime_train.loss}, Warmup: {model_cfg.runtime_train.warmup_steps}, Epochs: {model_cfg.runtime_train.epochs}), Corpus: {model_cfg.experiment.corpus.name}, Tokenizer: {model_cfg.experiment.tokenizer.name}, Time Signature: {model_cfg.experiment.music_context.time_signature}, Tonality: {model_cfg.experiment.music_context.tonality},\nStart time: {start_time}")

    # load the resolver and the files
    settings = Settings(root_path)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    sequences_file = resolver.sequences_file(model_cfg.experiment)
    vocab_file = resolver.vocab_file(model_cfg.experiment)
    model_file = resolver.model_file(model_cfg)
    model_cfg_json = resolver.model_cfg_file(model_cfg)
    log_file = resolver.loss_log_file(model_cfg)
    # ASEGURAR DIRECTORIOS EXISTEN
    for path in [model_file, model_cfg_json, log_file]:
        path.parent.mkdir(parents=True, exist_ok=True)

    #Load the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save model_cfg to json
    model_cfg.save_json(str(model_cfg_json))
    # Load vocab and sequences
    with open(sequences_file, "r") as f:
        sequences = json.load(f)
    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    dataset = MusicDataset(sequences, max_seq_len=model_cfg.architecture.max_seq_len, pad_token_id=0)
    dataloader = DataLoader(dataset, batch_size=model_cfg.runtime_train.batch_size, shuffle=True)

    #####################################################################################################
    # Construcción del modelo desde model_cfg.architecture
    #####################################################################################################
    model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)

    #####################################################################################################
    # Optimizador y loss
    #####################################################################################################
    optimizer = OptimizerFactory.build(model_cfg.runtime_train, model)
    criterion = LossFactory.build(model_cfg.runtime_train)


    #####################################################################################################
    # Scheduler PARA EL FUTURO
    #####################################################################################################
    # total_steps = len(dataloader) * model_cfg.runtime_train.epochs
    # scheduler = SchedulerFactory.build(
    #     model_cfg.runtime_train,
    #     optimizer,
    #     total_steps
    # )

    #####################################################################################################
    # Training loop
    #####################################################################################################
    for epoch in range(model_cfg.runtime_train.epochs):
        loss = train(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=None,
            pad_token_id=0,
            grad_clip=1.0
        )
        print(f"\n✅ Epoch {epoch + 1} completed — Average Loss: {loss:.4f}\n")
        save_loss_to_json(log_file, epoch + 1, loss)
        # Guardado de seguridad en cada epoch
        if save_each_epoch:
            torch.save(model.state_dict(), resolver.model_file_epoch(model_cfg, epoch))
            end_time = datetime.now()
            end_perf = time.perf_counter()
            total_time = end_perf - start_perf
            model_cfg.train_start_time = start_time.isoformat()
            model_cfg.train_end_time = end_time.isoformat()
            model_cfg.train_total_time = total_time
            model_cfg.train_date = start_time.date().isoformat()
            model_cfg.vocab_file = vocab_file.name  # muy importante guardar el vocal_file en el model
            model_cfg.save_json(str(resolver.model_epoch_cfg_file(model_cfg, epoch)))

    #####################################################################################################
    # Guardado
    #####################################################################################################
    torch.save(model.state_dict(), model_file)

    # -----------------------------
    # ⏱️ FIN
    # -----------------------------
    end_time = datetime.now()
    end_perf = time.perf_counter()
    total_time = end_perf - start_perf

    # -----------------------------
    # 💾 GUARDAR EN EL OBJETO
    # -----------------------------
    model_cfg.train_start_time = start_time.isoformat()
    model_cfg.train_end_time = end_time.isoformat()
    model_cfg.train_total_time = total_time
    model_cfg.train_date = start_time.date().isoformat()
    model_cfg.vocab_file=vocab_file.name #muy importante guardar el vocal_file en el model
    model_cfg.save_json(str(model_cfg_json))

    print(f"✅ MODEL TRAINING FINISHED at {end_time.isoformat()}")
    return model_cfg


def run_test_architecture(model_cfg: Model):

    # -----------------------------
    # ⏱️ INICIO
    # -----------------------------
    start_time = datetime.now()
    start_perf = time.perf_counter()

    print(
        f"🧪 ARCHITECTURE TEST START:\n"
        f"Model: {model_cfg.name}, "
        f"Architecture: {model_cfg.architecture.name} ({model_cfg.architecture.type}), "
        f"Runtime: ({model_cfg.runtime_train.optimizer}, {model_cfg.runtime_train.scheduler}, {model_cfg.runtime_train.loss}),\n"
        f"Corpus: {model_cfg.experiment.corpus.name}, "
        f"Tokenizer: {model_cfg.experiment.tokenizer.name}\n"
        f"Start time: {start_time}\n"
    )

    # -----------------------------
    # PATHS / RESOLVER
    # -----------------------------
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)

    sequences_file = resolver.sequences_file(model_cfg.experiment)
    vocab_file = resolver.vocab_file(model_cfg.experiment)

    # -----------------------------
    # DEVICE
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    with open(sequences_file, "r") as f:
        sequences = json.load(f)

    with open(vocab_file, "r") as f:
        vocab = json.load(f)

    dataset = MusicDataset(
        sequences,
        max_seq_len=model_cfg.architecture.max_seq_len,
        pad_token_id=0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=model_cfg.runtime_train.batch_size,
        shuffle=True
    )

    # -----------------------------
    # BUILD MODEL
    # -----------------------------
    model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)

    # -----------------------------
    # OPTIMIZER / LOSS
    # -----------------------------
    optimizer = OptimizerFactory.build(model_cfg.runtime_train, model)
    criterion = LossFactory.build(model_cfg.runtime_train)


    results = test_architecture(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        vocab_size=len(vocab),
        max_batches_overfit=10,
        pad_token_id=0,
        verbose=True,
    )

    # -----------------------------
    # ⏱️ FIN
    # -----------------------------
    end_time = datetime.now()
    end_perf = time.perf_counter()
    total_time = end_perf - start_perf

    print(f"\n🧪 TEST FINISHED at {end_time.isoformat()}")
    print(f"⏱️ Total test time: {total_time:.2f} seconds\n")

    # -----------------------------
    # 📊 VALIDACIÓN AUTOMÁTICA (CRÍTICA)
    # -----------------------------
    padding_ok = results.get("padding_diff", 1.0) < 1e-3
    causal_ok = results.get("causal_diff", 1.0) < 1e-3
    overfit_ok = results.get("overfit_final_loss", 999) < results.get("overfit_initial_loss", 0)

    print("===== VALIDATION =====")
    print(f"Padding mask OK: {padding_ok}")
    print(f"Causal mask OK:  {causal_ok}")
    print(f"Overfit OK:      {overfit_ok}")
    print("======================\n")

    # -----------------------------
    # ❌ HARD FAIL (opcional pero recomendable)
    # -----------------------------
    if not padding_ok:
        raise RuntimeError("❌ Padding mask incorrecta")

    if not causal_ok:
        raise RuntimeError("❌ Causal mask incorrecta")

    if not overfit_ok:
        raise RuntimeError("❌ Modelo no aprende (overfit test fallido)")

    return results




def run_generate(
        model_cfg: Model,
        runtime: RuntimeGenerate):

    print(f"🎶 GENERATION MODE: {model_cfg.name}, {model_cfg.experiment.corpus.name}, {model_cfg.experiment.tokenizer.name}, {model_cfg.experiment.music_context.time_signature}, {model_cfg.experiment.music_context.tonality}, {runtime.temperature}")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cfg.experiment)
    model_file = resolver.model_file(model_cfg)
    prod_dir = resolver.production_dir(model_cfg, runtime)

    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    #####################################################################################################
    # Construcción del modelo desde model_cfg.architecture
    #####################################################################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))

    for i in range(runtime.num_productions):
        try:
            start_token_id = vocab["START"]
            generated_tokens = generate_sequence(
                model=model,
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
            filepath = resolver.generated_new_file(model_cfg, runtime)

            # Guardar el archivo
            music_stream = tokens_to_music21_stream(generated_tokens,
                                                    model_cfg.experiment.allowed_durations.durations
                                                    )
            music_stream.write("musicxml", fp=filepath)
            print(f"✅ Archivo generado: {filepath}")
        except Exception as e:
            print(e)


def run_generate_from_musicxml_prompt(
        model_cfg: Model,
        runtime: RuntimeGenerate,
        file_xml_prompt_path:str):
    # 1. Leer el archivo XML y tokenizarlo
    errors = {}
    prompt_tokens = []
    #tokenizacion del promp xml
    prompt_tokens = process_musicxml_file(file_xml_prompt_path,
                                          model_cfg.experiment.tokenizer.name,
                                          model_cfg.experiment.music_context.time_signature,
                                          model_cfg.experiment.music_context.tonality,
                                          model_cfg.experiment.allowed_durations.durations,
                                          errors)

    if not prompt_tokens:
        raise RuntimeError("Prompt tokenization produced no tokens.")

    # load the resolver and the files
    settings = Settings()
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cfg.experiment)
    model_file = resolver.model_file(model_cfg)
    prod_dir = resolver.production_dir(model_cfg, runtime)

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
    trans_model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)
    # trans_model = MusicTransformer(vocab_size=len(vocab),
    #                                d_model=model_cfg.architecture.d_model,
    #                                nhead=model_cfg.architecture.n_heads,
    #                                num_layers=model_cfg.architecture.n_layers,
    #                                dim_feedforward=model_cfg.architecture.d_ff,
    #                                dropout=model_cfg.architecture.dropout,
    #                                max_seq_len=2 * model_cfg.architecture.max_seq_len).to(device)
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
        filepath = resolver.generated_new_file(model_cfg, runtime)

        # Guardar el archivo
        music_stream = tokens_to_music21_stream(generated_tokens,
                                                model_cfg.experiment.allowed_durations.durations
                                                )
        music_stream.write("musicxml", fp=filepath)
        print(f"✅ Archivo generado: {filepath}")

    except Exception as e:
        print(e)


def run_generate_from_TS_tonality(
        model_cfg: Model,
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
    vocab_file = resolver.vocab_file(model_cfg.experiment)
    model_file = resolver.model_file(model_cfg)
    prod_dir = resolver.production_dir(model_cfg, runtime)

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
    trans_model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)
    # trans_model = MusicTransformer(vocab_size=len(vocab),
    #                                d_model=model_cfg.architecture.d_model,
    #                                nhead=model_cfg.architecture.n_heads,
    #                                num_layers=model_cfg.architecture.n_layers,
    #                                dim_feedforward=model_cfg.architecture.d_ff,
    #                                dropout=model_cfg.architecture.dropout,
    #                                max_seq_len=2 * model_cfg.architecture.max_seq_len).to(device)
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
        filepath = resolver.generated_new_file(model_cfg, runtime)

        # Guardar el archivo
        music_stream = tokens_to_music21_stream(generated_tokens,
                                                model_cfg.experiment.allowed_durations.durations
                                                )
        music_stream.write("musicxml", fp=filepath)
        print(f"✅ Archivo generado: {filepath}")

    except Exception as e:
        print(e)




def run_generate_for_curves_style(
        model_cfg: Model,
        runtime: RuntimeGenerate,
        temperatures,
        dict_norm,
        pieces_per_temp,
        base_root,
        verbose=False):


    # 1. load the resolver and the files
    settings = Settings(base_root)
    paths = ProjectPaths(settings.root)
    resolver = PathResolver(paths)
    vocab_file = resolver.vocab_file(model_cfg.experiment)
    model_file = resolver.model_file(model_cfg)
    prod_dir = resolver.production_dir(model_cfg, runtime)

    # 2. Cargar vocabulario
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}


    # 3. Cargamos el modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans_model = ModelFactory.build(
        architecture=model_cfg.architecture,
        vocab_size=len(vocab)
    ).to(device)
    # trans_model = MusicTransformer(vocab_size=len(vocab),
    #                                d_model=model_cfg.architecture.d_model,
    #                                nhead=model_cfg.architecture.n_heads,
    #                                num_layers=model_cfg.architecture.n_layers,
    #                                dim_feedforward=model_cfg.architecture.d_ff,
    #                                dropout=model_cfg.architecture.dropout,
    #                                max_seq_len=model_cfg.architecture.max_seq_len).to(device)
    trans_model.load_state_dict(torch.load(model_file, map_location=device))

    for temperature in temperatures:
        runtime.temperature = temperature
        for (time_signature, tonality), value in dict_norm.items():
            num = int(value * pieces_per_temp)
            if num < 1: continue
            print(f"Generating {time_signature} {tonality}: {num} pieces")
            for i in range(num):
                # 4. Generar tokens iniciales
                prompt_tokens = []
                if not time_signature or not tonality:
                    raise exception("No Time Signature or Tonality was given.")
                if tonality.lower() not in ["minor", "major"]:
                    raise exception("Tonality not allowed.")

                # prompt_tokens.append(f"START")
                prompt_tokens.append(f"TS_{time_signature}")
                prompt_tokens.append(f"MODE_{tonality.lower()}")
                prompt_tokens.append(f"BAR")

                # 5. Convertir tokens del prompt a IDs
                try:
                    prompt_token_ids = [vocab[t] for t in prompt_tokens]
                    prompt_token_ids = [vocab["START"]] + prompt_token_ids
                except KeyError as e:
                    raise RuntimeError(f"Token not in vocabulary: {e}")


                try:
                    # 6. Generar obras
                    start_token_id = vocab["START"]
                    generated_tokens = generate_sequence_from_prompt(
                        model=trans_model,
                        start_token_id_list=prompt_token_ids,
                        max_len=runtime.max_len-10,
                        vocab=vocab,
                        inv_vocab=inv_vocab,
                        device=device,
                        temperature=temperature,
                        top_k=runtime.top_k,
                        top_p=runtime.top_p,
                        penalty=runtime.repetition_penalty
                    )


                    # Guardar el archivo
                    if verbose:
                        print("🎼 Tokens generados:", generated_tokens)

                    # Crear carpeta "productions" si no existe
                    os.makedirs(prod_dir, exist_ok=True)
                    filepath = resolver.generated_new_file(model_cfg, runtime)
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)

                    # Guardar el archivo
                    music_stream = tokens_to_music21_stream(generated_tokens,
                                                            model_cfg.experiment.allowed_durations.durations
                                                            )
                    music_stream.write("musicxml", fp=filepath)
                    print(f"✅ Archivo generado {i}/{num}: {filepath}")

                except Exception as e:
                    print(e)

