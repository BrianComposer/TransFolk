from transfolk.api.config import OUTPUT_DIR
from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi import File
import uuid

from transfolk.core.model_registry import MODEL_REGISTRY
from transfolk.core.pipeline_cache import get_pipeline
import inspect

router = APIRouter()

@router.post("/generate")
async def generate(
    model: str = Form(...),
    temperature: float = Form(1.2),
    max_len: int = Form(256),
    penalty: float = Form(1.1),
    topK: int = Form(25),
    topP: float = Form(0.9)
):
    if model not in MODEL_REGISTRY:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unknown model: {model}"}
        )

    input_path = None

    try:
        gen_id = uuid.uuid4().hex

        pipeline = get_pipeline(model)

        music_stream = pipeline.generate(
            temperature=temperature,
            max_len=max_len,
            penalty=penalty,
            topK=topK,
            topP=topP
        )

        xml_path = OUTPUT_DIR / f"{gen_id}.musicxml"
        midi_path = OUTPUT_DIR / f"{gen_id}.mid"

        music_stream.write("musicxml", fp=str(xml_path))
        music_stream.write("midi", fp=str(midi_path))

        return {
            "status": "ok",
            "generation_id": gen_id,
            "model": model,
            "artifacts": {
                "musicxml": f"/outputs/{xml_path.name}",
                "midi": f"/outputs/{midi_path.name}"
            },
            "sampling": {
                "temperature": temperature,
                "max_len": max_len,
                "penalty": penalty,
                "topK": topK,
                "topP": topP
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if input_path is not None and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


@router.post("/generate_from_xml")
async def generate_from_xml(
    file: UploadFile,
    model: str = Form(...),
    temperature: float = Form(1.2),
    max_len: int = Form(256),
    penalty: float = Form(1.1),
    topK: int = Form(25),
    topP: float = Form(0.9)
):
    if model not in MODEL_REGISTRY:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unknown model: {model}"}
        )

    input_path = None

    try:
        gen_id = uuid.uuid4().hex
        input_path = OUTPUT_DIR / f"{gen_id}_input.musicxml"

        with open(input_path, "wb") as f:
            f.write(await file.read())

        pipeline = get_pipeline(model)

        music_stream = pipeline.generate_from_xml(
            xml_path=str(input_path),
            temperature=temperature,
            max_len=max_len,
            penalty=penalty,
            topK=topK,
            topP=topP
        )

        xml_path = OUTPUT_DIR / f"{gen_id}.musicxml"
        midi_path = OUTPUT_DIR / f"{gen_id}.mid"

        music_stream.write("musicxml", fp=str(xml_path))
        music_stream.write("midi", fp=str(midi_path))

        return {
            "status": "ok",
            "generation_id": gen_id,
            "model": model,
            "artifacts": {
                "musicxml": f"/outputs/{xml_path.name}",
                "midi": f"/outputs/{midi_path.name}"
            },
            "sampling": {
                "temperature": temperature,
                "max_len": max_len,
                "penalty": penalty,
                "topK": topK,
                "topP": topP
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if input_path is not None and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass



@router.post("/generate_from_TS_tonality")
async def generate_from_TS_tonality(
    model: str = Form(...),
    time_signature : str = Form(...),
    tonality : str = Form(...),
    temperature: float = Form(1.2),
    max_len: int = Form(256),
    penalty: float = Form(1.1),
    topK: int = Form(25),
    topP: float = Form(0.9)
):
    if model not in MODEL_REGISTRY:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Unknown model: {model}"}
        )

    input_path = None

    try:
        gen_id = uuid.uuid4().hex
        pipeline = get_pipeline(model)

        music_stream = pipeline.generate_from_TS_tonality(
            time_signature=time_signature,
            tonality=tonality,
            temperature=temperature,
            max_len=max_len,
            penalty=penalty,
            topK=topK,
            topP=topP
        )

        xml_path = OUTPUT_DIR / f"{gen_id}.musicxml"
        midi_path = OUTPUT_DIR / f"{gen_id}.mid"

        music_stream.write("musicxml", fp=str(xml_path))
        music_stream.write("midi", fp=str(midi_path))

        return {
            "status": "ok",
            "generation_id": gen_id,
            "model": model,
            "artifacts": {
                "musicxml": f"/outputs/{xml_path.name}",
                "midi": f"/outputs/{midi_path.name}"
            },
            "sampling": {
                "temperature": temperature,
                "max_len": max_len,
                "penalty": penalty,
                "topK": topK,
                "topP": topP
            }
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        if input_path is not None and input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


print("ROUTES:", router.routes)
print("GENERATE FILE:", inspect.getfile(inspect.currentframe()))
