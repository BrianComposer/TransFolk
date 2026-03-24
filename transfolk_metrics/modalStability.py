from pathlib import Path
from music21 import converter, stream

def analyze_global_mode(score):
    try:
        k = score.analyze('key')
        return k.mode
    except Exception:
        return None

def analyze_window_mode(score_window):
    try:
        k = score_window.analyze('key')
        return k.mode
    except Exception:
        return None

def modal_stability_from_folder(xml_path, window_measures=2):
    xml_files = list(Path(xml_path).glob("*.xml")) + list(Path(xml_path).glob("*.musicxml"))
    if not xml_files:
        return 0.0, []

    results = []

    for f in xml_files:
        score = converter.parse(str(f))
        global_mode = analyze_global_mode(score)
        if global_mode is None:
            results.append(0.0)
            continue

        part = score.parts[0]
        measures = list(part.getElementsByClass('Measure'))

        window_modes = []
        for i in range(0, len(measures) - window_measures + 1):
            tmp = stream.Stream()
            for m in measures[i:i+window_measures]:
                tmp.append(m)

            mode = analyze_window_mode(tmp)
            if mode is not None:
                window_modes.append(mode)

        if not window_modes:
            results.append(0.0)
            continue

        stable = sum(1 for m in window_modes if m == global_mode)
        results.append(stable / len(window_modes))

    mean_stability = sum(results) / len(results)
    return mean_stability, results
