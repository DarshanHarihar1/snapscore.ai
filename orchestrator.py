import yaml
from agents.preprocess import PreprocessingAgent
from agents.detect import DetectionAgent
from agents.classify import ClassificationAgent
from utils.image_utils import crop_image
from agents.querygen import QueryGenAgent
from agents.search import SearchAggregatorAgent
from agents.curate import CurationAgent
from agents.format import FormatterAgent

def load_cfg(path="config.yml") -> dict:
    return yaml.safe_load(open(path))

if __name__ == "__main__":
    cfg = load_cfg()
    raw = cfg['input']['raw_image']

    cleaned = PreprocessingAgent(cfg).execute(raw)

    dets    = DetectionAgent(cfg).execute(cleaned)

    all_metadata, all_crops = [], []
    for d in dets:
        crop = crop_image(cleaned, d['bbox'])
        meta = ClassificationAgent(cfg).execute(crop)
        all_crops.append(crop)
        all_metadata.append(meta)

    all_queries = []
    for meta, crop in zip(all_metadata, all_crops):
        queries = QueryGenAgent(cfg).execute(meta, crop)['text_queries']
        all_queries.append(queries)

    paired = list(zip(all_queries, all_crops))
    raw_results = SearchAggregatorAgent(cfg).execute(paired)

    curated_results = CurationAgent(cfg).execute(raw_results)

    formatted_output = FormatterAgent(cfg).execute(curated_results)
    print(formatted_output)