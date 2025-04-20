# snapscore.ai

Snapshop.AI is a modular, multi-agent pipeline for fashion product discovery. Given a raw input image, it:

1. Preprocesses the image to normalize size, lighting and contrast.
2. Detects individual fashion items (bounding boxes).
3. Classifies each crop (e.g., "long sleeve shirt," "cargo pants").
4. Generates a set of text queries per item.
5. Searches Google Images Lens with each crop + query to find visual matches.
6. Curates the raw matches—deduplicating, filtering out missing URLs, and keeping top-K per item.
7. Formats the final results as human-readable text and writes them to disk.

Everything is orchestrated from a single orchestrator.py entry point and is fully configurable via config.yml.

---

**Project Structure**

```plaintext
.
├── agents/
│   ├── preprocess.py
│   ├── detect.py
│   ├── classify.py
│   ├── querygen.py
│   ├── search.py
│   ├── curate.py
│   └── format.py
├── data/               # sample crops, inputs, outputs
├── utils/
│   └── image_utils.py
├── orchestrator.py     # entrypoint
├── config.yml          # pipeline configuration
├── auth.json           # (ignored) local secrets
├── requirements.txt
└── search_results.json # raw search dump (example)
```
