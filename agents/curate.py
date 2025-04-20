class CurationAgent:
    """
    Curates raw search results by deduplicating, filtering, and selecting top matches per image.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.top_k = cfg.get('curation', {}).get('top_k', 5)
        self.drop_none_url = cfg.get('curation', {}).get('drop_none_url', True)

    def execute(self, results_by_image):
        """
        Args:
            results_by_image: dict {image_path: {query: [match_dict,...], ...}, ...}
        Returns:
            dict {image_path: [curated_match, ...], ...}
        """
        curated = {}
        for img_path, queries_dict in results_by_image.items():
            seen_urls = set()
            curated_list = []
            for query, matches in queries_dict.items():
                for m in matches:
                    url = m.get('url')
                    if self.drop_none_url and not url:
                        continue
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    curated_item = {
                        'query': query,
                        'title': m.get('title'),
                        'url': url,
                        'source': m.get('source')
                    }
                    curated_list.append(curated_item)
                    if len(curated_list) >= self.top_k:
                        break
                if len(curated_list) >= self.top_k:
                    break
            curated[img_path] = curated_list
        return curated