import json
import csv
from io import StringIO

class FormatterAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.format = cfg.get('formatter', {}).get('format', 'text')  # 'text', 'json', 'csv'
        self.output_path = cfg.get('formatter', {}).get('output_path')

    def execute(self, curated_results):
        """
        Args:
            curated_results: dict {image_path: [curated_match,...], ...}
        Returns:
            str: formatted output
        """
        if self.format == 'json':
            output = json.dumps(curated_results, indent=2)
        elif self.format == 'csv':
            rows = []
            for img, items in curated_results.items():
                for it in items:
                    rows.append({
                        'image': img,
                        'query': it['query'],
                        'title': it['title'],
                        'source': it['source'],
                        'url': it['url']
                    })
            sio = StringIO()
            writer = csv.DictWriter(sio, fieldnames=['image','query','title','source','url'])
            writer.writeheader()
            writer.writerows(rows)
            output = sio.getvalue()
        else:
            lines = []
            for img, items in curated_results.items():
                lines.append(f"Image: {img}")
                for i, it in enumerate(items, start=1):
                    lines.append(f"  {i}. Query='{it['query']}' | {it['title']} ({it['source']}): {it['url']}")
            output = "\n".join(lines)

        if self.output_path:
            with open(self.output_path, 'w') as f:
                f.write(output)
        return output