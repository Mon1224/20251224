import re
import json
import config
from utils import clear_directory_contents
from pathlib import Path
from typing import List, Dict

class ParentStoreManager:
    __store_path: Path

    def __init__(self, store_path=config.PARENT_STORE_PATH):
        self.__store_path = Path(store_path) 
        self.__store_path.mkdir(parents=True, exist_ok=True)

    def save(self, parent_id: str, content: str, metadata: Dict) -> None:
        file_path = self.__store_path / f"{parent_id}.json"
        file_path.write_text(
            json.dumps({"page_content": content,"metadata": metadata}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def save_many(self, parents: List) -> None:
        for parent_id, doc in parents:
            self.save(parent_id, doc.page_content, doc.metadata)

    def delete_many(self, parent_ids: List[str]) -> None:
        for parent_id in parent_ids:
            file_path = self.__store_path / f"{parent_id}.json"
            if file_path.exists():
                file_path.unlink()

    def load(self, parent_id: str) -> Dict:
        file_path = self.__store_path / (
            parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        )
        return json.loads(file_path.read_text(encoding="utf-8"))
    
    def load_content(self, parent_id: str) -> Dict:
        data = self.load(parent_id)
        return {
                "content": data["page_content"],
                "parent_id": parent_id,
                "metadata": data["metadata"]
            }

    @staticmethod
    def _get_sort_key(id_str):
        match = re.search(r'_(?:parent_|p)(\d+)$', id_str)
        return int(match.group(1)) if match else 0

    def load_content_many(self, parent_ids: List[str]) -> List[Dict]:
        unique_ids = set(parent_ids)
        return [self.load_content(pid) for pid in sorted(unique_ids, key=self._get_sort_key)]

    def list_sources(self) -> List[str]:
        sources = set()
        for file_path in self.__store_path.glob("*.json"):
            try:
                source = json.loads(file_path.read_text(encoding="utf-8")).get("metadata", {}).get("source")
                if source:
                    sources.add(source)
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(sources)
    
    def clear_store(self) -> None:
        self.__store_path.mkdir(parents=True, exist_ok=True)
        clear_directory_contents(self.__store_path)
