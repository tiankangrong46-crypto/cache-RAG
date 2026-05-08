# cache_utils.py
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

class SimpleRAGCache:
    """轻量级本地缓存：基于哈希的精确匹配 + TTL"""
    
    def __init__(self, cache_dir="./.rag_cache", ttl_hours=24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.db_path = self.cache_dir / "cache.db"
        self.ttl_hours = ttl_hours
        self._init_db()
    
    def _init_db(self):
        """初始化SQLite缓存表"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key_hash TEXT PRIMARY KEY,
                query TEXT,
                response TEXT,
                context_hash TEXT,
                image_hashes TEXT,
                created_at TIMESTAMP,
                hit_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON cache(created_at)")
        conn.commit()
        conn.close()
    
    def _make_key(self, query: str, context: str, image_hashes: list) -> str:
        """生成缓存Key：query + 检索结果 + 图像指纹"""
        content = f"{query.strip()}|{context.strip()}|{'|'.join(sorted(image_hashes))}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _hash_image(self, image_path: str) -> str:
        """生成图像文件的哈希指纹"""
        try:
            with open(image_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def get(self, query: str, context: str, image_paths: list) -> str | None:
        """获取缓存"""
        self._cleanup()
        
        image_hashes = [self._hash_image(p) for p in image_paths if Path(p).exists()]
        key = self._make_key(query, context, image_hashes)
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("SELECT response FROM cache WHERE key_hash = ?", (key,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                self._increment_hit(key)
                return result[0]
        except Exception as e:
            print(f"[Cache] Get error: {e}")
        return None
    
    def set(self, query: str, context: str, image_paths: list, response: str):
        """写入缓存"""
        try:
            image_hashes = [self._hash_image(p) for p in image_paths if Path(p).exists()]
            key = self._make_key(query, context, image_hashes)
            context_hash = hashlib.md5(context.encode('utf-8')).hexdigest()
            image_sig = "|".join(sorted(image_hashes))
            
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (key_hash, query, response, context_hash, image_hashes, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (key, query, response, context_hash, image_sig, datetime.now(), 0))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Cache] Set error: {e}")
    
    def _increment_hit(self, key_hash: str):
        """后台更新命中计数"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("UPDATE cache SET hit_count = hit_count + 1 WHERE key_hash = ?", (key_hash,))
            conn.commit()
            conn.close()
        except:
            pass
    
    def _cleanup(self):
        """清理过期缓存"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            expiry = datetime.now() - timedelta(hours=self.ttl_hours)
            conn.execute("DELETE FROM cache WHERE created_at < ?", (expiry,))
            conn.commit()
            conn.close()
        except:
            pass
    
    def clear(self):
        """🔥 清空所有缓存（类方法）"""
        if self.db_path.exists():
            self.db_path.unlink()
            self._init_db()
    
    def stats(self) -> dict:
        """🔥 返回缓存统计（类方法）"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            total_hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) FROM cache").fetchone()[0]
            conn.close()
            return {
                "total_entries": int(total),
                "total_hits": int(total_hits)
            }
        except Exception as e:
            print(f"[Cache] Stats error: {e}")
            return {"total_entries": 0, "total_hits": 0}