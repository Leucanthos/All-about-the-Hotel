"""
酒店评论RAG系统核心模块
包含：意图检测、双路召回、结果融合、重排序、答案生成
"""

import os
import re
import json
import time
import math
import pickle
import jieba
import nltk
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
from dashscope import TextEmbedding, TextReRank, Generation
import dashvector
from dashvector import Doc
import chromadb


# ============================================================================
# 1. BM25倒排索引类 (复用知识库中的实现)
# ============================================================================

class InvertedIndex:
    """基于 BM25 的倒排索引"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, stopwords_file: str = None):
        self.k1 = k1
        self.b = b
        self.index = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.num_docs = 0
        self.documents = {}

        self.stopwords = set()
        if stopwords_file and Path(stopwords_file).exists():
            with open(stopwords_file, encoding='utf-8') as f:
                self.stopwords.update([line.strip() for line in f])
            try:
                self.stopwords.update(nltk.corpus.stopwords.words('english'))
            except:
                pass
        
        jieba.initialize()
    
    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'\s+', '', text)
        tokens = jieba.lcut(text)
        pattern = re.compile(r'[^\u4e00-\u9fffa-zA-Z]')
        tokens = [token.lower() for token in tokens 
                  if token.lower() not in self.stopwords and not pattern.search(token)]
        return tokens
    
    def build(self, documents: Dict[str, str]):
        self.documents = documents
        self.num_docs = len(documents)
        
        total_length = 0
        for doc_id, text in documents.items():
            tokens = self.tokenize(text)
            doc_length = len(tokens)
            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length
            
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.index:
                    self.index[term] = {}
                self.index[term][doc_id] = freq
        
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
    
    def search(self, query: str, topk: int = 10) -> List[Tuple[str, float]]:
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        idf = {}
        for term in query_tokens:
            if term in self.index:
                df = len(self.index[term])
                idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        scores = {}
        for term in query_tokens:
            if term not in self.index:
                continue
            
            for doc_id, tf in self.index[term].items():
                if doc_id not in scores:
                    scores[doc_id] = 0
                
                doc_length = self.doc_lengths[doc_id]
                norm_factor = 1 - self.b + self.b * (doc_length / self.avg_doc_length)
                term_score = idf[term] * (tf * (self.k1 + 1)) / (tf + self.k1 * norm_factor)
                scores[doc_id] = scores.get(doc_id, 0) + term_score
        
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return sorted_docs
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'documents': self.documents,
                'k1': self.k1,
                'b': self.b,
                'stopwords': self.stopwords
            }, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.index = data['index']
            self.doc_lengths = data['doc_lengths']
            self.avg_doc_length = data['avg_doc_length']
            self.num_docs = data['num_docs']
            self.documents = data['documents']
            self.k1 = data['k1']
            self.b = data['b']
            self.stopwords = data.get('stopwords', set())


# ============================================================================
# 2. 嵌入模型客户端
# ============================================================================

class EmbeddingClient:
    """文本嵌入客户端"""
    def __init__(self, api_key: str, model: str = "text-embedding-v4", dimension: int = 1024):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
    
    def embed_batch(self, texts: List[str]) -> List:
        """批量生成 embedding"""
        if not texts:
            return []
        
        response = TextEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=texts,
            dimension=self.dimension
        )
        
        if response.status_code == 200:
            return [item['embedding'] for item in response.output['embeddings']]
        else:
            raise RuntimeError(f"Embedding 调用失败: {response.message}")
    
    def embed_query(self, query: str) -> List[float]:
        """生成带指令的查询向量"""
        response = TextEmbedding.call(
            api_key=self.api_key,
            model=self.model,
            input=query,
            text_type="query",
            instruct="Given a hotel review query, retrieve relevant hotel reviews that answer the query.",
            dimension=self.dimension
        )
        
        if response.status_code == 200:
            return response.output['embeddings'][0]['embedding']
        else:
            raise RuntimeError(f"Query Embedding 调用失败: {response.message}")


# ============================================================================
# 3. 主RAG系统类
# ============================================================================

class HotelReviewRAG:
    """酒店评论RAG系统"""
    
    def __init__(self, api_key: str, dashvector_api_key: str, dashvector_endpoint: str, 
                 data_dir: str = "data", batch_size: int = 10):
        """
        初始化RAG系统
        
        参数:
            api_key: DashScope API密钥
            dashvector_api_key: DashVector API密钥
            dashvector_endpoint: DashVector端点
            data_dir: 数据目录路径
            batch_size: embedding批量大小
        """
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        # 初始化Embedding客户端
        self.embedding_client = EmbeddingClient(api_key)
        
        # 初始化DashVector
        self.dashvector_client = dashvector.Client(
            api_key=dashvector_api_key,
            endpoint=dashvector_endpoint
        )
        
        # 初始化ChromaDB
        chroma_db_path = self.data_dir / "chroma_db"
        chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_db_path))
        
        # 加载评论数据
        self._load_data()
        
        # 初始化向量库连接
        self.comment_collection = self.dashvector_client.get("comment_database")
        self.query_collection = self.dashvector_client.get("reverse_query_database")
        self.summary_collection = self.chroma_client.get_collection("summary_database")
        
        # 初始化BM25索引
        self._init_inverted_index()
        
        # 意图类别
        self.intent_categories = {
            '房间': ['床', '房间', '卧室', '设施', '装修', '隔音', '空间'],
            '服务': ['服务', '前台', '前台员工', '态度', '效率', '热情'],
            '早餐': ['早餐', '自助', '点心', '虾饺', '肠粉', '品种'],
            '设施': ['泳池', '健身房', 'gym', '游泳', '温度'],
            '位置': ['位置', '地理', '地段', '地铁', '交通', '方便'],
        }
    
    def _load_data(self):
        """加载原始评论数据"""
        filtered_file = self.data_dir / "filtered_comments.csv"
        if filtered_file.exists():
            self.df_comments = pd.read_csv(filtered_file, index_col=0)
        else:
            self.df_comments = pd.DataFrame()
    
    def _init_inverted_index(self):
        """初始化BM25倒排索引"""
        index_file = self.data_dir / "inverted_index.pkl"
        stopwords_file = self.data_dir / "stopwords_chinese.txt"
        
        self.inverted_index = InvertedIndex(k1=1.5, b=0.75, stopwords_file=str(stopwords_file))
        
        if index_file.exists():
            self.inverted_index.load(str(index_file))
        else:
            # 从DataFrame重新构建
            if not self.df_comments.empty:
                documents = {idx: row['comment'] for idx, row in self.df_comments.iterrows()}
                self.inverted_index.build(documents)
                self.inverted_index.save(str(index_file))
    
    # ========================================================================
    # 意图检测
    # ========================================================================
    
    def detect_intent(self, query: str) -> Tuple[Dict, float]:
        """
        检测查询的意图类别

        参数:
            query: 用户查询

        返回:
            (意图检测字典, 置信度)
            意图检测字典格式: {'room_type': None, 'fuzzy_room_type': None, 'time_sensitivity': None}
        """
        query_lower = query.lower()
        intent_result = {
            'room_type': None,
            'fuzzy_room_type': None,
            'time_sensitivity': None
        }

        # 检测房型相关意图
        room_type_keywords = {
            '大床房': '大床房', '双床房': '双床房', '套房': '套房',
            '豪华房': '豪华房', '标准房': '标准房', '单人间': '单人间',
            '行政房': '行政房', '家庭房': '家庭房', '三人间': '三人间'
        }
        for kw, room in room_type_keywords.items():
            if kw in query_lower:
                intent_result['room_type'] = room
                break

        # 检测模糊房型意图
        fuzzy_keywords = {
            '安静': '安静', '靠窗': '靠窗',
            '楼层': '楼层', '高楼层': '高楼层', '低楼层': '低楼层',
            '朝向': '朝向', '花园': '花园', '景观': '景观'
        }
        for kw, fuzzy in fuzzy_keywords.items():
            if kw in query_lower:
                intent_result['fuzzy_room_type'] = fuzzy
                break

        # 检测时间敏感性
        time_keywords = {
            '最近': 'recent', '最新': 'recent', '近期': 'recent',
            '去年': 'past', '前年': 'past', '以前': 'past'
        }
        for kw, time_sens in time_keywords.items():
            if kw in query_lower:
                intent_result['time_sensitivity'] = time_sens
                break

        # 计算置信度
        scores = {}
        for category, keywords in self.intent_categories.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[category] = score

        if max(scores.values()) > 0:
            category = max(scores, key=scores.get)
            confidence = scores[category] / sum(scores.values())
            return intent_result, confidence
        else:
            return intent_result, 0.5

    def expand_intent(self, query: str) -> List[Tuple[str, float]]:
        """
        意图扩展：根据原始查询使用LLM扩展出2-3个相关问题，并自动分配权重

        参数:
            query: 用户查询

        返回:
            扩展问题列表 [(扩展问题, weight), ...]
        """
        try:
            # 使用LLM生成扩展问题和权重
            prompt = f"""基于用户的问题，生成2-3个相关的补充问题，帮助更全面地检索酒店评论信息。

用户问题：{query}

请生成2-3个相关的补充问题，并用0-1之间的权重表示每个问题的重要程度（权重越高表示对回答原问题越重要）。

要求：
1. 权重总和为1
2. 问题要简洁明了
3. 输出格式为每行一个问题和权重，用逗号分隔

输出示例：
酒店早餐的种类和品质如何？, 0.5
早餐有没有广州特色点心？, 0.3
早餐价格性价比怎么样？, 0.2

请直接输出，不要其他解释："""

            response = Generation.call(
                api_key=self.api_key,
                model='qwen-turbo',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=200,
                top_p=0.9,
                temperature=0.7
            )

            if response.status_code == 200:
                result_text = response.output.text.strip()
                expansions = []

                # 解析LLM输出
                for line in result_text.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    # 尝试分割问题和权重
                    if ',' in line:
                        parts = line.rsplit(',', 1)
                        if len(parts) == 2:
                            question = parts[0].strip().rstrip(',')
                            try:
                                weight = float(parts[1].strip())
                                if question and 0 <= weight <= 1:
                                    expansions.append((question, weight))
                            except:
                                pass

                # 如果解析成功且有结果，返回
                if expansions:
                    # 归一化权重
                    total = sum(w for _, w in expansions)
                    if total > 0:
                        expansions = [(q, w/total) for q, w in expansions]
                    return expansions[:3]

        except Exception as e:
            print(f"意图扩展失败: {e}")

        # 如果LLM失败，返回空列表
        return []
    
    # ========================================================================
    # 双路召回
    # ========================================================================
    
    def retrieve_vector(self, query: str, topk: int = 5) -> List[Dict]:
        """向量检索 - 使用DashVector"""
        try:
            query_embedding = self.embedding_client.embed_query(query)
            # DashVector API
            results = self.comment_collection.query(
                vector=query_embedding,
                topk=topk
            )

            retrieved = []
            for i, doc in enumerate(results):
                # 从df获取完整信息
                room_type = ''
                rating = 'N/A'
                publish_date = 'N/A'
                quality = 'N/A'
                comment = doc.fields.get('comment', '')

                if hasattr(self, 'df_comments') and not self.df_comments.empty:
                    if doc.id in self.df_comments.index:
                        row = self.df_comments.loc[doc.id]
                        room_type = row.get('room_type', '')
                        rating = row.get('score', 'N/A')
                        publish_date = str(row.get('publish_date', 'N/A'))
                        quality = row.get('quality_score', 'N/A')
                        # 如果comment为空，从df获取
                        if not comment:
                            comment = row.get('comment', '')

                retrieved.append({
                    'rank': i + 1,
                    'method': 'vector',
                    'doc_id': doc.id,
                    'score': doc.score,
                    'comment': comment,
                    'room_type': room_type,
                    'rating': rating,
                    'publish_date': publish_date,
                    'quality': quality
                })
            return retrieved
        except Exception as e:
            print(f"向量检索失败: {e}")
            return []
    
    def retrieve_bm25(self, query: str, topk: int = 5) -> List[Dict]:
        """BM25关键词检索"""
        try:
            results = self.inverted_index.search(query, topk=topk)

            retrieved = []
            for i, (doc_id, score) in enumerate(results):
                comment = self.inverted_index.documents.get(doc_id, '')

                # 尝试从df_comments获取更多元信息
                room_type = ''
                rating = 'N/A'
                publish_date = 'N/A'
                quality = 'N/A'

                if hasattr(self, 'df_comments') and not self.df_comments.empty:
                    if doc_id in self.df_comments.index:
                        row = self.df_comments.loc[doc_id]
                        room_type = row.get('room_type', '')
                        # 评分字段是score，不是rating
                        rating = row.get('score', 'N/A')
                        publish_date = str(row.get('publish_date', 'N/A'))
                        quality = row.get('quality_score', 'N/A')

                retrieved.append({
                    'rank': i + 1,
                    'method': 'bm25',
                    'doc_id': doc_id,
                    'score': score,
                    'comment': comment,
                    'room_type': room_type,
                    'rating': rating,
                    'publish_date': publish_date,
                    'quality': quality
                })
            return retrieved
        except Exception as e:
            print(f"BM25检索失败: {e}")
            return []
    
    def retrieve_reverse_query(self, query: str, topk: int = 5) -> List[Dict]:
        """反向查询检索 - 使用DashVector"""
        try:
            query_embedding = self.embedding_client.embed_query(query)
            results = self.query_collection.query(
                vector=query_embedding,
                topk=topk
            )

            retrieved = []
            for i, doc in enumerate(results):
                # 从df获取完整信息
                room_type = ''
                rating = 'N/A'
                publish_date = 'N/A'
                quality = 'N/A'
                comment = doc.fields.get('comment', '')

                if hasattr(self, 'df_comments') and not self.df_comments.empty:
                    if doc.id in self.df_comments.index:
                        row = self.df_comments.loc[doc.id]
                        room_type = row.get('room_type', '')
                        rating = row.get('score', 'N/A')
                        publish_date = str(row.get('publish_date', 'N/A'))
                        quality = row.get('quality_score', 'N/A')
                        # 如果comment为空，从df获取
                        if not comment:
                            comment = row.get('comment', '')

                retrieved.append({
                    'rank': i + 1,
                    'method': 'reverse_query',
                    'doc_id': doc.id,
                    'score': doc.score,
                    'comment': comment,
                    'room_type': room_type,
                    'rating': rating,
                    'publish_date': publish_date,
                    'quality': quality
                })
            return retrieved
        except Exception as e:
            print(f"反向查询检索失败: {e}")
            return []
    
    def retrieve_hyde(self, query: str, topk: int = 3) -> List[Dict]:
        """HyDE假设性文档生成"""
        try:
            # 使用LLM生成假设文档
            prompt = f"""基于以下问题，生成2个相关的酒店评论。问题：{query}

请只输出评论内容，用换行符分隔。"""
            
            response = Generation.call(
                api_key=self.api_key,
                model='qwen-turbo',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            if response.status_code == 200:
                generated_docs = response.output.text.strip().split('\n')
                retrieved = []
                
                for i, doc in enumerate(generated_docs[:2]):
                    if doc.strip():
                        # 对生成的文档进行embedding和检索
                        try:
                            embedding = self.embedding_client.embed_query(doc)
                            results = self.comment_collection.query(vector=embedding, topk=1)
                            if results:
                                retrieved.append({
                                    'rank': i + 1,
                                    'method': 'hyde',
                                    'doc_id': results[0].id,
                                    'score': results[0].score,
                                    'comment': results[0].fields.get('comment', ''),
                                    'room_type': results[0].fields.get('room_type', '')
                                })
                        except:
                            pass
                
                return retrieved
        except Exception as e:
            print(f"HyDE检索失败: {e}")

        return []

    def retrieve_summary_by_intent(self, query: str, intent: Dict) -> List[Dict]:
        """根据语义向量检索相关摘要"""
        try:
            # 对 query 做 embedding，使用向量相似度检索
            query_embedding = self.embedding_client.embed_query(query)
            results = self.summary_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )

            if not results or 'ids' not in results:
                return []

            retrieved = []
            ids = results['ids'][0]
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results.get('distances', [[]])[0]

            for i, (doc_id, document, metadata) in enumerate(zip(ids, documents, metadatas)):
                category = metadata.get('category', '')
                keywords = metadata.get('keywords', '')
                comment_count = metadata.get('comment_count', 0)
                # ChromaDB cosine distance → similarity score
                score = round(1 - distances[i], 4) if distances else 0.0

                retrieved.append({
                    'rank': i + 1,
                    'method': 'summary',
                    'doc_id': doc_id,
                    'score': score,
                    'category': category,
                    'keywords': keywords,
                    'comment_count': comment_count,
                    'summary': document  # 真实摘要内容
                })

            return retrieved
        except Exception as e:
            print(f"摘要检索失败: {e}")
            return []
    
    # ========================================================================
    # 结果融合 (RRF - Reciprocal Rank Fusion)
    # ========================================================================

    def _get_metadata_from_df(self, doc_id: str) -> Dict:
        """从df_comments获取元数据"""
        metadata = {
            'room_type': 'N/A',
            'rating': 'N/A',
            'publish_date': 'N/A',
            'quality': 'N/A'
        }

        if hasattr(self, 'df_comments') and not self.df_comments.empty:
            if doc_id in self.df_comments.index:
                row = self.df_comments.loc[doc_id]
                metadata['room_type'] = row.get('room_type', 'N/A') or 'N/A'
                metadata['rating'] = row.get('score', 'N/A')
                metadata['publish_date'] = str(row.get('publish_date', 'N/A'))
                metadata['quality'] = row.get('quality_score', 'N/A')

        return metadata

    def fuse_results(self, *retrieval_lists) -> List[Dict]:
        """
        使用RRF融合多路检索结果

        参数:
            *retrieval_lists: 多个检索结果列表

        返回:
            融合后的结果列表
        """
        rrf_scores = {}
        all_results = {}

        # 计算RRF分数
        for retrieval_list in retrieval_lists:
            for item in retrieval_list:
                doc_id = item['doc_id']
                rank = item['rank']
                rrf_score = 1.0 / (60 + rank)

                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0
                    all_results[doc_id] = item

                rrf_scores[doc_id] += rrf_score

        # 按RRF分数排序，并填充缺失的元数据
        fused = []
        for doc_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            item = all_results[doc_id].copy()
            item['fused_score'] = score

            # 确保元数据完整
            meta = self._get_metadata_from_df(doc_id)
            if not item.get('room_type') or item.get('room_type') in ['', 'N/A']:
                item['room_type'] = meta['room_type']
            if not item.get('rating') or item.get('rating') in ['', 'N/A']:
                item['rating'] = meta['rating']
            if not item.get('publish_date') or item.get('publish_date') in ['', 'N/A']:
                item['publish_date'] = meta['publish_date']
            if not item.get('quality') or item.get('quality') in ['', 'N/A']:
                item['quality'] = meta['quality']

            fused.append(item)

        return fused
    
    # ========================================================================
    # 重排序
    # ========================================================================
    
    def rerank(self, query: str, candidates: List[Dict], topk: int = 3) -> List[Dict]:
        """
        使用Qwen3-Rerank重排序候选文档
        
        参数:
            query: 用户查询
            candidates: 候选文档列表
            topk: 返回Top-K
        
        返回:
            重排序后的文档列表
        """
        if not candidates:
            return []
        
        try:
            documents = [item['comment'] for item in candidates]
            
            response = TextReRank.call(
                api_key=self.api_key,
                model="qwen3-rerank",
                query=query,
                documents=documents,
                top_n=min(topk, len(documents)),
                return_documents=False
            )
            
            reranked = []
            for item in response.output.results:
                original_item = candidates[item.index].copy()
                original_item['rerank_score'] = item.relevance_score
                reranked.append(original_item)
            
            return reranked
        except Exception as e:
            print(f"重排序失败: {e}")
            return candidates[:topk]
    
    # ========================================================================
    # 答案生成
    # ========================================================================
    
    def generate_answer(self, query: str, context: List[Dict], summaries: List[Dict] = None) -> str:
        """
        基于召回的文档和类别摘要生成答案

        参数:
            query: 用户查询
            context: 上下文评论列表
            summaries: 相关类别摘要列表（可选）

        返回:
            生成的答案
        """
        if not context:
            return "非常抱歉，目前暂时没有找到与您问题相关的评论信息。建议您查看最新的酒店评价，或联系酒店前台获取更准确的信息。"

        # 构建类别摘要背景块（如有）
        summary_block = ""
        if summaries:
            summary_parts = []
            for s in summaries:
                category = s.get('category', s.get('keywords', ''))
                content = s.get('summary', '')
                if content:
                    summary_parts.append(f"【{category}】{content[:300]}")
            if summary_parts:
                summary_block = "\n\n【类别背景知识】（基于大量评论的综合分析）：\n" + "\n\n".join(summary_parts) + "\n\n"

        # 构建评论上下文
        context_parts = []
        for i, item in enumerate(context[:5], 1):
            room_type = item.get('room_type', '未知房型') or '未知房型'
            rating = item.get('rating', item.get('score', 'N/A'))
            comment = item['comment']
            context_parts.append(f"评论{i}【{room_type}，评分{rating}】: {comment}")

        context_text = "\n\n".join(context_parts)

        prompt = f"""您是一个第三方酒店点评平台的智能助手，需要根据收集到的住客真实评论来回答用户的问题。

请用客观、专业、详细的语气回答用户的问题。**不要提及您是酒店的员工或客服，您是第三方点评平台**。
{summary_block}【真实住客评论】：
{context_text}

用户问题：{query}

请根据上述信息，给出详细、全面的回答。包括：
1. 正面评价（大多数住客的满意点）
2. 需要注意的方面（可能的不足）
3. 小贴士（如有）

回答要有条理，使用恰当的标题分隔内容。"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model='qwen-turbo',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=500,
                top_p=0.9,
                temperature=0.5
            )

            if response.status_code == 200:
                return response.output.text.strip()
        except Exception as e:
            print(f"生成答案失败: {e}")

        return "非常抱歉，目前暂时无法生成回答。请查看最新的酒店评价。"
    
    # ========================================================================
    # 主查询入口
    # ========================================================================
    
    def query(self, query: str, enable_hyde: bool = True, topk: int = 3) -> Dict:
        """
        端到端RAG查询

        参数:
            query: 用户查询
            enable_hyde: 是否使用HyDE
            topk: 最终返回Top-K

        返回:
            包含所有中间结果和最终答案的字典
        """
        import time

        # 初始化时间记录
        start_time = time.time()
        latency = {}

        result = {
            'query': query,
            'intent': None,
            'intent_confidence': 0.0,
            'intent_expand': [],
            'vector_results': [],
            'bm25_results': [],
            'reverse_query_results': [],
            'hyde_results': [],
            'fused_results': [],
            'reranked_results': [],
            'answer': '',
            'latency': {},
            'retrieval_summaries': []
        }

        # [1] 意图检测
        intent_start = time.time()
        intent, confidence = self.detect_intent(query)
        intent_detect_time = time.time() - intent_start
        result['intent'] = intent
        result['intent_confidence'] = confidence
        latency['意图识别'] = max(intent_detect_time, 0.001)

        # [1.5] 意图扩展
        expand_start = time.time()
        expanded_queries = self.expand_intent(query)
        result['intent_expand'] = expanded_queries
        intent_expand_time = time.time() - expand_start
        latency['意图扩展'] = max(intent_expand_time, 0.001)

        # [2] 双路召回
        retrieval_start = time.time()

        vec_start = time.time()
        result['vector_results'] = self.retrieve_vector(query, topk=5)
        vec_time = time.time() - vec_start

        bm25_start = time.time()
        result['bm25_results'] = self.retrieve_bm25(query, topk=5)
        bm25_time = time.time() - bm25_start

        rev_start = time.time()
        result['reverse_query_results'] = self.retrieve_reverse_query(query, topk=5)
        rev_time = time.time() - rev_start

        hyde_time = 0
        if enable_hyde:
            hyde_start = time.time()
            result['hyde_results'] = self.retrieve_hyde(query, topk=3)
            hyde_time = time.time() - hyde_start

        # 摘要召回（使用关键词匹配）
        summary_start = time.time()
        result['retrieval_summaries'] = self.retrieve_summary_by_intent(query, intent)
        summary_time = time.time() - summary_start

        retrieval_time = time.time() - retrieval_start
        latency['混合检索'] = retrieval_time
        latency['文本召回'] = bm25_time
        latency['向量召回'] = vec_time
        latency['反向召回'] = rev_time
        latency['HyDE召回'] = hyde_time
        latency['摘要召回'] = summary_time

        # [3] 结果融合
        fusion_start = time.time()
        retrieval_lists = [
            result['vector_results'],
            result['bm25_results'],
            result['reverse_query_results']
        ]
        if enable_hyde:
            retrieval_lists.append(result['hyde_results'])

        fused = self.fuse_results(*retrieval_lists)
        result['fused_results'] = fused[:10]
        fusion_time = time.time() - fusion_start
        # 确保至少显示0.001s
        latency['RRF融合'] = max(fusion_time, 0.001)

        # [4] 重排序 - 固定使用topk=10获取Top10评论
        rerank_start = time.time()
        reranked = self.rerank(query, fused[:10], topk=10)

        # 确保reranked结果中元数据完整
        for item in reranked:
            if item.get('room_type') in ['', 'N/A', None] or item.get('rating') in ['', 'N/A', None]:
                meta = self._get_metadata_from_df(item['doc_id'])
                if item.get('room_type') in ['', 'N/A', None]:
                    item['room_type'] = meta['room_type']
                if item.get('rating') in ['', 'N/A', None]:
                    item['rating'] = meta['rating']
                if item.get('publish_date') in ['', 'N/A', None]:
                    item['publish_date'] = meta['publish_date']
                if item.get('quality') in ['', 'N/A', None]:
                    item['quality'] = meta['quality']

        result['reranked_results'] = reranked
        rerank_time = time.time() - rerank_start
        latency['排序'] = rerank_time

        # [5] 生成答案
        generate_start = time.time()
        result['answer'] = self.generate_answer(query, reranked, summaries=result['retrieval_summaries'])
        generate_time = time.time() - generate_start

        total_time = time.time() - start_time

        latency['模型回复'] = generate_time
        latency['总延迟'] = total_time

        result['latency'] = latency

        return result


# ============================================================================
# 4. 输出格式化函数
# ============================================================================

def print_rag_result(result: Dict, verbose: bool = False):
    """美化打印RAG查询结果 - 最终答案放在最前面"""

    print("=" * 80)
    print(f"📝 查询: {result['query']}")
    print("=" * 80)

    # ========== 最终答案（放在最前面）==========
    print(f"\n💬 最终答案:")
    print("-" * 80)
    print(f"{result['answer']}")
    print("-" * 80)

    # ========== 意图检测 ==========
    print(f"\n🎯 意图检测: {result['intent']} (置信度: {result['intent_confidence']:.2%})")

    # 意图扩展输出
    if result.get('intent_expand'):
        print(f"\n🔍 意图扩展:")
        for exp_query, weight in result['intent_expand']:
            print(f"  • {exp_query} (weight={weight})")

    # 延迟统计输出
    if result.get('latency'):
        print(f"\n⏱️ 延迟统计:")
        latency = result['latency']
        for key, value in latency.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.3f}s")
            else:
                print(f"  • {key}: {value}")

    # 摘要召回输出
    if result.get('retrieval_summaries'):
        print(f"\n📚 召回摘要类别 ({len(result['retrieval_summaries'])}个):")
        for i, summary in enumerate(result['retrieval_summaries']):
            category = summary.get('category', summary.get('keywords', 'N/A'))
            score = summary.get('score', 0)
            comment_count = summary.get('comment_count', 0)
            summary_text = summary.get('summary', '')
            print(f"  [{i+1}] 类别: {category} (相似度: {score:.4f}, 评论数: {comment_count})")
            if summary_text:
                print(f"      摘要: {summary_text[:80]}...")

    if verbose:
        print(f"\n🔍 向量召回 ({len(result['vector_results'])} 条)")
        print(f"📚 BM25召回 ({len(result['bm25_results'])} 条)")
        print(f"🔄 反向Query召回 ({len(result['reverse_query_results'])} 条)")

    # 重排序结果（包含结构化特征和召回路由）
    print(f"\n🏆 Top 10 评论:")
    print(" " + "━" * 76)
    for i, item in enumerate(result['reranked_results'][:10], 1):
        # 提取各项分数
        rerank_score = item.get('rerank_score', item.get('fused_score', 0))
        retrieve_rank = item.get('rank', 0)
        retrieve_score = item.get('score', 0)

        # 评论结构化特征
        room_type = item.get('room_type', '')
        if not room_type or room_type in ['', 'N/A']:
            room_type = '未知房型'
        rating = item.get('rating', 'N/A')
        quality = item.get('quality', 'N/A')
        publish_date = item.get('publish_date', 'N/A')

        # 如果rating是数值，格式化显示
        if isinstance(rating, (int, float)) and rating != 'N/A':
            rating = f"{rating:.1f}"

        comment_info = f"房型: {room_type} | 评分: {rating}"
        if quality != 'N/A':
            comment_info += f" | 质量: {quality}"
        if publish_date != 'N/A':
            comment_info += f" | 发布: {publish_date}"

        # 召回路由 - 处理doc_id格式
        item_doc_id = str(item.get('doc_id', '')).strip()
        routes = []
        if result.get('vector_results'):
            for vr in result['vector_results']:
                if str(vr.get('doc_id', '')).strip() == item_doc_id:
                    routes.append(f"向量: #{vr.get('rank')}")
                    break
        if result.get('bm25_results'):
            for br in result['bm25_results']:
                if str(br.get('doc_id', '')).strip() == item_doc_id:
                    routes.append(f"文本: #{br.get('rank')}")
                    break
        if result.get('reverse_query_results'):
            for rr in result['reverse_query_results']:
                if str(rr.get('doc_id', '')).strip() == item_doc_id:
                    routes.append(f"反向: #{rr.get('rank')}")
                    break
        route_info = " | ".join(routes) if routes else "N/A"

        print(f"\n#{i} 综合得分: {rerank_score:.4f} | {comment_info}")
        print(f"   检索: #{retrieve_rank} 得分:{retrieve_score:.4f} | 召回: {route_info}")
        print(f"   {item.get('comment', 'N/A')[:120]}...")
        print(" " + "━" * 76)

    print("=" * 80)
