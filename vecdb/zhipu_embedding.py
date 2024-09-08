from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """`Zhipuai Embeddings` embedding models."""

    zhipuai_api_key: Optional[str] = None
    """Zhipuai application apikey"""
    
    client: Any = None  # 添加这个字段来存储客户端对象

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate whether zhipuai_api_key in the environment variables or
        configuration file are available or not.

        Args:
            values: a dictionary containing configuration information, must include the
            fields of zhipuai_api_key

        Returns:
            a dictionary containing configuration information. If zhipuai_api_key
            are not provided in the environment variables or configuration
            file, the original values will be returned; otherwise, values containing
            zhipuai_api_key will be returned.

        Raises:
            ValueError: zhipuai package not found, please install it with `pip install
            zhipuai`
        """
        values["zhipuai_api_key"] = get_from_dict_or_env(
            values,
            "zhipuai_api_key",
            "ZHIPUAI_API_KEY",
        )

        try:
            from zhipuai import ZhipuAI
            values["client"] = ZhipuAI(api_key=values["zhipuai_api_key"])

        except ImportError:
            raise ValueError(
                "Zhipuai package not found, please install it with "
                "`pip install zhipuai`"
            )
        return values

    def __call__(self, input: str) -> List[float]:
        """
        Embedding a single text input.
        
        Args:
            input (str): A text to be embedded.
        
        Returns:
            List[float]: An embedding list of input text, which is a list of floating-point values.
        """
        return self._embed(input)

    def _embed(self, text: str) -> List[float]:
        # send request
        try:
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text
            )
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")
        if not response.data and not response.data[0].embedding:
            raise ValueError("Invalid response received from the API")
        
        embeddings = response.data[0].embedding
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embedding a text.

        Args:
            text (str): A text to be embedded.

        Return:
            List [float]: An embedding list of input text, which is a list of floating-point values.
        """
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                            Each embedding is represented as a list of float values.
        """
        return [self._embed(text) for text in texts]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError(
            "Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError(
            "Please use `aembed_query`. Official does not support asynchronous requests")