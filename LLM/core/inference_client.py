"""
Inference Client
HTTP client for calling persistent LLM inference servers.
"""
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class InferenceClient:
    """HTTP client for LLM inference server"""
    
    def __init__(self, base_url: str):
        """
        Initialize inference client.
        
        Args:
            base_url: Base URL of the inference server (e.g., "http://127.0.0.1:9100")
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 256, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            
        Returns:
            Generated text (clean, no wrappers)
            
        Raises:
            requests.exceptions.RequestException: If request fails
            ValueError: If server returns invalid response
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
        
        logger.debug(f"Sending generation request to {url}")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        logger.debug(f"Max tokens: {max_new_tokens}, Temperature: {temperature}")
        
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=300  # 5 minute timeout for generation
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Generation request timed out after 300s. "
                f"Model may be too slow or unresponsive."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Generation request failed: {e}")
            raise
        
        # Parse response
        try:
            result = response.json()
        except ValueError as e:
            raise ValueError(f"Invalid JSON response from server: {e}")
        
        # Extract text from response
        if "text" not in result:
            raise ValueError(f"Server response missing 'text' field: {result}")
        
        text = result["text"]
        logger.debug(f"Generated text length: {len(text)} chars")
        
        return text
    
    def health_check(self) -> bool:
        """
        Check if server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
