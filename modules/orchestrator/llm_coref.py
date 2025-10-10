#!/usr/bin/env python3
"""
LLM-based coreference resolution for pronoun resolution.
Supports multiple LLM providers with fallback mechanisms.
"""

import os
import json
import requests
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM API."""
    provider: str  # 'openai', 'anthropic', 'ollama', 'custom'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 100
    temperature: float = 0.1
    timeout: int = 30

class LLMCorefResolver:
    """LLM-based coreference resolution."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'SRL-RDF-Pipeline/1.0'
        })
        
        if config.api_key:
            if config.provider == 'openai':
                self.session.headers['Authorization'] = f'Bearer {config.api_key}'
            elif config.provider == 'anthropic':
                self.session.headers['x-api-key'] = config.api_key
    
    def resolve_pronouns(self, text: str, context_entities: List[str], pronoun: str) -> str:
        """
        Resolve a pronoun using LLM.
        
        Args:
            text: The full text containing the pronoun
            context_entities: List of entity URIs that could be the referent
            pronoun: The specific pronoun to resolve (e.g., "he", "she", "it")
        
        Returns:
            Resolved entity URI or original pronoun if resolution fails
        """
        try:
            # Extract entity names from URIs for better LLM understanding
            entity_names = self._extract_entity_names(context_entities)
            
            if not entity_names:
                return pronoun
            
            # Create prompt for LLM
            prompt = self._create_prompt(text, entity_names, pronoun)
            
            # Call LLM API
            response = self._call_llm(prompt)
            
            # Parse response
            resolved_entity = self._parse_response(response, entity_names, pronoun)
            
            if resolved_entity:
                # Find the original URI for the resolved entity
                original_uri = self._find_original_uri(resolved_entity, context_entities)
                return original_uri if original_uri else pronoun
            
            return pronoun
            
        except Exception as e:
            print(f"LLM coref resolution failed: {e}")
            return pronoun
    
    def _extract_entity_names(self, entity_uris: List[str]) -> List[str]:
        """Extract readable entity names from URIs."""
        names = []
        for uri in entity_uris:
            if not uri:
                continue
            
            # Extract name from URI
            if 'dbpedia.org' in uri:
                name = uri.split('/')[-1].replace('_', ' ')
            elif 'wikipedia.org' in uri:
                name = uri.split('/')[-1].replace('_', ' ')
            else:
                name = uri.split('/')[-1].replace('_', ' ')
            
            if name and name not in names:
                names.append(name)
        
        return names
    
    def _create_prompt(self, text: str, entity_names: List[str], pronoun: str) -> str:
        """Create a prompt for LLM coreference resolution."""
        entities_str = ", ".join(entity_names)
        
        prompt = f"""You are a coreference resolution expert. Given a text and a list of entities, resolve the pronoun "{pronoun}" to the most likely entity.

Text: "{text}"

Available entities: {entities_str}

Rules:
1. Choose the entity that the pronoun "{pronoun}" most likely refers to
2. Consider grammatical context, proximity, and semantic coherence
3. If uncertain, choose the most recently mentioned suitable entity
4. Return only the entity name, nothing else

Resolved entity:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API based on provider."""
        if self.config.provider == 'openai':
            return self._call_openai(prompt)
        elif self.config.provider == 'anthropic':
            return self._call_anthropic(prompt)
        elif self.config.provider == 'ollama':
            return self._call_ollama(prompt)
        elif self.config.provider == 'custom':
            return self._call_custom(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        url = f"{self.config.base_url or 'https://api.openai.com/v1'}/chat/completions"
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content'].strip()
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        url = f"{self.config.base_url or 'https://api.anthropic.com/v1'}/messages"
        
        payload = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data['content'][0]['text'].strip()
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API (local)."""
        url = f"{self.config.base_url or 'http://localhost:11434'}/api/generate"
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }
        
        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data['response'].strip()
    
    def _call_custom(self, prompt: str) -> str:
        """Call custom API endpoint."""
        if not self.config.base_url:
            raise ValueError("Custom provider requires base_url")
        
        payload = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        response = self.session.post(self.config.base_url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        
        data = response.json()
        return data.get('response', data.get('text', '')).strip()
    
    def _parse_response(self, response: str, entity_names: List[str], pronoun: str) -> Optional[str]:
        """Parse LLM response to find the resolved entity."""
        response_lower = response.lower().strip()
        
        # Look for exact matches first
        for entity in entity_names:
            if entity.lower() in response_lower:
                return entity
        
        # Look for partial matches
        for entity in entity_names:
            entity_words = entity.lower().split()
            if any(word in response_lower for word in entity_words if len(word) > 2):
                return entity
        
        return None
    
    def _find_original_uri(self, entity_name: str, context_entities: List[str]) -> Optional[str]:
        """Find the original URI for a resolved entity name."""
        entity_lower = entity_name.lower()
        
        for uri in context_entities:
            if not uri:
                continue
            
            # Extract name from URI
            if 'dbpedia.org' in uri:
                uri_name = uri.split('/')[-1].replace('_', ' ').lower()
            elif 'wikipedia.org' in uri:
                uri_name = uri.split('/')[-1].replace('_', ' ').lower()
            else:
                uri_name = uri.split('/')[-1].replace('_', ' ').lower()
            
            if entity_lower == uri_name or entity_lower in uri_name or uri_name in entity_lower:
                return uri
        
        return None

def create_llm_resolver(provider: str = None, api_key: str = None, model: str = None) -> Optional[LLMCorefResolver]:
    """
    Create an LLM resolver with automatic configuration.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'ollama', 'custom')
        api_key: API key for the provider
        model: Model name to use
    
    Returns:
        Configured LLMCorefResolver or None if configuration fails
    """
    # Auto-detect provider if not specified
    if not provider:
        if os.getenv('OPENAI_API_KEY'):
            provider = 'openai'
            api_key = os.getenv('OPENAI_API_KEY')
        elif os.getenv('ANTHROPIC_API_KEY'):
            provider = 'anthropic'
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif os.getenv('OLLAMA_HOST'):
            provider = 'ollama'
        else:
            print("No LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_HOST environment variable.")
            return None
    
    # Set default model if not specified
    if not model:
        model_map = {
            'openai': 'gpt-3.5-turbo',
            'anthropic': 'claude-3-haiku-20240307',
            'ollama': 'llama2',
            'custom': 'custom-model'
        }
        model = model_map.get(provider, 'gpt-3.5-turbo')
    
    # Set base URL if needed
    base_url = None
    if provider == 'ollama':
        base_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    elif provider == 'custom':
        base_url = os.getenv('CUSTOM_LLM_URL')
    
    config = LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model
    )
    
    try:
        return LLMCorefResolver(config)
    except Exception as e:
        print(f"Failed to create LLM resolver: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test the resolver
    resolver = create_llm_resolver()
    
    if resolver:
        text = "Steve Jobs founded Apple. He was a visionary leader."
        context_entities = [
            "http://dbpedia.org/resource/Steve_Jobs",
            "http://dbpedia.org/resource/Apple_Inc."
        ]
        pronoun = "He"
        
        result = resolver.resolve_pronouns(text, context_entities, pronoun)
        print(f"Resolved '{pronoun}' to: {result}")
    else:
        print("LLM resolver not available")
