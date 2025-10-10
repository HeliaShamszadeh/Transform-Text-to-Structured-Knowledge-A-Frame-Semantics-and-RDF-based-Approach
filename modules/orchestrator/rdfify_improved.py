#!/usr/bin/env python3
"""
Improved RDF converter with enhanced pronoun resolution, entity linking, and frame completion.
This version is more robust and generic, not dependent on specific input patterns.
"""

import argparse
import json
import os
import re
import requests
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from pathlib import Path

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

class ImprovedDeepInfraCorefResolver:
    """Enhanced DeepInfra-based pronoun resolver with multiple strategies."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-2-70b-chat-hf"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepinfra.com/v1/openai/chat/completions"
        
    def resolve(self, pronoun: str, context_entities: List[Dict[str, str]], sentence: str, full_text: str = "") -> Optional[str]:
        """Enhanced pronoun resolution with LLM as default strategy."""
        
        # Strategy 1: LLM-based resolution (DEFAULT)
        resolved = self._llm_resolution(pronoun, context_entities, sentence, full_text)
        if resolved:
            return resolved
        
        # Strategy 2: Direct mention matching (fallback)
        resolved = self._direct_mention_matching(pronoun, context_entities)
        if resolved:
            return resolved
        
        # Strategy 3: Context-based heuristics (fallback)
        resolved = self._context_heuristics(pronoun, context_entities, sentence)
        if resolved:
            return resolved
        
        return None
    
    def _direct_mention_matching(self, pronoun: str, context_entities: List[Dict]) -> Optional[str]:
        """Direct matching based on pronoun type and entity characteristics."""
        if not context_entities:
            return None
        
        pronoun_lower = pronoun.lower()
        
        # Gender-based matching
        if pronoun_lower in ["he", "him", "his"]:
            for entity in context_entities:
                mention = entity.get("mention", "").strip()
                if mention and len(mention) > 2:
                    # Look for male indicators
                    if any(indicator in mention.lower() for indicator in ["mr", "sir", "dr", "prof", "king", "prince"]):
                        return mention
        elif pronoun_lower in ["she", "her"]:
            for entity in context_entities:
                mention = entity.get("mention", "").strip()
                if mention and len(mention) > 2:
                    # Look for female indicators
                    if any(indicator in mention.lower() for indicator in ["ms", "mrs", "miss", "dr", "prof", "queen", "princess"]):
                        return mention
        
        return None
    
    def _llm_resolution(self, pronoun: str, context_entities: List[Dict], sentence: str, full_text: str) -> Optional[str]:
        """LLM-based pronoun resolution with URI/name distinction."""
        if not self.api_key:
            print(f"    No API key available for LLM resolution")
            return None
        
        print(f"    Using DeepInfra LLM to resolve '{pronoun}'...")
        
        # Use all available entities for pronoun resolution
        person_entities = context_entities
        
        # Create context string with both mentions and URIs
        context_parts = []
        for entity in person_entities:
            mention = entity.get("mention", "")
            uri = entity.get("uri", "")
            if mention and uri:
                context_parts.append(f"{mention} -> {uri}")
        
        context_str = "Available entities: " + ", ".join(context_parts)
        
        system = (
            "You are a precise coreference resolver. Given a pronoun and context, determine what it refers to. "
            "Focus on PERSON entities (names of people). "
            "For pronouns like 'he', 'she', 'him', 'her', look for PERSON names in the context. "
            # "Ignore organizations, places, or other non-person entities. "
            "If the pronoun refers to a known person entity (has a URI), return the URI. "
            "If it refers to a person name/mention without a URI, return the name. "
            "Return 'NONE' if unclear. Return ONLY the URI or name, nothing else."
        )
        user = (
            f"Context: {context_str}\n"
            f"Sentence: {sentence}\n"
            f"Pronoun: {pronoun}\n\n"
            f"What does '{pronoun}' refer to? Return the URI if it's a known entity, or the name if it's just a mention, or NONE if unclear."
        )
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            text = data['choices'][0]['message']['content'].strip()
            
            if text.upper() == "NONE" or "unknown" in text.lower():
                return None
            
            # Check if it's a URI (starts with http)
            if text.startswith("http"):
                # Verify this URI exists in our context
                for entity in context_entities:
                    if entity.get("uri", "") == text:
                        return text
                return None
            
            # Check if it's a mention/name
            for entity in context_entities:
                if text.lower() in entity.get("mention", "").lower():
                    return entity.get("mention", "")
            
            # If not found in context, return as-is (might be a new name)
            return text
            
        except Exception as e:
            print(f"LLM resolution failed: {e}")
            return None
    
    def _context_heuristics(self, pronoun: str, context_entities: List[Dict], sentence: str) -> Optional[str]:
        """Context-based heuristics for pronoun resolution."""
        if not context_entities:
            return None
        
        sentence_lower = sentence.lower()
        pronoun_pos = sentence_lower.find(pronoun.lower())
        
        # Strategy 1: Look for entities mentioned before the pronoun in the same sentence
        entities_before_pronoun = []
        for entity in context_entities:
            mention = entity.get("mention", "")
            if mention and mention.lower() in sentence_lower:
                mention_pos = sentence_lower.find(mention.lower())
                if mention_pos < pronoun_pos:
                    entities_before_pronoun.append(entity)
        
        # If we found entities before the pronoun, use the most recent one
        if entities_before_pronoun:
            return entities_before_pronoun[-1].get("mention", "")
        
        # Strategy 2: Look for entities mentioned in the same sentence (anywhere)
        entities_in_sentence = []
        for entity in context_entities:
            mention = entity.get("mention", "")
            if mention and mention.lower() in sentence_lower:
                entities_in_sentence.append(entity)
        
        if entities_in_sentence:
            # For person pronouns, prefer multi-word names but include single words
            if pronoun.lower() in ["he", "him", "his", "she", "her"]:
                # For person pronouns, prefer multi-word names but include single words
                person_entities = [e for e in entities_in_sentence if len(e.get("mention", "").split()) >= 2]
                if person_entities:
                    return person_entities[0].get("mention", "")
                # Fall back to any entity in sentence
                return entities_in_sentence[0].get("mention", "")
            else:
                # For other pronouns, use any entity in sentence
                return entities_in_sentence[0].get("mention", "")
        
        # Strategy 2.5: If no entities in current sentence, look for person names in all context
        if pronoun.lower() in ["he", "him", "his", "she", "her"]:
            person_entities = [e for e in context_entities if len(e.get("mention", "").split()) >= 2]
            if person_entities:
                return person_entities[0].get("mention", "")
        
        # Strategy 3: Most recent high-confidence entity from all context
        high_conf_entities = [e for e in context_entities if e.get("confidence", 0) > 0.5]
        if high_conf_entities:
            return high_conf_entities[0].get("mention", "")
        
        # Strategy 4: Longest entity name (more specific)
        if context_entities:
            longest_entity = max(context_entities, key=lambda x: len(x.get("mention", "")))
            return longest_entity.get("mention", "")
        
        return None

class RDFConvertor:
    """Enhanced RDF converter with improved entity linking and frame completion."""
    
    def __init__(self, deepinfra_api_key: str, deepinfra_model: str = "meta-llama/Llama-2-70b-chat-hf"):
        self.resolver = ImprovedDeepInfraCorefResolver(deepinfra_api_key, deepinfra_model)
        self.graph = Graph()
        self.entity_index: Dict[str, str] = {}
        self.context_entities: List[Dict[str, str]] = []
        self.emitted_triples: Set[Tuple[str, str, str]] = set()  # Track emitted triples to prevent duplicates
        
        # Initialize namespaces and predicates
        self._setup_basic_structure()
        self.stats = {"frames": 0, "triples": 0, "entities_resolved": 0, "literals": 0, "pronouns": 0}
        self._current_sentence = ""
        self._full_text = ""

    def _setup_basic_structure(self):
        """Setup basic RDF structure."""
        FRAME = Namespace("http://example.org/frame/")
        ENTITY = Namespace("http://example.org/entity/")
        BASE = Namespace("http://example.org/")
        
        self.graph.bind("frame", FRAME)
        self.graph.bind("entity", ENTITY)
        self.graph.bind("", BASE)
        
        # Add frame-specific namespaces for cleaner output
        self.frame_namespaces = {}
        
        # Add predicates
        self.P = {}
        predicates = [
            'has_agent', 'has_theme', 'has_time', 'has_location', 'has_person',
            'has_entity', 'has_item', 'has_category', 'has_competitor', 'has_prize',
            'has_competition', 'has_leader', 'has_creator', 'has_speaker', 'has_authority'
        ]
        
        for pred in predicates:
            self.P[pred] = URIRef(f"http://example.org/{pred}")

    def convert(self, frames_file: str, entities_file: str, outfile: str):
        """Convert frames and entities to RDF with improvements."""
        print("Loading data...")
        
        # Load frames and entities
        with open(frames_file, 'r', encoding='utf-8') as f:
            frames_data = json.load(f)
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        
        # Extract full text for context
        self._full_text = " ".join([s.get("text", "") for s in frames_data.get("sentences", [])])
        
        # Build entity index
        self._build_entity_index(entities_data)
        
        # Process sentences and frames
        sentences = frames_data.get("sentences", [])
        all_entities = entities_data.get("entities", [])
        
        # Build a cumulative context that includes entities from all previous sentences
        cumulative_context = []
        
        for sentence_data in sentences:
            sentence_text = sentence_data.get("text", "")
            self._current_sentence = sentence_text
            
            # Add entities from current sentence to cumulative context
            self._add_entities_from_sentence(sentence_text, all_entities, cumulative_context)
            
            # Setup context entities using cumulative context
            self.context_entities = cumulative_context.copy()
            
            # Process frames
            frames = sentence_data.get("frames", [])
            for frame in frames:
                self._process_frame_improved(frame, sentence_text)
        
        # Save RDF in custom format
        self._save_custom_rdf(outfile)
        print(f"RDF saved to {outfile}")
        print(f"Stats: {self.stats}")

    def _build_entity_index(self, entities_data: Dict):
        """Build entity index for fast lookup."""
        entities = entities_data.get("entities", [])
        for entity in entities:
            mention = entity.get("mention", "").strip()
            uri = entity.get("uri", "")
            if mention and uri:
                self.entity_index[mention.lower()] = uri

    def _setup_context_entities_for_sentence(self, sentence_text: str, all_entities: List[Dict]):
        """Setup context entities for current sentence."""
        self.context_entities = []
        sentence_lower = sentence_text.lower()
        
        for entity in all_entities:
            mention = entity.get("mention", "").strip()
            if mention:
                # Only include entities that are mentioned in this sentence or are likely to be referenced
                if (mention.lower() in sentence_lower or 
                    entity.get("confidence", 0.0) > 0.7 or  # High confidence entities
                    any(word in mention.lower() for word in sentence_lower.split())):  # Partial matches
                    self.context_entities.append({
                        "mention": mention,
                        "uri": entity.get("uri", ""),
                        "confidence": entity.get("confidence", 0.0)
                    })
        
        # Sort by confidence
        self.context_entities.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

    def _add_entities_from_sentence(self, sentence_text: str, all_entities: List[Dict], cumulative_context: List[Dict]):
        """Add entities from current sentence to cumulative context."""
        sentence_lower = sentence_text.lower()
        
        for entity in all_entities:
            mention = entity.get("mention", "").strip()
            if mention and mention.lower() in sentence_lower:
                # Check if entity is already in cumulative context
                entity_already_exists = any(
                    existing.get("mention", "").lower() == mention.lower() 
                    for existing in cumulative_context
                )
                
                if not entity_already_exists:
                    cumulative_context.append({
                        "mention": mention,
                        "uri": entity.get("uri", ""),
                        "confidence": entity.get("confidence", 0.0)
                    })

    def _process_frame_improved(self, frame: Dict[str, Any], sentence_text: str):
        """Process frame with improvements - now generates entity-centric triples."""
        frame_name = frame.get("name", "Unknown")
        elements = frame.get("elements", [])
        
        if not elements:
            return
        
        # Improve frame completion
        frame = self._improve_frame_completion(frame, sentence_text)
        elements = frame.get("elements", [])
        
        # 1. PRE-RESOLVE pronouns in frame elements to get better context
        resolved_elements = self._pre_resolve_pronouns_in_elements(elements)
        
        # 2. Identify the main entity this frame is about using resolved elements
        main_entity = self._identify_main_entity(sentence_text, self.context_entities, frame_name, resolved_elements)
        if not main_entity:
            # Fallback: create a simple entity from the first meaningful element (use resolved elements)
            if resolved_elements:
                first_element = resolved_elements[0]
                element_text = first_element.get('text', '').strip()
                if element_text and len(element_text) > 1:
                    # Check if this element text is a subset of any known entity
                    matched_entity = self._find_entity_by_subset(element_text)
                    if matched_entity:
                        main_entity = matched_entity
                        # Handle Unicode characters safely
                        try:
                            print(f"  Using matched entity: {matched_entity['mention']} -> {matched_entity['uri']}")
                        except UnicodeEncodeError:
                            mention_safe = matched_entity['mention'].encode('ascii', 'replace').decode('ascii')
                            uri_safe = matched_entity['uri'].encode('ascii', 'replace').decode('ascii')
                            print(f"  Using matched entity: {mention_safe} -> {uri_safe}")
                    else:
                        main_entity = {
                            'mention': element_text,
                            'uri': '',
                            'confidence': 0.5
                        }
                        # Handle Unicode characters safely
                        try:
                            print(f"  Using fallback entity: {element_text}")
                        except UnicodeEncodeError:
                            text_safe = element_text.encode('ascii', 'replace').decode('ascii')
                            print(f"  Using fallback entity: {text_safe}")
                else:
                    print(f"  Warning: No suitable entity found for frame {frame_name}")
                    return
            else:
                print(f"  Warning: No elements found for frame {frame_name}")
                return
        
        # 2. Use URI if available, otherwise use text mention
        main_entity_uri = main_entity.get('uri', '')
        main_entity_mention = main_entity['mention']
        
        # Decide whether to use URI or text mention as subject
        if main_entity_uri and main_entity_uri.startswith('http'):
            # Use URI for known entities
            main_entity_subject = main_entity_uri
            # Handle Unicode characters safely
            try:
                print(f"    Using URI as subject: {main_entity_uri}")
            except UnicodeEncodeError:
                uri_safe = main_entity_uri.encode('ascii', 'replace').decode('ascii')
                print(f"    Using URI as subject: {uri_safe}")
        else:
            # Use text mention for unknown entities
            main_entity_subject = main_entity_mention
            # Handle Unicode characters safely
            try:
                print(f"    Using text as subject: {main_entity_mention}")
            except UnicodeEncodeError:
                mention_safe = main_entity_mention.encode('ascii', 'replace').decode('ascii')
                print(f"    Using text as subject: {mention_safe}")
        
        self.stats["entities_resolved"] += 1
        
        # 3. Process frame elements as properties of the main entity (use resolved elements)
        for element in resolved_elements:
            role = element.get("name", "")
            text = element.get("text", "")
            
            if not text or text.strip() in ['', 'to', 'from', 'about', 'in', 'on', 'at']:
                continue
            
            # Get predicate with frame name prefix
            base_pred_name = self._get_predicate_for_role(role, frame_name)
            pred_name = f"{frame_name}:{base_pred_name}"
            
            # Create frame-specific namespace if not exists
            if frame_name not in self.frame_namespaces:
                frame_ns = Namespace(f"http://example.org/frame/{frame_name}#")
                self.graph.bind(frame_name.lower(), frame_ns)
                self.frame_namespaces[frame_name] = frame_ns
            
            pred = URIRef(f"http://example.org/frame/{frame_name}#{base_pred_name}")
            
            # Emit triple: main_entity_subject predicate object
            self._emit_smart_subject(main_entity_subject, pred, text)
            
            print(f"    Triple: {main_entity_subject} {pred_name} {text}")
        
        # 4. Add frame as context (optional) - using text mention
        frame_uri = URIRef(f"http://example.org/frame/{frame_name}")
        main_entity_text_uri = URIRef(f"http://example.org/entity/{main_entity_mention.replace(' ', '_')}")
        self.graph.add((main_entity_text_uri, URIRef("http://example.org/participates_in"), frame_uri))
        self.graph.add((frame_uri, RDF.type, URIRef(f"http://example.org/frame/{frame_name}")))
        self.stats["frames"] += 1

    def _improve_frame_completion(self, frame: Dict[str, Any], sentence_text: str) -> Dict[str, Any]:
        """Improve frame completion by adding missing elements."""
        elements = frame.get("elements", [])
        
        # Add missing temporal information
        has_time = any(elem.get("name") == "Time" for elem in elements)
        if not has_time:
            time_match = re.search(r'\b(19|20)\d{2}\b', sentence_text)
            if time_match:
                elements.append({
                    "name": "Time",
                    "text": time_match.group()
                })
        
        # Add missing location information
        has_location = any(elem.get("name") in ["Place", "Location"] for elem in elements)
        if not has_location:
            location_patterns = [
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
            ]
            for pattern in location_patterns:
                match = re.search(pattern, sentence_text)
                if match:
                    elements.append({
                        "name": "Place",
                        "text": match.group(1)
                    })
                    break
        
        return frame

    def _pre_resolve_pronouns_in_elements(self, elements: List[Dict]) -> List[Dict]:
        """Pre-resolve pronouns in frame elements to improve main entity identification."""
        resolved_elements = []
        pronouns = {
            # Subject pronouns
            "he", "she", "it", "they", "i", "we", "you",
            # Object pronouns  
            "him", "her", "it", "them", "me", "us", "you",
            # Possessive pronouns
            "his", "her", "its", "their", "my", "our", "your",
            # Possessive determiners (same as possessive pronouns but used differently)
            "hers", "theirs", "mine", "ours", "yours"
        }
        
        for element in elements:
            element_text = element.get('text', '').strip()
            role = element.get('name', '')
            
            # Check if this element contains a pronoun
            if element_text.lower() in pronouns:
                print(f"    Pre-resolving pronoun '{element_text}' in {role} element...")
                resolved = self.resolver.resolve(element_text, self.context_entities, self._current_sentence, self._full_text)
                
                if resolved:
                    # Create a new element with resolved text
                    resolved_element = element.copy()
                    resolved_element['text'] = resolved
                    resolved_element['original_text'] = element_text  # Keep original for reference
                    resolved_elements.append(resolved_element)
                    print(f"    Resolved '{element_text}' to '{resolved}'")
                else:
                    # Keep original if resolution fails
                    resolved_elements.append(element)
                    print(f"    Could not resolve '{element_text}', keeping original")
            else:
                # Not a pronoun, keep as is
                resolved_elements.append(element)
        
        return resolved_elements

    def _identify_main_entity(self, sentence_text: str, sentence_entities: List[Dict], frame_name: str = None, frame_elements: List[Dict] = None) -> Optional[Dict]:
        """Enhanced frame-aware subject identification with multiple strategies."""
        if not sentence_entities:
            return None
        
        print(f"    Identifying main entity for frame '{frame_name}' with {len(sentence_entities)} entities")
        
        # Strategy 1: Frame-specific subject identification
        frame_aware_entity = self._identify_frame_aware_subject(sentence_text, sentence_entities, frame_name, frame_elements)
        if frame_aware_entity:
            print(f"    Frame-aware selection: {frame_aware_entity.get('mention')} ({frame_aware_entity.get('entity', 'unknown')})")
            return frame_aware_entity
        
        # Strategy 2: Person entity prioritization for person-centric frames
        if self._is_person_centric_frame(frame_name):
            person_entity = self._find_best_person_entity(sentence_entities, frame_elements)
            if person_entity:
                print(f"    Person-centric selection: {person_entity.get('mention')}")
                return person_entity
        
        # Strategy 3: Semantic analysis of frame elements (improved)
        if frame_elements:
            frame_element_entity = self._identify_from_frame_elements(sentence_entities, frame_elements, frame_name)
            if frame_element_entity:
                print(f"    Frame element selection: {frame_element_entity.get('mention')}")
                return frame_element_entity
        
        # Strategy 4: Early sentence appearance
        early_entity = self._find_early_sentence_entity(sentence_text, sentence_entities)
        if early_entity:
            print(f"    Early sentence selection: {early_entity.get('mention')}")
            return early_entity
        
        # Strategy 5: Highest confidence entity
        confidence_entity = self._find_highest_confidence_entity(sentence_entities)
        if confidence_entity:
            print(f"    Confidence-based selection: {confidence_entity.get('mention')}")
            return confidence_entity
        
        # Fallback: First entity
        print(f"    Fallback selection: {sentence_entities[0].get('mention')}")
        return sentence_entities[0]
    
    def _calculate_semantic_relevance(self, role: str, element_text: str, mention: str, entity: Dict) -> float:
        """Calculate semantic relevance score for an entity in a frame element."""
        score = 0.0
        
        # Base score from entity confidence
        score += entity.get('confidence', 0) * 0.3
        
        # Role-based scoring (semantic roles that typically indicate main subjects)
        main_subject_roles = {
            'Agent': 1.0, 'Author': 1.0, 'Child': 0.9, 'Entity': 0.8, 
            'Protagonist': 1.0, 'Ego': 0.9, 'Partner_1': 0.8, 'Cognizer': 0.9,
            'Speaker': 0.9, 'Creator': 0.9, 'Performer': 0.9
        }
        score += main_subject_roles.get(role, 0.5) * 0.4
        
        # Text analysis - prefer entities that are the main focus of the element
        element_lower = element_text.lower()
        mention_lower = mention.lower()
        
        # If the entity mention is the main part of the element text
        if mention_lower == element_lower:
            score += 0.3  # Exact match
        elif len(mention) > 5 and mention_lower in element_lower:
            # Entity is a significant part of the element
            coverage = len(mention) / len(element_text)
            score += coverage * 0.2
        
        # Prefer person entities for biographical/creative frames
        if 'person' in entity.get('entity', '').lower():
            score += 0.1
        
        # Prefer multi-word names (more specific)
        if len(mention.split()) >= 2:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def _find_entity_by_subset(self, element_text: str) -> Optional[Dict]:
        """Find an entity where the element text is a subset of the entity mention."""
        element_lower = element_text.lower().strip()
        
        # Skip very short words that could cause false matches
        if len(element_lower) < 3:
            return None
        
        for entity in self.context_entities:
            mention = entity.get('mention', '').strip()
            uri = entity.get('uri', '')
            
            if mention and uri and uri.startswith('http'):
                mention_lower = mention.lower()
                
                # Skip very short entity mentions
                if len(mention_lower) < 3:
                    continue
                
                # Check if element text is contained in the entity mention
                if element_lower in mention_lower:
                    return entity
                
                # Check if entity mention is contained in element text (reverse subset)
                if mention_lower in element_lower:
                    return entity
        
        return None

    def _get_predicate_for_role(self, role: str, frame_name: str) -> str:
        """Get predicate name for a semantic role with comprehensive frame support."""
        # Import comprehensive mappings
        import sys
        import os
        # Add the project root to path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(os.path.join(project_root, 'evaluation_pipeline'))
        from comprehensive_frame_mappings import get_predicate_for_role
        
        return get_predicate_for_role(role, frame_name)

    def _emit_improved(self, subj: URIRef, pred: URIRef, value: str):
        """Emit triple with improved entity resolution."""
        if not value:
            return
        
        # Clean value
        cleaned_value = self._clean_value(value)
        if not cleaned_value:
            return
        
        # Check for pronouns
        pronouns = {
            # Subject pronouns
            "he", "she", "it", "they", "i", "we", "you",
            # Object pronouns  
            "him", "her", "it", "them", "me", "us", "you",
            # Possessive pronouns
            "his", "her", "its", "their", "my", "our", "your",
            # Possessive determiners
            "hers", "theirs", "mine", "ours", "yours"
        }
        if cleaned_value.lower() in pronouns:
            print(f"    Resolving pronoun '{cleaned_value}' using LLM...")
            resolved = self.resolver.resolve(cleaned_value, self.context_entities, self._current_sentence, self._full_text)
            if resolved:
                # Check if resolved is a URI or a name
                if resolved.startswith("http"):
                    obj = URIRef(resolved)
                    self.stats["entities_resolved"] += 1
                    self.stats["pronouns"] += 1
                else:
                    # It's a name, try to link it to an entity
                    is_entity, result = self._improved_entity_linking(resolved)
                    if is_entity:
                        obj = URIRef(result)
                        self.stats["entities_resolved"] += 1
                    else:
                        obj = self._create_better_literal(result)
                        self.stats["literals"] += 1
                    self.stats["pronouns"] += 1
                self.graph.add((subj, pred, obj))
                self.stats["triples"] += 1
                return
        
        # Try entity linking for non-pronouns
        is_entity, result = self._improved_entity_linking(cleaned_value)
        if is_entity:
            obj = URIRef(result)
            self.stats["entities_resolved"] += 1
        else:
            obj = self._create_better_literal(result)
            self.stats["literals"] += 1
        
        self.graph.add((subj, pred, obj))
        self.stats["triples"] += 1

    def _emit_improved_text(self, subject_text: str, pred: URIRef, value: str):
        """Emit triple using text mention as subject."""
        if not value or not subject_text:
            return
        
        # Clean value
        cleaned_value = self._clean_value(value)
        if not cleaned_value:
            return
        
        # Check for pronouns in object
        pronouns = {
            # Subject pronouns
            "he", "she", "it", "they", "i", "we", "you",
            # Object pronouns  
            "him", "her", "it", "them", "me", "us", "you",
            # Possessive pronouns
            "his", "her", "its", "their", "my", "our", "your",
            # Possessive determiners
            "hers", "theirs", "mine", "ours", "yours"
        }
        if cleaned_value.lower() in pronouns:
            print(f"    Resolving pronoun '{cleaned_value}' using LLM...")
            resolved = self.resolver.resolve(cleaned_value, self.context_entities, self._current_sentence, self._full_text)
            if resolved:
                # Check if resolved is a URI or a name
                if resolved.startswith("http"):
                    # It's a URI, use it directly
                    obj = URIRef(resolved)
                else:
                    # It's a name, create a literal
                    obj = Literal(resolved)
                print(f"    Resolved to: {resolved}")
            else:
                print(f"    Could not resolve pronoun '{cleaned_value}', skipping")
                return
        else:
            # Not a pronoun, create literal
            obj = Literal(cleaned_value)
        
        # Create subject as text literal
        subj = Literal(subject_text)
        
        # Check for duplicates
        triple_key = (str(subj), str(pred), str(obj))
        if triple_key in self.emitted_triples:
            return
        
        # Add to graph
        self.graph.add((subj, pred, obj))
        self.emitted_triples.add(triple_key)
        self.stats["triples"] += 1

    def _emit_smart_subject(self, subject: str, pred: URIRef, value: str):
        """Emit triple with smart subject selection (URI or text)."""
        if not value or not subject:
            return
        
        # Clean value
        cleaned_value = self._clean_value(value)
        if not cleaned_value:
            return
        
        # Check for pronouns in object
        pronouns = {
            # Subject pronouns
            "he", "she", "it", "they", "i", "we", "you",
            # Object pronouns  
            "him", "her", "it", "them", "me", "us", "you",
            # Possessive pronouns
            "his", "her", "its", "their", "my", "our", "your",
            # Possessive determiners
            "hers", "theirs", "mine", "ours", "yours"
        }
        if cleaned_value.lower() in pronouns:
            print(f"    Resolving pronoun '{cleaned_value}' using LLM...")
            resolved = self.resolver.resolve(cleaned_value, self.context_entities, self._current_sentence, self._full_text)
            if resolved:
                # Check if resolved is a URI or a name
                if resolved.startswith("http"):
                    # It's a URI, use it directly
                    obj = URIRef(resolved)
                else:
                    # It's a name, create a literal
                    obj = Literal(resolved)
                print(f"    Resolved to: {resolved}")
            else:
                print(f"    Could not resolve pronoun '{cleaned_value}', skipping")
                return
        else:
            # Check if the object text contains a known entity
            obj = self._smart_object_resolution(cleaned_value)
        
        # Create subject - URI if it starts with http, otherwise literal
        if subject.startswith("http"):
            subj = URIRef(subject)
        else:
            subj = Literal(subject)
        
        # Check for self-referential triples
        subj_str = str(subj)
        obj_str = str(obj)
        if subj_str == obj_str:
            print(f"    Skipping self-referential triple: {subj_str} -> {obj_str}")
            return
        
        # Check for duplicates
        triple_key = (str(subj), str(pred), str(obj))
        if triple_key in self.emitted_triples:
            return
        
        # Add to graph
        self.graph.add((subj, pred, obj))
        self.emitted_triples.add(triple_key)
        self.stats["triples"] += 1

    def _smart_object_resolution(self, text: str) -> Union[URIRef, Literal]:
        """Smart object resolution - check if text contains known entities."""
        # Skip very short or generic texts that shouldn't be converted to URIs
        if len(text.strip()) < 3:
            return Literal(text)
        
        # Skip possessive and descriptive texts that shouldn't be URIs
        skip_patterns = [
            'her ', 'his ', 'their ', 'its ', 'our ',  # Possessive
            'the ', 'a ', 'an ',  # Articles
            'this ', 'that ', 'these ', 'those ',  # Demonstratives
            'novels', 'stories', 'plays', 'books',  # Generic works
            'marriage', 'child', 'mother', 'father',  # Relationships
            'knowledge', 'headlines', 'copies',  # Abstract concepts
            'writer', 'author', 'detective',  # Generic roles
            'world', 'hospital', 'family',  # Generic places/concepts
        ]
        
        text_lower = text.lower().strip()
        if any(pattern in text_lower for pattern in skip_patterns):
            return Literal(text)
        
        # Check if any known entity mention is a subset of the text
        for entity in self.context_entities:
            mention = entity.get('mention', '').strip()
            uri = entity.get('uri', '')
            
            if mention and uri and uri.startswith('http'):
                # Only convert if the text is exactly the entity mention (exact match only)
                if mention.lower() == text_lower:
                    print(f"    Found exact entity match '{mention}' in text '{text}' -> using URI {uri}")
                    return URIRef(uri)
        
        # If no exact entity found, return as literal
        return Literal(text)

    def _improved_entity_linking(self, value: str) -> Tuple[bool, str]:
        """Enhanced entity linking with improved matching logic."""
        cleaned_value = value.strip().lower()
        
        # Direct exact match (highest confidence)
        if cleaned_value in self.entity_index:
            return True, self.entity_index[cleaned_value]
        
        # Calculate similarity scores for all entities
        best_match = None
        best_score = 0.0
        
        for entity_mention, uri in self.entity_index.items():
            score = self._calculate_similarity(cleaned_value, entity_mention)
            if score > best_score and score > 0.6:  # Higher threshold for better precision
                best_score = score
                best_match = uri
        
        if best_match:
            return True, best_match
        
        return False, value
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # Boost score for exact substring matches
        if text1 in text2 or text2 in text1:
            jaccard = min(1.0, jaccard + 0.3)
        
        # Boost score for longer matches
        if len(text1) > 3 and len(text2) > 3:
            jaccard = min(1.0, jaccard + 0.1)
        
        return jaccard

    def _clean_value(self, value: str) -> str:
        """Clean value by removing leading prepositions."""
        if not value:
            return ""
        
        cleaned = value.strip()
        prepositions = ['to', 'from', 'about', 'in', 'on', 'at', 'with', 'by', 'for', 'of', 'the', 'a', 'an']
        
        words = cleaned.split()
        while words and words[0].lower() in prepositions:
            words.pop(0)
        
        result = ' '.join(words).strip()
        return result if result and result.lower() not in prepositions else ""

    def _create_better_literal(self, value: str) -> Literal:
        """Create better typed literals."""
        v = value.strip()
        
        # Time patterns
        if re.fullmatch(r'\d{4}', v):
            return Literal(v, datatype=XSD.gYear)
        elif re.fullmatch(r'\d{4}-\d{2}-\d{2}', v):
            return Literal(v, datatype=XSD.date)
        
        # Number patterns
        if re.fullmatch(r'\d+', v):
            return Literal(int(v), datatype=XSD.integer)
        elif re.fullmatch(r'\d+\.\d+', v):
            return Literal(float(v), datatype=XSD.decimal)
        
        # Boolean patterns
        if v.lower() in ['true', 'false']:
            return Literal(v.lower() == 'true', datatype=XSD.boolean)
        
        return Literal(v)

    def _save_custom_rdf(self, outfile: str):
        """Save RDF in custom format with each triple on its own line."""
        with open(outfile, 'w', encoding='utf-8') as f:
            # Write header
            f.write("# RDF Triples in custom format\n")
            f.write("# Format: subject predicate object\n\n")
            
            # Get all triples from the graph
            for subj, pred, obj in self.graph:
                # Skip type declarations and frame definitions
                if str(pred) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type":
                    continue
                if str(subj).startswith("http://example.org/frame/") and str(obj).startswith("http://example.org/frame/"):
                    continue
                if str(pred) == "http://example.org/participates_in":
                    continue
                
                # Format the triple
                subj_str = str(subj)
                pred_str = str(pred)
                obj_str = str(obj)
                
                # Clean up the predicate to show frame:predicate format
                if "#" in pred_str:
                    frame_part = pred_str.split("/")[-1].split("#")[0]
                    pred_part = pred_str.split("#")[-1]
                    pred_str = f"{frame_part}:{pred_part}"
                
                # Clean up object formatting
                if obj_str.startswith('"') and obj_str.endswith('"'):
                    # Remove quotes and datatype info for literals
                    obj_str = obj_str.split('"')[1]
                elif obj_str.startswith('<') and obj_str.endswith('>'):
                    # Keep URIs as is
                    pass
                
                # Write the triple
                f.write(f"{subj_str} {pred_str} {obj_str}\n")

    def _identify_frame_aware_subject(self, sentence_text: str, sentence_entities: List[Dict], frame_name: str, frame_elements: List[Dict]) -> Optional[Dict]:
        """Enhanced frame-aware subject identification using comprehensive frame mappings."""
        if not frame_elements or not sentence_entities:
            return None
        
        print(f"    Frame-aware analysis for '{frame_name}' with {len(frame_elements)} elements")
        
        # Get frame-specific subject roles from comprehensive mappings
        subject_roles = self._get_subject_roles_for_frame(frame_name)
        if not subject_roles:
            return None
        
        # Find entities that match the preferred subject roles
        best_entity = None
        best_score = 0.0
        
        for element in frame_elements:
            role = element.get('name', '')
            element_text = element.get('text', '')
            
            if role in subject_roles:
                print(f"    Found subject role '{role}' with text: '{element_text}'")
                
                # Find the best matching entity for this element
                matching_entity = self._find_best_entity_for_element(element_text, sentence_entities)
                if matching_entity:
                    # Calculate score based on role priority and entity quality
                    role_priority = subject_roles[role]
                    entity_quality = matching_entity.get('confidence', 0)
                    score = role_priority * 0.7 + entity_quality * 0.3
                    
                    print(f"    Entity '{matching_entity.get('mention')}' score: {score:.2f} (role: {role_priority}, quality: {entity_quality})")
                    
                    if score > best_score:
                        best_score = score
                        best_entity = matching_entity
        
        return best_entity

    def _get_subject_roles_for_frame(self, frame_name: str) -> Dict[str, float]:
        """Get subject roles for a frame with priority scores."""
        # Import comprehensive mappings
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append(os.path.join(project_root, 'evaluation_pipeline'))
        
        try:
            from comprehensive_frame_mappings import get_comprehensive_frame_mappings
            frame_specific, generic_mapping = get_comprehensive_frame_mappings()
            
            if frame_name in frame_specific:
                frame_mapping = frame_specific[frame_name]
                # Return roles that typically indicate subjects with priority scores
                subject_roles = {}
                for role, predicate in frame_mapping.items():
                    if predicate in ['has_person', 'has_agent', 'has_author', 'has_creator', 'has_achiever', 
                                   'has_competitor', 'has_leader', 'has_speaker', 'has_authority', 'has_governor',
                                   'has_traveler', 'has_arriver', 'has_employee', 'has_student', 'has_recipient',
                                   'has_judge', 'has_evaluator', 'has_assessor', 'has_classifier', 'has_typer',
                                   'has_perceiver', 'has_seer', 'has_hearer', 'has_thinker', 'has_believer',
                                   'has_knower', 'has_experiencer', 'has_lover', 'has_hater', 'has_hoper',
                                   'has_owner', 'has_possessor', 'has_user', 'has_consumer', 'has_buyer',
                                   'has_seller', 'has_donor', 'has_manufacturer', 'has_producer']:
                        subject_roles[role] = 1.0  # High priority for person/agent roles
                    elif predicate in ['has_theme', 'has_item', 'has_category']:
                        subject_roles[role] = 0.5  # Medium priority for theme roles
                    else:
                        subject_roles[role] = 0.3  # Lower priority for other roles
                
                return subject_roles
        except ImportError:
            print(f"    Warning: Could not import comprehensive mappings")
        
        # Fallback: generic subject roles
        return {
            'Agent': 1.0, 'Author': 1.0, 'Child': 1.0, 'Person': 1.0, 'Entity': 0.8,
            'Protagonist': 1.0, 'Ego': 1.0, 'Partner_1': 0.8, 'Cognizer': 0.9,
            'Speaker': 0.9, 'Creator': 0.9, 'Performer': 0.9, 'Theme': 0.5
        }

    def _find_best_entity_for_element(self, element_text: str, sentence_entities: List[Dict]) -> Optional[Dict]:
        """Find the best matching entity for a frame element."""
        if not element_text or not sentence_entities:
            return None
        
        element_lower = element_text.lower().strip()
        best_entity = None
        best_score = 0.0
        
        for entity in sentence_entities:
            mention = entity.get('mention', '').strip()
            if not mention:
                continue
            
            mention_lower = mention.lower()
            
            # Calculate matching score
            score = 0.0
            
            # Exact match gets highest score
            if mention_lower == element_lower:
                score = 1.0
            # Partial match gets medium score
            elif mention_lower in element_lower or element_lower in mention_lower:
                coverage = min(len(mention), len(element_text)) / max(len(mention), len(element_text))
                score = coverage * 0.8
            # Word overlap gets lower score
            elif any(word in element_lower for word in mention_lower.split() if len(word) > 2):
                score = 0.3
            
            # Boost score for entities with URIs (more reliable)
            if entity.get('uri') and entity['uri'].startswith('http'):
                score *= 1.2
            
            # Boost score for higher confidence
            confidence = entity.get('confidence', 0)
            score += confidence * 0.1
            
            if score > best_score:
                best_score = score
                best_entity = entity
        
        return best_entity if best_score > 0.2 else None

    def _is_person_centric_frame(self, frame_name: str) -> bool:
        """Check if a frame is person-centric (typically has a person as main subject)."""
        person_centric_frames = {
            'Being_born', 'Death', 'Marriage', 'Divorce', 'Education', 'Employment',
            'Retirement', 'Win_prize', 'Award', 'Achievement', 'Leadership', 'Authority',
            'Travel', 'Arrival', 'Departure', 'Communication', 'Speaking', 'Writing',
            'Reading', 'Meeting', 'Social_event', 'Friendship', 'Relationship',
            'Creation', 'Production', 'Consumption', 'Use', 'Purchase', 'Sale',
            'Possession', 'Ownership', 'Transfer', 'Perception', 'Seeing', 'Hearing',
            'Thinking', 'Belief', 'Knowledge', 'Emotion', 'Love', 'Hate', 'Fear',
            'Hope', 'Judgment', 'Evaluation', 'Assessment', 'Categorization',
            'Classification', 'Typing'
        }
        return frame_name in person_centric_frames

    def _find_best_person_entity(self, sentence_entities: List[Dict], frame_elements: List[Dict]) -> Optional[Dict]:
        """Find the best person entity for person-centric frames."""
        if not sentence_entities:
            return None
        
        # Prioritize entities that are clearly people
        person_entities = []
        for entity in sentence_entities:
            mention = entity.get('mention', '')
            entity_type = entity.get('entity', '')
            
            # Check if it's likely a person based on various indicators
            is_person = (
                'person' in entity_type.lower() or
                any(name_indicator in mention.lower() for name_indicator in [
                    'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sir', 'dame', 'lord', 'lady'
                ]) or
                # Check if it's a proper name (capitalized, multiple words)
                (len(mention.split()) >= 2 and mention[0].isupper() and all(word[0].isupper() for word in mention.split()))
            )
            
            if is_person:
                person_entities.append(entity)
        
        if person_entities:
            # Return the highest confidence person entity
            return max(person_entities, key=lambda e: e.get('confidence', 0))
        
        # Fallback: return highest confidence entity
        return max(sentence_entities, key=lambda e: e.get('confidence', 0))

    def _identify_from_frame_elements(self, sentence_entities: List[Dict], frame_elements: List[Dict], frame_name: str) -> Optional[Dict]:
        """Identify subject from frame elements using semantic analysis."""
        if not frame_elements or not sentence_entities:
            return None
        
        best_entity = None
        best_score = 0.0
        
        for element in frame_elements:
            role = element.get('name', '')
            element_text = element.get('text', '')
            
            # Calculate semantic relevance for each entity
            for entity in sentence_entities:
                score = self._calculate_semantic_relevance(role, element_text, entity.get('mention', ''), entity)
                if score > best_score:
                    best_score = score
                    best_entity = entity
        
        return best_entity if best_score > 0.3 else None

    def _find_early_sentence_entity(self, sentence_text: str, sentence_entities: List[Dict]) -> Optional[Dict]:
        """Find entity that appears early in the sentence."""
        if not sentence_entities:
            return None
        
        sentence_lower = sentence_text.lower()
        best_entity = None
        earliest_position = float('inf')
        
        for entity in sentence_entities:
            mention = entity.get('mention', '')
            if mention:
                position = sentence_lower.find(mention.lower())
                if position != -1 and position < earliest_position:
                    earliest_position = position
                    best_entity = entity
        
        return best_entity

    def _find_highest_confidence_entity(self, sentence_entities: List[Dict]) -> Optional[Dict]:
        """Find entity with highest confidence score."""
        if not sentence_entities:
            return None
        
        return max(sentence_entities, key=lambda e: e.get('confidence', 0))

def main():
    parser = argparse.ArgumentParser(description="RDF converter with enhanced entity resolution")
    parser.add_argument("--frames", required=True, help="Path to frames.json")
    parser.add_argument("--entities", required=True, help="Path to entities.json")
    parser.add_argument("--outfile", default="outputs/rdf_output_improved.ttl", help="Output Turtle file")
    parser.add_argument("--deepinfra-model", default="meta-llama/Llama-2-70b-chat-hf", help="DeepInfra model")
    parser.add_argument("--deepinfra-api-key", default="your-api-key", help="DeepInfra API key")
    
    args = parser.parse_args()
    
    # Get API key - try argument first, then environment, then default
    api_key = args.deepinfra_api_key or os.getenv("DEEPINFRA_API_KEY") or "your-api-key"
    if not api_key:
        print("Warning: No DeepInfra API key provided. Pronoun resolution will use heuristics only.")
        api_key = ""
    else:
        print(f"Using DeepInfra API key: {api_key[:10]}...")
    
    # Convert
    converter = RDFConvertor(api_key, args.deepinfra_model)
    converter.convert(args.frames, args.entities, args.outfile)

if __name__ == "__main__":
    main()
