import streamlit as st
import time
import json
import random
import re
import uuid
import os
import io
import base64
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from enum import Enum
import asyncio
from pydantic import BaseModel, Field, validator

# Import LLM clients
import openai
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langgraph.prebuilt import ToolExecutor
import pydantic_core

# Constants
DEFAULT_SYSTEM_PROMPT = """
You are DungeonLM, an expert Dungeons & Dragons Dungeon Master (DM). 
Your task is to create and narrate an engaging, immersive D&D adventure.
You will create NPCs, describe environments, manage combat, and respond to player actions.

Follow these principles:
1. Be creative and descriptive in your narration
2. Create challenging but fair encounters
3. Allow players significant freedom of choice
4. Remember key details about the campaign world and player decisions
5. Follow D&D 5th Edition rules, but prioritize fun over strict rule adherence
6. Present clear options to players, but also respond to unexpected choices

General structure:
- Start with a campaign setup based on the player's chosen setting/theme
- Begin with an engaging scene that introduces the main quest
- Narrate scenes vividly, using all five senses
- Guide player(s) through the adventure with hooks and challenges
- Track player stats and inventory accurately
- Implement combat rules for turn-based battles
- Award XP, gold, and items appropriately
- Maintain continuity and consequences for player choices

In your responses, after narrating the scene, include a section with:
[AVAILABLE ACTIONS] - List 3-5 suggested actions the player might take
[NEARBY LOCATIONS] - List locations the player can visit from here
[COMBAT INFO] - Include this section only during combat with initiative and enemy status

When generating a map, provide a simplified ASCII map like:
```
    N
  W + E
    S
~~~~~~~~~~~~~~~~~
|     |     |   |
| Inn | Town|For|
|     |Square|est|
~~~~~~~~~~~~~~~~~
|     |     |   |  
|Shop |Well |   |
|     |     |   |
~~~~~~~~~~~~~~~~~
```

Never break character as the DM. You are a creative storyteller who responds to player actions with engaging narration and appropriate consequences.
"""

# LLM Models
OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
]

# Pydantic Models
class ActionType(str, Enum):
    MOVE = "move"
    ATTACK = "attack"
    TALK = "talk"
    SEARCH = "search"
    USE = "use"
    CAST = "cast"
    REST = "rest"
    OTHER = "other"

class EntityType(str, Enum):
    PLAYER = "player"
    ENEMY = "enemy"
    NPC = "npc"

class Stats(BaseModel):
    STR: int = Field(10, description="Strength stat")
    DEX: int = Field(10, description="Dexterity stat")
    CON: int = Field(10, description="Constitution stat")
    INT: int = Field(10, description="Intelligence stat")
    WIS: int = Field(10, description="Wisdom stat")
    CHA: int = Field(10, description="Charisma stat")
    
    @validator('STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA', pre=True)
    def validate_stats(cls, v):
        """Validate stats are between 1 and 20"""
        v = int(v)
        if v < 1 or v > 30:
            raise ValueError(f"Stats must be between the ranges of 1-30, got {v}")
        return v

class Character(BaseModel):
    id: str
    name: str
    race: str
    character_class: str
    level: int = Field(1, ge=1, le=20)
    hp: int
    max_hp: int
    stats: Stats
    inventory: List[str] = Field(default_factory=list)
    gold: int = Field(0, ge=0)
    xp: int = Field(0, ge=0)
    spells: List[str] = Field(default_factory=list)
    background: str = ""
    alignment: str = "Neutral"
    
    class Config:
        validate_assignment = True

class NPC(BaseModel):
    id: str
    name: str
    description: str
    disposition: str = "Neutral"  # Friendly, Neutral, Hostile
    location: str = ""
    dialogue: Dict[str, str] = Field(default_factory=dict)
    
class Location(BaseModel):
    id: str
    name: str
    description: str
    connected_locations: List[str] = Field(default_factory=list)
    npcs: List[str] = Field(default_factory=list)
    items: List[str] = Field(default_factory=list)
    visited: bool = False
    
class QuestStatus(str, Enum):
    ACTIVE = "Active"
    COMPLETED = "Completed"
    FAILED = "Failed"
    
class Quest(BaseModel):
    id: str
    title: str
    description: str
    status: QuestStatus = QuestStatus.ACTIVE
    rewards: Dict[str, Any] = Field(default_factory=dict)
    steps: List[str] = Field(default_factory=list)
    current_step: int = 0
    
class Enemy(BaseModel):
    id: str
    name: str
    hp: int
    max_hp: int
    ac: int
    attack_bonus: int
    damage_dice: str
    dex_mod: int
    
class InitiativeEntry(BaseModel):
    id: str
    name: str
    initiative: int
    entity_type: EntityType

class CombatEncounter(BaseModel):
    id: str
    enemies: List[Enemy]
    initiative_order: List[InitiativeEntry]
    current_turn_idx: int = 0
    round: int = 1
    active: bool = True
    
class CampaignState(BaseModel):
    id: str
    title: str
    setting: str
    theme: str
    characters: List[Character] = Field(default_factory=list)
    npcs: List[NPC] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    quests: List[Quest] = Field(default_factory=list)
    current_location_id: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    current_combat: Optional[CombatEncounter] = None
    
class UserAction(BaseModel):
    action_type: ActionType
    target: Optional[str] = None
    content: str
    
class GameState(BaseModel):
    campaign: CampaignState
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    last_user_action: Optional[UserAction] = None
    system_messages: List[str] = Field(default_factory=list)

# LLM Client Wrapper
class LLMClient:
    def __init__(self, provider, api_key, model=None):
        self.provider = provider
        self.api_key = api_key

        if provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate_async(self, system_prompt, conversation_history, max_tokens=2000, temperature=0.7):
        try:
            messages = [{"role": "system", "content": system_prompt}] + conversation_history

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error generating content: {str(e)}")
            return f"The Dungeon Master seems to be taking a break. Please try again. Error: {str(e)}"

    def generate(self, system_prompt, conversation_history, max_tokens=2000, temperature=0.7):
        try:
            messages = [{"role": "system", "content": system_prompt}] + conversation_history

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        except Exception as e:
            st.error(f"Error generating content: {str(e)}")
            return f"The Dungeon Master seems to be taking a break. Please try again. Error: {str(e)}"

# LangGraph Tools
class CharacterTools:
    def add_item_to_inventory(self, state: GameState, item_name: str) -> GameState:
        """Add an item to the player's inventory"""
        if not state.campaign.characters:
            return state
            
        # Add item to first character's inventory
        state.campaign.characters[0].inventory.append(item_name)
        state.system_messages.append(f"Added {item_name} to inventory")
        return state
        
    def add_gold(self, state: GameState, amount: int) -> GameState:
        """Add gold to the player's wallet"""
        if not state.campaign.characters:
            return state
            
        state.campaign.characters[0].gold += amount
        state.system_messages.append(f"Added {amount} gold")
        return state
        
    def modify_hp(self, state: GameState, amount: int) -> GameState:
        """Modify character's HP by given amount (positive for healing, negative for damage)"""
        if not state.campaign.characters:
            return state
            
        char = state.campaign.characters[0]
        char.hp = max(0, min(char.max_hp, char.hp + amount))
        
        if amount > 0:
            state.system_messages.append(f"Healed {amount} HP")
        else:
            state.system_messages.append(f"Took {abs(amount)} damage")
            
        return state

class LocationTools:
    def move_to_location(self, state: GameState, location_id: str) -> GameState:
        """Move the player to a new location"""
        # Check if location exists
        location_exists = False
        for loc in state.campaign.locations:
            if loc.id == location_id:
                location_exists = True
                loc.visited = True
                break
                
        if location_exists:
            state.campaign.current_location_id = location_id
            state.system_messages.append(f"Moved to new location")
        else:
            state.system_messages.append(f"Location not found")
            
        return state
        
    def create_location(self, 
                      state: GameState, 
                      name: str, 
                      description: str, 
                      connect_to_current: bool = True) -> GameState:
        """Create a new location and optionally connect it to current location"""
        location_id = str(uuid.uuid4())
        new_location = Location(
            id=location_id,
            name=name,
            description=description
        )
        
        state.campaign.locations.append(new_location)
        
        # Connect to current location if requested
        if connect_to_current and state.campaign.current_location_id:
            current_loc = None
            for loc in state.campaign.locations:
                if loc.id == state.campaign.current_location_id:
                    current_loc = loc
                    break
                    
            if current_loc:
                current_loc.connected_locations.append(location_id)
                new_location.connected_locations.append(current_loc.id)
                
        state.system_messages.append(f"Created new location: {name}")
        return state

class NPCTools:
    def add_npc(self, 
              state: GameState, 
              name: str, 
              description: str, 
              disposition: str = "Neutral") -> GameState:
        """Add a new NPC to the current location"""
        npc_id = str(uuid.uuid4())
        
        npc = NPC(
            id=npc_id,
            name=name,
            description=description,
            disposition=disposition,
            location=state.campaign.current_location_id
        )
        
        state.campaign.npcs.append(npc)
        
        # Add NPC to current location
        if state.campaign.current_location_id:
            for loc in state.campaign.locations:
                if loc.id == state.campaign.current_location_id:
                    loc.npcs.append(npc_id)
                    break
                    
        state.system_messages.append(f"Added NPC: {name}")
        return state

class CombatTools:
    def start_combat(self, state: GameState, enemy_names: List[str]) -> GameState:
        """Start a combat encounter with specified enemies"""
        if state.campaign.current_combat and state.campaign.current_combat.active:
            state.system_messages.append("Combat already in progress")
            return state
            
        # Create enemies
        enemies = []
        for name in enemy_names:
            enemy_id = str(uuid.uuid4())
            hp = random.randint(10, 30)
            
            enemy = Enemy(
                id=enemy_id,
                name=name,
                hp=hp,
                max_hp=hp,
                ac=random.randint(10, 16),
                attack_bonus=random.randint(2, 6),
                damage_dice=f"{random.randint(1, 3)}d{random.choice([4, 6, 8])}",
                dex_mod=random.randint(-1, 3)
            )
            enemies.append(enemy)
            
        # Roll initiative
        initiative_entries = []
        
        # Player initiative
        for character in state.campaign.characters:
            initiative = random.randint(1, 20) + (character.stats.DEX - 10) // 2
            initiative_entries.append(
                InitiativeEntry(
                    id=character.id,
                    name=character.name,
                    initiative=initiative,
                    entity_type=EntityType.PLAYER
                )
            )
            
        # Enemy initiative
        for enemy in enemies:
            initiative = random.randint(1, 20) + enemy.dex_mod
            initiative_entries.append(
                InitiativeEntry(
                    id=enemy.id,
                    name=enemy.name,
                    initiative=initiative,
                    entity_type=EntityType.ENEMY
                )
            )
            
        # Sort by initiative
        initiative_entries.sort(key=lambda x: x.initiative, reverse=True)
        
        # Create combat encounter
        combat_id = str(uuid.uuid4())
        combat = CombatEncounter(
            id=combat_id,
            enemies=enemies,
            initiative_order=initiative_entries,
            current_turn_idx=0,
            round=1,
            active=True
        )
        
        state.campaign.current_combat = combat
        state.system_messages.append(f"Started combat with {', '.join(enemy_names)}")
        return state
        
    def end_combat(self, state: GameState) -> GameState:
        """End the current combat encounter"""
        if not state.campaign.current_combat or not state.campaign.current_combat.active:
            state.system_messages.append("No active combat to end")
            return state
            
        state.campaign.current_combat.active = False
        state.system_messages.append("Combat ended")
        return state
        
    def next_turn(self, state: GameState) -> GameState:
        """Advance to the next turn in combat"""
        if not state.campaign.current_combat or not state.campaign.current_combat.active:
            state.system_messages.append("No active combat")
            return state
            
        combat = state.campaign.current_combat
        combat.current_turn_idx = (combat.current_turn_idx + 1) % len(combat.initiative_order)
        
        # If we've gone through everyone, increase round counter
        if combat.current_turn_idx == 0:
            combat.round += 1
            
        # Get current entity
        current_entity = combat.initiative_order[combat.current_turn_idx]
        state.system_messages.append(f"Combat round {combat.round}: {current_entity.name}'s turn")
        
        return state
        
    def damage_enemy(self, state: GameState, enemy_id: str, damage: int) -> GameState:
        """Apply damage to an enemy"""
        if not state.campaign.current_combat:
            state.system_messages.append("No active combat")
            return state
            
        for i, enemy in enumerate(state.campaign.current_combat.enemies):
            if enemy.id == enemy_id:
                enemy.hp = max(0, enemy.hp - damage)
                state.campaign.current_combat.enemies[i] = enemy
                
                # Check if enemy is defeated
                if enemy.hp <= 0:
                    state.system_messages.append(f"{enemy.name} has been defeated!")
                    
                    # Remove from initiative if necessary
                    state.campaign.current_combat.initiative_order = [
                        entry for entry in state.campaign.current_combat.initiative_order 
                        if entry.id != enemy_id
                    ]
                    
                    # Check if all enemies are defeated
                    all_defeated = all(e.hp <= 0 for e in state.campaign.current_combat.enemies)
                    if all_defeated:
                        state.campaign.current_combat.active = False
                        state.system_messages.append("All enemies defeated. Combat ended.")
                else:
                    state.system_messages.append(f"{enemy.name} took {damage} damage")
                break
                
        return state

class QuestTools:
    def add_quest(self, 
                state: GameState, 
                title: str, 
                description: str,
                steps: List[str] = None) -> GameState:
        """Add a new quest to the campaign"""
        quest_id = str(uuid.uuid4())
        
        quest = Quest(
            id=quest_id,
            title=title,
            description=description,
            steps=steps or []
        )
        
        state.campaign.quests.append(quest)
        state.system_messages.append(f"Added new quest: {title}")
        return state
        
    def complete_quest(self, state: GameState, quest_id: str) -> GameState:
        """Mark a quest as completed"""
        for i, quest in enumerate(state.campaign.quests):
            if quest.id == quest_id:
                quest.status = QuestStatus.COMPLETED
                state.campaign.quests[i] = quest
                state.system_messages.append(f"Completed quest: {quest.title}")
                
                # Add rewards if specified
                if "gold" in quest.rewards:
                    if state.campaign.characters:
                        state.campaign.characters[0].gold += quest.rewards["gold"]
                        
                if "xp" in quest.rewards:
                    if state.campaign.characters:
                        state.campaign.characters[0].xp += quest.rewards["xp"]
                break
                
        return state

# Custom LangGraph nodes
def parse_user_action(state: GameState) -> Dict[str, Any]:
    """Parse the user's input to determine their action"""
    if not state.conversation_history or state.conversation_history[-1]["role"] != "user":
        return {"state": state}
        
    user_input = state.conversation_history[-1]["content"].lower()
    
    # Determine action type
    action_type = ActionType.OTHER
    target = None
    
    # Movement actions
    movement_patterns = [
        r"go to (.*)", r"enter (.*)", r"visit (.*)", r"travel to (.*)",
        r"head to (.*)", r"move to (.*)", r"explore (.*)", r"walk to (.*)"
    ]
    
    for pattern in movement_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.MOVE
            target = match.group(1).strip()
            break
            
    # Attack actions
    attack_patterns = [
        r"attack (.*)", r"fight (.*)", r"strike (.*)", r"hit (.*)",
        r"shoot (.*)", r"stab (.*)", r"slash (.*)"
    ]
    
    for pattern in attack_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.ATTACK
            target = match.group(1).strip()
            break
            
    # Talk actions
    talk_patterns = [
        r"talk to (.*)", r"speak to (.*)", r"ask (.*)", r"communicate with (.*)",
        r"greet (.*)", r"chat with (.*)", r"converse with (.*)", r"address (.*)"
    ]
    
    for pattern in talk_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.TALK
            target = match.group(1).strip()
            break
            
    # Search actions
    search_patterns = [
        r"search (.*)", r"look for (.*)", r"examine (.*)", r"investigate (.*)",
        r"inspect (.*)", r"check (.*)", r"look at (.*)", r"search for (.*)"
    ]
    
    for pattern in search_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.SEARCH
            target = match.group(1).strip()
            break
            
    # Use actions
    use_patterns = [
        r"use (.*)", r"activate (.*)", r"drink (.*)", r"employ (.*)",
        r"utilize (.*)", r"apply (.*)", r"consume (.*)", r"wield (.*)"
    ]
    
    for pattern in use_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.USE
            target = match.group(1).strip()
            break
            
    # Cast actions
    cast_patterns = [
        r"cast (.*)", r"spell (.*)", r"incantation (.*)", r"magic (.*)"
    ]
    
    for pattern in cast_patterns:
        match = re.search(pattern, user_input)
        if match:
            action_type = ActionType.CAST
            target = match.group(1).strip()
            break
            
    # Rest actions
    if any(word in user_input for word in ["rest", "sleep", "camp", "nap"]):
        action_type = ActionType.REST
    
    # Update user action in state
    state.last_user_action = UserAction(
        action_type=action_type,
        target=target,
        content=user_input
    )
    
    return {"state": state}

def process_movement(state: GameState) -> Dict[str, Union[GameState, str]]:
    """Process movement actions"""
    if not state.last_user_action or state.last_user_action.action_type != ActionType.MOVE:
        return {"state": state}
        
    target = state.last_user_action.target
    if not target:
        return {"state": state}
        
    # Check if target is a connected location
    current_loc = None
    target_loc = None
    
    for loc in state.campaign.locations:
        if loc.id == state.campaign.current_location_id:
            current_loc = loc
            
        # Case-insensitive match on location name
        if loc.name.lower() == target.lower():
            target_loc = loc
            
    # If we found both locations, check if they're connected
    if current_loc and target_loc:
        if target_loc.id in current_loc.connected_locations:
            # Move to location
            state.campaign.current_location_id = target_loc.id
            target_loc.visited = True
            state.system_messages.append(f"Moved to {target_loc.name}")
        else:
            state.system_messages.append(f"Cannot move directly to {target_loc.name}")
    elif target and current_loc:
        # Create new location and connect it
        location_tools = LocationTools()
        state = location_tools.create_location(
            state,
            name=target.title(),
            description=f"A newly discovered area: {target}",
            connect_to_current=True
        )
        
        # Find the newly created location
        for loc in state.campaign.locations:
            if loc.name.lower() == target.title().lower():
                state.campaign.current_location_id = loc.id
                break
                
    return {"state": state}

def process_combat(state: GameState) -> Dict[str, Union[GameState, str]]:
    """Process combat and attack actions"""
    if not state.last_user_action:
        return {"state": state}
        
    # Check if we should start combat
    if state.last_user_action.action_type == ActionType.ATTACK and not (state.campaign.current_combat and state.campaign.current_combat.active):
        # Start combat with target
        if state.last_user_action.target:
            enemy_names = [state.last_user_action.target.title()]
            combat_tools = CombatTools()
            state = combat_tools.start_combat(state, enemy_names)
    
    # Process combat actions if in combat
    if state.campaign.current_combat and state.campaign.current_combat.active:
        combat = state.campaign.current_combat
        current_turn = combat.initiative_order[combat.current_turn_idx]
        
        # Only process player actions on player turn
        if current_turn.entity_type == EntityType.PLAYER:
            # Attack action
            if state.last_user_action.action_type == ActionType.ATTACK:
                target = state.last_user_action.target
                if target:
                    # Find matching enemy
                    for enemy in combat.enemies:
                        if enemy.name.lower() in target.lower() or target.lower() in enemy.name.lower():
                            # Roll attack
                            attack_roll = random.randint(1, 20)
                            
                            # Get player's attack bonus based on class
                            attack_bonus = 0
                            if state.campaign.characters:
                                char = state.campaign.characters[0]
                                if char.character_class.lower() in ["fighter", "barbarian", "paladin"]:
                                    attack_bonus = (char.stats.STR - 10) // 2 + char.level // 3
                                elif char.character_class.lower() in ["rogue", "ranger"]:
                                    attack_bonus = (char.stats.DEX - 10) // 2 + char.level // 3
                                elif char.character_class.lower() in ["wizard", "sorcerer", "warlock"]:
                                    attack_bonus = (char.stats.INT - 10) // 2 + char.level // 3
                                elif char.character_class.lower() in ["cleric", "druid"]:
                                    attack_bonus = (char.stats.WIS - 10) // 2 + char.level // 3
                                    
                            total_attack = attack_roll + attack_bonus
                            
                            # Check if hit
                            if total_attack >= enemy.ac:
                                # Roll damage
                                damage_roll = random.randint(1, 6) + attack_bonus // 2
                                
                                # Apply damage
                                combat_tools = CombatTools()
                                state = combat_tools.damage_enemy(state, enemy.id, damage_roll)
                                
                                state.system_messages.append(
                                    f"Attack hit! Rolled {attack_roll} + {attack_bonus} = {total_attack} vs AC {enemy.ac}. "
                                    f"Dealt {damage_roll} damage."
                                )
                            else:
                                state.system_messages.append(
                                    f"Attack missed! Rolled {attack_roll} + {attack_bonus} = {total_attack} vs AC {enemy.ac}."
                                )
                            
                            # Move to next turn
                            combat_tools = CombatTools()
                            state = combat_tools.next_turn(state)
                            break
                    
            # Cast spell action
            elif state.last_user_action.action_type == ActionType.CAST:
                # Move to next turn after spell cast
                combat_tools = CombatTools()
                state = combat_tools.next_turn(state)
                
            # Move to enemy turn if we're still in player turn
            elif state.last_user_action.content.lower() == "end turn":
                combat_tools = CombatTools()
                state = combat_tools.next_turn(state)
        
    return {"state": state}

def process_npc_interaction(state: GameState) -> Dict[str, Union[GameState, str]]:
    """Process talking to NPCs"""
    if not state.last_user_action or state.last_user_action.action_type != ActionType.TALK:
        return {"state": state}
        
    target = state.last_user_action.target
    if not target:
        return {"state": state}
        
    # Find NPCs in current location
    current_location_npcs = []
    if state.campaign.current_location_id:
        for loc in state.campaign.locations:
            if loc.id == state.campaign.current_location_id:
                current_location_npcs = loc.npcs
                break
    
    # Check if target NPC exists
    target_npc = None
    for npc in state.campaign.npcs:
        if npc.id in current_location_npcs and npc.name.lower() in target.lower():
            target_npc = npc
            break
            
    # If NPC not found, create them
    if not target_npc:
        npc_tools = NPCTools()
        state = npc_tools.add_npc(
            state,
            name=target.title(),
            description=f"A person you've met in your travels.",
            disposition="Neutral"
        )
        
    return {"state": state}

def process_search(state: GameState) -> Dict[str, Union[GameState, str]]:
    """Process search actions"""
    if not state.last_user_action or state.last_user_action.action_type != ActionType.SEARCH:
        return {"state": state}
        
    # Chance to find items or gold
    if random.random() < 0.3:
        item_options = [
            "Small Potion", "Rusty Key", "Map Fragment", "Old Coin", 
            "Scroll", "Gemstone", "Amulet", "Lockpick"
        ]
        item = random.choice(item_options)
        
        character_tools = CharacterTools()
        state = character_tools.add_item_to_inventory(state, item)
        state.system_messages.append(f"Found: {item}")
    
    if random.random() < 0.2:
        gold_amount = random.randint(1, 10)
        
        character_tools = CharacterTools()
        state = character_tools.add_gold(state, gold_amount)
        state.system_messages.append(f"Found {gold_amount} gold coins")
        
    # Chance to discover a hidden area or feature
    if random.random() < 0.15:
        # Add a message about discovering something
        state.system_messages.append("You discovered something interesting...")
        
        # Chance to find a secret door to a new location
        if random.random() < 0.5:
            location_name = f"Hidden {random.choice(['Chamber', 'Cavern', 'Passage', 'Alcove', 'Room'])}"
            
            location_tools = LocationTools()
            state = location_tools.create_location(
                state,
                name=location_name,
                description=f"A secret area you discovered by searching carefully.",
                connect_to_current=True
            )
            
    return {"state": state}

def generate_dm_response(state: GameState) -> Dict[str, Union[GameState, str]]:
    """Generate DM response using the LLM"""
    # Create system prompt with context about the campaign
    system_prompt = DEFAULT_SYSTEM_PROMPT + "\n\n"
    
    # Add campaign info
    system_prompt += f"Campaign: {state.campaign.title}\n"
    system_prompt += f"Setting: {state.campaign.setting}\n"
    system_prompt += f"Theme: {state.campaign.theme}\n\n"
    
    # Add character info
    if state.campaign.characters:
        char = state.campaign.characters[0]
        system_prompt += f"Player Character: {char.name}, Level {char.level} {char.race} {char.character_class}\n"
        system_prompt += f"HP: {char.hp}/{char.max_hp}, XP: {char.xp}\n"
        system_prompt += f"Inventory: {', '.join(char.inventory)}\n"
        system_prompt += f"Gold: {char.gold}\n\n"
    
    # Add current location info
    current_location = None
    for loc in state.campaign.locations:
        if loc.id == state.campaign.current_location_id:
            current_location = loc
            break
    
    if current_location:
        system_prompt += f"Current Location: {current_location.name}\n"
        system_prompt += f"Description: {current_location.description}\n"
        
        # Add NPCs in location
        location_npcs = []
        for npc_id in current_location.npcs:
            for npc in state.campaign.npcs:
                if npc.id == npc_id:
                    location_npcs.append(npc.name)
                    break
        
        if location_npcs:
            system_prompt += f"NPCs present: {', '.join(location_npcs)}\n"
        
        # Add connected locations
        connected_locations = []
        for loc_id in current_location.connected_locations:
            for loc in state.campaign.locations:
                if loc.id == loc_id:
                    connected_locations.append(loc.name)
                    break
        
        if connected_locations:
            system_prompt += f"Connected areas: {', '.join(connected_locations)}\n"
    
    # Add active combat info
    if state.campaign.current_combat and state.campaign.current_combat.active:
        combat = state.campaign.current_combat
        system_prompt += f"\nCOMBAT IN PROGRESS - Round {combat.round}\n"
        
        # Current initiative
        current_entity = combat.initiative_order[combat.current_turn_idx]
        system_prompt += f"Current turn: {current_entity.name}\n"
        
        # Initiative order
        system_prompt += "Initiative order:\n"
        for entry in combat.initiative_order:
            system_prompt += f"- {entry.name}: {entry.initiative}\n"
        
        # Enemy status
        system_prompt += "Enemy status:\n"
        for enemy in combat.enemies:
            hp_percent = enemy.hp / enemy.max_hp * 100
            status = "Healthy"
            if hp_percent <= 25:
                status = "Near death"
            elif hp_percent <= 50:
                status = "Badly wounded"
            elif hp_percent <= 75:
                status = "Wounded"
            
            system_prompt += f"- {enemy.name}: {enemy.hp}/{enemy.max_hp} HP ({status})\n"
    
    # Add active quests
    active_quests = [q for q in state.campaign.quests if q.status == QuestStatus.ACTIVE]
    if active_quests:
        system_prompt += "\nActive Quests:\n"
        for quest in active_quests:
            system_prompt += f"- {quest.title}: {quest.description}\n"
    
    # Add system messages as additional context
    if state.system_messages:
        system_prompt += "\nRecent Events:\n"
        # Only include the last 5 system messages to avoid prompt too long
        for msg in state.system_messages[-5:]:
            system_prompt += f"- {msg}\n"
    
    # Get conversation history for LLM context
    # Only include the last 10 turns to avoid token limits
    conversation_context = state.conversation_history[-10:]
    
    # Call the LLM
    try:
        # Create LLM client if not already in session state
        if "llm_client" not in st.session_state:
            if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
                model = st.session_state.get("selected_model", "gpt-4o")
                st.session_state.llm_client = LLMClient("openai", st.session_state.openai_api_key, model)
            else:
                return {"state": state, "next": "error"}
        
        llm_client = st.session_state.llm_client
        response = llm_client.generate(system_prompt, conversation_context)
        
        # Add the response to conversation history
        state.conversation_history.append({"role": "assistant", "content": response})
        
        return {"state": state, "next": "output"}
    except Exception as e:
        st.error(f"Error generating DM response: {str(e)}")
        return {"state": state, "next": "error"}

# Set up LangGraph
def setup_dungeon_graph() -> StateGraph:
    """Set up the LangGraph for the dungeon game"""
    # Create a graph
    dungeon_graph = StateGraph(GameState)
    
    # Add all the different tool executors
    character_tools_executor = ToolExecutor(tools=[CharacterTools()])
    location_tools_executor = ToolExecutor(tools=[LocationTools()])
    npc_tools_executor = ToolExecutor(tools=[NPCTools()])
    combat_tools_executor = ToolExecutor(tools=[CombatTools()])
    quest_tools_executor = ToolExecutor(tools=[QuestTools()])
    
    # Add nodes to the graph
    dungeon_graph.add_node("parse_user_action", parse_user_action)
    dungeon_graph.add_node("process_movement", process_movement)
    dungeon_graph.add_node("process_combat", process_combat)
    dungeon_graph.add_node("process_npc_interaction", process_npc_interaction)
    dungeon_graph.add_node("process_search", process_search)
    dungeon_graph.add_node("generate_dm_response", generate_dm_response)
    
    # Add edges
    dungeon_graph.add_edge("parse_user_action", "process_movement")
    dungeon_graph.add_edge("process_movement", "process_combat")
    dungeon_graph.add_edge("process_combat", "process_npc_interaction")
    dungeon_graph.add_edge("process_npc_interaction", "process_search")
    dungeon_graph.add_edge("process_search", "generate_dm_response")
    
    # Add conditional edges from generate_dm_response
    dungeon_graph.add_conditional_edges(
        "generate_dm_response",
        lambda x: x.get("next", "output"),
        {
            "output": END,
            "error": END
        }
    )
    
    # Compile the graph
    dungeon_graph.compile()
    
    return dungeon_graph

# Streamlit UI functions
def create_character():
    """Create a new character"""
    st.header("Create Your Character")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Character Name", "Adventurer")
        race = st.selectbox("Race", ["Human", "Elf", "Dwarf", "Halfling", "Half-Elf", "Half-Orc", "Dragonborn", "Tiefling"])
        character_class = st.selectbox("Class", ["Fighter", "Wizard", "Rogue", "Cleric", "Ranger", "Paladin", "Barbarian", "Bard", "Druid", "Sorcerer", "Warlock", "Monk"])
        level = st.number_input("Level", min_value=1, max_value=20, value=1)
        
    with col2:
        # Generate random stats with a slight boost
        stats = {
            "STR": st.slider("Strength", 3, 18, random.randint(8, 16)),
            "DEX": st.slider("Dexterity", 3, 18, random.randint(8, 16)),
            "CON": st.slider("Constitution", 3, 18, random.randint(8, 16)),
            "INT": st.slider("Intelligence", 3, 18, random.randint(8, 16)),
            "WIS": st.slider("Wisdom", 3, 18, random.randint(8, 16)),
            "CHA": st.slider("Charisma", 3, 18, random.randint(8, 16))
        }
    
    background = st.text_area("Character Background (Optional)", "")
    alignment = st.selectbox("Alignment", ["Lawful Good", "Neutral Good", "Chaotic Good", "Lawful Neutral", "True Neutral", "Chaotic Neutral", "Lawful Evil", "Neutral Evil", "Chaotic Evil"], index=4)
    
    # Calculate HP based on class and constitution
    base_hp = {
        "Barbarian": 12,
        "Fighter": 10, 
        "Paladin": 10,
        "Ranger": 10,
        "Monk": 8,
        "Rogue": 8,
        "Bard": 8,
        "Cleric": 8,
        "Druid": 8,
        "Warlock": 8,
        "Wizard": 6,
        "Sorcerer": 6
    }
    
    con_modifier = (stats["CON"] - 10) // 2
    max_hp = base_hp.get(character_class, 8) + con_modifier
    max_hp = max(1, max_hp)  # Ensure minimum of 1 HP
    
    # Create character
    if st.button("Create Character"):
        character_id = str(uuid.uuid4())
        
        character = Character(
            id=character_id,
            name=name,
            race=race,
            character_class=character_class,
            level=level,
            hp=max_hp,
            max_hp=max_hp,
            stats=Stats(**stats),
            inventory=["Backpack", "Rations (1 day)", "Waterskin"],
            gold=random.randint(5, 20),
            background=background,
            alignment=alignment
        )
        
        # Store in session state
        st.session_state.character = character
        st.session_state.character_created = True
        
        st.success("Character created! Now you can start your adventure.")
        st.experimental_rerun()

def create_campaign():
    """Create a new campaign"""
    st.header("Create Your Campaign")
    
    title = st.text_input("Campaign Title", "The Lost Artifacts")
    
    setting_options = [
        "Medieval Fantasy", "High Fantasy", "Dark Fantasy", "Urban Fantasy",
        "Mythological", "Post-Apocalyptic", "Steampunk", "Sci-Fi Fantasy"
    ]
    setting = st.selectbox("Setting", setting_options)
    
    theme_options = [
        "Heroic Adventure", "Mystery", "Horror", "Political Intrigue",
        "Exploration", "War", "Heist", "Survival", "Redemption", "Revenge"
    ]
    theme = st.selectbox("Theme", theme_options)
    
    if st.button("Create Campaign"):
        # Create starting location
        start_location_id = str(uuid.uuid4())
        
        # Generate starting location based on setting
        if setting == "Medieval Fantasy":
            start_name = "Small Village"
            start_desc = "A quaint village with thatched-roof buildings and a central marketplace."
        elif setting == "Urban Fantasy":
            start_name = "City Streets"
            start_desc = "The busy streets of a large city where magic and modern technology coexist."
        elif setting == "Dark Fantasy":
            start_name = "Gloomy Hamlet"
            start_desc = "A small settlement shrouded in mist with suspicious townsfolk."
        else:
            start_name = "Starting Area"
            start_desc = "The beginning of your adventure."
            
        start_location = Location(
            id=start_location_id,
            name=start_name,
            description=start_desc,
            visited=True
        )
        
        # Create campaign state
        campaign = CampaignState(
            id=str(uuid.uuid4()),
            title=title,
            setting=setting,
            theme=theme,
            locations=[start_location],
            current_location_id=start_location_id
        )
        
        # Add character to campaign
        if hasattr(st.session_state, "character"):
            campaign.characters.append(st.session_state.character)
        
        # Create game state
        initial_state = GameState(
            campaign=campaign,
            conversation_history=[]
        )
        
        # Store in session state
        st.session_state.game_state = initial_state
        st.session_state.campaign_created = True
        
        # Initialize LangGraph
        # Use in-memory checkpointing instead of MemorySaver
        app = setup_dungeon_graph()
        st.session_state.dungeon_app = app
        
        st.success("Campaign created! Your adventure awaits.")
        st.experimental_rerun()

def display_character_sheet():
    """Display character sheet in sidebar"""
    if not hasattr(st.session_state, "character"):
        return
        
    char = st.session_state.character
    
    st.sidebar.header(f"{char.name}")
    st.sidebar.write(f"Level {char.level} {char.race} {char.character_class}")
    
    # HP bar
    hp_percent = char.hp / char.max_hp * 100
    st.sidebar.progress(hp_percent / 100)
    st.sidebar.write(f"HP: {char.hp}/{char.max_hp}")
    
    # Stats
    st.sidebar.subheader("Stats")
    cols = st.sidebar.columns(3)
    stats = ["STR", "DEX", "CON", "INT", "WIS", "CHA"]
    
    for i, stat in enumerate(stats):
        col_idx = i % 3
        cols[col_idx].metric(stat, getattr(char.stats, stat))
    
    # Inventory
    st.sidebar.subheader("Inventory")
    st.sidebar.write(f"Gold: {char.gold}")
    
    if char.inventory:
        items = ", ".join(char.inventory)
        st.sidebar.write(items)
    else:
        st.sidebar.write("Empty")

def main():
    st.title("🧙‍♂️ DungeonLM - AI Dungeon Master")
    
    # Initialize session state
    if "character_created" not in st.session_state:
        st.session_state.character_created = False
        
    if "campaign_created" not in st.session_state:
        st.session_state.campaign_created = False
        
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Setup sidebar
    with st.sidebar:
        st.header("Game Settings")
        
        # API Key input
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            
        # Model selection
        selected_model = st.selectbox("Select Model", OPENAI_MODELS)
        st.session_state.selected_model = selected_model
        
        # Display character sheet if character exists
        if st.session_state.character_created:
            display_character_sheet()
            
        # Reset game button
        if st.button("Start New Game"):
            # Reset all game state
            for key in list(st.session_state.keys()):
                if key not in ["openai_api_key", "selected_model"]:
                    del st.session_state[key]
            st.experimental_rerun()
    
    # Check for API key
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to begin.")
        return
        
    # Character creation
    if not st.session_state.character_created:
        create_character()
        return
        
    # Campaign creation
    if not st.session_state.campaign_created:
        create_campaign()
        return
    
    # Main game interaction
    if st.session_state.character_created and st.session_state.campaign_created:
        # Display conversation history
        display_conversation()
        
        # User input
        user_input = st.chat_input("What do you do?")
        
        if user_input:
            process_user_input(user_input)

def display_conversation():
    """Display the conversation history"""
    for message in st.session_state.game_state.conversation_history:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🧙‍♂️"):
                st.write(message["content"])
        elif message["role"] == "user":
            with st.chat_message("user", avatar="🧝"):
                st.write(message["content"])
                
    # If no conversation yet, display intro
    if not st.session_state.game_state.conversation_history:
        # Generate intro message
        intro_message = f"Welcome to {st.session_state.game_state.campaign.title}, brave adventurer! "
        intro_message += f"You are about to embark on a journey in a {st.session_state.game_state.campaign.setting} world "
        intro_message += f"with themes of {st.session_state.game_state.campaign.theme}.\n\n"
        
        # Add starting location description
        for loc in st.session_state.game_state.campaign.locations:
            if loc.id == st.session_state.game_state.campaign.current_location_id:
                intro_message += f"You find yourself in {loc.name}. {loc.description}\n\n"
                break
                
        intro_message += "What would you like to do?"
        
        with st.chat_message("assistant", avatar="🧙‍♂️"):
            st.write(intro_message)
            
        # Add to conversation history
        st.session_state.game_state.conversation_history.append({"role": "assistant", "content": intro_message})

def process_user_input(user_input):
    """Process user input and advance the game state"""
    # Add user input to conversation history
    st.session_state.game_state.conversation_history.append({"role": "user", "content": user_input})
    
    # Run the LangGraph app with user input
    dungeon_app = st.session_state.dungeon_app
    game_state = st.session_state.game_state
    
    # Show spinner while processing
    with st.spinner("The Dungeon Master is thinking..."):
        try:
            # Run the graph
            result = dungeon_app.invoke({"state": game_state})
            st.session_state.game_state = result["state"]
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error processing game state: {str(e)}")

if __name__ == "__main__":
    main()