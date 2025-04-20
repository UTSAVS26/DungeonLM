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
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

# Import LLM clients
import openai

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

# Data Models
@dataclass
class Character:
    id: str
    name: str
    race: str
    character_class: str
    level: int
    hp: int
    max_hp: int
    stats: Dict[str, int]
    inventory: List[str]
    gold: int
    xp: int
    spells: List[str] = field(default_factory=list)
    background: str = ""
    alignment: str = "Neutral"
    
@dataclass
class NPC:
    id: str
    name: str
    description: str
    disposition: str = "Neutral"  # Friendly, Neutral, Hostile
    location: str = ""
    
@dataclass
class Location:
    id: str
    name: str
    description: str
    connected_locations: List[str] = field(default_factory=list)
    npcs: List[str] = field(default_factory=list)
    visited: bool = False
    
@dataclass
class Quest:
    id: str
    title: str
    description: str
    status: str = "Active"  # Active, Completed, Failed
    rewards: Dict[str, Any] = field(default_factory=dict)
    steps: List[str] = field(default_factory=list)
    current_step: int = 0
    
@dataclass
class Campaign:
    id: str
    title: str
    setting: str
    theme: str
    characters: List[Character] = field(default_factory=list)
    npcs: List[NPC] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    quests: List[Quest] = field(default_factory=list)
    current_location_id: str = ""
    history: List[Dict[str, Any]] = field(default_factory=list)
    
@dataclass
class CombatEncounter:
    id: str
    enemies: List[Dict[str, Any]]
    initiative_order: List[str]
    current_turn: int = 0
    round: int = 1
    active: bool = True

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

# DungeonLM Core
class DungeonLM:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.campaign = None
        self.current_combat = None
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.conversation_history = []
            
    def create_campaign(self, title, setting, theme):
        campaign_id = str(uuid.uuid4())
        self.campaign = Campaign(
            id=campaign_id,
            title=title,
            setting=setting,
            theme=theme
        )
        
        # Add campaign setup to conversation history
        self.conversation_history.append({
            "role": "system", 
            "content": f"Campaign '{title}' created with setting '{setting}' and theme '{theme}'. Initialize the world and starting location."
        })
        
        return self.campaign
    
    def add_character(self, name, race, character_class, level=1):
        char_id = str(uuid.uuid4())
        
        # Set default stats based on class
        base_stats = {"STR": 10, "DEX": 10, "CON": 10, "INT": 10, "WIS": 10, "CHA": 10}
        if character_class.lower() in ["warrior", "fighter", "barbarian", "paladin"]:
            base_stats["STR"] += 4
            base_stats["CON"] += 2
        elif character_class.lower() in ["rogue", "ranger", "monk"]:
            base_stats["DEX"] += 4
            base_stats["WIS"] += 2
        elif character_class.lower() in ["wizard", "sorcerer", "warlock"]:
            base_stats["INT"] += 4
            base_stats["CHA"] += 2
        elif character_class.lower() in ["cleric", "druid"]:
            base_stats["WIS"] += 4
            base_stats["CON"] += 2
        
        # Calculate HP based on class and level
        base_hp = 8
        if character_class.lower() in ["barbarian"]:
            base_hp = 12
        elif character_class.lower() in ["fighter", "paladin", "ranger"]:
            base_hp = 10
        elif character_class.lower() in ["wizard", "sorcerer", "warlock"]:
            base_hp = 6
            
        max_hp = base_hp + ((base_hp // 2) * (level - 1))
        
        character = Character(
            id=char_id,
            name=name,
            race=race,
            character_class=character_class,
            level=level,
            hp=max_hp,
            max_hp=max_hp,
            stats=base_stats,
            inventory=["Backpack", "Bedroll", "Rations (5)"],
            gold=50,
            xp=0
        )
        
        # Add starting equipment based on class
        if character_class.lower() in ["warrior", "fighter", "barbarian", "paladin"]:
            character.inventory.extend(["Longsword", "Shield", "Chainmail"])
        elif character_class.lower() in ["rogue", "ranger"]:
            character.inventory.extend(["Shortbow", "Quiver with 20 arrows", "Leather Armor", "Dagger"])
        elif character_class.lower() in ["wizard", "sorcerer", "warlock"]:
            character.inventory.extend(["Spellbook", "Wand", "Robes", "Component Pouch"])
        elif character_class.lower() in ["cleric", "druid"]:
            character.inventory.extend(["Mace", "Shield", "Holy Symbol", "Scale Mail"])
            
        self.campaign.characters.append(character)
        
        # Add character creation to conversation history
        self.conversation_history.append({
            "role": "system", 
            "content": f"Character created: {name}, a level {level} {race} {character_class}. Remember their abilities and stats."
        })
        
        return character
    
    def add_location(self, name, description, connected_locations=None):
        location_id = str(uuid.uuid4())
        location = Location(
            id=location_id,
            name=name,
            description=description,
            connected_locations=connected_locations or []
        )
        
        self.campaign.locations.append(location)
        
        # Set as current location if it's the first one
        if not self.campaign.current_location_id:
            self.campaign.current_location_id = location_id
            
        return location
    
    def add_npc(self, name, description, disposition="Neutral", location_id=None):
        npc_id = str(uuid.uuid4())
        npc = NPC(
            id=npc_id,
            name=name,
            description=description,
            disposition=disposition,
            location=location_id or self.campaign.current_location_id
        )
        
        self.campaign.npcs.append(npc)
        
        # Add NPC to location
        if location_id:
            for loc in self.campaign.locations:
                if loc.id == location_id:
                    loc.npcs.append(npc_id)
                    break
        elif self.campaign.current_location_id:
            for loc in self.campaign.locations:
                if loc.id == self.campaign.current_location_id:
                    loc.npcs.append(npc_id)
                    break
                    
        return npc
    
    def add_quest(self, title, description, steps=None, rewards=None):
        quest_id = str(uuid.uuid4())
        quest = Quest(
            id=quest_id,
            title=title,
            description=description,
            steps=steps or [],
            rewards=rewards or {"gold": 100, "xp": 200}
        )
        
        self.campaign.quests.append(quest)
        return quest
    
    def current_location(self):
        for loc in self.campaign.locations:
            if loc.id == self.campaign.current_location_id:
                return loc
        return None
    
    def move_to_location(self, location_id):
        # Mark as visited
        for loc in self.campaign.locations:
            if loc.id == location_id:
                loc.visited = True
                self.campaign.current_location_id = location_id
                break
    
    def get_location_npcs(self, location_id=None):
        loc_id = location_id or self.campaign.current_location_id
        npc_ids = []
        
        for loc in self.campaign.locations:
            if loc.id == loc_id:
                npc_ids = loc.npcs
                break
                
        npcs = []
        for npc_id in npc_ids:
            for npc in self.campaign.npcs:
                if npc.id == npc_id:
                    npcs.append(npc)
                    
        return npcs
    
    def start_combat(self, enemies):
        """Start a combat encounter with given enemies"""
        combat_id = str(uuid.uuid4())
        
        # Initialize initiative order
        initiative_order = []
        for character in self.campaign.characters:
            initiative = random.randint(1, 20) + (character.stats["DEX"] - 10) // 2
            initiative_order.append({"id": character.id, "name": character.name, "initiative": initiative, "type": "player"})
            
        for enemy in enemies:
            initiative = random.randint(1, 20) + (enemy.get("dex_mod", 0))
            initiative_order.append({"id": enemy["id"], "name": enemy["name"], "initiative": initiative, "type": "enemy"})
            
        # Sort by initiative
        initiative_order.sort(key=lambda x: x["initiative"], reverse=True)
        initiative_order_ids = [entry["id"] for entry in initiative_order]
        
        self.current_combat = CombatEncounter(
            id=combat_id,
            enemies=enemies,
            initiative_order=initiative_order_ids,
            current_turn=0,
            round=1,
            active=True
        )
        
        # Add combat start to conversation history
        enemy_names = ", ".join([e["name"] for e in enemies])
        self.conversation_history.append({
            "role": "system", 
            "content": f"Combat has started against: {enemy_names}. Initiative order: {', '.join([entry['name'] for entry in initiative_order])}"
        })
        
        return self.current_combat
    
    def end_combat(self):
        """End the current combat encounter"""
        if self.current_combat:
            self.current_combat.active = False
            self.conversation_history.append({
                "role": "system", 
                "content": "Combat has ended."
            })
            self.current_combat = None
    
    def next_turn(self):
        """Advance to the next turn in combat"""
        if self.current_combat:
            self.current_combat.current_turn = (self.current_combat.current_turn + 1) % len(self.current_combat.initiative_order)
            
            # If we've gone through everyone, increase round counter
            if self.current_combat.current_turn == 0:
                self.current_combat.round += 1
                
            # Get current entity
            current_id = self.current_combat.initiative_order[self.current_combat.current_turn]
            current_entity = None
            
            # Check if it's a player
            for char in self.campaign.characters:
                if char.id == current_id:
                    current_entity = char.name
                    break
                    
            # Check if it's an enemy
            if not current_entity:
                for enemy in self.current_combat.enemies:
                    if enemy["id"] == current_id:
                        current_entity = enemy["name"]
                        break
            
            self.conversation_history.append({
                "role": "system", 
                "content": f"Round {self.current_combat.round}, Turn: {current_entity}'s turn."
            })
            
            return current_entity
        return None
    
    def generate_map(self, size=(15, 10)):
        """Generate a simple ASCII map of the area"""
        current_loc = self.current_location()
        if not current_loc:
            return "No current location to map."
        
        # Create a blank map grid
        map_grid = [[' ' for _ in range(size[0])] for _ in range(size[1])]
        
        # Place current location in center
        center_x, center_y = size[0] // 2, size[1] // 2
        map_grid[center_y][center_x] = 'X'  # X marks current location
        
        # Get connected locations
        connected_loc_ids = current_loc.connected_locations
        connected_locs = []
        
        for loc_id in connected_loc_ids:
            for loc in self.campaign.locations:
                if loc.id == loc_id:
                    connected_locs.append(loc)
                    
        # Place connected locations around center
        directions = [
            (0, -1, "North"), (1, 0, "East"), 
            (0, 1, "South"), (-1, 0, "West"),
            (1, -1, "Northeast"), (1, 1, "Southeast"),
            (-1, 1, "Southwest"), (-1, -1, "Northwest")
        ]
        
        for idx, loc in enumerate(connected_locs[:len(directions)]):
            dx, dy, direction = directions[idx]
            x, y = center_x + dx, center_y + dy
            
            # Check if position is valid
            if 0 <= x < size[0] and 0 <= y < size[1]:
                map_grid[y][x] = loc.name[0].upper()  # First letter of location name
        
        # Convert grid to ASCII art
        map_str = "```\n    N    \n  W + E  \n    S    \n"
        map_str += "-" * (size[0] + 2) + "\n"
        
        for row in map_grid:
            map_str += "|" + "".join(row) + "|\n"
            
        map_str += "-" * (size[0] + 2) + "\n"
        map_str += "```\n"
        map_str += f"X = {current_loc.name} (Current Location)\n"
        
        # Add legend for connected locations
        for idx, loc in enumerate(connected_locs[:len(directions)]):
            dx, dy, direction = directions[idx]
            x, y = center_x + dx, center_y + dy
            
            if 0 <= x < size[0] and 0 <= y < size[1]:
                map_str += f"{loc.name[0].upper()} = {loc.name} ({direction})\n"
        
        return map_str
    
    def process_user_input(self, user_input):
        """Process user input and generate response"""
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get current game state to include in prompt
        game_state = self._get_game_state_summary()
        
        # Combine system prompt with game state
        full_system_prompt = self.system_prompt + "\n\n" + game_state
        
        # Limit conversation history to last 10 exchanges to avoid token limits
        limited_history = self.conversation_history[-20:]
        
        # Generate response
        if self.llm_client:
            response = self.llm_client.generate(
                system_prompt=full_system_prompt,
                conversation_history=limited_history
            )
        else:
            response = "No LLM client configured. Please set up an API key and model."
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Update game state based on NLP understanding of response
        self._update_game_state_from_response(response, user_input)
        
        return response
    
    def _get_game_state_summary(self):
        """Create a summary of current game state for the LLM"""
        summary = "CURRENT GAME STATE:\n"
        
        # Campaign info
        summary += f"Campaign: {self.campaign.title} (Setting: {self.campaign.setting}, Theme: {self.campaign.theme})\n\n"
        
        # Character info
        summary += "CHARACTERS:\n"
        for char in self.campaign.characters:
            summary += f"- {char.name}: Level {char.level} {char.race} {char.character_class}, HP: {char.hp}/{char.max_hp}, Gold: {char.gold}, XP: {char.xp}\n"
            summary += f"  Inventory: {', '.join(char.inventory[:5])}" + (f" and {len(char.inventory) - 5} more items" if len(char.inventory) > 5 else "") + "\n"
        
        # Current location
        current_loc = self.current_location()
        if current_loc:
            summary += f"\nCURRENT LOCATION: {current_loc.name}\n"
            summary += f"Description: {current_loc.description[:100]}...\n"
            
            # Connected locations
            connected_locs = []
            for loc_id in current_loc.connected_locations:
                for loc in self.campaign.locations:
                    if loc.id == loc_id:
                        connected_locs.append(loc.name)
            
            if connected_locs:
                summary += f"Connected locations: {', '.join(connected_locs)}\n"
            
            # NPCs in location
            npcs = self.get_location_npcs()
            if npcs:
                summary += f"NPCs present: {', '.join([npc.name for npc in npcs])}\n"
        
        # Active quests
        active_quests = [q for q in self.campaign.quests if q.status == "Active"]
        if active_quests:
            summary += "\nACTIVE QUESTS:\n"
            for quest in active_quests[:3]:  # Limit to 3 active quests
                summary += f"- {quest.title}: {quest.description[:100]}...\n"
        
        # Combat info
        if self.current_combat and self.current_combat.active:
            summary += "\nCOMBAT STATUS:\n"
            summary += f"Round: {self.current_combat.round}\n"
            
            # Get current turn entity
            current_id = self.current_combat.initiative_order[self.current_combat.current_turn]
            current_entity = "Unknown"
            
            # Check if it's a player
            for char in self.campaign.characters:
                if char.id == current_id:
                    current_entity = char.name
                    break
                    
            # Check if it's an enemy
            if current_entity == "Unknown":
                for enemy in self.current_combat.enemies:
                    if enemy["id"] == current_id:
                        current_entity = enemy["name"]
                        break
            
            summary += f"Current turn: {current_entity}\n"
            
            # List enemies with health
            summary += "Enemies:\n"
            for enemy in self.current_combat.enemies:
                hp_percent = (enemy["hp"] / enemy["max_hp"]) * 100
                status = "Healthy"
                if hp_percent <= 25:
                    status = "Near Death"
                elif hp_percent <= 50:
                    status = "Badly Wounded"
                elif hp_percent <= 75:
                    status = "Wounded"
                
                summary += f"- {enemy['name']}: {status} ({enemy['hp']}/{enemy['max_hp']} HP)\n"
        
        return summary
    
    def _update_game_state_from_response(self, response, user_input):
        """Update game state based on NLP understanding of the response"""
        # Extract possible location change
        location_change_pattern = r"You (?:have|are now in|arrive at|enter|reach) (?:the|a) (.+?)(?:\.|\n|$)"
        location_matches = re.search(location_change_pattern, response, re.IGNORECASE)
        
        if location_matches:
            new_location_name = location_matches.group(1).strip()
            # Check if this is a known location
            found_location = False
            for loc in self.campaign.locations:
                if loc.name.lower() == new_location_name.lower():
                    self.move_to_location(loc.id)
                    found_location = True
                    break
            
            # If not found but should be created, create it
            if not found_location and "enter" in user_input.lower() or "go to" in user_input.lower():
                # Create a new location
                new_loc = self.add_location(
                    new_location_name,
                    f"A location the party has discovered: {new_location_name}"
                )
                
                # Connect it to current location
                current_loc = self.current_location()
                if current_loc:
                    current_loc.connected_locations.append(new_loc.id)
                    new_loc.connected_locations.append(current_loc.id)
                
                # Move to the new location
                self.move_to_location(new_loc.id)
        
        # Check for combat start
        combat_start_pattern = r"Roll initiative|Combat begins|prepares to attack|hostile creatures|COMBAT INFO"
        if re.search(combat_start_pattern, response, re.IGNORECASE) and not (self.current_combat and self.current_combat.active):
            # Try to extract enemy information
            enemies = []
            enemy_pattern = r"(?:A|An|The) ([^.!,]+) (?:appears|attacks|charges|emerges)"
            enemy_matches = re.finditer(enemy_pattern, response, re.IGNORECASE)
            
            for match in enemy_matches:
                enemy_name = match.group(1).strip()
                enemy_id = str(uuid.uuid4())
                # Create basic enemy stats
                enemy = {
                    "id": enemy_id,
                    "name": enemy_name,
                    "hp": random.randint(10, 30),
                    "max_hp": random.randint(10, 30),
                    "ac": random.randint(10, 16),
                    "attack_bonus": random.randint(2, 6),
                    "damage": f"{random.randint(1, 3)}d{random.choice([4, 6, 8])}",
                    "dex_mod": random.randint(-1, 3)
                }
                enemies.append(enemy)
            
            # If we found enemies, start combat
            if enemies:
                self.start_combat(enemies)
            # Otherwise create a generic goblin
            elif "combat" in user_input.lower() or "attack" in user_input.lower() or "fight" in user_input.lower():
                enemy_id = str(uuid.uuid4())
                enemy = {
                    "id": enemy_id,
                    "name": "Goblin",
                    "hp": 15,
                    "max_hp": 15,
                    "ac": 13,
                    "attack_bonus": 4,
                    "damage": "1d6+2",
                    "dex_mod": 2
                }
                self.start_combat([enemy])
        
        # Check for combat end
        combat_end_pattern = r"combat ends|defeated|victorious|The battle is over"
        if re.search(combat_end_pattern, response, re.IGNORECASE) and (self.current_combat and self.current_combat.active):
            self.end_combat()
        
        # Check for items received
        item_pattern = r"You (?:receive|got|find|acquire|loot) (?:a|an|the) ([^.!,]+)"
        item_matches = re.finditer(item_pattern, response, re.IGNORECASE)
        
        for match in item_matches:
            item_name = match.group(1).strip()
            if self.campaign.characters:
                self.campaign.characters[0].inventory.append(item_name)
                
        # Check for gold received
        gold_pattern = r"You (?:receive|got|find|acquire|loot) (\d+) gold"
        gold_matches = re.search(gold_pattern, response, re.IGNORECASE)
        
        if gold_matches:
            gold_amount = int(gold_matches.group(1))
            if self.campaign.characters:
                self.campaign.characters[0].gold += gold_amount
                
        # Check for quest updates
        quest_pattern = r"New quest: (.+?)(?:\.|\n)"
        quest_matches = re.search(quest_pattern, response, re.IGNORECASE)
        
        if quest_matches:
            quest_title = quest_matches.group(1).strip()
            self.add_quest(quest_title, f"A quest to {quest_title}")
        
        # Check for NPC introduction
        npc_pattern = r"(?:A|An|The) ([^.!,]+) (?:approaches|greets|welcomes|addresses) you"
        npc_matches = re.finditer(npc_pattern, response, re.IGNORECASE)
        
        for match in npc_matches:
            npc_name = match.group(1).strip()
            found_npc = False
            
            # Check if NPC already exists
            for npc in self.campaign.npcs:
                if npc.name.lower() == npc_name.lower():
                    found_npc = True
                    break
            
            # Add new NPC if not found
            if not found_npc:
                self.add_npc(npc_name, f"An NPC encountered in {self.current_location().name if self.current_location() else 'the world'}")

    def save_to_file(self, filename):
        """Save campaign to file"""
        with open(filename, 'w') as f:
            # Convert campaign to dict
            campaign_dict = asdict(self.campaign)
            json.dump(campaign_dict, f)
    
    def load_from_file(self, filename):
        """Load campaign from file"""
        with open(filename, 'r') as f:
            campaign_dict = json.load(f)
            
            # Convert dict back to Campaign object
            self.campaign = Campaign(**campaign_dict)
            
            # Convert lists of dicts back to proper objects
            characters = []
            for char_dict in self.campaign.characters:
                characters.append(Character(**char_dict))
            self.campaign.characters = characters
            
            npcs = []
            for npc_dict in self.campaign.npcs:
                npcs.append(NPC(**npc_dict))
            self.campaign.npcs = npcs
            
            locations = []
            for loc_dict in self.campaign.locations:
                locations.append(Location(**loc_dict))
            self.campaign.locations = locations
            
            quests = []
            for quest_dict in self.campaign.quests:
                quests.append(Quest(**quest_dict))
            self.campaign.quests = quests

# UI Helpers
def generate_ascii_minimap(dm):
    """Generate a simple ASCII minimap for display"""
    if not dm.campaign or not dm.current_location():
        return "No map available"
    
    current_loc = dm.current_location()
    
    # Create a simple 3x3 grid
    grid = [['.' for _ in range(3)] for _ in range(3)]
    grid[1][1] = 'X'  # Current location in center
    
    # Add connected locations
    connected_loc_names = []
    directions = [(-1, -1), (0, -1), (1, -1), 
                  (-1, 0),          (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]
    
    connected_locs = []
    for loc_id in current_loc.connected_locations:
        for loc in dm.campaign.locations:
            if loc.id == loc_id:
                connected_locs.append(loc)
                
    for i, loc in enumerate(connected_locs[:len(directions)]):
        dx, dy = directions[i]
        x, y = 1 + dx, 1 + dy
        grid[y][x] = loc.name[0].upper()  # First letter of location name
        connected_loc_names.append(f"{loc.name[0].upper()} = {loc.name}")
    
    # Convert grid to string
    map_str = "    N    \n  W + E  \n    S    \n"
    map_str += "---------\n"
    
    for row in grid:
        map_str += "| " + " ".join(row) + " |\n"
        
    map_str += "---------\n"
    map_str += f"X = {current_loc.name} (Current)\n"
    
    for name in connected_loc_names:
        map_str += f"{name}\n"
        
    return map_str

def generate_character_sheet(character):
    """Generate a simple character sheet display"""
    sheet = f"# {character.name}\n"
    sheet += f"Level {character.level} {character.race} {character.character_class}\n"
    sheet += f"HP: {character.hp}/{character.max_hp}  |  XP: {character.xp}  |  Gold: {character.gold}\n\n"
    
    # Stats
    sheet += "## Stats\n"
    for stat, value in character.stats.items():
        mod = (value - 10) // 2
        sheet += f"{stat}: {value} ({'+' if mod >= 0 else ''}{mod})\n"
    
    # Inventory
    sheet += "\n## Inventory\n"
    for item in character.inventory:
        sheet += f"- {item}\n"
    
    # Spells if any
    if character.spells:
        sheet += "\n## Spells\n"
        for spell in character.spells:
            sheet += f"- {spell}\n"
            
    return sheet

def parse_messages_for_display(messages):
    """Parse messages to highlight game elements"""
    formatted_messages = []
    
    for msg in messages:
        if msg['role'] == 'user':
            # User message is displayed as-is
            formatted_messages.append({
                'role': 'user',
                'content': msg['content']
            })
        elif msg['role'] == 'assistant':
            # Process DM responses to highlight sections
            content = msg['content']
            
            # Highlight sections
            content = re.sub(r'\[AVAILABLE ACTIONS\]', '## AVAILABLE ACTIONS', content)
            content = re.sub(r'\[NEARBY LOCATIONS\]', '## NEARBY LOCATIONS', content)
            content = re.sub(r'\[COMBAT INFO\]', '## COMBAT INFO', content)
            
            formatted_messages.append({
                'role': 'assistant',
                'content': content
            })
    
    return formatted_messages

# Streamlit UI
def main():
    st.set_page_config(
        page_title="DungeonLM - Interactive D&D Agent DM",
        page_icon="🎲",
        layout="wide"
    )
    
    # Title and description
    st.title("🎲 DungeonLM - Interactive D&D Agent Dungeon Master")
    st.markdown("""
    Welcome to DungeonLM! This AI-powered Dungeon Master will guide you through 
    an immersive D&D adventure. Create a character, explore the world, and embark on epic quests!
    """)
    
    # Initialize session state
    if 'dungeon_master' not in st.session_state:
        st.session_state.dungeon_master = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'character_created' not in st.session_state:
        st.session_state.character_created = False
    if 'campaign_created' not in st.session_state:
        st.session_state.campaign_created = False
    if 'adventure_started' not in st.session_state:
        st.session_state.adventure_started = False
    
    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("Game Setup")
        
        # LLM provider selection
        provider_tab, campaign_tab, character_tab = st.tabs(["LLM Setup", "Campaign", "Character"])
        
        with provider_tab:
            llm_provider = st.selectbox("Select LLM Provider", ["OpenAI"])
            
            if llm_provider == "OpenAI":
                openai_api_key = st.text_input("OpenAI API Key", type="password")
                model = st.selectbox("Select Model", OPENAI_MODELS)
                
                if st.button("Set OpenAI API"):
                    if openai_api_key:
                        try:
                            llm_client = LLMClient("openai", openai_api_key, model)
                            st.session_state.dungeon_master = DungeonLM(llm_client)
                            st.success("OpenAI API configured successfully!")
                        except Exception as e:
                            st.error(f"Error configuring OpenAI API: {str(e)}")
                    else:
                        st.warning("Please enter an API key")
        
        with campaign_tab:
            if st.session_state.dungeon_master:
                campaign_title = st.text_input("Campaign Title", value="The Lost Artifacts of Eldoria")
                campaign_setting = st.selectbox(
                    "Setting", 
                    ["Fantasy Medieval", "High Fantasy", "Dark Fantasy", "Urban Fantasy", 
                     "Steampunk", "Post-Apocalyptic", "Sci-Fi Fantasy"]
                )
                campaign_theme = st.selectbox(
                    "Theme",
                    ["Epic Quest", "Political Intrigue", "Mystery", "Dungeon Crawl", 
                     "Monster Hunting", "Exploration", "Revenge", "War"]
                )
                
                if st.button("Create Campaign"):
                    st.session_state.dungeon_master.create_campaign(
                        campaign_title, campaign_setting, campaign_theme
                    )
                    st.session_state.campaign_created = True
                    st.success(f"Campaign '{campaign_title}' created!")
            else:
                st.warning("Please configure an LLM provider first")
                
        with character_tab:
            if st.session_state.dungeon_master and st.session_state.campaign_created:
                character_name = st.text_input("Character Name", value="Thalion")
                character_race = st.selectbox(
                    "Race",
                    ["Human", "Elf", "Dwarf", "Halfling", "Half-Elf", "Half-Orc", "Gnome", "Tiefling", "Dragonborn"]
                )
                character_class = st.selectbox(
                    "Class",
                    ["Fighter", "Wizard", "Cleric", "Rogue", "Ranger", "Paladin", "Barbarian", 
                     "Bard", "Druid", "Monk", "Sorcerer", "Warlock"]
                )
                character_level = st.slider("Level", 1, 10, 1)
                
                if st.button("Create Character"):
                    st.session_state.dungeon_master.add_character(
                        character_name, character_race, character_class, character_level
                    )
                    st.session_state.character_created = True
                    st.success(f"Character '{character_name}' created!")
            else:
                st.warning("Please create a campaign first")
        
        # Start adventure button
        if st.session_state.dungeon_master and st.session_state.campaign_created and st.session_state.character_created:
            if st.button("🎮 Start Adventure"):
                # Add starting prompt
                dm = st.session_state.dungeon_master
                
                # Create starting location if none exists
                if not dm.campaign.locations:
                    dm.add_location("Village of Eldoria", 
                                   "A small village nestled in the hills. It has a few buildings including an inn, a blacksmith, and a general store.")
                    dm.add_location("Forest Path", 
                                   "A winding path leading through the dense forest north of the village.")
                    dm.add_location("Town Square", 
                                   "The central gathering place of the village with a well and notice board.")
                    
                    # Connect locations
                    dm.campaign.locations[0].connected_locations = [dm.campaign.locations[1].id, dm.campaign.locations[2].id]
                    dm.campaign.locations[1].connected_locations = [dm.campaign.locations[0].id]
                    dm.campaign.locations[2].connected_locations = [dm.campaign.locations[0].id]
                    
                    # Add an NPC
                    dm.add_npc("Mayor Galwin", "The elderly mayor of Eldoria village with a white beard and kind eyes.", "Friendly")
                
                # Start the adventure with a prompt
                character = dm.campaign.characters[0]
                start_prompt = f"Let's begin our adventure! I am {character.name}, a level {character.level} {character.race} {character.character_class}. Please introduce the setting and give me my first quest."
                response = dm.process_user_input(start_prompt)
                
                # Add to messages for display
                st.session_state.messages.append({"role": "user", "content": start_prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.session_state.adventure_started = True
        
        # Save and load functionality
        if st.session_state.dungeon_master and st.session_state.campaign_created:
            st.subheader("Save/Load Game")
            save_file = st.text_input("Save File Name", "dungeon_lm_save.json")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Game"):
                    try:
                        st.session_state.dungeon_master.save_to_file(save_file)
                        st.success(f"Game saved to {save_file}")
                    except Exception as e:
                        st.error(f"Error saving game: {str(e)}")
            
            with col2:
                if st.button("Load Game"):
                    try:
                        st.session_state.dungeon_master.load_from_file(save_file)
                        st.success(f"Game loaded from {save_file}")
                        st.session_state.campaign_created = True
                        st.session_state.character_created = len(st.session_state.dungeon_master.campaign.characters) > 0
                    except Exception as e:
                        st.error(f"Error loading game: {str(e)}")
        
        # Display character sheet if character exists
        if st.session_state.dungeon_master and st.session_state.character_created:
            st.subheader("Character Sheet")
            if st.session_state.dungeon_master.campaign.characters:
                character = st.session_state.dungeon_master.campaign.characters[0]
                st.markdown(generate_character_sheet(character))
    
    # Main game area
    if st.session_state.dungeon_master and st.session_state.adventure_started:
        # Display message history
        messages = parse_messages_for_display(st.session_state.messages)
        
        # Two-column layout for game
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display chat messages
            for msg in messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])
            
            # Player input
            if user_input := st.chat_input("What do you do? (type your action)"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Get response from DM
                response = st.session_state.dungeon_master.process_user_input(user_input)
                
                # Add DM response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to update UI
                st.rerun()
        
        with col2:
            # Display minimap
            st.subheader("Location Map")
            st.text(generate_ascii_minimap(st.session_state.dungeon_master))
            
            # Display current location info
            current_loc = st.session_state.dungeon_master.current_location()
            if current_loc:
                st.subheader("Current Location")
                st.markdown(f"**{current_loc.name}**")
                st.write(current_loc.description)
            
            # Display NPCs in area
            npcs = st.session_state.dungeon_master.get_location_npcs()
            if npcs:
                st.subheader("NPCs Here")
                for npc in npcs:
                    st.markdown(f"**{npc.name}** ({npc.disposition})")
            
            # Display active quests
            active_quests = [q for q in st.session_state.dungeon_master.campaign.quests if q.status == "Active"]
            if active_quests:
                st.subheader("Active Quests")
                for quest in active_quests:
                    st.markdown(f"**{quest.title}**")
                    if quest.current_step > 0:
                        st.write(f"Progress: {quest.current_step}/{len(quest.steps)}")
            
            # Show combat info if in combat
            if st.session_state.dungeon_master.current_combat and st.session_state.dungeon_master.current_combat.active:
                st.subheader("⚔️ Combat")
                st.markdown("**Initiative Order:**")
                
                # Get initiative order with names
                initiative_order = []
                for entity_id in st.session_state.dungeon_master.current_combat.initiative_order:
                    # Check if it's a character
                    found = False
                    for char in st.session_state.dungeon_master.campaign.characters:
                        if char.id == entity_id:
                            initiative_order.append(f"👤 {char.name}")
                            found = True
                            break
                    
                    # Check if it's an enemy
                    if not found:
                        for enemy in st.session_state.dungeon_master.current_combat.enemies:
                            if enemy["id"] == entity_id:
                                initiative_order.append(f"👹 {enemy['name']}")
                                break
                
                # Mark current turn
                current_idx = st.session_state.dungeon_master.current_combat.current_turn
                for i, entity in enumerate(initiative_order):
                    if i == current_idx:
                        st.markdown(f"➡️ **{entity}** (Current Turn)")
                    else:
                        st.markdown(f"- {entity}")
                
                # Enemy status
                st.markdown("**Enemies:**")
                for enemy in st.session_state.dungeon_master.current_combat.enemies:
                    hp_percent = (enemy["hp"] / enemy["max_hp"]) * 100
                    hp_bar = "🟥" * int(hp_percent // 20) + "⬜" * (5 - int(hp_percent // 20))
                    st.markdown(f"{enemy['name']}: {enemy['hp']}/{enemy['max_hp']} HP {hp_bar}")
                
                # Combat actions helper
                st.markdown("**Common Combat Actions:**")
                st.markdown("- Attack [target]")
                st.markdown("- Cast [spell] at [target]")
                st.markdown("- Use potion/item")
                st.markdown("- Dodge/Disengage")
                
                # Next turn button
                if st.button("Next Turn"):
                    next_entity = st.session_state.dungeon_master.next_turn()
                    st.write(f"Now it's {next_entity}'s turn!")
                    st.rerun()
    else:
        # Welcome screen with instructions
        st.subheader("Getting Started")
        st.markdown("""
        1. 🛠️ Configure an LLM provider in the sidebar (OpenAI)
        2. 🌍 Create a campaign with a title, setting, and theme
        3. 👤 Create your character with a name, race, class, and level
        4. 🎮 Click "Start Adventure" to begin!
        
        The AI Dungeon Master will guide your adventure through a text-based interface.
        You can type commands like:
        - "Explore the forest"
        - "Talk to the innkeeper"
        - "Search for treasure"
        - "Attack the goblin"
        
        The DM will respond with descriptions, dialogue, and options for your next actions!
        """)
        
        st.image("https://placehold.co/600x400?text=DungeonLM+Fantasy+Adventure", caption="Embark on an epic D&D adventure!")

if __name__ == "__main__":
    main()