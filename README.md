<p align="center">
  <img src="https://imgur.com/iQ4YTRf.gif" alt="DungeonLM Banner" width="700"/>
</p>

<h1 align="center">🎲 DungeonLM - Your AI Dungeon Master</h1>

---

## 🌟 Features

<p align="center">
  <img src="https://media.giphy.com/media/f9k1tV7HyORcngKF8v/giphy.gif" alt="Features Animation" width="400"/>
</p>

- 🧙 **AI Dungeon Master** — A GPT-powered DM that builds your world, quests, and encounters in real time.
- 🧝 **Custom Characters** — Choose your race, class, and level. Auto-stats, inventory, and spells included.
- 🗺️ **Dynamic World** — NPCs, quests, and locations spawn based on your choices.
- ⚔️ **Turn-Based Combat** — With initiative tracking, enemy HP, and battle narration.
- 💾 **Save and Resume** — Pause anytime and pick up your journey later.
- 🎮 **Streamlit UI** — A modern, real-time interface with zero setup.

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- OpenAI API Key

### 📦 Installation

```bash
git clone https://github.com/yourusername/dungeonlm.git
cd dungeonlm
pip install -r requirements.txt
```

### 🔐 Set Your OpenAI API Key

Either input in the app sidebar or:

```bash
export OPENAI_API_KEY=your-api-key
```

### ▶️ Launch the App

```bash
streamlit run dungeonlm.py
```

---

## 🧠 How to Play

<p align="center">
  <img src="https://media.giphy.com/media/3o7abB06u9bNzA8lu8/giphy.gif" alt="How to Play" width="400"/>
</p>

1. **Select Provider** → Use OpenAI and choose your model (gpt-4o, gpt-4-turbo, etc.)
2. **Create Campaign** → Set a title, setting, and theme.
3. **Build Character** → Choose name, race, class, and level.
4. **Start Adventure** → Let the AI narrate, challenge, and surprise you.
5. **Combat** → Initiative, turns, strategy — full D&D-style battles.
6. **Save Anytime** → Store your campaign in one click and resume later.

---

## 🌐 World Mechanics

<p align="center">
  <img src="https://imgur.com/Vy7Cs1K.gif" alt="World Building" width="450"/>
</p>

- Automatically creates and connects locations like forests, villages, caves, etc.
- NPCs are dynamic with names, traits, and attitudes (friendly, hostile, neutral).
- ASCII minimap shows your current position and surroundings.
- Quest system tracks tasks, rewards, and status (active, completed).

---

## 🔧 Tech Stack

<p align="center">
  <img src="https://media.giphy.com/media/5VKbvrjxpVJCM/giphy.gif" alt="Tech Stack" width="400"/>
</p>

- 🧠 **LLM**: OpenAI GPT (gpt-4o, gpt-3.5, etc.)
- 🖥️ **Frontend**: Streamlit
- 🐍 **Backend**: Python (all-in-one script)
- 🧳 **Storage**: JSON Save Files
- 🎨 **Rendering**: ASCII-style minimaps and battle displays

---

## 📁 Project Structure

```bash
dungeonlm.py          # All logic and UI
README.md             # You're reading it!
requirements.txt      # Dependencies
```

---

## 💡 Ideas for Expansion

- 🗣️ Add voice commands or text-to-speech narration
- 👥 Expand to support co-op multiplayer
- 🧙 Add magic schools, item crafting, or divine blessings
- 🛡️ Visualize characters with avatar generators
- 💬 Include memory or journaling for previous choices

---

> 🎲 DungeonLM brings the magic of tabletop roleplaying to your screen — driven by imagination and AI.
