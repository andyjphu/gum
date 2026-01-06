# System Prompt: to precede all subsequent prompts so as to improve reliability therein by helping LLM understand role, task, vocabulary
SYSTEM_PROMPT = """You are a behavioral state extraction system analyzing sequential screen captures from desktop computing sessions.

## Your Role
Extract structured, machine-readable descriptions of user-computer interactions from visual observations. Your outputs will be used to:
1. Build finite state machine models of user workflows
2. Detect activity transitions (change points)
3. Enable human review of behavioral patterns

## Vocabulary Constraints
You MUST use ONLY terms from these controlled vocabularies:

### Commands (what action is occurring)
INPUT: click, double_click, right_click, type, select, drag, drop, scroll_up, scroll_down, hover
NAVIGATION: zoom_in, zoom_out, switch_tab, switch_window, open, close
SYSTEM_RESPONSE: appear, disappear, loading
PASSIVE: reading, idle

### Widgets (what UI element is involved)
ATOMIC: button, checkbox, radio_button, dropdown, text_field, slider
DISPLAY: icon, image, text, link
CONTAINER: window, dialog, popup, menu, tab, panel, page
FALLBACK: other

### Locations (where in the UI)
SPATIAL: toolbar, menubar, sidebar, statusbar, header, footer, navigation
FUNCTIONAL: search_bar, address_bar, title_bar, content_area, workspace, canvas
CONTAINER: popup, dialog, dropdown_menu, context_menu, tab_bar, ribbon, palette

## Output Principles
1. Report ONLY what is directly observable - never infer intent or goals
2. Use vocabulary terms exactly as specified - no synonyms or variations
3. If uncertain between two terms, choose the more general one
4. "idle" means hands visible but not engaged with input devices
5. "reading" means gaze directed at content with no input activity
6. Always output valid JSON matching the requested schema"""


# From SeeAction + GUI automation tools
COMMAND_VOCABULARY = [
    # User Input Actions
    "click",          # Single mouse click
    "double_click",   # Double mouse click  
    "right_click",    # Context menu click
    "type",           # Keyboard text entry
    "select",         # Text/item selection
    "drag",           # Click-hold-move-release
    "drop",           # Release after drag
    "scroll_up",      # Scroll wheel up
    "scroll_down",    # Scroll wheel down
    "hover",          # Mouse over without click
    
    # View/Navigation Actions  
    "zoom_in",
    "zoom_out",
    "switch_tab",
    "switch_window",
    "open",
    "close",
    
    # System Responses (observable outcomes)
    "appear",         # UI element appears
    "disappear",      # UI element disappears
    "loading",        # Progress indicator visible
    
    # Passive States
    "reading",        # Eyes on screen, no input
    "idle",           # No observable activity
]

WIDGET_VOCABULARY = [
    # Atomic Input Widgets
    "button",
    "checkbox", 
    "radio_button",
    "dropdown",
    "text_field",
    "slider",
    
    # Display Widgets
    "icon",
    "image", 
    "text",
    "link",
    
    # Container Widgets
    "window",
    "dialog",
    "popup",
    "menu",
    "tab",
    "panel",
    "page",
    
    # Other
    "other",
]

LOCATION_VOCABULARY = [
    # Spatial positions
    "toolbar", "menubar", "sidebar", "statusbar",
    "header", "footer", "navigation",
    
    # Functional areas
    "search_bar", "address_bar", "title_bar",
    "content_area", "workspace", "canvas",
    
    # Container references
    "popup", "dialog", "dropdown_menu", "context_menu",
    "tab_bar", "ribbon", "palette",
]


TRANSCRIPTION_PROMPT = """Transcribe in markdown ALL observable content from the provided visual input.

NEVER SUMMARIZE. 
1. Transcribe everything EXACTLY as observed, word for word 
2. With a description of where the text was observed
- Example: `top left: HELLO WORLD`


## Capture

**Digital (screens, displays):**
- Application names, window titles, file paths, URLs
- All visible text: menus, dialogs, notifications, content
- Active/focused elements and cursor position

**Physical (real-world, egocentric):**
- Visible text: signs, labels, documents, screens
- Objects being interacted with
- Hands and current action being performed

Output a single structured markdown transcription of the current state."""


ATOMIC_PROMPT = """Analyze the provided image and extract the atomic-level interaction state.

## Task
Identify the single most prominent user-computer interaction occurring in this frame.
This is the LOWEST level of the activity hierarchy - primitive actions at the input device level.

## Output Schema
Return a JSON object:
```json
{{
  "command": "<term from COMMAND vocabulary>",
  "widget": "<term from WIDGET vocabulary or null if no widget involved>",
  "location": "<term from LOCATION vocabulary or null if not determinable>",
  "confidence": "<high|medium|low>",
  "evidence": "<brief description of visual evidence supporting classification>"
}}
```

## Decision Rules
1. If cursor is visible over a clickable element with pressed appearance → "click"
2. If text cursor (I-beam) is active in a field → "type"
3. If highlighted/selected region exists → "select"
4. If content position has shifted vertically → "scroll_up" or "scroll_down"
5. If no input activity but eyes on content → "reading"
6. If no observable interaction → "idle"
7. For system-initiated changes (popups, loading) → "appear", "disappear", or "loading"

## Examples
- Cursor on blue highlighted button → {{"command": "click", "widget": "button", "location": "toolbar", ...}}
- Text cursor blinking in form → {{"command": "type", "widget": "text_field", "location": "dialog", ...}}
- Mouse pointer over menu item → {{"command": "hover", "widget": "menu", "location": "menubar", ...}}
- No hands on keyboard/mouse, looking at code → {{"command": "reading", "widget": "text", "location": "content_area", ...}}

Analyze the image now and return ONLY the JSON object."""


ACTIVITY_PROMPT = """Analyze the provided sequence of {n_images} images (captured at {interval}-second intervals) and extract the activity-level state for each frame.

## Task
For each image, identify the MID-LEVEL activity being performed. Activities are composed of atomic actions and represent coherent units of work (e.g., "editing code", "browsing documentation", "configuring settings").

## Output Schema
Return a JSON array with one object per image:
```json
[
  {{
    "image": 1,
    "command": "<primary action from COMMAND vocabulary>",
    "widget": "<primary widget from WIDGET vocabulary>",
    "location": "<location from LOCATION vocabulary>",
    "application": "<application name visible in title bar or inferred>",
    "activity_label": "<short verb phrase describing the activity>",
    "details": "<observable specifics: filename, URL, menu item, etc.>"
  }}
]
```

## Activity Labeling Guidelines
Compose activity_label as: [verb] + [object] + [optional context]
- "editing code in models.py"
- "reading Stack Overflow answer"
- "navigating file browser"
- "configuring VS Code settings"
- "reviewing pull request diff"
- "composing email reply"
- "searching documentation"

## Transition Detection
Note when the activity CHANGES between consecutive images:
- Application switch (window title change)
- Context switch (same app, different file/page)
- Mode switch (same context, different action type)

If the activity is IDENTICAL across consecutive images, use the same activity_label.

## Rules
1. One activity_label per image - choose the dominant activity if multiple are occurring
2. activity_label should be human-readable, not just vocabulary terms
3. details should capture specifics that distinguish this instance (filenames, search queries, etc.)
4. If a frame shows a transition mid-action, describe the END state

Analyze all {n_images} images and return ONLY the JSON array."""