TRANSCRIPTION_PROMPT = """Transcribe in markdown ALL observable content from the provided visual input.

NEVER SUMMARIZE. Transcribe everything EXACTLY as observed, word for word.

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

SUMMARY_PROMPT = """Describe the actions occurring across the provided images, presented in chronological order.

## Guidelines

- Refer to individuals as "the person" or by observable role (e.g., "the driver", "the presenter") rather than assuming a single "user"
- Describe observable actions only—avoid inferring intent, beliefs, or attention
- For screen content: distinguish between what is displayed vs. what is being actively manipulated
- For physical actions: describe what hands/body are doing and objects being interacted with

## Output

Provide concise bullet points describing specific, observable actions across the sequence."""