# openai-chat

A simple terminal chat application compatible with OpenAI API, supporting streaming responses and reasoning display.

## Features

- 🚀 **Real-time streaming** of tokens as they arrive
- 🧠 **Reasoning display** - Toggle with `/r` or `/reasoning` to see model's thinking process
- 📝 **Multi-line input** support - paste long texts easily
- 🎨 **Clean terminal output** - "Tú:" and "Asistente:" on separate lines with colors
- 📋 **Copy-friendly** - No panel borders around reasoning text
- ✅ **Well tested** - 12 unit tests with pytest
- ✅ **Linted** - Clean code with ruff

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Access to an OpenAI-compatible API (local or remote)

## Installation

```bash
# Clone the repository
git clone https://github.com/aguspiza/openai-chat.git
cd openai-chat

# Install dependencies with uv
uv sync
```

## Usage

Start chat with default settings (connects to `http://127.0.0.1:8079/v1` with Gemma model):

```bash
./chat.py
```

Or specify custom API endpoint and model:

```bash
./chat.py -u https://api.example.com/v1 -m gpt-4 -k your-api-key
```

### Commands

- `/r` or `/reasoning` - Toggle reasoning display on/off
- `/exit`, `/quit`, or `/q` - Exit the chat

### Example with llama-server

If you have a local llama-server running on port 8079:

```bash
# Start llama-server with a model
llama-server -m model.gguf -c 2048

# In another terminal, run the chat
./chat.py
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run linter
uv run ruff check .

# Run tests
uv run pytest test_chat.py -v

# Fix linting issues automatically
uv run ruff check --fix .
```

## How it works

The chat reads streaming responses from any OpenAI-compatible API. When reasoning is enabled (`/r`), it shows the model's reasoning process (if supported by the model) in real-time before displaying the final response.

## License

MIT (or specify your license)
