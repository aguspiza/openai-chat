#!/usr/bin/env python3
"""
Chat por terminal compatible con OpenAI API.
Uso: python chat.py -u http://127.0.0.1:8079/v1

Características:
- Streaming de tokens en tiempo real
- /reasoning: mostrar/ocultar zona de razonamiento
- Soporte para modelos con razonamiento (como Gemma)
"""

import argparse
import sys
from openai import OpenAI
from rich.console import Console

console = Console()
show_reasoning = False

def parse_args():
    parser = argparse.ArgumentParser(description="Chat por terminal compatible con OpenAI API")
    parser.add_argument("-u", "--url", default="http://127.0.0.1:8079/v1", 
                        help="URL de la API compatible con OpenAI (default: http://127.0.0.1:8079/v1)")
    parser.add_argument("-k", "--api-key", default="dummy", help="API key (default: dummy)")
    parser.add_argument("-m", "--model", default="unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M", 
                        help="Modelo a usar (default: unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M)")
    return parser.parse_args()

def read_multiline_input(prompt_text="Tú:"):
    """Lee input que puede ser multi-línea.
    
    Si se pega texto rápido, lee todas las líneas.
    Si se escribe lento, es como input() normal.
    Enter vacío + Enter termina la entrada multi-línea.
    """
    print(f"\033[1;34m{prompt_text}\033[0m")
    lines = []
    
    try:
        while True:
            line = input()
            
            # Si la línea está vacía y ya tenemos algo, terminar
            if line == "" and lines:
                break
            # Si la línea está vacía y no tenemos nada, ignorar
            elif line == "" and not lines:
                continue
            
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        pass
    
    return "\n".join(lines)


def main():
    global show_reasoning
    
    args = parse_args()
    
    client = OpenAI(
        base_url=args.url,
        api_key=args.api_key
    )
    
    messages = []
    
    console.print("[bold green]Chat iniciado[/bold green]")
    console.print(f"[dim]API: {args.url}[/dim]")
    console.print(f"[dim]Modelo: {args.model}[/dim]")
    console.print("[dim]/r o /reasoning: toggle razonamiento | /exit: salir[/dim]\n")
    
    try:
        while True:
            # Leer input del usuario (soporta múltiples líneas)
            try:
                user_input = read_multiline_input("Tú:")
            except (EOFError, KeyboardInterrupt):
                break
            
            if not user_input.strip():
                continue
            
            # Comandos especiales
            if user_input.lower() in ['/exit', '/quit', '/q']:
                break
            elif user_input.lower() in ['/reasoning', '/r']:
                show_reasoning = not show_reasoning
                console.print(f"[dim]Razonamiento: {'ON' if show_reasoning else 'OFF'}[/dim]\n")
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            # Preparar para mostrar respuesta
            print("\033[1;36mAsistente:\033[0m")
            sys.stdout.flush()
            
            answer_text = []
            reasoning_text = []
            in_reasoning_phase = False
            in_response_phase = False
            
            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=True
                )
                
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    
                    # Razonamiento llegando
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        if show_reasoning and not in_reasoning_phase:
                            print("\n\033[33m═══ Razonamiento ═══\033[0m")
                            in_reasoning_phase = True
                        if show_reasoning:
                            sys.stdout.write(delta.reasoning_content)
                            sys.stdout.flush()
                        reasoning_text.append(delta.reasoning_content)
                    
                    # Respuesta llegando
                    if delta.content:
                        if show_reasoning and in_reasoning_phase and not in_response_phase:
                            print("\n\033[32m═══ Respuesta ═══\033[0m")
                            in_response_phase = True
                        content = delta.content
                        answer_text.append(content)
                        sys.stdout.write(content)
                        sys.stdout.flush()
                
                sys.stdout.write("\n")
                sys.stdout.flush()
                
                # Guardar respuesta en el historial
                full_answer = "".join(answer_text)
                if full_answer:
                    messages.append({"role": "assistant", "content": full_answer})
                
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
            
            console.print()
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Saliendo...[/bold yellow]")
    
    console.print("[bold green]Chat finalizado[/bold green]")

if __name__ == "__main__":
    main()
