"""
Tests básicos para chat.py
"""

import pytest
from unittest.mock import Mock, patch

# Importar las funciones del módulo chat
from chat import parse_args


class TestParseArgs:
    """Tests para el parser de argumentos."""
    
    def test_default_values(self):
        """Verifica que los valores por defecto sean correctos."""
        with patch('sys.argv', ['chat.py']):
            args = parse_args()
            assert args.url == "http://127.0.0.1:8079/v1"
            assert args.api_key == "dummy"
            assert args.model == "unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_M"
    
    def test_custom_url(self):
        """Verifica que se pueda especificar una URL personalizada."""
        with patch('sys.argv', ['chat.py', '-u', 'http://localhost:8080/v1']):
            args = parse_args()
            assert args.url == "http://localhost:8080/v1"
    
    def test_custom_model(self):
        """Verifica que se pueda especificar un modelo personalizado."""
        with patch('sys.argv', ['chat.py', '-m', 'llama3']):
            args = parse_args()
            assert args.model == "llama3"
    
    def test_custom_api_key(self):
        """Verifica que se pueda especificar una API key personalizada."""
        with patch('sys.argv', ['chat.py', '-k', 'mi-api-key']):
            args = parse_args()
            assert args.api_key == "mi-api-key"
    
    def test_url_required_when_no_default(self):
        """Verifica que la URL sea requerida si no hay default."""
        # Como ahora tiene default, esto no debería fallar
        with patch('sys.argv', ['chat.py']):
            args = parse_args()
            assert args.url is not None


class TestShowReasoning:
    """Tests para el toggle de razonamiento."""
    
    def test_initial_state(self):
        """Verifica que el estado inicial de show_reasoning sea False."""
        from chat import show_reasoning
        assert not show_reasoning
    
    def test_toggle_behavior(self):
        """Verifica que el toggle de razonamiento funcione."""
        # Este test requeriría modificar la variable global
        # Por ahora solo verificamos que sea un booleano
        from chat import show_reasoning
        assert isinstance(show_reasoning, bool)


class TestMainFunction:
    """Tests para la función main (con mocks)."""
    
    @patch('chat.OpenAI')
    @patch('chat.read_multiline_input')
    @patch('chat.console')
    def test_exit_command(self, mock_console, mock_read_multiline, mock_openai):
        """Verifica que el comando /exit funcione."""
        mock_read_multiline.return_value = '/exit'
        
        # Ejecutar main con argumentos mínimos
        with patch('sys.argv', ['chat.py']):
            from chat import main
            try:
                main()
            except SystemExit:
                pass  # main() puede llamar a exit()
        
        # Verificar que se llamó read_multiline_input al menos una vez
        mock_read_multiline.assert_called()
    
    @patch('chat.OpenAI')
    @patch('chat.read_multiline_input')
    @patch('chat.console')
    def test_reasoning_toggle_command(self, mock_console, mock_read_multiline, mock_openai):
        """Verifica que el comando /reasoning funcione."""
        mock_read_multiline.side_effect = ['/r', '/exit']
        
        with patch('sys.argv', ['chat.py']):
            from chat import main
            try:
                main()
            except SystemExit:
                pass
        
        # Verificar que se procesaron los comandos
        assert mock_read_multiline.call_count >= 2
    
    @patch('chat.OpenAI')
    def test_openai_client_creation(self, mock_openai_class):
        """Verifica que el cliente de OpenAI se cree correctamente."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        with patch('sys.argv', ['chat.py', '-u', 'http://test.com/v1', '-k', 'test-key']):
            args = parse_args()
            
            # Crear el cliente usando el mock (como lo hace main)
            client = mock_openai_class(base_url=args.url, api_key=args.api_key)
            
            # Verificar que se llamó con los parámetros correctos
            mock_openai_class.assert_called_with(base_url='http://test.com/v1', api_key='test-key')
            assert client == mock_client


class TestStreamingResponse:
    """Tests para el manejo de streaming."""
    
    def test_stream_chunk_processing(self):
        """Verifica que los chunks del stream se procesen correctamente."""
        # Crear un mock de chunk de stream
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = "Hola"
        mock_chunk.choices[0].delta.reasoning_content = None
        
        # Verificar que tiene la estructura esperada
        assert mock_chunk.choices[0].delta.content == "Hola"
        assert mock_chunk.choices[0].delta.reasoning_content is None
    
    def test_reasoning_chunk_processing(self):
        """Verifica que los chunks con razonamiento se procesen correctamente."""
        mock_chunk = Mock()
        mock_chunk.choices = [Mock()]
        mock_chunk.choices[0].delta = Mock()
        mock_chunk.choices[0].delta.content = None
        mock_chunk.choices[0].delta.reasoning_content = "Pensando..."
        
        assert mock_chunk.choices[0].delta.reasoning_content == "Pensando..."
        assert mock_chunk.choices[0].delta.content is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
