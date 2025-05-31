"""
Chat session example using the refactored Mobile Crashes RCA Agent.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.llm import get_llm_client
from src.utils import setup_logging
from src.prompt_engineering import CrashAnalysisTemplates

def main():
    """Interactive chat session example."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting chat session example")
    
    # Get LLM client
    llm_client = get_llm_client('openai')
    
    # Get system prompt
    system_template = CrashAnalysisTemplates.get_system_prompt()
    system_prompt = system_template.template
    
    print("🤖 Mobile Crash Analysis Chat Session")
    print("=" * 50)
    print("Type 'quit' to exit")
    print()
    
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Build conversation with system prompt
            messages = llm_client.format_messages(user_input, system_prompt)
            
            # Add conversation history
            for msg in conversation_history:
                messages.insert(-1, msg)  # Insert before the latest user message
            
            response = llm_client.chat(messages)
            
            print(f"🤖 Assistant: {response}")
            print()
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})
            
            # Keep only last 10 messages to avoid token limits
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            print(f"❌ Error: {e}")
            print()

if __name__ == "__main__":
    main() 