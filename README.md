# Mobile Crashes RCA Agent

An AI-powered mobile application crash analysis agent using LangGraph and OpenAI GPT models. This tool automatically analyzes crash data, identifies patterns, and provides actionable recommendations for fixing mobile app stability issues.

## 🏗️ Architecture

This project follows a modular, generative AI architecture:

```
mobile_crashes_rca_agent/
├── config/                    # Configuration files
│   ├── __init__.py
│   ├── model_config.yaml     # LLM model settings
│   ├── prompt_templates.yaml # Prompt templates
│   └── logging_config.yaml   # Logging configuration
├── src/                      # Source code
│   ├── llm/                  # LLM client implementations
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract base class
│   │   ├── gpt_client.py    # OpenAI GPT client
│   │   ├── claude_client.py # Anthropic Claude client
│   │   └── utils.py         # LLM utilities
│   ├── prompt_engineering/   # Prompt engineering tools
│   │   ├── __init__.py
│   │   ├── templates.py     # Template management
│   │   ├── few_shot.py      # Few-shot learning
│   │   └── chainer.py       # Prompt chaining
│   ├── utils/               # Utility modules
│   │   ├── __init__.py
│   │   ├── logger.py        # Logging utilities
│   │   ├── rate_limiter.py  # API rate limiting
│   │   ├── token_counter.py # Token counting
│   │   └── cache.py         # Caching system
│   ├── handlers/            # Error and event handlers
│   │   ├── __init__.py
│   │   └── error_handler.py # Error handling
│   └── mobile_crash_agent.py # Main agent implementation
├── data/                    # Data directories
│   ├── cache/              # Cached responses
│   ├── prompts/            # Prompt storage
│   ├── outputs/            # Analysis outputs
│   └── embeddings/         # Vector embeddings
├── examples/               # Usage examples
│   ├── basic_completion.py
│   ├── chat_session.py
│   └── chain_prompts.py
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── README.md              # This file
└── Dockerfile             # Container configuration
```

## 🚀 Features

- **Automated Crash Analysis**: Fetches and analyzes mobile app crash data
- **Pattern Recognition**: Identifies trends across devices, OS versions, and app versions
- **Root Cause Analysis**: Uses AI to determine likely causes of crashes
- **Actionable Recommendations**: Provides specific, prioritized fix suggestions
- **Quality Evaluation**: LLM-powered assessment of analysis quality
- **Visualization**: Generates charts and workflow diagrams
- **LLM Observability**: Integrated Langfuse for comprehensive monitoring and tracing
- **Performance Tracking**: Monitor token usage, latency, and costs
- **Production-Ready**: Docker-based deployment with observability stack
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Configuration-Driven**: YAML-based configuration for easy customization
- **Caching System**: Intelligent caching to reduce API costs
- **Error Handling**: Comprehensive error handling and retry mechanisms

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

## 🛠️ Installation

### Option 1: Standard Installation

```bash
# Clone the repository
git clone https://github.com/your-org/mobile-crashes-rca-agent.git
cd mobile-crashes-rca-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Docker Installation with Langfuse Observability

```bash
# Quick setup with automated script
chmod +x docker-setup.sh
./docker-setup.sh

# Or manual setup:
# 1. Copy environment template
cp env.template .env

# 2. Edit .env with your configuration
# Add your OPENAI_API_KEY and other settings

# 3. Start all services (PostgreSQL, Langfuse, and the Agent)
docker-compose up -d

# 4. Access Langfuse at http://localhost:3000
# Create a project and copy the keys to .env

# 5. Restart the agent with new keys
docker-compose restart mobile-rca-agent
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp env.template .env
```

Edit `.env` with your configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_API_BASE=https://api.openai.com/v1
DEFAULT_LLM_MODEL=gpt-4o-mini

# Langfuse Configuration (for observability)
LANGFUSE_SECRET_KEY=lf-sk-your-secret-key-here
LANGFUSE_PUBLIC_KEY=lf-pk-your-public-key-here
LANGFUSE_HOST=http://localhost:3000

# Development Configuration
DEBUG=true
LOG_LEVEL=INFO
```

### Configuration Files

- `config/model_config.yaml`: LLM model settings and API configuration
- `config/prompt_templates.yaml`: Prompt templates for different analysis tasks
- `config/logging_config.yaml`: Logging configuration and output settings

## 🔍 Observability with Langfuse

This project includes comprehensive observability using Langfuse, an open-source LLM observability platform. Langfuse helps you:

- **Monitor LLM Interactions**: Track all LLM API calls, inputs, outputs, and performance
- **Trace Agent Workflows**: See the complete execution flow of your crash analysis
- **Performance Monitoring**: Monitor token usage, latency, and costs in real-time
- **Error Tracking**: Identify and debug issues in production
- **Quality Assessment**: Score and evaluate the quality of AI-generated analyses

### Accessing Langfuse

1. **Start the observability stack**: `docker-compose up -d`
2. **Open Langfuse UI**: http://localhost:3000
3. **Create a project** and copy the API keys to your `.env` file
4. **Restart the agent**: `docker-compose restart mobile-rca-agent`
5. **Run analyses** and view traces in real-time

### Key Observability Features

- **Session Tracing**: Each crash analysis creates a complete trace
- **LLM Call Monitoring**: Track all GPT interactions with input/output
- **Performance Metrics**: Response times, token usage, and costs
- **Error Logging**: Automatic error capture and reporting
- **Quality Scores**: AI-powered quality evaluation of analyses

### Example Integration

```python
from src.observability import LangfuseHandler, ObservabilityConfig
from src.observability.decorators import trace_llm_call, trace_agent_step

# Initialize observability
obs_handler = LangfuseHandler(ObservabilityConfig.from_env())

# Trace an entire analysis session
with obs_handler.trace_session("mobile_crash_analysis") as trace:
    # Your crash analysis code here
    result = analyze_crash_data(crash_data)
    
    # Add quality score
    obs_handler.add_score("analysis_quality", 0.85)
```

## 🎯 Usage

### Basic Usage

```python
from src.mobile_crash_agent import create_crash_analysis_agent

# Create and run the agent
agent = create_crash_analysis_agent()
result = agent.run_analysis()

if result["success"]:
    print("✅ Analysis completed successfully!")
    print(f"Results: {result['result']}")
else:
    print(f"❌ Analysis failed: {result['error']}")
```

### Command Line

```bash
# Run the main analysis workflow
python -m src.mobile_crash_agent

# Or use the console script (after installation)
mobile-rca
```

### Examples

```bash
# Basic completion example
python examples/basic_completion.py

# Interactive chat session
python examples/chat_session.py

# Prompt chaining example
python examples/chain_prompts.py
```

## 🔧 Development

### Project Structure

The project follows a clean architecture with:

- **Configuration Layer**: YAML-based configuration management
- **LLM Layer**: Abstracted LLM client implementations
- **Prompt Engineering**: Template management and chaining
- **Utilities**: Logging, caching, rate limiting, token counting
- **Error Handling**: Centralized error management
- **Main Agent**: Core business logic and workflow orchestration

### Adding New LLM Providers

1. Create a new client class inheriting from `BaseLLMClient`
2. Implement required methods (`complete`, `chat`, `validate_connection`)
3. Add configuration to `config/model_config.yaml`
4. Update `src/llm/utils.py` to include the new provider

### Extending Functionality

1. **New Tools**: Add tools to `src/mobile_crash_agent.py`
2. **New Templates**: Add templates to `config/prompt_templates.yaml`
3. **New Chains**: Create chains in `src/prompt_engineering/chainer.py`

## 📊 Output

The agent generates:

- **Markdown Reports**: Comprehensive analysis reports in `data/outputs/`
- **Visualizations**: Charts and diagrams in `data/outputs/images/`
- **Logs**: Detailed execution logs in `logs/YYYYMMDD/`
- **Cache**: Cached responses in `data/cache/`

## 🐳 Docker Commands Quick Reference

```bash
# Setup everything (automated)
./docker-setup.sh

# Manual commands
docker-compose up -d              # Start all services
docker-compose down               # Stop all services
docker-compose logs -f [service]  # View logs
docker-compose restart [service]  # Restart a service
docker-compose pull               # Update images
docker-compose up --build         # Rebuild and start

# Access points
# Langfuse UI: http://localhost:3000
# Agent API: http://localhost:8000
# PostgreSQL: localhost:5432 (langfuse/langfuse)
```

## 🧪 Testing

```bash
# Run basic tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific example
python examples/basic_completion.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the project root
2. **API Errors**: Check your OpenAI API key and rate limits
3. **Configuration Errors**: Validate YAML syntax in config files
4. **Permission Errors**: Ensure write permissions for `logs/` and `data/` directories

### Debug Mode

Enable debug logging by setting the log level in `config/logging_config.yaml`:

```yaml
loggers:
  RCA_Agent:
    level: DEBUG
```

## 📈 Performance

- **Caching**: Reduces API calls by caching responses
- **Rate Limiting**: Prevents API rate limit violations
- **Token Optimization**: Efficient prompt engineering to minimize costs
- **Parallel Processing**: Concurrent tool execution where possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- LangChain team for the framework
- LangGraph for agent orchestration
- Contributors and maintainers

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information
4. Contact the development team

---

**Built with ❤️ for mobile app stability** 