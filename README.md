# Mobile Crashes RCA Agent

An AI-powered agent for analyzing mobile app crashes using AppDynamics data. This agent automatically identifies patterns, generates reports, and provides actionable recommendations for fixing mobile app crashes.

## Features

- **Automated Crash Data Analysis**: Fetches and analyzes crash data from AppDynamics
- **Pattern Recognition**: Identifies common factors in crashes (device, OS, app version, exception types)
- **Visual Reports**: Generates charts and visualizations of crash patterns
- **Actionable Recommendations**: Provides specific fix recommendations based on crash analysis
- **Impact Evaluation**: Measures the effectiveness of implemented fixes
- **📝 Comprehensive Reporting**: Generates timestamped markdown reports with full analysis
- **📊 Organized Output**: Saves reports in `reports/` and images in `reports/images/`
- **📋 Detailed Logging**: Comprehensive logging system with timestamped log files in `logs/`
- **🔄 Complete Workflow**: End-to-end automation from data fetch to report generation

## Prerequisites

- Python 3.8 or higher
- Valid OpenAI API access (configured for custom endpoint)

## Quick Setup

1. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Verify configuration:**
   Check that `config.env` contains your API credentials:
   ```
   OPENAI_API_KEY=your-api-key
   OPENAI_API_BASE=https://vip.apiyi.com/v1
   DEFAULT_LLM_MODEL=gpt-4o-mini
   ```

4. **Run the agent:**
   ```bash
   python src/rca_agent_langgraph.py
   ```

## Manual Setup

If you prefer to set up manually:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   Update `config.env` with your API credentials

## Usage

The agent follows this workflow:

1. **Fetch Recent Crashes**: Retrieves crash data from AppDynamics
2. **Analyze Patterns**: Identifies common factors and patterns
3. **Generate Report**: Creates structured analysis report
4. **Create Visualizations**: Generates charts highlighting key patterns
5. **Recommend Fixes**: Provides specific recommendations
6. **Evaluate Impact**: Measures fix effectiveness (post-implementation)

### Example Usage

```python
from src.rca_agent_langgraph import create_crash_analysis_agent

# Create the agent
crash_agent = create_crash_analysis_agent()

# Run analysis
result = crash_agent.invoke({
    "messages": [("human", "Analyze the recent mobile app crashes and provide recommendations.")]
})

print(result["messages"][-1].content)
```

## Available Tools

- `fetch_recent_crashes`: Get crash data from AppDynamics
- `analyze_crash_patterns`: Identify patterns in crash data
- `generate_crash_report`: Create structured analysis report
- `visualize_crash_data`: Generate charts and save to `reports/images/`
- `recommend_fixes`: Provide actionable fix recommendations
- `save_markdown_report`: Generate comprehensive timestamped markdown reports
- `evaluate_resolution_impact`: Measure fix effectiveness

## Configuration

The agent uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_BASE`: Custom API endpoint (default: https://vip.apiyi.com/v1)
- `DEFAULT_LLM_MODEL`: LLM model to use (default: gpt-4o-mini)

## Output Files

The agent automatically creates organized directories and generates:

### Directory Structure
```
mobile_crashes_rca_agent/
├── reports/                    # Timestamped markdown reports
│   ├── crash_analysis_report_YYYYMMDD_HHMMSS.md
│   └── images/                 # All charts and visualizations
│       └── crash_analysis_YYYYMMDD_HHMMSS.png
└── logs/                       # Detailed execution logs
    └── rca_agent_YYYYMMDD_HHMMSS.log
```

### Generated Files
- **Markdown Reports**: Comprehensive analysis reports with timestamps
- **Visualization Charts**: Professional charts saved in `reports/images/`
- **Execution Logs**: Detailed logs for debugging and audit trails
- **Console Output**: Real-time progress and summary information

## Evaluation

The agent includes a built-in evaluation framework that measures:
- Pattern detection accuracy (precision/recall)
- Recommendation quality
- Overall performance score

## Troubleshooting

**Import Errors**: Ensure virtual environment is activated and dependencies are installed
**API Errors**: Verify your API key and endpoint in `config.env`
**Permission Errors**: Make sure `setup.sh` is executable (`chmod +x setup.sh`)

## Architecture

- **Agent Core**: LangGraph-based ReAct agent (modern, stable implementation)
- **Data Integration**: AppDynamics API client (currently with mock data)
- **Analysis Engine**: Pattern recognition and statistical analysis
- **Visualization**: Matplotlib-based charting (non-interactive backend)
- **Evaluation**: Performance metrics and quality assessment

### Implementation Details

The project includes two implementations:
- `src/rca_agent.py`: Legacy LangChain AgentExecutor (may have compatibility issues)
- `src/rca_agent_langgraph.py`: **Recommended** LangGraph implementation (stable and working)

Dependencies:
- `langchain` - LLM framework
- `langchain-core` - Core LangChain components  
- `langchain-openai` - OpenAI integration
- `langgraph` - Modern graph-based agent framework
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computing
- `python-dotenv` - Environment variable management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 