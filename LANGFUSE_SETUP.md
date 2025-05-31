# Langfuse Observability Setup Guide

This guide helps you set up comprehensive LLM observability for the Mobile Crashes RCA Agent using Langfuse.

## 🎯 What is Langfuse?

Langfuse is an open-source LLM observability platform that provides:
- **Request Monitoring**: Track all LLM API calls and responses
- **Performance Analytics**: Monitor token usage, latency, and costs
- **Trace Visualization**: See complete execution flows
- **Error Tracking**: Identify and debug issues
- **Quality Scoring**: Evaluate AI output quality

## 🚀 Quick Start

### 1. Automated Setup (Recommended)

```bash
# Make setup script executable
chmod +x docker-setup.sh

# Run automated setup
./docker-setup.sh
```

This script will:
- Check Docker installation
- Create environment file
- Start all services (PostgreSQL, Langfuse, Agent)
- Provide setup instructions

### 2. Manual Setup

```bash
# 1. Copy environment template
cp env.template .env

# 2. Edit .env with your OpenAI API key
nano .env  # or your preferred editor

# 3. Start all services
docker-compose up -d

# 4. Check status
docker-compose ps
```

## 🔧 Configuration Steps

### Step 1: Access Langfuse UI
1. Open http://localhost:3000 in your browser
2. Complete the initial setup (create admin account)
3. Create a new project for your mobile crash agent

### Step 2: Get API Keys
1. In Langfuse, go to **Settings** → **API Keys**
2. Copy the **Public Key** and **Secret Key**
3. Update your `.env` file:
   ```env
   LANGFUSE_PUBLIC_KEY=lf-pk-your-actual-public-key
   LANGFUSE_SECRET_KEY=lf-sk-your-actual-secret-key
   ```

### Step 3: Restart Agent
```bash
docker-compose restart mobile-rca-agent
```

## 🔍 Using Observability

### Example: Running with Observability

```bash
# Run the example with observability
cd examples
python langfuse_integration_example.py
```

This will:
1. Analyze sample crash data
2. Track all LLM calls and agent steps
3. Create traces in Langfuse
4. Add quality scores

### View Results in Langfuse

1. Open http://localhost:3000
2. Navigate to **Traces** to see execution flows
3. Click on a trace to see detailed steps
4. View **Analytics** for performance metrics

## 📊 What You'll See in Langfuse

### Traces
- **Session Traces**: Complete crash analysis workflows
- **LLM Generations**: Individual GPT API calls with inputs/outputs
- **Spans**: Agent steps like preprocessing, pattern analysis
- **Performance Data**: Duration, token usage, costs

### Analytics
- **Cost Tracking**: API usage costs over time
- **Performance Metrics**: Response times and token consumption
- **Error Rates**: Success/failure rates
- **Quality Scores**: AI-evaluated analysis quality

## 🛠️ Advanced Configuration

### Environment Variables

```env
# Langfuse Configuration
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key

# Observability Features
TRACE_LLM_CALLS=true
TRACE_AGENT_STEPS=true
TRACK_TOKEN_USAGE=true
TRACK_COSTS=true

# Session Configuration
SESSION_ID=custom-session-id
USER_ID=your-user-id
```

### Adding Observability to Your Code

```python
from src.observability import LangfuseHandler, ObservabilityConfig
from src.observability.decorators import trace_llm_call, trace_agent_step

# Initialize
obs_handler = LangfuseHandler(ObservabilityConfig.from_env())

# Trace a complete session
with obs_handler.trace_session("my_analysis") as trace:
    result = my_analysis_function()
    obs_handler.add_score("quality", 0.85)

# Use decorators for automatic tracing
@trace_llm_call(model="gpt-4o-mini")
def my_llm_function():
    # Your LLM call here
    pass

@trace_agent_step(name="data_processing")
def process_data():
    # Your processing logic here
    pass
```

## 🔧 Troubleshooting

### Common Issues

**1. Langfuse not accessible**
```bash
# Check if services are running
docker-compose ps

# Check logs
docker-compose logs langfuse-server
```

**2. Database connection issues**
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check PostgreSQL logs
docker-compose logs postgres
```

**3. Agent not connecting to Langfuse**
```bash
# Verify environment variables
docker-compose exec mobile-rca-agent env | grep LANGFUSE

# Check agent logs
docker-compose logs mobile-rca-agent
```

### Reset Everything

```bash
# Stop all services
docker-compose down

# Remove data (⚠️ This deletes all traces!)
docker-compose down -v

# Start fresh
docker-compose up -d
```

## 📈 Production Considerations

### Security
- Use environment-specific API keys
- Set up proper authentication
- Configure network security

### Performance
- Monitor resource usage
- Configure retention policies
- Set up log rotation

### Scaling
- Use external PostgreSQL for production
- Consider Langfuse Cloud for managed solution
- Implement proper backup strategies

## 🔗 Useful Links

- **Langfuse Documentation**: https://langfuse.com/docs
- **Langfuse GitHub**: https://github.com/langfuse/langfuse
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **Mobile RCA Agent README**: ./README.md

## 🎉 Next Steps

1. **Explore the UI**: Navigate through traces and analytics
2. **Customize Tracking**: Add more observability to your agent
3. **Set Alerts**: Configure notifications for errors or performance issues
4. **Optimize Performance**: Use insights to improve your agent
5. **Scale Up**: Move to production with proper configuration

Happy observing! 🔍✨ 